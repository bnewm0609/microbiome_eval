import json
import random

from microbiome_eval.tasks.base import BaseTask, PROJ_PATH
from microbiome_eval.llm import LLM, wait_for_vllm

class DiseaseClassificationTask(BaseTask):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--taxa", default="genus", choices=["genus", "species"], help="Which taxonomy level to use. 2075/2120 have genus taxonomies. 512/2120 have species-level.")
        parser.add_argument("--include_choices", action="store_true", help="Give model the 200 disease options.")
        parser.add_argument("--judge_model", default="Qwen/Qwen-32B", help="Model to use for judging the correctness of the model's responses. Should be a local vLLM model to avoid excessive costs.")
        parser.add_argument("--judge_generation_kwargs", default="{}", help="Generation kwargs to use for the judge model calls, e.g. --judge_generation_kwargs '{\"temperature\": 0.7, \"max_tokens\": 512}'")

    def get_prompts(self) -> list[dict]:
        """
        Returns a list of prompts for the disease classification task.
        Each prompt is a dict with the following keys:
        - "prompt": The prompt to send to the model
        - "label": The correct label for the prompt

        Can download the dataset from:
        
        """
        # load the dataset
        with open(PROJ_PATH / f"data/gmrepo/hf_dataset/gmrepo_eval.jsonl") as f:
            dataset = [json.loads(line) for line in f]

        if self.config["taxa"] == "genus":
            prompt_key = "prompt_genus"
        else:
            prompt_key = "prompt_species"

        system_message = "You are a helpful and precise assistant for answering medical diagnostic questions. The last line of your answer should be a short word or phrase corresponding to the disease you think is most likely given the patient's gut microbiome taxonomy."

        prompt_template = "{prompt}"
        if self.config.get("include_choices", False):
            disease_options = {
                sample["gold_label_set"][0] for sample in dataset if sample["gold_label_set"]
            }
            print(len(disease_options), "unique disease options")
            options = "\n".join(f"- {disease}" for disease in sorted(disease_options))
            prompt_template += f"\n\nHere are the possible disease options to choose from:\n{options}"

        prompts = []
        for sample in dataset:
            if not sample[prompt_key] or isinstance(sample[prompt_key], float):  # skip samples with empty prompts (which are represented as NaNs in the jsonl)
                continue
            prompts.append(
                sample | {
                    "prompt": prompt_template.format(prompt=sample["prompt_genus"]),
                    "system_message": system_message,
                })
        

        # shuffle the dataset if seed is set
        if self.config["seed"] is not None:
            random.seed(self.config["seed"])
            random.shuffle(prompts)
        
        # apply start and limit
        start = self.config.get("start", 0)
        limit = self.config.get("limit", len(prompts))
        prompts = prompts[start: start + limit]

        return prompts

    def evaluate_responses(self, results):
        llm_judge = LLM(self.config["judge_model"])
        judge_generation_kwargs = json.loads(self.config.get("judge_generation_kwargs", "{}"))
        wait_for_vllm(llm_judge.hostname, llm_judge.port)
        
        judge_prompt_template = """
A model has been asked to identify a disease based on a patient's gut microbiome taxonomy. The model's response is below, along with the a list of different names for the correct disease (the "gold label set"). The model's response may not exactly match any of the gold labels, but it may still be correct if it is a reasonable synonym of the condition. Output "correct" if the model's response is a reasonable synonym of any of the gold labels, and "incorrect" otherwise. The final line of your responses should only contain "correct" or "incorrect" and no other text.

Model's response:
{response}
Gold label set:
{gold_labels}
""".strip()
        
        
        eval_messages = []
        for result in results:
            eval_prompt = judge_prompt_template.format(
                response=result["response"],
                gold_labels=result["gold_label_set"],
            )
            messages = [
                {"role": "system", "content": "You are a helpful and precise assistant for evaluating the quality of disease classification predictions based on gut microbiome taxonomy."},
                {"role": "user", "content": eval_prompt},
            ]
            eval_messages.append(messages)
    
        responses = llm_judge.batch_call(eval_messages, max_workers=self.config.get("max_workers", 5), **judge_generation_kwargs)
        
        instance_metrics = []
        for result, response in zip(results, responses):
            correctness = response["content"].splitlines()[-1].strip().lower()
            instance_metrics.append(result | {
                "judge_response": response,
                "is_correct": int(correctness == "correct"),
            })
        n = len(instance_metrics)
        n_correct = sum(o["is_correct"] for o in instance_metrics)
        summary_metrics = {
            "accuracy": n_correct / n if n > 0 else 0.0,
            "n": n,
        }
        return instance_metrics, summary_metrics

        