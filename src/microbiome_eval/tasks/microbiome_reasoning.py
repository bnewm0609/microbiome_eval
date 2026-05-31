from pathlib import Path
import json
from typing import Any

import datasets

from microbiome_eval.tasks.base import BaseTask, PROJ_PATH

class MicrobiomeReasoningTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "microbiome_reasoning"

    @staticmethod
    def add_arguments(parser):
        pass
        # parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Which split of the dataset to use.")

    
    def get_prompts(self) -> list[dict[str, Any]]:
        """
        """

        # load the dataset from huggingface
        dataset = datasets.load_dataset("Eubiota/Microbiome-Reasoning", split="test", cache_dir=PROJ_PATH / "data/eubiota_microbiome_reasoning/")

        system_message = "You are a helpful and precise assistant for answering microbiome-related questions. The last line of your answer should be a single letter corresponding to the correct answer choice on its own line."
        if self.config.get("model").lower() == "google/medgemma-1.5-4b-it":
            # medgemma isn't a reasoning model, so we should specifically prompt it to think step-by-step
            # similar to here: https://github.com/Google-Health/medgemma/blob/main/notebooks/evaluate_on_medqa.ipynb
            system_message += " Think step by step. **Your answer should end with a single letter corresponding to the correct answer choice on its own line.** For example, if the answer is choice A, your response should look like:\nsome thinking...\nsome more thinking\nA"
        prompts = []
        for sample in dataset:
            # convert to list of dicts
            sample_dict = dict(sample)
            prompts.append(sample_dict | {
                "prompt": sample_dict["question"],
                "system_messsage": system_message,
            })

        # shuffle the dataset if seed is set
        if self.config["seed"] is not None:
            import random
            random.seed(self.config["seed"])
            random.shuffle(prompts)
        
        # apply start and limit
        limit = self.config.get("limit", len(prompts))
        start = self.config.get("start", 0)
        prompts = prompts[start: start + limit]

        return prompts

    def evaluate_responses(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        
        
        # if self.config.get("model").lower() == "google/medgemma-1.5-4b-it":
            # this model can't follow instructions, so we're going to instead parse out the answer letter from the response using an LLM
        from microbiome_eval.llm import LLM
        judge_llm = LLM("google/gemma-4-31B-it")
        
        judge_prompt_template = """
Given the following model response to a multiple choice question, extract the single letter corresponding to the model's final answer choice.
Your response should be a single letter (A, B, C, or D) on its own line or the word "unknown" if you cannot determine the answer choice from the model's response.

Model response:
{response}
""".strip()
        
        judge_prompts = []
        for result in results:
            judge_prompts.append([{"role": "user", "content": judge_prompt_template.format(response=result["response"])}])

            response = result["response"]
        judge_responses = judge_llm.batch_call(judge_prompts, temperature=1.0, max_tokens=5, max_workers=self.config.get("max_workers", 5))
        for result, judge_response in zip(results, judge_responses):
            judge_response_content = judge_response["content"].strip()
            pred_letter = judge_response_content[0].upper() if judge_response_content else None
            result["raw_response"] = result["response"]
            result["response"] = pred_letter


        results_metrics = []    
        for result in results:
            response = result["response"]
            answer = result["ground_truth"]

            # The system prompt asks for the answer letter on the last line
            lines = [line.strip() for line in response.strip().splitlines()]
            last_line = next((line for line in reversed(lines) if line), "")
            # Strip common punctuation suffixes like "D." or "D)"
            last_line = last_line.strip(".:)( ")
            pred_letter = last_line[0].upper() if last_line else None
            answer_letter = answer.strip().strip("*.:)(")[0].upper()

            results_metrics.append(result | {
                "correct": pred_letter == answer_letter,
            })

        n = len(results_metrics)
        n_correct = sum(r["correct"] for r in results_metrics)
        summary_metrics = {
            "accuracy": n_correct / n if n > 0 else 0.0,
            "n": n,
            "n_correct": n_correct,
        }
        return results_metrics, summary_metrics