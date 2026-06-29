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


from pathlib import Path

class DiseaseClassification_Generation:

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--limit", default=None, help="count")
        parser.add_argument("--start", default=0, help="start idx")
        parser.add_argument("--model", help="Which model to use for all steps")
        parser.add_argument("--generation_kwargs", help="model generation kwargs")
        parser.add_argument("--steps", type=str, default="qa,distractors,quality_filter_surface_form", help="Comma-separated list of steps to run.")
        # parser.add_argument("--start_step", default="qa", choices=["qa", "distractors", "quality_filter_surface_form"], help="Step to start running from. Defaults to the first step.")
        # parser.add_argument("--end_step", default="all", choices=["qa", "distractors", "quality_filter_surface_form", "all"], help="Run up to and including this step. By default, runs all steps.")
        parser.add_argument("--input_file", help="Path to a jsonl file to use as input to --start_step. Defaults to the paper info file for the first step.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--out_dir", help="Output file")


    def run(self, args):
        random.seed(args.seed)
        llm = LLM(args.model)
        generation_kwargs = json.loads(args.generation_kwargs) if args.generation_kwargs else {}
        out_dir = Path(args.out_dir)
        cache_dir = out_dir / "cache/"
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "config.json", "w") as f:
            json.dump(vars(args), f)

        if args.input_file:
            with open(args.input_file, "r") as f:
                step_inputs = [json.loads(line) for line in f]
        else:
            with open("data/gmrepo/papers/s2_paper_info.jsonl", "r") as f:
                step_inputs = [json.loads(line) for line in f]# [:10]

        steps = []
        for step_label in args.steps.split(","):
            if step_label not in self.steps:
                raise ValueError(f"Invalid step '{step_label}' specified. Must be one of: {list(self.steps.keys())}")
            steps.append((step_label, self.steps[step_label]))

        # start_idx = next(i for i, (s, _) in enumerate(steps) if s == args.start_step)

        start = int(args.start)
        limit = int(args.limit) if args.limit is not None else len(step_inputs)

        force_rerun = False
        for step_i, (step, step_fn) in enumerate(steps):

            step_inputs = step_inputs[start: start + limit]
            print(f"Running step {step_i + 1}/{len(steps)}: {step}")

            cache_path = cache_dir / f"{step}.jsonl"
            if force_rerun and cache_path.exists():
                cache_path.unlink()
            cache_existed = cache_path.exists()

            step_results = step_fn(
                step_inputs,
                llm,
                cache_path,
                generation_kwargs
            )

            if not cache_existed:
                force_rerun = True

            print(f'Step results at: {cache_path}')
            # if step == args.end_step:
            #     print(f"Reached specified end step {args.end_step}. Stopping.")
            #     break
            step_inputs = step_results
        

class DiseaseClassification_GenerationV1(DiseaseClassification_Generation):

    def __init__(self):
        self.steps = {
            "gen_pathways": self.gen_pathways,
            "gen_trace": self.gen_trace,
        }

    def gen_pathways(self, qa_pairs, llm, cache_path, generation_kwargs):
        prompt_template = """
Based on the following scientific paper abstracts, what are some pathways or mechanisms influencing the effect of the bacteria {bacteria_name} on the disease {disease_name}? Please provide a concise summary of the key findings and mechanisms discussed in the abstracts. If no relevant information is found, please state "No relevant information found."

{abstracts}
""".strip()
        
        search_query_template = "{disease_name} AND {bacteria_name} AND microbiome"

        from microbiome_eval.tools.pubmed_search import PubMed_Search_Tool
        search_tool = PubMed_Search_Tool()

        def format_publications(result):
            formatted_publications = ""
            for i, pub in enumerate(result['All Journals']['publications']):
                text = f"""
            Publication Abstract ([{i+1}] PMID: {pub['PMID']}):
            Title: {pub['Title']}
            {pub['Abstract']}
            """
                formatted_publications += text
            return formatted_publications

        valid_idxs = []
        for sample in qa_pairs:
            if sample["gold_label_set"]:
                disease_name = sample["gold_label_set"][0]
            else:
                continue

            cache_filename = cache_path.with_name(f"{cache_path.stem}_step_{sample['Index']}.json")
            if cache_filename.exists():
                valid_idxs.append(sample["Index"])
                print(f"Found cached example: {sample['Index']}")
                continue
            # else:
            #     print(f"Didn't find file: {cache_filename}")

            # start with genus
            search_results = []
            for bacteria in sample.get("taxonomic_profile_genus", []):
                if bacteria.get("scientific_name", "Unknown") == "Unknown":
                    continue
                # search
                query = search_query_template.format(disease_name=disease_name, bacteria_name=bacteria["scientific_name"])
                result = search_tool.run(query=query)
                if result['All Journals']['citations']:  # citations is always there
                    search_results.append((bacteria["scientific_name"], result))
            
            # We didn't find any relevant papers for this disease
            if not search_results:
                continue

            valid_idxs.append(sample["Index"])
            # generate pathways (batched)
            prompts = []
            for bacteria_name, papers in search_results:
                prompts.append([{
                    "role": "user",
                    "content": prompt_template.format(
                        bacteria_name=bacteria_name,
                        disease_name=disease_name,
                        abstracts=format_publications(papers)
                    )
                }])
            
            responses = llm.batch_call(prompts, **generation_kwargs)

            # cache intermediate results?
            cache_filename = cache_path.with_name(f"{cache_path.stem}_step_{sample['Index']}.json")
            with open(cache_filename, "w") as f:
                f.write(json.dumps({
                    bacteria_name: {"description": response["content"], "search_results": papers}
                    for (bacteria_name, papers), response in zip(
                        search_results,
                        responses
                    )
                }))


        # finally, write the info in the valid index files to the cache path
        outputs = []
        with open(cache_path, "w") as f:
            for sample in qa_pairs:
                if sample["Index"] in valid_idxs:
                    cache_filename = cache_path.with_name(f"{cache_path.stem}_step_{sample['Index']}.json")
                    with open(cache_filename, "r") as cf:
                        step_results = json.load(cf)
                    outputs.append(sample | {"pathways": step_results})
                    f.write(json.dumps(sample | {"pathways": step_results}) + "\n")
        return outputs


    def gen_trace(self, qa_pairs, llm, cache_path, generation_kwargs):
        # create prompts
        prompt_template = """
A sample of the microbiome was taken from a patient with the disease {disease_name}, and the following bacteria were identified in the sample:

{abundance_table}

---
Using the mechanism descriptions below, simulate a reasoning process that starts with the the bacterial profile and ends up concluding what disease the patient has.
Your response should include a step-by-step reasoning trace, highlighting potential interactions between bacteria and their mechanisms. The relevant pathways for each of the bacteria types are highlighted below:

{pathways}
---

Using the bacteria profile and the pathway descriptions, simulating a reasoning process that starts with the profile and ends with concluding what disease the patient has ({disease_name}). This should be formatted as a step-by-step reasoning trace.
Do NOT include any citations to papers or explicitly mention the pathway descriptions in your answer.
""".strip()

        prompts = []
        for sample in qa_pairs:
            # form the abundance table
            abundance_table_str = "\n".join([
                f'{bact["scientific_name"]}: {bact["relative_abundance"]:.3f}%'
                for bact in sample["taxonomic_profile_genus"]
            ])

            formatted_pathways = ""
            for bacteria in sample["taxonomic_profile_genus"]:
                pathway_descr = sample["pathways"].get(bacteria["scientific_name"])
                if pathway_descr is None:
                    continue
                formatted_pathways += f"\n## Pathways for {bacteria}:\n{pathway_descr}\n"
            formatted_pathways = formatted_pathways.strip()

            prompts.append([{
                "role": "user",
                "content": prompt_template.format(
                    disease_name=sample["gold_label_set"][0],
                    abundance_table=abundance_table_str,
                    pathways=formatted_pathways,
                )
            }])

        # generate the traces
        generation_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        responses = llm.batch_call(prompts, **generation_kwargs)
        outputs = []
        for sample, response in zip(qa_pairs, responses):
            outputs.append(sample | {"trace": response['content']})

        with open(cache_path, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

        return outputs


    def run(self, args):
        super().run(args)

    

if __name__ == "__main__":
    import argparse

    pipelines = {
        "disease_classification_generation_v1": DiseaseClassification_GenerationV1,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=pipelines.keys())
    args, _ = parser.parse_known_args()

    pipeline_cls = pipelines[args.pipeline]
    if pipeline_cls is None:
        raise NotImplementedError(f"Pipeline '{args.pipeline}' is not implemented yet.")

    pipeline_cls.add_arguments(parser)
    args = parser.parse_args()

    pipeline_cls().run(args)