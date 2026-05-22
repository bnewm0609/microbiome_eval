from pathlib import Path
import json
from typing import Any

from microbiome_eval.tasks.base import BaseTask, PROJ_PATH
from microbiome_eval.llm import LLM


class MicrobiomeLitQA(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "microbiome_litqa"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Which split of the dataset to use.")


# For creating the dataset:


class MicrobiomeLitQA_Generation:

    def __init__(self):
        self.steps = {
            "qa": self.gen_qa_pairs,
            "distractors": self.gen_distractors,
            "quality_filter": self.quality_filter,
        }

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--model", help="Which split of the dataset to use.")
        parser.add_argument("--generation_kwargs", help="Which split of the dataset to use.")
        parser.add_argument("--step", default="all", choices=["qa", "distractors", "quality_filter", "all"], help="Runs up to and including the provided step. By default, runs all steps.")
        parser.add_argument("--out_dir", help="Output file")


    def gen_qa_pairs(self, paper_info, llm, cache_path, generation_kwargs):
        # Check if cache file exists
        if cache_path.exists():
            with open(cache_path, "r") as f:
                qa_pairs = [json.loads(line) for line in f]
            print(f"Loaded {len(qa_pairs)} QA pairs from cache.")
            return qa_pairs
    
        prompt_template = """
Given the following paper abstract in the field of microbiome research, identify the main research question(s) that the paper is trying to answer. The research question should be specific and focused on the scientific inquiry being conducted in the paper.

You should return the research questions in the following JSON format:
[{{"question": "<insert question here>", "answer": "<insert answer here>"}}]
Put your final answer in between ```json and ``` to make it easy to parse.

Additional notes:
- Most papers will have 1-2 main research questions.
- Your answer should be based solely on the abstract provided.
- Each question must be **independent**.
- Questions are going to be asked without providing the original abstract, so
    - make sure all necessary context is included in both the questions and the answers.
    - Do not refer to "the paper", "the authors", "this study", etc. in your question or answer. Instead, rephrase to be self-contained.

The title and abstract of the paper are as follows:
Title: {title}
Abstract: {abstract}

--- 
Identify the main research question(s) that the paper is trying to answer along with their corresponding answer(s) using the json format specified above.
""".strip()
        
        prompts_batched = []
        for paper in paper_info:
            prompt = prompt_template.format(
                title=paper["title"],
                abstract=paper["abstract"]
            )
            prompts_batched.append([{"role": "user", "content": prompt}])
        
        responses = llm.batch_call(prompts_batched, max_workers=10, **generation_kwargs)
        outputs = []
        for response in responses:
            response_content = response["content"].strip()

            # extract the json from between ```json and ```
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
                try:
                    qa_pairs = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for paper: {paper['title']}")
                    print(f"Response content: {response_content}")
                    continue
            else:
                print(f"No JSON found in response for paper: {paper['title']}")
                print(f"Response content: {response_content}")
                continue

            for qa_pair in qa_pairs:
                qa_pair_dict = {
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "options": None,  # to be filled in the next step
                    "answer_idx": None,  # to be filled in the next step
                    "quality": None,
                    "metadata": {
                        "type": "resarch_question",
                    },
                    "paper_data": {
                        "corpusId": paper["corpusId"],
                        "PMCID": paper["PMC_ID"],
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "authors": [author["name"] for author in paper["authors"]],
                        "year": paper["year"],
                    }
                }
                outputs.append(qa_pair_dict)
            
            with open(cache_path, "w") as f:
                for qa_pair in outputs:
                    f.write(json.dumps(qa_pair) + "\n")
            
            return outputs

    def gen_distractors(self, qa_pairs, llm, cache_path, generation_kwargs):
        # Check if cache file exists
        if cache_path.exists():
            with open(cache_path, "r") as f:
                qa_pairs_distractors = [json.loads(line) for line in f]
            print(f"Loaded {len(qa_pairs_distractors)} QA pairs with distractors from cache.")
            return qa_pairs_distractors
    
    def quality_filter(self, qa_pairs_distractors, llm, cache_path, generation_kwargs):
        # Check if cache file exists
        if cache_path.exists():
            with open(cache_path, "r") as f:
                qa_pairs_quality_filter = [json.loads(line) for line in f]
            print(f"Loaded {len(qa_pairs_quality_filter)} QA pairs with quality filter from cache.")
            return qa_pairs_quality_filter
        

    def run(self, args):
        # First, generate the qa pairs
        llm = LLM(args.model)
        generation_kwargs = json.loads(args.generation_kwargs) if args.generation_kwargs else {}
        out_dir = Path(args.out_dir)
        cache_dir = out_dir / "cache/"
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "config.json", "w") as f:
            json.dump(vars(args), f)

        with open("data/gmrepo/papers/s2_paper_info.jsonl", "r") as f:
            paper_info = [json.loads(line) for line in f]
        
        step_inputs = paper_info
        for step_i, (step, step_fn) in enumerate(self.steps.items()):
            print(f"Running step {step_i + 1}/{len(self.steps)}: {step}")

            step_results = step_fn(
                step_inputs,
                llm,
                cache_dir / f"{step}.jsonl",
                generation_kwargs
            )

            print(f'Step results at: {cache_dir}/{step}.jsonl')
            if step == args.step:
                print(f"Reached specified step {args.step}. Stopping.")
                break
            step_inputs = step_results
        
        if step_i == len(self.steps) - 1:
            with open(out_dir / "research_qa_pairs.jsonl", "w") as f:
                for qa in step_results:
                    if qa["quality"] == "good":
                        f.write(json.dumps(qa) + "\n")
            print(f"All steps completed. Final results at: {cache_dir}/quality_filter.jsonl")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    MicrobiomeLitQA_Generation.add_arguments(parser)
    args = parser.parse_args()

    generator = MicrobiomeLitQA_Generation()
    generator.run(args)
