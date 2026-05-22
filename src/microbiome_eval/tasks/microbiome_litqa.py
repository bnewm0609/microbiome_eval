from pathlib import Path
import hashlib
import json
import random
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


class MicrobiomeLitQA_ResearchQAGeneration:

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
        parser.add_argument("--start_step", default="qa", choices=["qa", "distractors", "quality_filter"], help="Step to start running from. Defaults to the first step.")
        parser.add_argument("--end_step", default="all", choices=["qa", "distractors", "quality_filter", "all"], help="Run up to and including this step. By default, runs all steps.")
        parser.add_argument("--input_file", help="Path to a jsonl file to use as input to --start_step. Defaults to the paper info file for the first step.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
                question_id = hashlib.sha256(
                    f"{paper['title']}\n{qa_pair['question']}".encode()
                ).hexdigest()[:12]
                qa_pair_dict = {
                    "question_id": question_id,
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
                        "PMCID": paper["PMCID"],
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "authors": [author["name"] for author in paper["authors"]],
                        "year": paper["year"],
                        "it_ref": f"{paper['authors'][0]['name'].split()[-1]}{paper['year']}{paper['title'].split()[0].title()}"
                    }
                }
                outputs.append(qa_pair_dict)
            
            with open(cache_path, "w") as f:
                for qa_pair in outputs:
                    f.write(json.dumps(qa_pair) + "\n")
            
            return outputs

    def gen_distractors(self, qa_pairs, llm, cache_path, generation_kwargs):
        if cache_path.exists():
            with open(cache_path, "r") as f:
                qa_pairs_distractors = [json.loads(line) for line in f]
            print(f"Loaded {len(qa_pairs_distractors)} QA pairs with distractors from cache.")
            return qa_pairs_distractors

        prompt_template = """
You are creating multiple-choice questions for a biomedical assessment on microbiome research.

Your task: given a question and its correct answer, generate exactly 4 plausible but incorrect distractor options.

Requirements for distractors:
- Each distractor should sound scientifically reasonable and relate directly to the question
- Distractors should be comparable in length and specificity to the correct answer
- Distractors must be clearly distinct from each other and from the correct answer
- A knowledgeable reader who doesn't know the specific result should find them plausible
- Do not use vague or obviously wrong answers (e.g., "there was no effect")
- Do not reference "the paper", "the authors", or "this study"

Question: {question}
Correct answer: {answer}

Respond with a JSON array of exactly 4 distractor strings:
```json
["distractor 1", "distractor 2", "distractor 3", "distractor 4"]
```
""".strip()

        prompts_batched = []
        for qa in qa_pairs:
            prompt = prompt_template.format(
                question=qa["question"],
                answer=qa["answer"],
            )
            prompts_batched.append([{"role": "user", "content": prompt}])

        responses = llm.batch_call(prompts_batched, max_workers=10, **generation_kwargs)

        letters = ["A", "B", "C", "D", "E"]
        outputs = []
        for qa, response in zip(qa_pairs, responses):
            response_content = response["content"].strip()

            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
                try:
                    distractors = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for question: {qa['question'][:80]}")
                    print(f"Response content: {response_content}")
                    continue
            else:
                print(f"No JSON found in response for question: {qa['question'][:80]}")
                print(f"Response content: {response_content}")
                continue

            if len(distractors) != 4:
                print(f"Expected 4 distractors, got {len(distractors)} for: {qa['question'][:80]}")
                continue

            # Gold answer is at index 0; shuffle which letter it receives
            all_options = [qa["answer"]] + distractors
            shuffled_letters = letters[:]
            random.shuffle(shuffled_letters)
            gold_letter = shuffled_letters[0]

            options = sorted(
                [f"{shuffled_letters[i]}. {all_options[i]}" for i in range(5)]
            )

            output = {**qa}
            output["answer"] = f"{gold_letter}. {qa['answer']}"
            output["answer_idx"] = gold_letter
            output["options"] = options
            outputs.append(output)

        with open(cache_path, "w") as f:
            for item in outputs:
                f.write(json.dumps(item) + "\n")

        return outputs
    
    def quality_filter(self, qa_pairs_distractors, llm, cache_path, generation_kwargs):
        # Check if cache file exists
        if cache_path.exists():
            with open(cache_path, "r") as f:
                qa_pairs_quality_filter = [json.loads(line) for line in f]
            print(f"Loaded {len(qa_pairs_quality_filter)} QA pairs with quality filter from cache.")
            return qa_pairs_quality_filter
        

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
                step_inputs = [json.loads(line) for line in f]

        steps = list(self.steps.items())
        start_idx = next(i for i, (s, _) in enumerate(steps) if s == args.start_step)

        force_rerun = False
        for step_i, (step, step_fn) in enumerate(steps):
            if step_i < start_idx:
                continue

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
            if step == args.end_step:
                print(f"Reached specified end step {args.end_step}. Stopping.")
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

    pipelines = {
        "research_qa_generation": MicrobiomeLitQA_ResearchQAGeneration,
        "entity_qa_generation": None,  # not implemented yet
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
