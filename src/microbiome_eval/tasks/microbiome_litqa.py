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
        # parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Which split of the dataset to use.")
        parser.add_argument("--litqa_setting", default="research_qa_generation", choices=["research_qa_generation", "high_citation_qa_generation"], help="Which setting of the MicrobiomeLitQA dataset to use.")
    
    def get_prompts(self):
        if self.config["litqa_setting"] == "research_qa_generation":
            with open(PROJ_PATH / "data/microbiome_litqa/research_qa_pairs.jsonl", "r") as f:
                samples = [json.loads(line) for line in f]
        elif self.config["litqa_setting"] == "high_citation_qa_generation":
            # with open(PROJ_PATH / "data/microbiome_litqa/high_citation_qa_pairs.jsonl", "r") as f:
            with open(PROJ_PATH / "data/gmrepo/papers/high_citation_qa_generation_v1_gemma4_31B_100/cache/fix_distractors.jsonl") as f:
                samples = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Invalid litqa_setting: {self.config.litqa_setting}")
    
        system_message = "You are a helpful and precise assistant for answering microbiome-related questions."
        prompt_template = """
Answer the following multiple choice question:

The last line of your answer should be a single letter corresponding to the correct answer choice on its own line.

For example, if the answer is choice A, the last line of your response should be:

A

---
Now answer the following question:

Question: {question}
{options}
""".strip()
        prompts = []
        for sample in samples:
            # convert to list of dicts
            sample_dict = dict(sample)
            prompt_text = prompt_template.format(
                question=sample_dict["question"],
                options="\n".join(sample_dict["options"]),
            )
            if self.config.get("model").lower() == "google/medgemma-1.5-4b-it":
                # medgemma isn't a reasoning model, so we should specifically prompt it to think step-by-step
                prompt_text = " Think step by step and a" + prompt_text[1:]

            prompts.append(sample_dict | {
                "prompt": prompt_text,
                "system_messsage": system_message,
            })

        # shuffle the dataset if seed is set
        if self.config["seed"] is not None:
            import random
            random.seed(self.config["seed"])
            random.shuffle(prompts)
        
        # apply start and limit
        limit = self.config.get("limit", len(prompts))
        if limit is None:
            limit = len(prompts)
        start = self.config.get("start", 0)
        prompts = prompts[start: start + limit]

        return prompts
    
    def evaluate_responses(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        from microbiome_eval.llm import LLM
        judge_llm = LLM("google/gemma-4-31B-it")
        
        # Parse out the qa response with a judge. This is especially important for models like medgemma that can't reliably follow the instruction to put the answer letter on its own line, so we need a judge to parse out the answer letter from the rest of the response. Even for models that can follow that instruction, this adds robustness
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

        # now evaluate the parsed answer letters against the ground truth answer letters
        results_metrics = []
        for result in results:
            response = result["response"]
            answer = result["answer_idx"]  # the letter corresponding to the correct answer choice, e.g. "B"

            # The system prompt asks for the answer letter on the last line
            lines = [line.strip() for line in response.strip().splitlines()]
            last_line = next((line for line in reversed(lines) if line), "")
            # Strip common punctuation suffixes like "D." or "D)"
            last_line = last_line.strip(".:)( ")
            pred_letter = last_line[0].upper() if last_line else None
            answer_letter = answer.strip().strip(".:)(")[0].upper()

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


# For creating the dataset:
class MicrobiomeLitQA_Generation:

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--limit", default=None, help="count")
        parser.add_argument("--start", default=0, help="start idx")
        parser.add_argument("--model", help="Which model to use for all steps")
        parser.add_argument("--generation_kwargs", help="Which split of the dataset to use.")
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
            # if step_i < start_idx:
            #     continue

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
        

class MicrobiomeLitQA_ResearchQAGeneration(MicrobiomeLitQA_Generation):

    def __init__(self):
        self.steps = {
            "qa": self.gen_qa_pairs,
            "distractors": self.gen_distractors,
            "quality_filter_surface_form": self.quality_filter_surface_form,
        }


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
- Do not compose two questions into a single one - your question should not include the word "and". If there are multiple questions, list them separately.
- Questions are going to be asked without providing the original title or abstract, so
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
        
        responses = llm.batch_call(prompts_batched, **generation_kwargs)
        # breakpoint()
        outputs = []
        for paper, response in zip(paper_info, responses):
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
            
        
        # choose some representative samples to inspect the quality of the generated QA pairs before moving on to the next step:
        print("Sample generated QA pairs:")
        random.seed(42)
        for sample in random.sample(outputs, min(5, len(outputs))):
            print(f"Question ID: {sample['question_id']}")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer']}")
            print("-" * 80)
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

        responses = llm.batch_call(prompts_batched, **generation_kwargs)

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


        random.seed(42)
        def preview_sample(samples, sample_i):
            sample = samples[sample_i]
            print(f"{sample_i}. Question ID:", sample["question_id"])
            print("Question:", sample["question"])
            print("\n".join(sample["options"]))
            print("\nAnswer:", sample["answer"])
            print("\n", "-" * 80, "\n")

        for sample_i in random.sample(list(range(len(outputs))), 5):
            preview_sample(outputs, sample_i)
        return outputs
    
    def quality_filter_surface_form(self, qa_pairs_distractors, llm, cache_path, generation_kwargs):
        # Check if cache file exists
        if cache_path.exists():
            with open(cache_path, "r") as f:
                qa_pairs_quality_filter = [json.loads(line) for line in f]
            print(f"Loaded {len(qa_pairs_quality_filter)} QA pairs with quality filter from cache.")
            return qa_pairs_quality_filter
    
        prompt_template = """
Your task is to rate the quality of a generated multiple choice question as "high" or "low" quality.
The questions are meant to be difficult and to test reasoning about clinical applications of the microbiome.

As a first step, you should focus on the surface forms of the answer choices. For some questions, the gold answer uses langauge or spellings that differ from the distractors (e.g. "B. fragilis" vs "Bacteroids fragilis"). This is a sign of a low quality question, even if the distractors are scientifically reasonable.

If the question has distractors that differ in language or spelling from the gold answer, output "bad". Otherwise, output "good".
Additionally, if the question or answer choices reference a "study", "paper", or "authors", that's a sign of a low quality question, so output "bad" in that case as well.

Your response should end with just the word "good" or "bad" on its own line.
The question you should judge is below:

Question: {question}
{options}
Answer: {answer}
""".strip()

        prompts_batched = []
        for qa in qa_pairs_distractors:
            prompt = prompt_template.format(
                question=qa["question"],
                options="\n".join(qa["options"]),
                answer=qa["answer"],
            )
            prompts_batched.append([{"role": "user", "content": prompt}])
        responses = llm.batch_call(prompts_batched, **generation_kwargs)
        outputs = []
        for qa, response in zip(qa_pairs_distractors, responses):
            response_content = response["content"].strip().lower()
            quality_rating = "bad" if "bad" in response_content.splitlines()[-1].strip() else "good"
            
            output = {**qa}
            output["quality"] = {
                "surface_form": quality_rating,
            }
            outputs.append(output)
        
        # save outputs
        with open(cache_path, "w") as f:
            for item in outputs:
                f.write(json.dumps(item) + "\n")
        
        # preview some samples to check the quality of the ratings before moving on to the next step:
        print("Sample quality_filter_surface_form ratings:")
        random.seed(42)
        def preview_sample(samples, sample_i):
            sample = samples[sample_i]
            print(f"{sample_i}. Question ID:", sample["question_id"])
            print("Question:", sample["question"])
            print("\n".join(sample["options"]))
            print("\nAnswer:", sample["answer"])
            print(f"\nQuality rating:", sample["quality"]["surface_form"])
            print("\n", "-" * 80, "\n")

        for sample_i in random.sample(list(range(len(outputs))), 5):
            preview_sample(outputs, sample_i)
        
        return outputs


    # def run(self, args):
        
        # if step_i == len(self.steps) - 1:
        #     with open(out_dir / "research_qa_pairs.jsonl", "w") as f:
        #         for qa in step_results:
        #             if qa["quality"] == "good":
        #                 f.write(json.dumps(qa) + "\n")
        #     print(f"All steps completed. Final results at: {cache_dir}/quality_filter.jsonl")


class MicrobiomeLitQA_HighCitationQAGeneration(MicrobiomeLitQA_ResearchQAGeneration):
    """
    Very similar to the ResearchQAGeneration pipeline, but with a different set of papers as input.
    We're using the top 100 most cited papers in the GMRepo dataset as the input to this pipeline
    because these represent facts that we might expect a model to know/reason about.

    Which papers to look at are computed in Exploration 1.ipynb, cell 2.3. The number of citations for the papers range from
    13 - 74 citations from our dataset.

    We're going to switch the initial qa generation away from "research question" toward "clinical takeaways"
    because some of these papers are methods contributions, so the research question is less well-defined.
    """
    def __init__(self):
        self.steps = {
            "qa": self.gen_qa_pairs,
            "distractors": self.gen_distractors,
            "quality_filter_surface_form": self.quality_filter_surface_form,
            "quality_filter_obvious_answer": self.quality_filter_obvious_answer,
            "fix_distractors": self.fix_distractors,
        }
    def gen_qa_pairs(self, paper_info, llm, cache_path, generation_kwargs):
        # Check if cache file exists
        if cache_path.exists():
            with open(cache_path, "r") as f:
                qa_pairs = [json.loads(line) for line in f]
            print(f"Loaded {len(qa_pairs)} QA pairs from cache.")
            return qa_pairs
    
        prompt_template = """
Given the following paper title and abstract in the field of microbiome research, identify the main takeaway(s) from the paper for clinicians. Rather than outputing the takeaways verbatim, frame them as questions whose answers are the takeaways. The questions should be framed as general questions that one scientist might ask another rather than focusing on the paper itself.

You should return the research questions in the following JSON format:
[{{"question": "<insert question here>", "answer": "<insert answer here>"}}]
Put your final answer in between ```json and ``` to make it easy to parse.

Additional notes:
- Most papers will have 1-2 takeaway questions.
- Your answer should be based solely on the abstract provided.
- Each question must be **independent** and only ask about one particular takeaway. Do not combine multiple takeaways into one question. (ie your question should not include conjunctions like "and")
- Questions are going to be asked without providing the original abstract, so
    - make sure all necessary context is included in both the questions and the answers.
    - Do not refer to "the paper", "the authors", "this study", etc. in your question or answer. Instead, rephrase to be self-contained.

The title and abstract of the paper are as follows:
Title: {title}
Abstract: {abstract}

--- 
Identify the main takeaway question(s) that the paper is trying to answer along with their corresponding answer(s) using the json format specified above.
""".strip()
        
        prompts_batched = []
        for paper in paper_info:
            prompt = prompt_template.format(
                title=paper["title"],
                abstract=paper["abstract"]
            )
            prompts_batched.append([{"role": "user", "content": prompt}])

        responses = llm.batch_call(prompts_batched, **generation_kwargs)
        # breakpoint()
        outputs = []
        for paper, response in zip(paper_info, responses):
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
                        # "PMCID": paper["PMCID"],
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

    
    def quality_filter_obvious_answer(self, qa_pairs_distractors, llm, cache_path, generation_kwargs):
        prompt_template = """
Your task is to rate the quality of a generated multiple choice question as "good" or "bad" quality.

The questions are meant to test knowledge of important clinical aspects of the microbiome.
Rate a question as "bad" if the answer is obvious from the wording of the question and answer choices.
For example if the question is:
Question: Which of the following bacteria is most commonly associated with the human gut microbiome?
A. Bacteroides fragilis is associated with the human gut microbiome
B. Escherichia coli is associated with sickness
C. Staphylococcus aureus is associated with skin infections
D. Lactobacillus acidophilus is associated with yogurt
Answer: A. Bacteroides fragilis is associated with the human gut microbiome

Then, the question is "bad" because the answer choice A has a lot of overlap in surface form with the question, making the answer obvious.

If the answer is not obvious in this way, then rate the question as "good".

Your response should end with just the word "good" or "bad" on its own line.
The question you should judge is below:

Question: {question}
{choices}
Answer: {answer}
""".strip()
        prompts_batched = []
        for qa in qa_pairs_distractors:
            prompt = prompt_template.format(
                question=qa["question"],
                choices="\n".join(qa["options"]),
                answer=qa["answer"],
            )
            prompts_batched.append([{"role": "user", "content": prompt}])
        responses = llm.batch_call(prompts_batched, **generation_kwargs)
        outputs = []
        for qa, response in zip(qa_pairs_distractors, responses):
            response_content = response["content"].strip().lower()
            quality_rating = "bad" if "bad" in response_content.splitlines()[-1].strip() else "good"
            
            output = {**qa}
            output["quality"] |= {
                "obvious_answer": quality_rating,
            }
            outputs.append(output)
        
        with open(cache_path, "w") as f:
            for item in outputs:
                f.write(json.dumps(item) + "\n")
        
        return outputs

    def fix_distractors(self, qa_pairs_distractors, llm, cache_path, generation_kwargs):
        # This is a placeholder for a potential future step where we might want to fix low-quality distractors rather than just filtering out the whole question.
        # For example, we could ask the LLM to generate new distractors that are more similar in surface form to the gold answer if it rated the original distractors as low quality.
        prompt_template = """
The following multiple choice question is too easy because the correct answer has too much overlap in surface form with the question.

Your job is to rewrite the answer choices to make the question more difficult, while still keeping the same correct answer and keeping the question scientifically accurate.
The answers should be made more concise.


Question: {question}
{options}
Answer: {answer}

Respond with a JSON array containing the rewritten answer choice strings in the same order as the original answer choices. Do not include the letter labels (A, B, C, D, E) in your response. The desired format is below:
```json
["choice 1", "choice 2", "choice 3", "choice 4", "choice 5"]
```
""".strip()
        
        prompts = []
        qis = []
        for qi, qa in enumerate(qa_pairs_distractors):
            if qa["quality"]["obvious_answer"] == "bad":
                qis.append(qi)
                prompt = prompt_template.format(
                    question=qa["question"],
                    options="\n".join(qa["options"]),
                    answer=qa["answer"],
                )
                prompts.append([{"role": "user", "content": prompt}])
        
        responses = llm.batch_call(prompts, **generation_kwargs)
        for qi, response in zip(qis, responses):
            response_content = response["content"].strip()

            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
                try:
                    new_choices = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for question: {qa_pairs_distractors[qi]['question'][:80]}")
                    print(f"Response content: {response_content}")
                    continue
            else:
                print(f"No JSON found in response for question: {qa_pairs_distractors[qi]['question'][:80]}")
                print(f"Response content: {response_content}")
                continue
            
            
            # update the options in the original qa pair with the new choices (keeping the same correct answer)
            
            # output["answer_idx"] = gold_letter
            letters = "ABCDE"
            new_choices = [f"{letters[i]}. {new_choice}" for i, new_choice in enumerate(new_choices)]
            qa_pairs_distractors[qi]["answer"] = new_choices[letters.index(qa_pairs_distractors[qi]["answer_idx"])]  # update the answer to match the new choice format
            qa_pairs_distractors[qi]["options"] = new_choices

        with open(cache_path, "w") as f:
            for item in qa_pairs_distractors:
                f.write(json.dumps(item) + "\n")
        return qa_pairs_distractors



class MicrobiomeLitQA_MultihopGeneration(MicrobiomeLitQA_Generation):
    
    def __init__(self):
        self.steps = {
            "extract_entities": self.extract_entities,
            "filter_entities": self.filter_entities,
            "quality_filter_surface_form": self.quality_filter_surface_form,
        }
    
    def extract_entities(self, paper_info, llm, cache_path, generation_kwargs):
        prompt_template = """
As a knowledge analyzer, your task is to dissect and understand a section of of a scientific paper in the medical field focusing on the microbiome. You are required to perform the following step:
1. Extract Named Entities: Identify and list all significant named entities mentioned within the section. These entities be nouns and should be specific names of methods, bacteria, drugs, proteins, genes, techniques, etc.

Ensure that your list of named entities is comprehensive and accurate. Structure your response in a JSON format to organize the information effectively.

Here is the format you should use for your response: 
``json
{{
"entities": ["named_entity1", "named_entity2", ...]
}}
```

The title and abstract are below:

{title}

{abstract}

""".strip()

        prompts = [
            [{
                "role": "user",
                "content": prompt_template.format(
                    title=paper_info[pi]["title"],
                    abstract=paper_info[pi]["abstract"])
            }] 
            for pi in range(len(paper_info))
        ]

        responses = llm.batch_call(prompts, **generation_kwargs)



        



if __name__ == "__main__":
    import argparse

    pipelines = {
        "research_qa_generation": MicrobiomeLitQA_ResearchQAGeneration,
        "high_citation_qa_generation": MicrobiomeLitQA_HighCitationQAGeneration,
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
