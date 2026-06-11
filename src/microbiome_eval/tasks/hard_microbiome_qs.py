from collections import defaultdict
from pathlib import Path
import json
import re
from typing import Any

from microbiome_eval.tasks.base import BaseTask, PROJ_PATH

import pymupdf
# helper functions for Healthbench evaluation



class Healthbench:
    """Based on: https://github.com/openai/simple-evals/blob/main/healthbench_eval.py"""
    GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

    @classmethod
    def parse_json_to_dict(cls, json_string: str) -> dict:
        # Remove markdown-style ```json``` markers if present
        if isinstance(json_string, dict):
            json_string = json_string["content"]
        json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
        return json.loads(json_cleaned)
        

    @classmethod
    def calculate_score(
        cls, rubric_items: list, grading_response_list: list[dict]
    ) -> float | None:
        total_possible_points = sum(
            rubric_item["points"] for rubric_item in rubric_items if rubric_item["points"] > 0
        )
        if total_possible_points == 0:
            # should not happen for overall score, but may happen for tags
            return None

        achieved_points = sum(
            rubric_item["points"]
            for rubric_item, grading_response in zip(
                rubric_items, grading_response_list, strict=True
            )
            if grading_response["criteria_met"]
        )
        overall_score = achieved_points / total_possible_points
        return overall_score

    @staticmethod
    def get_prompts(samples, src_filename) -> list[dict[str, Any]]:
        prompts = []
        for sample in samples:
            system_prompt = "You are a helpful assistant."
            prompt = sample["sample"]["prompt"]
            # remove some info that was used for filtering
            sample_subset = {k: v for k, v in sample.items() if k not in ["_model", "_gen_kwargs", "_filter_response"]}
            prompts.append({
                "system_prompt": system_prompt,
                "prompt": prompt,
                "sample": sample["sample"],
                "_src_filename": str(src_filename),
                **sample_subset,
            })
        return prompts


    @classmethod
    def evaluate_responses(cls, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        from microbiome_eval.llm import LLM
        grader_llm = LLM("google/gemma-4-31B-it")
        grader_prompts = []
        for result in results:
            convo_with_response = result["sample"]["prompt"] + [{"role": "assistant", "content": result["response"]}]
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response]
            )
            for rubric_item in result["sample"]["rubrics"]:
                rubric_item_str = f"[{rubric_item['points']}] {rubric_item['criterion']}"
                grader_prompt = cls.GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", rubric_item_str)
                grader_prompts.append([{"role": "user", "content": grader_prompt}])
        

        grader_responses = grader_llm.batch_call(grader_prompts, validation_fn=cls.parse_json_to_dict, temperature=0.5, max_tokens=2048, max_workers=5)
        results_metrics = []
        grader_response_idx = 0
        for result in results:
            grader_responses_dicts = []
            for rubric_item in result["sample"]["rubrics"]:
                grader_response = grader_responses[grader_response_idx]
                grader_response_idx += 1
                try:
                    grader_json = cls.parse_json_to_dict(grader_response["content"])
                except json.JSONDecodeError as e:
                    print(f"JSON decoding failed: {e}")
                    grader_json = {}
                grader_responses_dicts.append(grader_json)

            overall_score = cls.calculate_score(result["sample"]["rubrics"], grader_responses_dicts)
            results_metrics.append(result | {
                "grader_responses": grader_responses_dicts,
                "score": overall_score,
            })
        
        summary_metrics = {
            "average_score": sum(r["score"] for r in results_metrics if r["score"] is not None) / len(results_metrics),
            "n": len(results_metrics),
        }
        return results_metrics, summary_metrics


class HealthbenchProfessional(Healthbench):
    
    @staticmethod
    def get_prompts(samples, src_filename) -> list[dict[str, Any]]:
        """Similar to parent class, just has some minor differences in the data keys"""
        prompts = []
        for sample in samples:
            system_prompt = "You are a helpful assistant."
            prompt = sample["sample"]["conversation"]["messages"]
            # remove some info that was used for filtering
            sample_subset = {k: v for k, v in sample.items() if k not in ["_model", "_gen_kwargs", "_filter_response"]}
            prompts.append({
                "system_prompt": system_prompt,
                "prompt": prompt,
                "sample": sample["sample"],
                "_src_filename": str(src_filename),
                **sample_subset,
            })
        return prompts

    @classmethod
    def evaluate_responses(cls, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        from microbiome_eval.llm import LLM
        grader_llm = LLM("google/gemma-4-31B-it")
        grader_prompts = []
        for result in results:
            convo_with_response = result["sample"]["conversation"]["messages"] + [{"role": "assistant", "content": result["response"]}]
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response]
            )
            for rubric_item in result["sample"]["rubric_items"]:
                rubric_item_str = f"[{rubric_item['points']}] {rubric_item['criterion_text']}"
                grader_prompt = cls.GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", rubric_item_str)
                grader_prompts.append([{"role": "user", "content": grader_prompt}])
        

        grader_responses = grader_llm.batch_call(grader_prompts, validation_fn=cls.parse_json_to_dict, temperature=0.5, max_tokens=2048, max_workers=5)
        results_metrics = []
        grader_response_idx = 0
        for result in results:
            grader_responses_dicts = []
            for rubric_item in result["sample"]["rubric_items"]:
                grader_response = grader_responses[grader_response_idx]
                grader_response_idx += 1
                try:
                    grader_json = cls.parse_json_to_dict(grader_response["content"])
                except json.JSONDecodeError as e:
                    print(f"JSON decoding failed: {e}")
                    grader_json = {}
                    grader_responses_dicts.append(grader_json)
                    continue
                grader_responses_dicts.append(grader_json)

            if len(grader_responses_dicts) != len(result["sample"]["rubric_items"]):
                print(f"Warning: number of grader responses ({len(grader_responses_dicts)}) does not match number of rubric items ({len(result['sample']['rubric_items'])}) for result with idx {result['_idx']}. This may be due to JSON parsing errors. Skipping score calculation for this result.")
                overall_score = None
            else:
                overall_score = cls.calculate_score(result["sample"]["rubric_items"], grader_responses_dicts)
            results_metrics.append(result | {
                "grader_responses": grader_responses_dicts,
                "score": overall_score,
            })
        
        summary_metrics = {
            "average_score": sum(r["score"] for r in results_metrics if r["score"] is not None) / len(results_metrics),
            "n": len(results_metrics),
        }
        return results_metrics, summary_metrics


class MedXpertQA:

    @staticmethod
    def get_prompts(samples, src_filename) -> list[dict[str, Any]]:
        # filter out some unnecessary fields (mostly related to filtering)
        prompts = []
        for sample in samples:
            system_prompt = "You are a helpful assistant for answering medical questions."

            prompt = """
Answer the following multiple choice question:

Question:
{question}

The final line of your answer should be a single letter corresponding to the correct answer choice, on its own line. For example, if the correct answer is choice A, your answer should end with a line that just says "A".
""".strip()

            # remove some info that was used for filtering
            sample_subset = {k: v for k, v in sample.items() if k not in ["_model", "_gen_kwargs", "_filter_response"]}
            prompts.append({
                "system_prompt": system_prompt,
                "prompt": prompt.format(question=sample["sample"]["question"]),
                "sample": sample["sample"],
                "_src_filename": str(src_filename),
                **sample_subset,
            })
        return prompts

    @staticmethod
    def evaluate_responses(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        from microbiome_eval.llm import LLM
        judge_llm = LLM("google/gemma-4-31B-it")
        
        judge_prompt_template = """
Given the following model response to a multiple choice question, extract the single letter corresponding to the model's final answer choice.
Your response should be a single letter on its own line or the word "unknown" if you cannot determine the answer choice from the model's response.

Model response:
{response}
""".strip()
        
        judge_prompts = []
        for result in results:
            judge_prompts.append([{"role": "user", "content": judge_prompt_template.format(response=result["response"])}])

            response = result["response"]
        judge_responses = judge_llm.batch_call(judge_prompts, temperature=1.0, max_tokens=5, max_workers=5)
        for result, judge_response in zip(results, judge_responses):
            judge_response_content = judge_response["content"].strip()
            pred_letter = judge_response_content[0].upper() if judge_response_content else None
            result["raw_response"] = result["response"]
            result["response"] = pred_letter


        results_metrics = []
        for result in results:
            response = result["response"]
            answer = result["sample"]["label"]

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


class BixBench:

    @staticmethod
    def get_prompts(samples, src_filename) -> list[dict[str, Any]]:
        import random
        prompts = []
        for sample in samples:
            system_prompt = "You are a helpful assistant."

            question = sample["sample"]["question"]
            ideal = sample["sample"]["ideal"]
            distractors = sample["sample"]["distractors"]

            choices = [ideal] + list(distractors)
            rng = random.Random(sample["sample"]["id"])
            rng.shuffle(choices)
            correct_letter = chr(ord("A") + choices.index(ideal))

            choices_str = "\n".join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices))

            prompt = f"""Answer the following multiple choice question:

Question:
{question}

{choices_str}

The final line of your answer should be a single letter corresponding to the correct answer choice, on its own line. For example, if the correct answer is choice A, your answer should end with a line that just says "A"."""

            sample_subset = {k: v for k, v in sample.items() if k not in ["_model", "_gen_kwargs", "_filter_response"]}
            prompts.append({
                "system_prompt": system_prompt,
                "prompt": prompt,
                "sample": sample["sample"],
                "correct_letter": correct_letter,
                "_src_filename": str(src_filename),
                **sample_subset,
            })
        return prompts

    @staticmethod
    def evaluate_responses(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        from microbiome_eval.llm import LLM
        judge_llm = LLM("google/gemma-4-31B-it")

        judge_prompt_template = """Given the following model response to a multiple choice question, extract the single letter corresponding to the model's final answer choice.
Your response should be a single letter on its own line or the word "unknown" if you cannot determine the answer choice from the model's response.

Model response:
{response}""".strip()

        judge_prompts = [
            [{"role": "user", "content": judge_prompt_template.format(response=result["response"])}]
            for result in results
        ]
        judge_responses = judge_llm.batch_call(judge_prompts, temperature=1.0, max_tokens=5, max_workers=5)

        results_metrics = []
        for result, judge_response in zip(results, judge_responses):
            judge_content = judge_response["content"].strip()
            pred_letter = judge_content[0].upper() if judge_content else None
            correct_letter = result["correct_letter"]
            results_metrics.append(result | {
                "pred_letter": pred_letter,
                "correct": pred_letter == correct_letter,
            })

        n = len(results_metrics)
        n_correct = sum(r["correct"] for r in results_metrics)
        summary_metrics = {
            "accuracy": n_correct / n if n > 0 else 0.0,
            "n": n,
            "n_correct": n_correct,
        }
        return results_metrics, summary_metrics


class Labbench2:

    @staticmethod
    def _build_prompt(sample: dict) -> str | list:
        import base64
        from microbiome_eval.tasks.labbench2_utils import (
            download_question_files, download_sources, GCS_BUCKET,
            is_text_injectable_format, get_media_type,
        )

        files_path = sample["files"]
        sources = sample["sources"]
        mode = sample.get("mode", {})
        question = sample["question"]
        prompt_suffix = sample.get("prompt_suffix", "").strip()

        injected_text = ""
        media_files = []

        if files_path:
            local_dir = download_question_files(GCS_BUCKET, files_path.strip("/"))
            if local_dir.exists():
                for f in sorted(local_dir.iterdir()):
                    if not f.is_file():
                        continue
                    if is_text_injectable_format(f) or mode.get("inject", False):
                        injected_text += f"\n--- {f.name} ---\n{f.read_text()}"
                    else:
                        media_files.append(f)
        elif sources:
            # for now, skip if there are not files.
            return None
            # content = download_sources(sources)
            # if content:
            #     injected_text = content

        parts = []
        if injected_text:
            parts.append(injected_text.strip())
        parts.append(question)
        if prompt_suffix:
            parts.append(prompt_suffix)
        full_text = "\n\n".join(parts)

        if not media_files:
            return full_text

        content_blocks = [{"type": "text", "text": full_text}]
        for f in media_files:
            media_type = get_media_type(f.suffix)
            if media_type.startswith("image/"):
                data_b64 = base64.b64encode(f.read_bytes()).decode()
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data_b64}"},
                })
            elif media_type == "application/pdf":
                # convert pdf to png
                doc = pymupdf.open(f)
                page = doc.load_page(0)  # load the first page
                pix = page.get_pixmap(dpi=150)  # render page to an image
                data_b64 = base64.b64encode(pix.tobytes("png")).decode()
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{data_b64}"},
                })
                # content_blocks.append({
                #     "type": "file",
                #     "file": {"filename": f.name, "file_data": f"data:application/pdf;base64,{data_b64}"},
                # })
        return [{"role": "user", "content": content_blocks}]

    @staticmethod
    def get_prompts(samples, src_filename) -> list[dict[str, Any]]:
        prompts = []
        for sample in samples:
            prompt = Labbench2._build_prompt(sample["sample"])
            if prompt is None:
                continue
            sample_subset = {k: v for k, v in sample.items() if k not in ["_model", "_gen_kwargs", "_filter_response"]}
            prompts.append({
                "system_prompt": "You are a helpful assistant.",
                "prompt": prompt,
                "sample": sample["sample"],
                "_src_filename": str(src_filename),
                **sample_subset,
            })
        return prompts

    @staticmethod
    def evaluate_responses(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        from microbiome_eval.llm import LLM
        from collections import defaultdict

        judge_llm = LLM("google/gemma-4-31B-it")

        judge_template = """You are evaluating whether a model's response correctly answers a short-answer question.

Question: {question}
Correct answer: {ideal}
Model response: {response}

Does the model's response contain the correct answer? Accept partial matches, different phrasings, and equivalent numeric representations (e.g. "2x10^6" == "2,000,000"). Respond with just "correct" or "incorrect"."""

        judge_prompts = [
            [{"role": "user", "content": judge_template.format(
                question=r["sample"]["question"],
                ideal=r["sample"]["ideal"],
                response=r["response"],
            )}]
            for r in results
        ]
        judge_responses = judge_llm.batch_call(judge_prompts, temperature=0.0, max_tokens=10, max_workers=5)

        results_metrics = []
        for result, judge_response in zip(results, judge_responses):
            correct = judge_response["content"].lower().strip().startswith("correct")
            results_metrics.append(result | {"correct": correct})

        by_tag: dict[str, list] = defaultdict(list)
        for r in results_metrics:
            by_tag[r["sample"]["tag"]].append(r)

        n = len(results_metrics)
        n_correct = sum(r["correct"] for r in results_metrics)
        summary_metrics = {
            "accuracy": n_correct / n if n > 0 else 0.0,
            "n": n,
            "n_correct": n_correct,
            "by_tag": {
                tag: {
                    "accuracy": sum(r["correct"] for r in tag_results) / len(tag_results),
                    "n": len(tag_results),
                }
                for tag, tag_results in sorted(by_tag.items())
            },
        }
        return results_metrics, summary_metrics


DATASET_CLS_MAP = {
    "healthbench": Healthbench,
    "healthbench_professional": HealthbenchProfessional,
    "MedXpertQA": MedXpertQA,
    "BixBench": BixBench,
    "labbench2": Labbench2,
}


class HardMicrobiomeQsTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "medqa"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--data_config", type=str, help="What composition of the eval dataset to use")
    
    def get_prompts(self) -> list[dict]:
        data_config = json.loads(self.config["data_config"])
        print(data_config)

        prompts = []
        for dataset_name, run_num in data_config.items():
            data_dir = PROJ_PATH / f"data/filtered_difficult_datasets/{dataset_name}"
            print(data_dir)
            datafns = data_dir.glob(f"run_{run_num}-*.jsonl")
            # assert len(datafns) == 1, f"Expected exactly one data file for dataset {dataset_name} and run {run_num}, but found {len(datafns)}"
            print(datafns)
            dataset = []
            # datafn = datafns[0]
            for datafn in datafns:
                print(f"Loading prompts from {datafn}...")
                with open(datafn) as f:
                    dataset.extend([json.loads(line) for line in f])
            
            if dataset:
                # load the prompts for this dataset
                prompts.extend((DATASET_CLS_MAP[dataset_name].get_prompts(dataset, datafn)))
        
        limit = self.config.get("limit", len(prompts))
        start = self.config.get("start", 0)
        prompts = prompts[start: start + limit]
        return prompts


    def evaluate_responses(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        # group the results by dataset
        results_by_dataset = defaultdict(list)
        for result in results:
            results_by_dataset[result["_dataset"]].append(result)
        
        # evaluate each dataset separately and aggregate the metrics
        all_results_metrics = []
        summary_metrics = {}
        for dataset_name, dataset_results in results_by_dataset.items():
            dataset_cls = DATASET_CLS_MAP[dataset_name]
            dataset_results_metrics, dataset_summary_metrics = dataset_cls.evaluate_responses(dataset_results)
            all_results_metrics.extend(dataset_results_metrics)
            summary_metrics[dataset_name] = dataset_summary_metrics
        
        return all_results_metrics, summary_metrics
            


        # data_config = json.loads(self.config["data_config"])
        # for dataset_name in data_config.keys():

