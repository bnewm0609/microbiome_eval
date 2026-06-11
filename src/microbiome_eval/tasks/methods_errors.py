import json
import re
from pathlib import Path

from microbiome_eval.tasks.base import BaseTask, PROJ_PATH


DATA_PATH = PROJ_PATH / "data/cMDA/methodology_qs_hand_shaped/qs.jsonl"

PROMPT_TEMPLATE = """The following is a claim from a scientific paper:

{claim}

The following is a methodological description of a microbiome study. Your job is to identify any methodological flaws in the described approach. If there are no flaws, then respond with "No flaws".

{incorrect_methodology}

Please identify any methodological flaws in the described approach. If there are no flaws, then respond with "No flaws"."""

JUDGE_TEMPLATE = """You are evaluating whether a model response correctly identifies a specific methodological error.

# Methodology
{incorrect_methodology}

# Model Response
{model_response}

# Rubric Item
{rubric_item}

# Instructions
Evaluate whether the model's response covers the error described in the rubric item. We only care about recall — it is fine if the model identifies additional errors not mentioned in the rubric.

Return a JSON object with:
- "explanation": a string explaining your reasoning
- "verdict": one of "covered", "partially covered", or "missing"
  - "covered": the model clearly identifies the error described in the rubric item
  - "partially covered": the model hints at or partially addresses the error
  - "missing": the model does not address the error described in the rubric item

Return just the JSON object in markdown format. Do not include any other text."""


class MethodsErrors(BaseTask):

    @staticmethod
    def add_arguments(parser):
        pass

    def get_prompts(self):
        prompts = []
        with open(DATA_PATH) as f:
            for line in f:
                item = json.loads(line)
                prompts.append({
                    "system_prompt": "You are a helpful assistant.",
                    "prompt": PROMPT_TEMPLATE.format(
                        claim=item["claim"],
                        incorrect_methodology=item["incorrect_methodology"],
                    ),
                    **item,  # include question_id, rubric, and incorrect_methodology for later use in evaluation
                })
        return prompts

    @staticmethod
    def parse_verdict(response) -> dict:
        if isinstance(response, dict):
            response = response["content"]
        json_cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())
        return json.loads(json_cleaned)

    def evaluate_responses(self, results):
        from microbiome_eval.llm import LLM
        judge_llm = LLM("google/gemma-4-31B-it")

        judge_prompts = [
            [{"role": "user", "content": JUDGE_TEMPLATE.format(
                incorrect_methodology=r["incorrect_methodology"],
                model_response=r["response"],
                rubric_item=r["rubric"],
            )}]
            for r in results
        ]

        judge_responses = judge_llm.batch_call(
            judge_prompts,
            validation_fn=self.parse_verdict,
        )

        results_metrics = []
        for result, judge_response in zip(results, judge_responses):
            try:
                verdict_dict = self.parse_verdict(judge_response)
                verdict = verdict_dict.get("verdict", "missing")
            except (json.JSONDecodeError, KeyError, TypeError):
                verdict = "missing"
            results_metrics.append(result | {
                "judge_response": judge_response,
                "verdict": verdict,
            })

        n = len(results_metrics)
        covered = sum(1 for r in results_metrics if r["verdict"] == "covered")
        at_least_partial = sum(1 for r in results_metrics if r["verdict"] in ("covered", "partially covered"))

        summary_metrics = {
            "covered_rate": covered / n if n > 0 else 0,
            "at_least_partial_rate": at_least_partial / n if n > 0 else 0,
            "n": n,
        }
        return results_metrics, summary_metrics
