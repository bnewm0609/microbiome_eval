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