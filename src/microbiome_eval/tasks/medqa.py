from pathlib import Path
import json
from typing import Any

from microbiome_eval.tasks.base import BaseTask



class MedQATask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "medqa"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--split", default="dev", options=["dev", "test"], help="Which split of the dataset to use.")

    
    def get_prompts(self):
        """
        Uses the same question and answer choice format as:
        https://huggingface.co/datasets/Eubiota/data/viewer/default/train?f%5Bsource%5D%5Bvalue%5D=%27medqa%27&row=16

        Sample:
        {
        "question": "A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given?",
        "answer": "Ceftriaxone",
        "options": {
            "A": "Chloramphenicol",
            "B": "Gentamicin",
            "C": "Ciprofloxacin",
            "D": "Ceftriaxone",
            "E": "Trimethoprim"
        },
        "meta_info": "step1",
        "answer_idx": "D"
        }
        """

        # load the dataset
        split = self.config["split"]
        with open(PROJ_PATH / f"data/med_qa/US/{split}.json") as f:
            dataset = json.load(f)
        
        # shuffle the dataset if seed is set
        if self.config["seed"] is not None:
            import random
            random.seed(self.config["seed"])
            random.shuffle(dataset)
        
        # apply start and limit        start = self.config["start"]
        limit = self.config.get("limit", len(dataset))
        start = self.config.get("start", 0)
        dataset = dataset[start: start + limit]

        system_message = "You are a helpful and precise assistant for answering medical questions. The last line of your answer should be a single letter corresponding to the correct answer choice on its own line."
        prompts = []
        for sample in dataset:
            question = sample["question"]
            options = sample["options"]
            prompt = f"{question}\n"
            for letter, option in options.items():
                prompt += f"{letter}. {option}\n"
            prompts.append(sample
            | {
                "prompt": prompt,
                # Any system prompt can be included here
            })
        
        return prompts

    def evaluate_responses(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        results_metrics = []
        for result in results:
            response = result["response"]
            answer_idx = result["answer_idx"]
            options = result["options"]

            # The system prompt asks for the answer letter on the last line
            lines = [line.strip() for line in response.strip().splitlines()]
            last_line = next((line for line in reversed(lines) if line), "")
            # Strip common punctuation suffixes like "D." or "D)"
            last_line = last_line.strip(".:)( ")
            pred = last_line[0].upper() if last_line and last_line[0].upper() in options else None

            results_metrics.append(result | {
                "pred": pred,
                "correct": pred == answer_idx,
            })

        n = len(results_metrics)
        n_correct = sum(r["correct"] for r in results_metrics)
        summary_metrics = {
            "accuracy": n_correct / n if n > 0 else 0.0,
            "n": n,
            "n_correct": n_correct,
        }
        return results_metrics, summary_metrics