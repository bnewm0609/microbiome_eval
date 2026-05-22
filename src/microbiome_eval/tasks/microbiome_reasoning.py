import json
import random

from microbiome_eval.tasks.base import BaseTask, PROJ_PATH


class MicrobiomeReasoningTask(BaseTask):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--taxa", default="genus", choices=["genus", "species"], help="Which taxonomy level to use. 2075/2120 have genus taxonomies. 512/2120 have species-level.")

    def get_prompts(self) -> list[dict]:
        """
        Returns a list of prompts for the disease classification task.
        Each prompt is a dict with the following keys:
        - "prompt": The prompt to send to the model
        - "label": The correct label for the prompt

        Can download the dataset from:
        
        """
        # load the dataset
        split = self.config["split"]
        with open(PROJ_PATH / f"data/gmrepo/hf_dataset/gmrepo_eval.jsonl") as f:
            dataset = json.load(f)

        if self.config["taxa"] == "genus":
            prompt_key = "prompt_genus"
        else:
            prompt_key = "prompt_species"

        system_message = "You are a helpful and precise assistant for answering medical diagnostic questions. The last line of your answer should be a short word or phrase corresponding to the disease you think is most likely given the patient's gut microbiome taxonomy."
        prompts = []
        for sample in dataset:
            if not sample[prompt_key] or isinstance(sample[prompt_key], float):  # skip samples with empty prompts (which are represented as NaNs in the jsonl)
                continue
            prompts.append(
                sample | {"prompt": sample["prompt_genus"]})
        

        # shuffle the dataset if seed is set
        if self.config["seed"] is not None:
            random.seed(self.config["seed"])
            random.shuffle(prompts)
        
        # apply start and limit
        start = self.config.get("start", 0)
        limit = self.config.get("limit", len(prompts))
        prompts = prompts[start: start + limit]

        return prompts