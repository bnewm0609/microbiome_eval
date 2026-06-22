
from pathlib import Path
import hashlib
import json
import random
from typing import Any

from microbiome_eval.tasks.base import BaseTask, PROJ_PATH
from microbiome_eval.llm import LLM


def get_negatives(papers, method, num_negatives, relevant_corpus_ids, paper_info):
    candidates = [
        p for p in papers
        if p["corpusId"] not in relevant_corpus_ids
        and p["year"] <= paper_info["year"]
        and p["corpusId"] != paper_info["corpusId"]
    ]
    if method == "rand":
        return random.sample(candidates, min(num_negatives, len(candidates)))
    if method == "hard_negatives":
        from sentence_transformers import SentenceTransformer, util
        if not candidates:
            return []

        def query_text(p):
            question = p["question"]
            query_text = f"Identify which of papers are relevant to answer the following question: {question}"
            return query_text.strip()

        def paper_text(p):
            title = p.get("title", "")
            abstract = p.get("abstract", "") or ""
            return f"Title: {title}\nAbstract: {abstract}".strip()

        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode(query_text(paper_info), convert_to_tensor=True)
        candidate_embs = model.encode([paper_text(p) for p in candidates], convert_to_tensor=True)

        scores = util.cos_sim(query_emb, candidate_embs)[0]
        top_indices = scores.topk(min(num_negatives, len(candidates))).indices.tolist()
        return [candidates[i] for i in top_indices]
    else:
        raise ValueError(f"Unsupported method: {method}")


class LitReviewReranking(BaseTask):
    """
    The task here is given set of paper abstracts and a research question, identify which papers are relevant for answering the research questions.

    The idea is that this collection of papers would be the result of calling some retrieval model.

    For now, there are 700 questions about 300 sources papers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "microbiome_litqa"

    @staticmethod
    def add_arguments(parser):
        # parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Which split of the dataset to use.")
        parser.add_argument("--num_abstracts", type=int, default=100)
        parser.add_argument("--num_relevant", default=None, type=int, help="Number of relevant papers to include in the prompt (default: include all up to num_abstracts)")
        parser.add_argument("--negative_selection_method", default="rand", choices=["rand", "hard_negatives"], help="Method for selecting negative papers to include in the prompt")
    
    def get_prompts(self):
        with open("data/gmrepo/papers/lit_retrieval/relevances.jsonl") as f:
            relevances = [json.loads(line) for line in f]

        with open("data/gmrepo/papers/lit_retrieval/papers.jsonl") as f:
            papers = [json.loads(line) for line in f]
            papers_by_corpus_id = {p["corpusId"]: p for p in papers}
        
        def format_paper_for_prompt(paper):
            return f"CorpusId: {paper['corpusId']}\nTitle: {paper['title']}\nAbstract: {paper['abstract']}".strip() 

        
        prompt_template = """
I'm writing a paper looking at answering the following question. Which of the following papers should I cite in my work?
Question: {question}

Your answer should be a json list of the corpusIds of the relevant papers, like this:
["12345", "67890", ...]

Here are the papers:
{papers}
""".strip()

        random.seed(self.config["seed"])
        num_abstracts = self.config["num_abstracts"]
        num_relevant = self.config["num_relevant"]
        method = self.config["negative_selection_method"]
        prompts = []
        for sample_qa in relevances:
            relevant_papers_all = [papers_by_corpus_id[cid] for cid in sample_qa["relevant_references"] if cid in papers_by_corpus_id]

            # sample the relevant papers
            relevant_papers = random.sample(relevant_papers_all, min(num_relevant, len(relevant_papers_all)))

            # sample irrelevant papers
            negatives = get_negatives(papers, method, num_abstracts - num_relevant, sample_qa["relevant_references"], sample_qa)

            # shuffle the relevant_papers and negatives together
            prompt_papers = relevant_papers + negatives
            random.shuffle(prompt_papers)

            # format the papers for the prompt
            formatted_papers = "\n\n".join([format_paper_for_prompt(p) for p in prompt_papers])

            prompt_text = prompt_template.format(
                question=sample_qa["question"],
                papers=formatted_papers
            )
            prompts.append(sample_qa | {
                "prompt": prompt_text,
                "system_messsage": "You are a helpful assistant.",
                "relevant_papers": relevant_papers,
                "negatives": negatives,
            })
    
        # shuffle the dataset if seed is set
        if self.config["seed"] is not None:
            random.seed(self.config["seed"])
            random.shuffle(prompts)
        
        # apply start and limit
        limit = self.config.get("limit", len(prompts))
        if limit is None:
            limit = len(prompts)
        start = self.config.get("start", 0)
        prompts = prompts[start: start + limit]
        return prompts
    

    def get_metrics(self, predicted_corpus_ids, relevant_papers):
        pred_ids = {int(ref) for ref in predicted_corpus_ids}
        true_ids = {rp["corpusId"] for rp in relevant_papers}

        tp = len(true_ids.intersection(pred_ids))
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }


    def evaluate_responses(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        outputs = []
        for result in results:

            response = result["response"]
            try:
                predicted_corpus_ids = json.loads(response.split("```json")[-1].split("```")[0].strip())
            except json.JSONDecodeError:
                predicted_corpus_ids = []

            pids = []
            for cid in predicted_corpus_ids:
                try:
                    pids.append(int(cid))
                except ValueError:
                    pass
            predicted_corpus_ids = pids
            metrics = self.get_metrics(predicted_corpus_ids, result["relevant_papers"])
            outputs.append(result | {
                "metrics": metrics,
                "predicted_corpus_ids": predicted_corpus_ids,
            })
        
        n = len(outputs)
        summary_metrics = {
            "f1": sum(r["metrics"]["f1"] for r in outputs) / n if n > 0 else 0.0,
            "precision": sum(r["metrics"]["precision"] for r in outputs) / n if n > 0 else 0.0,
            "recall": sum(r["metrics"]["recall"] for r in outputs) / n if n > 0 else 0.0,
            "n": n,
        }
        return outputs, summary_metrics