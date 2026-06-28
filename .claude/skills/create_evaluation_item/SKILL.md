---
name: create_evaluation_item
description: Extracts claims and methodologies from microbiome papers for evaluating weaker LLMs.
---

I want to curate a language model evaluation task that requires reasoning about microbiome-related experiments, protocols and workflows. The way the task is designed is as follows:
The model will be povided with a claim and a description of the methodology followed to support the claim.
However, we will intentionally introduce a single error into the methodology description that renders the methodolgy incorrect.
The model's task will be to identify which part of the methodology is incorrect.

You will help create evaluation items for this task.
You will be given the path to a scientific paper (its a markdown file), and a path to a particular `claim.md` file, that contains a claim and a summary of the methodology used to support that claim.
You will then complete the following steps:
1. Identify a few *key decision points* made in the methdology. These are places where the authors made a decision that helped support their claim, but there was a pathway for a different decision.
2. For each of these decision points, identify an *incorrect alternative*---this is something the authors could have done, but would have been incorrect and undermined the validity of their evidence. We are interested in evaluating microbiome-specific knowledge, so steps 1-2 should focus on microbiome-related knolwedge and concepts rather than basic statistics. After step 2, you should have a list of key decision points and their corresponding incorrect alternatives.
3. For each element in this list:
  a. create an *incorrect methodology description*. This is a copy of the methodology description where the incorrect alternative is subtly incorporated. The incorrect methdology should otherwise be the same as the original.
  b. Create a clear and concise rubric to evaluate whether the model identified the correct error. The rubric should start with language like "The model should identify..." 

The final output should be a line appended to a jsonl file for each element of the key decision point - incorrect alternatives list. The file lives in the same directory as the paper called `eval_items.jsonl`. There should be one row added per item. Each added row should have the following fields:

{
"paper_id": paper id (from the paper path),
"claim_id": number (from the claim path),
"claim": the claim from the (claim path),
"methodology": the methodology from the claim path,
"decision point": summary of the decision point,
"incorrect alternative": summary of the incorrect alternative,
"incorrect_methodology": the generated incorrect version of the methodology,
"rubric": the generated rubric,
}

Do not worry too much about the quality of the items at this point. The goal is to generate a diverse set of items that are relevant to the microbiome, the paper, and the claim.


The paper is at this path: $1
The claim is at this path: $2