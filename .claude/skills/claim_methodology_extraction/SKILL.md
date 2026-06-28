---
name: claim_methodology_extraction
description: Extracts claims and methodologies from microbiome papers for evaluating weaker LLMs.
disable-model-invocation: true
---

I want to curate an evaluation task that requires reproducing key quantitative results from bioinformatics papers about the gut microbiome. 
You are provided with a path to a OCRed version of a paper, and your job is to:
1. Identify the top most important quantitative or qualitative claims the paper makes that are important to reproduce. You should have at most three claims.
2. For each claim, provide a detailed description of the methodology the paper used to support the claim. We are interested in evaluating microbiome-specific knowledge/methodologies, so make sure your summary focuses on the microbiome-specific parts (in addition to other parts like statistical tests.)

The model we are evaluating will **not** have access to the paper itself, so your claim and description of the methodology should contain all necessary context for understanding them.

Here is an example claim and methodology description:

## Claim
There are no significant differences in alpha-diversity and bacterial prevalence between responders and non-responders to immune checkpoint inhibitor therapy in metastatic melanoma patients.

## Methodology
A researcher has pre-treatment stool samples from 25 metastatic melanoma patients (12 responders, 13 non-responders) who received immune checkpoint inhibitor therapy. Samples were processed with metagenomic shotgun sequencing and taxonomically profiled using MetaPhlAn2, yielding species-level relative abundances. To assess alpha-diversity, the researcher computes a Shannon diversity index for each sample and tests for a difference between responders and non-responders using a Kruskal–Wallis rank-sum test. To assess prevalence differences, the researcher binarizes abundance values (1 if abundance > 0, 0 otherwise) and runs a Fisher's exact test and a logistic regression for each taxon individually, then corrects for multiple testing using the Holm method. The significance threshold is set at P < 0.05 for both nominal and corrected values. The researcher finds that there are no significant differences in alpha-diversity and bacterial prevalence (P > 0.05).


You should output your claims and methodology descriptions in markdown format, in the same directory as the paper. 
**Each claim and methodology pair should be in a separate markdown file, and the filename should be the claim number followed by `.md`, e.g. `claim_1.md`, `claim_2.md`, etc.**

Here is the path to the paper you should analyze: $ARGUMENTS