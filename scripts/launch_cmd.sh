#!/usr/bin/bash
set -x

cd /gscratch/xlab/blnewman/microbiome_eval/
uv run -- python src/microbiome_eval/tasks/microbiome_litqa.py --pipeline research_qa_generation --model google/gemma-4-31B-it --generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}' --end_step distractors --out_dir data/gmrepo/papers/research_qa_v1_gemma4_31B/