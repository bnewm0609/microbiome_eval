#!/usr/bin/bash
set -x

# cd /gscratch/xlab/blnewman/microbiome_eval/
# uv run -- python src/microbiome_eval/tasks/microbiome_litqa.py --pipeline research_qa_generation --model google/gemma-4-31B-it --generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}' --end_step distractors --out_dir data/gmrepo/papers/research_qa_v1_gemma4_31B/



# med qa
# model="Qwen/Qwen3.5-9B"
# gen_kwargs='{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": true}}'
# out_dir="results/run-1_500"
# uv run -- python src/microbiome_eval/evaluate.py --model "$model" --generation_kwargs "$gen_kwargs" --task med_qa --seed 31 --limit 500 --out_dir "$out_dir" --max_workers 20

# # microbiome reasoning
# # also uses a judge model, it's just hidden
# uv run -- python src/microbiome_eval/evaluate.py --model "$model" --generation_kwargs "$gen_kwargs" --task microbiome_reasoning --seed 31 --limit 500 --out_dir "$out_dir" --max_workers 20

# # disease classification
# uv run -- python src/microbiome_eval/evaluate.py --model "$model" --generation_kwargs "$gen_kwargs" --task disease_classification --taxa genus --seed 31 --limit 500 --out_dir "$out_dir" --max_workers 20 --judge_model google/gemma-4-31B-it --judge_generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}'



### Next, run on our v1 "high_citation" setting. This dataset is small and not very good, but it's an initial poc for in-context litqa
uv run -- python src/microbiome_eval/evaluate.py --model google/gemma-4-31B-it --generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}' --task microbiome_litqa --litqa_setting high_citation_qa_generation --seed 31 --out_dir results/run-1_500 --max_workers 20

uv run -- python src/microbiome_eval/evaluate.py --model google/medgemma-1.5-4b-it --generation_kwargs '{"temperature": 0.0}' --task microbiome_litqa --litqa_setting high_citation_qa_generation --seed 31  --out_dir results/run-1_500/ --max_workers 20


uv run -- python src/microbiome_eval/evaluate.py --model Qwen/Qwen3.5-9B --generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": true}}' --task microbiome_litqa --litqa_setting high_citation_qa_generation --seed 31 --out_dir results/run-1_500 --max_workers 20