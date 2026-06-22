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
# uv run -- python src/microbiome_eval/evaluate.py --model google/gemma-4-31B-it --generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}' --task microbiome_litqa --litqa_setting high_citation_qa_generation --seed 31 --out_dir results/run-1_500 --max_workers 20

# uv run -- python src/microbiome_eval/evaluate.py --model google/medgemma-1.5-4b-it --generation_kwargs '{"temperature": 0.0}' --task microbiome_litqa --litqa_setting high_citation_qa_generation --seed 31  --out_dir results/run-1_500/ --max_workers 20


# uv run -- python src/microbiome_eval/evaluate.py --model Qwen/Qwen3.5-9B --generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": true}}' --task microbiome_litqa --litqa_setting high_citation_qa_generation --seed 31 --out_dir results/run-1_500 --max_workers 20


# launch the servers
# uv run python scripts/submit_job.py medgemma-1.5-4b-it "uv run -- python scripts/launch_vllm_server.py \"uv run -- vllm serve google/medgemma-1.5-4b-it --data-parallel-size 2 --host 0.0.0.0 --api-key synthesis_rc --max-model-len 36032   --chat-template vllm_servers/tool_chat_templates/chat_template_medgemma15.jinja --enable-prefix-caching --log-error-stack --max-num-batched-tokens 2496\"" --ngpu 2 --partition gpu-a100 --time 24:00:00 --mem 128G

# uv run python scripts/submit_job.py Qwen/Qwen3.5-9B "uv run -- python scripts/launch_vllm_server.py \"uv run -- vllm serve Qwen/Qwen3.5-9B --data-parallel-size 2 --host 0.0.0.0 --api-key synthesis_rc --max-model-len 36032 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching --log-error-stack\"" --ngpu 2 --partition gpu-a100 --time 24:00:00 --mem 128G

# uv run python scripts/submit_job.py gemma-4-31B-it "uv run -- python scripts/launch_vllm_server.py \"uv run -- vllm serve google/gemma-4-31B-it --data-parallel-size 2 --host 0.0.0.0 --api-key synthesis_rc --max-model-len 36032 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template vllm_servers/tool_chat_templates/tool_chat_template_gemma4.jinja --enable-prefix-caching --log-error-stack --max-num-batched-tokens 2496\"" --ngpu 2 --partition gpu-a100 --time 24:00:00 --mem 128G

# run the evals:
# models=(
#     # "google/medgemma-1.5-4b-it"
#     "Qwen/Qwen3.5-9B"
#     # "google/gemma-4-31B-it"
# )
# gen_kwargs=(
#     #'{"temperature": 0.0}'
#     '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": true}}'
#     #'{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}'
# )
# out_dir="results/2_hard_microbiome_qs_v1/"
# data_config='{"healthbench": "001", "healthbench_professional": "001", "MedXpertQA": "001"}'

# for i in "${!models[@]}"; do
#     model="${models[$i]}"
#     gen_kwargs="${gen_kwargs[$i]}"
    
#     uv run -- python scripts/submit_job.py "$model-2_hard_microbiome_qs_v1" "uv run -- python src/microbiome_eval/evaluate.py --model \"$model\" --generation_kwargs '$gen_kwargs' --seed 31 --limit 500 --out_dir \"$out_dir\" --max_workers 20 --task hard_microbiome_qs --data_config '$data_config'" --ngpu 0 --partition gpu-a100 --time 24:00:00 --mem 64G
# done

# gen_kwargs='{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}'

# uv run -- python scripts/submit_job.py "gemma_method_errors_v2" "uv run -- python src/microbiome_eval/evaluate.py --model google/gemma-4-31B-it --generation_kwargs '$gen_kwargs' --task methods_errors --seed 32 --limit 500 --out_dir results/debugging_methods_errors/ --max_workers 20" --ngpu 0 --partition gpu-a100 --time 24:00:00 --mem 64G --out_dir results/debugging_methods_errors/
