# Microbiome Foundation Model Eval Harness

## Installation:
```bash
uv sync
```

## Downloading Data
```bash
bash scripts/download_data.sh
```


## Running:

### Running local model servers:


```bash
# launch the proxy/load balancing server to enable scaling up number of slurm jobs for any given collection of models and provide a unified endpoint to hit:
# Because of the proxy server, you can launch a gemma 31B 2gpu job, then later another gemma 31B 4gpu job, and you can hit both models with a single endpoint.
uv run python scripts/submit_job.py vllm-proxy "uv run -- python scripts/launch_vllm_proxy_server.py" --ngpu 0 --partition gpu-a100 --time "7-22" --mem 64G

# launch the vllm servers:
# E.g. gemma-4-31B-it on 2 gpus
uv run python scripts/submit_job.py gemma-4-31B-it-2gpu "uv run -- python scripts/launch_vllm_server.py \"uv run -- vllm serve google/gemma-4-31B-it --data-parallel-size 2 --host 0.0.0.0 --api-key synthesis_rc --max-model-len 36032 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template vllm_servers/tool_chat_templates/tool_chat_template_gemma4.jinja --enable-prefix-caching --log-error-stack --max-num-batched-tokens 2496\"" --ngpu 2 --partition gpu-a100 --time 24:00:00 --mem 128G

# qwen
uv run python scripts/submit_job.py Qwen/Qwen3.5-9B "uv run -- python scripts/launch_vllm_server.py \"uv run -- vllm serve Qwen/Qwen3.5-9B --data-parallel-size 2 --host 0.0.0.0 --api-key synthesis_rc --max-model-len 36032 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching --log-error-stack\"" --ngpu 2 --partition gpu-a100 --time 24:00:00 --mem 128G
```

### Run the evals



```bash


# microbiome reasoning - google/gemma-4-31B-it
uv run -- python src/microbiome_eval/evaluate.py --model google/gemma-4-31B-it --generation_kwargs '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}' --task microbiome_reasoning --seed 31 --limit 500 --out_dir results/run-1_500 --max_workers 20


# med_qa
uv run -- python src/microbiome_eval/evaluate.py --model Qwen/Qwen3-4B-Thinking-2507 --generation_kwargs '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 16384}' --task med_qa --split dev --seed 31 --limit 10 --out_dir results/debugging

# {'accuracy': 0.6, 'n': 10, 'n_correct': 6}
# Metric results saved to: results/debugging/med_qa_Qwen_Qwen3-4B-Thinking-2507_10_sd31_gkw_ctk-enable_thinking-true-mt-16384-mp-0-t-0.6-tk-20-tp-0.95/metrics.jsonl


# disease_classification
uv run -- python src/microbiome_eval/evaluate.py --model Qwen/Qwen3-4B-Thinking-2507 --generation_kwargs '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 16384}' --task disease_classification --taxa genus --seed 31 --limit 10 --out_dir results/debugging
```