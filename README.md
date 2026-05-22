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
```bash
# med_qa
uv run -- python src/microbiome_eval/evaluate.py --model Qwen/Qwen3-4B-Thinking-2507 --generation_kwargs '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 16384}' --task med_qa --split dev --seed 31 --limit 10 --out_dir results/debugging

# {'accuracy': 0.6, 'n': 10, 'n_correct': 6}
# Metric results saved to: results/debugging/med_qa_Qwen_Qwen3-4B-Thinking-2507_10_sd31_gkw_ctk-enable_thinking-true-mt-16384-mp-0-t-0.6-tk-20-tp-0.95/metrics.jsonl


# disease_classification
uv run -- python src/microbiome_eval/evaluate.py --model Qwen/Qwen3-4B-Thinking-2507 --generation_kwargs '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 16384}' --task disease_classification --taxa genus --seed 31 --limit 10 --out_dir results/debugging
```