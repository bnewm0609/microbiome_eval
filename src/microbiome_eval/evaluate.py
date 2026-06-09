"""
An updated version of `evaluate.py` started on May 13, 2026

Main differences from `evaluate.py`:
- Runs against api models rather than loading in models themselves
- Doesn't handling filling prompts or truncating examples. That will be handled by the task or agents when appropriate, so this is a bit mroe streamlined
- Saves intermediate results more frequently
- Cuts out some of the cruft from unused evals

"""
from argparse import ArgumentParser
from tqdm import tqdm
import json
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent


from microbiome_eval.llm import LLM, wait_for_vllm
from microbiome_eval.tasks import load_task


def compress_generation_kwargs(kwargs: dict) -> str:
    """Return a compact, path-safe string representation of generation kwargs."""
    if not kwargs:
        return ""
    parts = []
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (dict, list)):
            v_str = json.dumps(v, separators=(",", "-")).replace('"', "").replace(" ", "").replace("{", "").replace("}", "")
        else:
            v_str = str(v)
        k="".join([k_word[0] for k_word in k.split("_")])
        parts.append(f"{k}-{v_str}")
    return "-".join(parts)


def compress_config(config: dict, exclude: set = None) -> str:
    """Return a compact, path-safe string of all config values, excluding specified keys."""
    exclude = exclude or set()
    parts = []
    for k, v in sorted(config.items()):
        if k in exclude:
            continue
        k_abbr = "".join(word[0] for word in k.split("_"))
        if k == "generation_kwargs" or k == "judge_generation_kwargs" or k == "data_config":
            gkw = json.loads(v) if isinstance(v, str) and v else {}
            v_str = compress_generation_kwargs(gkw) if gkw else "default"
        elif isinstance(v, str):
            v_str = v.replace("/", "_")
        elif v is None:
            v_str = "none"
        elif isinstance(v, bool):
            v_str = str(v).lower()
        else:
            v_str = str(v)
        parts.append(f"{k_abbr}-{v_str}")
    return "_".join(parts)


# def run(model, prompt, config):
#     generation_kwargs = config.get("generation_kwargs", "{}")
#     generation_kwargs = json.loads(generation_kwargs)  # No explicit defaults
#     if not generation_kwargs:
#         print("No generation kwargs provided, using defaults.")
    
#     messages = []
#     if prompt.get("system_message"):
#         messages.append({"role": "system", "content": prompt["system_message"]})
#     messages.append({"role": "user", "content": prompt["prompt"]})
#     response = model.call(messages, **generation_kwargs)
#     messages.append({"role": "assistant"} | response)
#     response = response["content"]

#     return {
#         "response": response,
#         **prompt,
#         "messages": messages,
#     }


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", help="Name of the task to run", choices=["disease_classification", "microbiome_reasoning", "microbiome_litqa", "med_qa", "hard_microbiome_qs", "methods_errors"])
    parser.add_argument("--model", help="Name of the model to evaluate")
    parser.add_argument("--generation_kwargs", help="Generation kwargs to use for the model calls, e.g. --generation_kwargs '{\"temperature\": 0.7, \"max_tokens\": 512}'", default=None)
    parser.add_argument("--start", type=int, default=0, help="Which example index to start running at (useful when parallelizing synthetic data generation",)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to run for debugging, e.g. --limit 5")
    parser.add_argument("--out_dir", default="./dag_agent_outputs")
    parser.add_argument("--max_workers", type=int, default=5, help="Max parallel workers for number of simultaneous agent calls (default: 5)")
    parser.add_argument("--seed", default=None, help="Random seed for reproducibility. If not set, data will not be shuffled.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode, which runs examples sequentially and prints out the full response for each example (default: False)")
    
    # TODO: move these to be task specific arguments that get added in the task's add_arguments method
    # parser.add_argument("--judge", help="Name of the judge model to use for evaluation, e.g. gpt-4-0613")
    # parser.add_argument("--judge_kwargs", help="Judge model generation kwargs to use for evaluation, e.g. gpt-4-0613")

    args, _ = parser.parse_known_args()
    task_cls = load_task(args.task)
    task_cls.add_arguments(parser)
    args = parser.parse_args()


    config = vars(args)
    task = task_cls(config)

    # form the output directory name based on the config
    generation_kwargs = json.loads(config.get("generation_kwargs", "{}"))  # No explicit defaults
    gkw_suffix = f"_gkw_{compress_generation_kwargs(generation_kwargs)}" if generation_kwargs else ""
    # out_dir = Path(args.out_dir) / f"{args.task}_{args.model.replace('/', '_')}_{args.limit if args.limit is not None else 'all'}_sd{args.seed}{gkw_suffix}"
    out_dir = Path(args.out_dir) / compress_config(config, exclude={"max_workers", "out_dir"})
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {out_dir}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(config | config, f)


    prompts = task.get_prompts()

    generations_file = out_dir / "generations.jsonl"
    if generations_file.exists():
        print(f"Generations file already exists at: {generations_file}, skipping generation.")
        with open(generations_file) as f:
            generations = [json.loads(line) for line in f]
    else:

        # load model
        model = LLM(args.model, use_cache=False, debug=args.debug)

        # create prompts
        messages_batched = []
        for prompt in prompts:
            messages = []
            if prompt.get("system_message"):
                messages.append({"role": "system", "content": prompt["system_message"]})
            
            if isinstance(prompt["prompt"], list):
                messages.extend(prompt["prompt"])
            else:
                messages.append({"role": "user", "content": prompt["prompt"]})
            messages_batched.append(messages)

        responses = model.batch_call(messages_batched, max_workers=args.max_workers, **generation_kwargs)
        generations = []
        for prompt, messages, response in zip(prompts, messages_batched, responses):
            generations.append({
                "response": response["content"],
                **prompt,
                "messages": messages + [{"role": "assistant"} | response],
            })

        generations_file = out_dir / "generations.jsonl"
        with open(generations_file, "w") as f:
            for gen in generations:
                f.write(json.dumps(gen) + "\n")
        print(f"Saved {len(generations)} generations to:\n{generations_file}")

    # generations_file = out_dir / "generations.jsonl"
    # if generations_file.exists():
    #     print(f"Generations file already exists at: {generations_file}, skipping generation.")
    #     with open(generations_file) as f:
    #         generations = [json.loads(line) for line in f]

    # next, run evaluation:
    metrics_file = out_dir / "metrics.jsonl"
    if not metrics_file.exists():
        results = []
        with open(generations_file) as f:
            for line in f:
                result = json.loads(line)
                results.append(result)

        if results:
            results_metrics, summary_metrics = task.evaluate_responses(results)
            
            with open(metrics_file, "w") as f:
                for result_metrics in results_metrics:
                    f.write(json.dumps({k: v for k, v in result_metrics.items()}) + "\n")
            
            print(summary_metrics)
            with open(out_dir / "summary_metrics.json", "w") as f:
                json.dump(summary_metrics, f)

            print(f'Metric results saved to: {metrics_file}')
    else:
        print(f'Metric results already exist at: {metrics_file}, skipping evaluation.')

if __name__ == "__main__":
    main()