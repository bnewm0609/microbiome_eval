from argparse import ArgumentParser
import json
import hashlib
from pathlib import Path
from datasets import load_dataset

from microbiome_eval.llm import LLM

PROMPTS = {
    "v1": """
Help me filter the following medical-related prompts provided to a language model. I want to keep only prompts that potentially have to do with the human gut microbiome.
You are provided with the prompt and a set of criteria that are used to evaluate the model's response to the prompt.
You should respond with "keep" if the main topic of the question or rubric features the human gut microbiome, and "discard" if it is not.
If you are unsure, you should respond with "keep" to be safe. 

Here is the prompt:
{prompt}

And here are the criteria:
{criteria}

---

Now it's your turn. Identify whether this prompt is about the microbiome and if it should be kept or discarded. The last line of your response should be either "keep" or "discard", but lines before that can be your reasoning.
""".strip(),

    # v1 multiturn version - it provides the entire conversation rather than just the initial question.
    "v1_mt": """
Help me filter the following medical-related conversations provided to a language model. I want to keep only conversations whose next turn requires knowledge of the human gut microbiome.
You are provided with the conversation and a set of criteria that are used to evaluate the model's response in the next turn of the conversation.
You should respond with "keep" if the next conversation turn requires knowledge of the human gut microbiome, and "discard" if it does not.
If you are unsure, you should respond with "keep" to be safe. 

Here is the conversation:
{prompt}

---

And here are the criteria to judge the model's next response:
{criteria}

---

Now it's your turn. Identify whether the conversation is about the microbiome and if it should be kept or discarded. The last line of your response should be either "keep" or "discard", but lines before that can be your reasoning.
""".strip(),
}



def healthbench(args):
    with open("data/healthbench/2025-05-07-06-14-12_oss_eval.jsonl") as f:
        data = [json.loads(line) for line in f]
    
    # model = LLM("google/gemma-4-31B-it")
    model = LLM(args.model)
    gen_kwargs = json.loads(args.gen_kwargs)
    out_dir = Path(args.output_path) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    run_num = len(list(out_dir.glob("run_*.jsonl")))
    out_file = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.jsonl"
    print(f"Saving to {out_file}")



    prompt_template = PROMPTS[args.prompt]


    prompts = []
    for sample in data:
        all_criteria = []
        for ri, rubric in enumerate(sample["rubrics"]):
            all_criteria.append(
                f"Rubric {ri + 1}:\n{rubric['criterion']}"
            )
        conversation= "\n\n".join([f"{turn['role']}:\n{turn['content']}" for turn in sample['prompt']])
        prompts.append([{
            "role": "user",
            "content": prompt_template.format(
                prompt=conversation,
                criteria="\n\n".join(all_criteria)
            )
        }])

    gen_kwargs = {}
    resp = model.batch_call(prompts, **gen_kwargs)
    keep_idxs = []
    discard_idxs = []
    amgiguous_idxs = []
    for i, r in enumerate(resp):
        r = r['content'].splitlines()[-1].strip().lower()
        if "keep" in r:
            keep_idxs.append(i)
        elif "discard" in r:
            discard_idxs.append(i)
        else:
            amgiguous_idxs.append(i)
    

    config = vars(args)
    config = {f"_{k}": v for k, v in config.items() if k not in ("output_path",)}
    with open(out_file, "w") as f:
        # filter_hash = hashlib.md5((model.model_name + json.dumps(gen_kwargs) + prompt_template).encode('utf-8')).hexdigest()
        for i in keep_idxs:
            f.write(
                json.dumps({"_idx": i, "_run": run_num, **config, "_filter_response": resp[i], "sample": data[i]}) + "\n"
            )
    
    print(f"Kept {len(keep_idxs)} samples, discarded {len(discard_idxs)} samples, and marked {len(amgiguous_idxs)} samples as ambiguous.")
    print("Saved filtered samples to", args.output_path)



def healthbench_professional(args):

    data = load_dataset("openai/healthbench-professional")["test"]
    
    model = LLM("google/gemma-4-31B-it")

    prompt_template = """
Help me filter the following medical-related prompts provided to a language model. I want to keep only prompts that potentially have to do with the human gut microbiome.
You are provided with the prompt and a set of criteria that are used to evaluate the model's response to the prompt.
You should respond with "keep" if the main topic of the question or rubric features the human gut microbiome, and "discard" if it is not.
If you are unsure, you should respond with "keep" to be safe. 

Here is the prompt:
{prompt}

And here are the criteria:
{criteria}

---

Now it's your turn. Identify whether this prompt is about the microbiome and if it should be kept or discarded. The last line of your response should be either "keep" or "discard", but lines before that can be your reasoning.
""".strip()

    model = LLM(args.model)
    gen_kwargs = json.loads(args.gen_kwargs)
    out_dir = Path(args.output_path) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    run_num = len(list(out_dir.glob("run_*.jsonl")))
    out_file = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.jsonl"
    print(f"Saving to {out_file}")


    prompt_template = PROMPTS[args.prompt]


    prompts = []
    for sample in data:
        all_criteria = []
        for ri, rubric in enumerate(sample["rubric_items"]):
            all_criteria.append(
                f"- {rubric['criterion_text']}"
            )
        # breakpoint()
        conversation= "\n\n".join([f"{turn['role']}:\n{turn['content']}" for turn in sample['conversation']['messages']])
        prompts.append([{
            "role": "user",
            "content": prompt_template.format(
            prompt=conversation,
            criteria="\n".join(all_criteria)
            )
        }])

    gen_kwargs = {}
    resp = model.batch_call(prompts, **gen_kwargs)
    keep_idxs = []
    discard_idxs = []
    amgiguous_idxs = []
    for i, r in enumerate(resp):
        r = r['content'].splitlines()[-1].strip().lower()
        if "keep" in r:
            keep_idxs.append(i)
        elif "discard" in r:
            discard_idxs.append(i)
        else:
            amgiguous_idxs.append(i)
    

    config = vars(args)
    config = {f"_{k}": v for k, v in config.items() if k not in ("output_path",)}
    with open(out_file, "w") as f:
        # filter_hash = hashlib.md5((model.model_name + json.dumps(gen_kwargs) + prompt_template).encode('utf-8')).hexdigest()
        for i in keep_idxs:
            f.write(
                json.dumps({"_idx": i, "_run": run_num, **config, "_filter_response": resp[i], "sample": data[i]}) + "\n"
            )
    
    print(f"Kept {len(keep_idxs)} samples, discarded {len(discard_idxs)} samples, and marked {len(amgiguous_idxs)} samples as ambiguous.")
    print("Saved filtered samples to", out_file)




def main():
    argp = ArgumentParser()
    argp.add_argument("--dataset")
    # argp.add_argument("--dataset_path")
    argp.add_argument("--model")
    argp.add_argument("--gen_kwargs", default="{}")
    argp.add_argument("--prompt", default="prompt_template")
    argp.add_argument("--output_path", default="data/filtered_difficult_datasets/")
    args = argp.parse_args()

    globals()[args.dataset](args)
    


if __name__ == "__main__":
    main()