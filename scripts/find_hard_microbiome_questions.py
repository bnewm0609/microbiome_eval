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

    "medxpert_v1": """
Help me filter the following medical-related multiple choice question provided to evaluate a language model. I want to keep only questions that require knowledge of the human gut microbiome to answer.
You are provided with the question and answer choices, and you should respond with "keep" if the question requires knowledge of the human gut microbiome to answer, and "discard" if it does not.
If you are unsure, you should respond with "keep" to be safe. 

Here is the question:
{question}
{options}

Answer: {answer}

Now it's your turn. Identify whether the question requires knowledge of the human gut microbiome and if it should be kept or discarded. The last line of your response should be either "keep" or "discard", but lines before that can be your reasoning.
""".strip(),

    "BixBench_v1": """
Help me filter the following bioinformatics question designed to evaluate a language model. I want to keep questions whose data comes from or is related to the human gut microbiome.
You are provided with the question and answer choices, and you should respond with "keep" if the question is related to the human gut micriobiome, and "discard" if it does not.
If you are unsure, you should respond with "keep" to be safe. You are provided with the question, and some metadata including the hypothesis and result from the paper the question was derived from. This should inform your decision.

The last line of your response should be either "keep" or "discard", but lines before that should contain your reasoning.

Question: {question}
Hypothesis: {hypothesis}
Result: {result}
""".strip(),

    "labbench2_v1": """
Help me filter the following question designed to evaluate a language model. I want to keep questions whose data comes from or is related to the human gut microbiome.
You are provided with the question and some related information, and you should respond with "keep" if the question is related to the human gut micriobiome, and "discard" if it does not.
If you are unsure, you should respond with "keep" to be safe. You are provided with the question, and some metadata including a key passage that supports the answer (in some cases) and the ideal answer to the question. This should inform your decision.

The last line of your response should be either "keep" or "discard", but lines before that should contain your reasoning.

Question: {question}
Key Passage (might be empty): {key_passage}
Ideal answer: {answer}
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
    # load data/model/prompts
    data = load_dataset("openai/healthbench-professional")["test"]
    model = LLM(args.model)
    gen_kwargs = json.loads(args.gen_kwargs)
    prompt_template = PROMPTS[args.prompt]
    
    # prepare out directory
    out_dir = Path(args.output_path) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    run_num = len(list(out_dir.glob("run_*.jsonl")))
    out_file = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.jsonl"
    print(f"Saving to {out_file}")

    # format prompts
    prompts = []
    for sample in data:
        all_criteria = []
        for ri, rubric in enumerate(sample["rubric_items"]):
            all_criteria.append(
                f"- {rubric['criterion_text']}"
            )
        conversation= "\n\n".join([f"{turn['role']}:\n{turn['content']}" for turn in sample['conversation']['messages']])
        prompts.append([{
            "role": "user",
            "content": prompt_template.format(
            prompt=conversation,
            criteria="\n".join(all_criteria)
            )
        }])

    # run model and parse responses
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
    
    # save outputs
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


def MedXpertQA(args):
    """
    Valid prompts:
    - medxpert_v1
    """
    valid_prompts = {"medxpert_v1"}
    if args.prompt not in valid_prompts:
        raise ValueError(f"Invalid prompt: {args.prompt}. Valid prompts are: {valid_prompts}")

    # load data/model/prompts
    data = load_dataset("TsinghuaC3I/MedXpertQA", "Text")
    model = LLM(args.model)
    gen_kwargs = json.loads(args.gen_kwargs)
    prompt_template = PROMPTS[args.prompt]
    
    # prepare out directory
    out_dir = Path(args.output_path) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    run_num = len(list(out_dir.glob("run_*.jsonl")))
    out_file_dev = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.dev.jsonl"
    out_file_test = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.test.jsonl"
    print(f"Saving to {out_file_dev}")
    print(f"Saving to {out_file_test}")

    # let's filter both the dev and the test splits
    def filter_split(split):
        # format prompts
        prompts = []
        for sample in data[split]:
            options = "\n".join([f"{k}. {choice}" for k, choice in sample['options'].items()])
            prompts.append([{
                "role": "user",
                "content": prompt_template.format(
                    question=sample['question'],
                    options=options,
                    answer=sample['label']
                )
            }])

        # run model and parse responses
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
        
        # save outputs
        config = vars(args)
        config = {f"_{k}": v for k, v in config.items() if k not in ("output_path",)}
        out_file = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.{split}.jsonl"
        with open(out_file, "w") as f:
            # filter_hash = hashlib.md5((model.model_name + json.dumps(gen_kwargs) + prompt_template).encode('utf-8')).hexdigest()
            for i in keep_idxs:
                f.write(
                    json.dumps({"_idx": i, "_run": run_num, **config, "_split": split, "_filter_response": resp[i], "sample": data[split][i]}) + "\n"
                )
        
        print(f"Split {split}: Kept {len(keep_idxs)} samples, discarded {len(discard_idxs)} samples, and marked {len(amgiguous_idxs)} samples as ambiguous.")
        print("Saved filtered samples to", out_file)
        
        return keep_idxs, discard_idxs, amgiguous_idxs, resp
    
    filter_split("dev")
    filter_split("test")

def BixBench(args):
    valid_prompts = {"BixBench_v1"}
    if args.prompt not in valid_prompts:
        raise ValueError(f"Invalid prompt: {args.prompt}. Valid prompts are: {valid_prompts}")
    
    data = load_dataset("futurehouse/BixBench")
    model = LLM(args.model)
    gen_kwargs = json.loads(args.gen_kwargs)
    prompt_template = PROMPTS[args.prompt]

    # prepare out directory
    out_dir = Path(args.output_path) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    run_num = len(list(out_dir.glob("run_*.jsonl")))
    out_file = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.jsonl"
    print(f"Saving to {out_file}")

    # format prompts
    prompts = []
    for sample in data['train']:
        prompts.append([{
            "role": "user",
            "content": prompt_template.format(
                question=sample['question'],
                hypothesis=sample['hypothesis'],
                result=sample['result']
            )
        }])

    # run model and parse responses
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
    
    # save outputs
    config = vars(args)
    config = {f"_{k}": v for k, v in config.items() if k not in ("output_path",)}
    with open(out_file, "w") as f:
        # filter_hash = hashlib.md5((model.model_name + json.dumps(gen_kwargs) + prompt_template).encode('utf-8')).hexdigest()
        for i in keep_idxs:
            f.write(
                json.dumps({"_idx": i, "_run": run_num, **config, "_filter_response": resp[i], "sample": data["train"][i]}) + "\n"
            )
    
    print(f"Kept {len(keep_idxs)} samples, discarded {len(discard_idxs)} samples, and marked {len(amgiguous_idxs)} samples as ambiguous.")
    print("Saved filtered samples to", out_file)


def labbench2(args):
    valid_prompts = {"labbench2_v1"}
    if args.prompt not in valid_prompts:
        raise ValueError(f"Invalid prompt: {args.prompt}. Valid prompts are: {valid_prompts}")
    
    data = load_dataset("EdisonScientific/labbench2", "all", token=open(Path.home()/ ".hf_token_fs").read().strip())
    model = LLM(args.model)
    gen_kwargs = json.loads(args.gen_kwargs)
    prompt_template = PROMPTS[args.prompt]

    # prepare out directory
    out_dir = Path(args.output_path) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    run_num = len(list(out_dir.glob("run_*.jsonl")))
    out_file = out_dir / f"run_{run_num:03d}-prompt_{args.prompt}.jsonl"
    print(f"Saving to {out_file}")

    # format prompts
    prompts = []
    for sample in data['train']:
        prompts.append([{
            "role": "user",
            "content": prompt_template.format(
                question=sample['question'],
                key_passage=sample['key_passage'],
                answer=sample['ideal']
            )
        }])

    # run model and parse responses
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
    
    # save outputs
    config = vars(args)
    config = {f"_{k}": v for k, v in config.items() if k not in ("output_path",)}
    with open(out_file, "w") as f:
        # filter_hash = hashlib.md5((model.model_name + json.dumps(gen_kwargs) + prompt_template).encode('utf-8')).hexdigest()
        for i in keep_idxs:
            f.write(
                json.dumps({"_idx": i, "_run": run_num, **config, "_filter_response": resp[i], "sample": data["train"][i]}) + "\n"
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