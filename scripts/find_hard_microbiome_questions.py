from argparse import ArgumentParser
import json
import hashlib
from datasets import load_dataset

from microbiome_eval.llm import LLM

def healthbench(args):
    with open("data/healthbench/2025-05-07-06-14-12_oss_eval.jsonl") as f:
        data = [json.loads(line) for line in f]
    
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

    prompts = []
    for sample in data:
        all_criteria = []
        for ri, rubric in enumerate(sample["rubrics"]):
            all_criteria.append(
                f"Rubric {ri + 1}:\n{rubric['criterion']}"
            )
        prompts.append([{
            "role": "user",
            "content": prompt_template.format(
            prompt=sample['prompt'][0]['content'],
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
    
    with open(args.output_path, "a") as f:
        filter_hash = hashlib.md5((model.model_name + json.dumps(gen_kwargs) + prompt_template).encode('utf-8')).hexdigest()
        for i in keep_idxs:
            f.write(
                json.dumps(
                    {
                        "_dataset_name": "healthbench",
                        "_idx": i,
                        "_filter_model": model.model_name,
                        "_filter_gen_kwargs": gen_kwargs,
                        "_filter_hash": filter_hash,
                        "sample": data[i],
                    }
                ) + "\n"
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

    prompts = []
    for sample in data:
        all_criteria = []
        for ri, rubric in enumerate(sample["rubric_items"]):
            all_criteria.append(
                f"- {rubric['criterion_text']}"
            )
        prompts.append([{
            "role": "user",
            "content": prompt_template.format(
            prompt=sample['conversation']['messages'][0]['content'],
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
    
    with open(args.output_path, "a") as f:
        filter_hash = hashlib.md5(("healthbench_professional" + model.model_name + json.dumps(gen_kwargs) + prompt_template).encode('utf-8')).hexdigest()
        for i in keep_idxs:
            f.write(
                json.dumps(
                    {
                        "_dataset_name": "healthbench_professional",
                        "_idx": i,
                        "_filter_model": model.model_name,
                        "_filter_gen_kwargs": gen_kwargs,
                        "_filter_hash": filter_hash,
                        "_version": 0,
                        "sample": data[i],
                    }
                ) + "\n"
            )
    
    print(f"Kept {len(keep_idxs)} samples, discarded {len(discard_idxs)} samples, and marked {len(amgiguous_idxs)} samples as ambiguous.")
    print("Saved filtered samples to", args.output_path)




def main():
    argp = ArgumentParser()
    argp.add_argument("--dataset_name")
    # argp.add_argument("--dataset_path")
    # argp.add_argument("--model")
    argp.add_argument("--output_path", default="data/master_list_hard_microbiome.jsonl")
    args = argp.parse_args()

    globals()[args.dataset_name](args)
    


if __name__ == "__main__":
    main()