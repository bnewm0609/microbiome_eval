from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import json


import openai
import requests
from tqdm import tqdm

from microbiome_eval.cache import LLMCache

VLLM_EXTRA_PARAMETERS = {
    "use_beam_search",
    "top_k",
    "min_p",
    "repetition_penalty",
    "length_penalty",
    "stop_token_ids",
    "include_stop_str_in_output",
    "ignore_eos",
    "min_tokens",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "truncate_prompt_tokens",
    "allowed_token_ids",
    "prompt_logprobs",
    "chat_template_kwargs",
}

def wait_for_vllm(hostname: str, port: int, model_name: str | None = None, timeout: int = 1200):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"http://{hostname}:{port}/health").status_code != 200:
                raise requests.exceptions.RequestException("not healthy")
            if model_name is not None:
                models = requests.get(f"http://{hostname}:{port}/v1/models").json()
                loaded = {m["id"] for m in models.get("data", [])}
                if model_name not in loaded:
                    print(f"Waiting for model '{model_name}' to be registered by proxy (loaded: {loaded})...", flush=True)
                    time.sleep(5)
                    continue
            return
        except requests.exceptions.RequestException:
            print(f"Waiting for vllm to startup...", flush=True)
            time.sleep(5)
    raise TimeoutError("vLLM server did not become ready")

class LLM:
    def __init__(self, model_name: str, use_cache: bool = True, debug: bool = False):
        if use_cache:
            self.cache = LLMCache(cache_dir=str(Path(__file__).parent.parent.parent / ".llm_cache"), enabled=True)
        else:
            self.cache = None

        self.model_name = model_name
        self.debug = debug

        # This junk is to make sure we can connect to the llm as a judge server if the model is a local model.
        if not model_name.startswith("gpt-"):
            self.port = None
            self.hostname = None
            # we have a local model... we need to check if the vllm server is up, and if not tell the user to start it.
            # if it's up, we need to create an ssh tunnel to it if needed and wait for it to be ready to accept requests.

            # read server info: prefer the proxy, fall back to a direct vllm server file
            vllm_servers_dir = Path(__file__).parent.parent.parent / "vllm_servers"
            proxy_file = vllm_servers_dir / "proxy.json"
            model_safe = model_name.replace("/", "_")
            while True:
                if proxy_file.exists():
                    with open(proxy_file) as f:
                        server_info = json.load(f)
                    wait_model = model_name  # ask wait_for_vllm to confirm model is registered
                    break
                matches = list(vllm_servers_dir.glob(f"vllm-{model_safe}-*.json"))
                if matches:
                    with open(matches[0]) as f:
                        server_info = json.load(f)
                    wait_model = None  # direct vllm server; no /v1/models check needed
                    break
                print(
                    f"No proxy or direct server file found for '{model_name}'. "
                    f"Start the proxy (launch_vllm_proxy_server.py) or a direct vllm server.",
                    flush=True,
                )
                time.sleep(60)

            self.port = server_info["port"]
            self.hostname = server_info["hostname"]
            wait_for_vllm(server_info["hostname"], self.port, model_name=wait_model)
        
            self.client = openai.OpenAI(
                base_url=f"http://{server_info['hostname']}:{self.port}/v1",
                api_key="synthesis_rc",
            )
        else:
            self.client = openai.OpenAI(
                api_key=open(Path.home() / ".openai_api_key_uwtrust").read().strip(),
            )

    
    def call(self, messages, validation_fn=None, **generation_kwargs):
        """
        Implements a single message call to the model.
        with retries and caching. (currently caching isn't implemented)
        """
    
        extra_body_params = {}
        params = {}
        for k, v in generation_kwargs.items():
            if k in VLLM_EXTRA_PARAMETERS:
                extra_body_params[k] = v
            else:
                params[k] = v
        
        for attempt in range(3):
            try:
                if self.model_name.startswith("gpt") or self.model_name.startswith("o"):
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=messages,
                        extra_body=extra_body_params,
                        **params,
                    )
                    return response.model_dump()
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        extra_body=extra_body_params,
                        **params,
                    )

                    # check validation fn if present
                    if validation_fn is not None:
                        try:
                            validation_fn(response)
                        except Exception as e:
                            print(f"Validation failed for response: {e}. Retrying...")
                            continue

                    # check if we're we're in non-thinking mode but the content is none bc the reasoning parser is stupid.
                    # It sometimes parses content into reasoning_content
                    response_dict = response.model_dump()['choices'][0]['message']
                    if not response_dict["content"] and ((not generation_kwargs.get("chat_template_kwargs", {}).get("enable_thinking", False)) or ("Thinking" not in self.model_name)):
                        response_dict["content"] = response_dict.get("reasoning_content", "")
                        response_dict["reasoning_content"] = None
                    return response_dict
                
            except Exception as e:
                print(f"Error during LLM call (attempt {attempt+1}/3): {e}")
                time.sleep(5)  # wait a bit before retrying
        
        raise RuntimeError("LLM call failed after 3 attempts")

    def batch_call(self, batch_messages, max_workers=10, validation_fn=None, **generation_kwargs):
        results = []
        if self.debug:
            for prompt in batch_messages:
                run_output = self.call(prompt, **generation_kwargs)
                print(run_output["content"])
                results.append(run_output)
        else:
            results = [None] * len(batch_messages)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.call, prompt, validation_fn, **generation_kwargs): i for i, prompt in enumerate(batch_messages)}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    i = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"Worker failed with exception: {e}")
                        continue
                    results[i] = result
        return results