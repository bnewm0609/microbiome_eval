import argparse
import fcntl
import glob
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VLLM_SERVERS_DIR = Path(__file__).parent.parent / "vllm_servers"
DEFAULT_PORT = 37140


def get_open_port() -> int:
    json_files = glob.glob(str(VLLM_SERVERS_DIR / "*.json"))
    if not json_files:
        return DEFAULT_PORT

    used_ports = []
    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)
        used_ports.append(data["port"])
    
    next_port = DEFAULT_PORT
    while next_port in used_ports:
        next_port += 1
    return next_port


def parse_model_from_command(command: str) -> str:
    tokens = shlex.split(command)
    try:
        serve_idx = tokens.index("serve")
    except ValueError:
        print("Error: command must contain 'serve'", file=sys.stderr)
        sys.exit(1)
    if serve_idx + 1 >= len(tokens):
        print("Error: no model argument after 'serve'", file=sys.stderr)
        sys.exit(1)
    return tokens[serve_idx + 1]


def parse_num_gpus_from_command(command: str) -> int:
    tokens = shlex.split(command)

    def get_flag(flags):
        for flag in flags:
            if flag in tokens:
                idx = tokens.index(flag)
                if idx + 1 < len(tokens):
                    return int(tokens[idx + 1])
        return 1

    tp = get_flag(("--tensor-parallel-size", "-tp"))
    dp = get_flag(("--data-parallel-size", "-dp"))
    return tp * dp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="The full 'vllm serve ...' command to run")
    args = parser.parse_args()

    # If we're not in an apptainer container with the correct conda environment, re-launch this script inside one with the command to start the vLLM server as an argument. The "launch_vllm_server.sh" script will handle starting the server and then re-launching this script inside the container with the correct environment variables set to connect to the server.
    in_apptainer = bool(os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"))
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    in_correct_conda = conda_env in ("nvcc129", "nvcc130")
    print(in_apptainer, in_correct_conda)

    if not (in_apptainer and in_correct_conda):
        script = Path(__file__).parent / "launch_vllm_server.sh"
        python_cmd = " ".join(shlex.quote(arg) for arg in [sys.executable] + sys.argv)
        print(str(script), python_cmd)
        os.execvp("/usr/bin/bash", ["/usr/bin/bash", str(script), python_cmd])
        return

    model = parse_model_from_command(args.command)
    num_gpus = parse_num_gpus_from_command(args.command)
    model_safe = model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    VLLM_SERVERS_DIR.mkdir(parents=True, exist_ok=True)
    server_file = VLLM_SERVERS_DIR / f"vllm-{model_safe}-{num_gpus}-{timestamp}.json"
    with open(VLLM_SERVERS_DIR / ".port.lock", "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        port = get_open_port()
        with open(server_file, "w") as fh:
            json.dump(
                {"hostname": socket.gethostname(), "port": port, "model_name": model, "num_gpus": num_gpus},
                fh,
                indent=2,
            )

    tokens = shlex.split(args.command)
    if "--port" not in tokens:
        tokens += ["--port", str(port)]
    else:
        port_idx = tokens.index("--port")
        tokens[port_idx + 1] = str(port)

    def cleanup(signum=None, frame=None):
        print("Cleaning up server in `clenaup` handler")
        server_file.unlink(missing_ok=True)
        if proc is not None:
            proc.terminate()
        sys.exit(0)

    proc = None
    # This signal handling thing doesn't work. I don't know why and claude code isn't helping me debug it.
    # The `cleanup` function is never called. I even tried to send a SIGKILL 60 seconds before.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, cleanup)

    # set the hf_token env variable for the subprocess
    env = os.environ.copy()
    env["HF_TOKEN"] = open(f"{os.path.expanduser('~')}/.hf_token_fs").read().strip()

    print(f"Launching '{model}' on {socket.gethostname()}:{port}", flush=True)

    proc = subprocess.Popen(
        tokens,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    import select

    streams = {proc.stdout: sys.stdout, proc.stderr: sys.stderr}
    open_streams = set(streams)
    try:
        while open_streams:
            readable, _, _ = select.select(list(open_streams), [], [])
            for stream in readable:
                line = stream.readline()
                if line:
                    streams[stream].write(line)
                    streams[stream].flush()
                else:
                    open_streams.discard(stream)
    finally:
        server_file.unlink(missing_ok=True)

    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

