"""vLLM proxy / load-balancer.

Polls vllm_servers/ for new server files and routes /v1/completions and
/v1/responses to a GPU-weighted random backend.  Run via launch_vllm_server.sh
(same apptainer/conda setup as the vllm workers) or directly if the right
packages are already available.
"""

import argparse
import asyncio
import fcntl
import json
import os
import random
import shlex
import socket
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

VLLM_SERVERS_DIR = Path(__file__).parent.parent / "vllm_servers"
PROXY_FILE = VLLM_SERVERS_DIR / "proxy.json"
POLL_INTERVAL = 60  # seconds
DEFAULT_PORT = 37139  # one below the vllm default so they don't collide

registry: dict[str, list["Server"]] = {}
http_client: httpx.AsyncClient | None = None


@dataclass
class Server:
    hostname: str
    port: int
    model_name: str
    num_gpus: int
    server_filename: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Server) and self.server_filename == other.server_filename


# ---------------------------------------------------------------------------
# Registry management
# ---------------------------------------------------------------------------

async def scan_for_new_servers() -> None:
    known = {s.server_filename for servers in registry.values() for s in servers}
    for path in VLLM_SERVERS_DIR.glob("vllm-*.json"):
        if path.name in known:
            continue
        try:
            data = json.loads(path.read_text())
            server = Server(
                hostname=data["hostname"],
                port=data["port"],
                model_name=data["model_name"],
                num_gpus=data["num_gpus"],
                server_filename=path.name,
            )
            registry.setdefault(server.model_name, []).append(server)
            print(f"[proxy] Registered {server.model_name} at {server.hostname}:{server.port}", flush=True)
            resp = await http_client.get(f"http://{server.hostname}:{server.port}/health", timeout=10)
            if resp.status_code == 503:
                print(f"[proxy] Health check returned 503 for {server.hostname}:{server.port}, deregistering", flush=True)
                deregister(server)
        except Exception as exc:
            print(f"[proxy] Skipping {path.name}: {exc}", flush=True)


def deregister(server: Server) -> None:
    model = server.model_name
    if model in registry:
        registry[model] = [s for s in registry[model] if s != server]
        if not registry[model]:
            del registry[model]
    (VLLM_SERVERS_DIR / server.server_filename).unlink(missing_ok=True)
    print(f"[proxy] Deregistered {server.model_name} at {server.hostname}:{server.port}", flush=True)


def pick_server(model_name: str) -> "Server | None":
    servers = registry.get(model_name)
    if not servers:
        return None
    weights = [s.num_gpus for s in servers]
    return random.choices(servers, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# HTTP forwarding
# ---------------------------------------------------------------------------

async def forward(server: Server, path: str, request: Request) -> Response:
    url = f"http://{server.hostname}:{server.port}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    # print(f"[proxy] Forwarding to {url} with headers {headers} and body {body[:100]}...", flush=True)
    for attempt in range(3):
        try:
            resp = await http_client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
                timeout=600,
            )
            return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError):
            if attempt < 2:
                await asyncio.sleep(2**attempt)
            else:
                deregister(server)
                return JSONResponse(
                    status_code=503,
                    content={"error": f"Server {server.hostname}:{server.port} is unreachable and has been deregistered."},
                )


async def proxy(path: str, request: Request) -> Response:
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    model_name = data.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "Missing 'model' field in request body"})

    server = pick_server(model_name)
    if server is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_name}' has not been initialized yet. Available: {list(registry.keys())}"},
        )
    print("[proxy] Forwarding request for model", model_name, "to", server.hostname, server.port, flush=True)
    return await forward(server, path, request)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(trust_env=False)
    await scan_for_new_servers()
    poll_task = asyncio.create_task(_poll_loop())
    yield
    poll_task.cancel()
    await http_client.aclose()
    PROXY_FILE.unlink(missing_ok=True)


async def _poll_loop() -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        await scan_for_new_servers()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "num_servers": len(servers)}
            for name, servers in registry.items()
        ],
    }


@app.post("/v1/chat/completions")
async def completions(request: Request):
    # print("Running completions proxy handler", flush=True)
    return await proxy("v1/chat/completions", request)


@app.post("/v1/responses")
async def responses(request: Request):
    return await proxy("v1/responses", request)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _get_open_port() -> int:
    used: list[int] = []
    for f in VLLM_SERVERS_DIR.glob("vllm*.json"):
        try:
            used.append(json.loads(f.read_text())["port"])
        except Exception:
            pass
    port = DEFAULT_PORT
    while port in used:
        port -= 1 # count down to not collide with vllm servers
    return port


def main():
    # in_apptainer = bool(os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"))
    # in_correct_conda = os.environ.get("CONDA_DEFAULT_ENV", "") in ("nvcc129", "nvcc130")

    # if not (in_apptainer and in_correct_conda):
    #     script = Path(__file__).parent / "launch_vllm_server.sh"
    #     python_cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)
    #     os.execvp("/usr/bin/bash", ["/usr/bin/bash", str(script), python_cmd])
    #     return

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (auto-selected if omitted)")
    args = parser.parse_args()

    VLLM_SERVERS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VLLM_SERVERS_DIR / ".port.lock", "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        port = args.port if args.port is not None else _get_open_port()
        PROXY_FILE.write_text(json.dumps({"hostname": socket.gethostname(), "port": port}, indent=2))

    print(f"[proxy] Starting on {socket.gethostname()}:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
