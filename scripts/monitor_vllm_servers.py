import time
from pathlib import Path
import json
import requests

if __name__ == "__main__":
    while True:
        print("Monitoring VLLM servers...")
        # check every 8 hours if the servers are still alive
        # if they aren't, then we should clean up the server files

        for server_file in (Path(__file__).parent.parent / "vllm_servers").glob("*.json"):
            if not server_file.is_file():
                continue
            with open(server_file) as fh:
                data = json.load(fh)
            port = data["port"]
            hostname = data.get("hostname", "localhost")
            try:
                response = requests.get(f"http://{hostname}:{port}/health")
                print(response)
                if response.status_code != 200:
                    print(f"Server on port {port} is not healthy, removing server file")
                    server_file.unlink()
            except requests.exceptions.ConnectionError:
                print(f"Server on port {port} is not responding, removing server file")
                server_file.unlink()

        # wait for 8 hours before checking again
        time.sleep(8 * 60 * 60)
        