# global imports
from pathlib import Path
import re
import argparse
import json
import urllib
import shlex
import subprocess
import time

#  project imports
from src.metadata import Metadata
from src.metrics_collector import *
from src.utils import *
from src.graph_constructor import *
from src.rca_module import *

# Initial functions

def check_requirements():
    # add a check against all required components against requirements.txt
    import importlib.util

    requirements_file = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not requirements_file.exists():
        raise FileNotFoundError(f"requirements.txt not found at: {requirements_file}")

    missing = []
    for raw in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # Extract package name from common requirement formats.
        name = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip()
        module_name = name.replace("-", "_")

        if not importlib.util.find_spec(module_name):
            missing.append(name)

    if missing:
        missing_list = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"Missing required components: {missing_list}. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    return True

def parse_args():
    parser = argparse.ArgumentParser(description="RCA inference entry point")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name/path to be run by vLLM",
    )

    # optional arguments
    parser.add_argument(
        "--vllm-options",
        default=None,
        help="vLLM options as a string",
    )

    args = parser.parse_args()

    if args.model is None or not isinstance(args.model, str) or not args.model.strip():
        parser.error("--model is required and must be a non-empty string")

    if args.vllm_options is not None:
        if not isinstance(args.vllm_options, str):
            parser.error("--vllm-options must be a string")

        args.vllm_options = args.vllm_options.strip()

        # Enforce enclosing in double quotes, e.g. "--tensor-parallel-size 2 --dtype bfloat16"
        if not (len(args.vllm_options) >= 2 and args.vllm_options.startswith('"') and args.vllm_options.endswith('"')):
            parser.error('--vllm-options must be enclosed in double quotes ("...")')

        # Remove outer quotes for internal use
        args.vllm_options = args.vllm_options[1:-1].strip()

        if len(args.vllm_options) > 2000:
            parser.error("--vllm-options is too long")

        if re.search(r"[\x00-\x1f\x7f]", args.vllm_options):
            parser.error("--vllm-options contains invalid control characters")

        # Basic shell-injection guard if this string is later passed to a shell command.
        if any(tok in args.vllm_options for tok in (";", "&&", "||", "|", "`")):
            parser.error("--vllm-options contains unsupported shell operators")

    return args

def initiate_run_time(model: str, vllm_options: str):
    # execute the commands to start the model run time with vLLM and the provided options
    
    print(f"Loading model in background")

    # Build command safely (no shell)
    cmd = ["vllm", "serve", model]
    if vllm_options:
        cmd.extend(shlex.split(vllm_options))

    # Persist logs so startup errors can be shown
    log_path = Path(__file__).resolve().parents[1] / "vllm_startup.log"
    log_file = open(log_path, "a", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # keep running after this script exits
        )
    except Exception as e:
        print(f"Failed to start vLLM process: {e}")
        log_file.close()
        return

    print(f"vLLM process started in background (PID: {proc.pid}). Verifying readiness...")

    def _is_ready() -> bool:
        port = 8000
        if vllm_options:
            try:
                opts = shlex.split(vllm_options)
                for i, tok in enumerate(opts):
                    if tok == "--port" and i + 1 < len(opts):
                        port = int(opts[i + 1])
                        break
                    if tok.startswith("--port="):
                        port = int(tok.split("=", 1)[1])
                        break

                if not (1 <= port <= 65535):
                    port = 8000
            
            except Exception:
                port = 8000

        urls = (
            f"http://127.0.0.1:{port}/health",
            f"http://127.0.0.1:{port}/v1/models",
        )
        for u in urls:
            try:
                with urllib.request.urlopen(u, timeout=2) as r:
                    if 200 <= r.status < 300:
                        return True
            except Exception:
                pass

        return False

    timeout_s = 600
    start = time.time()

    while time.time() - start < timeout_s:
        # If process crashed, show startup logs
        if proc.poll() is not None:
            log_file.flush()
            log_file.close()
            try:
                tail = log_path.read_text(encoding="utf-8")[-4000:]
            except Exception:
                tail = "<unable to read startup log>"
            print("Error: vLLM failed during startup.")
            print("---- vLLM startup log (tail) ----")
            print(tail)
            print("---- end log ----")
            raise RuntimeError("vLLM process terminated unexpectedly during startup")

        if _is_ready():
            print("Success: vLLM is up and serving API requests.")
            log_file.close()
            root_process = proc.pid
            print(f"Root process started (PID: {root_process})")
            Metadata.process_id = root_process
            return

        time.sleep(10)

    # Timed out waiting for readiness
    log_file.flush()
    log_file.close()
    try:
        tail = log_path.read_text(encoding="utf-8")[-4000:]
    except Exception:
        tail = "<unable to read startup log>"
    print(f"Error: vLLM did not become ready within {timeout_s} seconds.")
    print("---- vLLM startup log (tail) ----")
    print(tail)
    print("---- end log ----")
    raise RuntimeError("vLLM did not become ready within the specified timeout.")


## core logic functions

if __name__ == "__main__":
    # define arguments required
    args = parse_args()

    # check requirements
    check_requirements()

    # initiate model run time
    initiate_run_time(args.model, args.vllm_options)

    # run the metrics collection orchestrator, construct graph and run RCA module
    orchestrator.collect_all_metrics(metrics_interval=20)