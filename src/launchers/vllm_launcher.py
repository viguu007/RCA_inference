# src/launchers/vllm_launcher.py

import subprocess
import os
import signal
import time
import urllib.request


class VLLMLauncher:
    def __init__(self, model_name: str, host: str = "127.0.0.1", port: int = 8000):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.process = None
        self.pid = None
        self.pgid = None

    def launch(self):
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        self.process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self.pid = self.process.pid
        self.pgid = os.getpgid(self.pid)

        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "process": self.process,
        }

    def wait_until_ready(self, timeout=120, interval=2):
        """
        Wait until vLLM HTTP server is ready.
        """
        url = f"http://{self.host}:{self.port}/v1/models"
        start = time.time()

        print("⏳ Waiting for vLLM to be ready...")

        while time.time() - start < timeout:
            # If process died early → fail fast
            if self.process.poll() is not None:
                raise RuntimeError("vLLM process exited before becoming ready")

            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        print("✅ vLLM is ready\n")
                        return
            except Exception:
                pass

            time.sleep(interval)

        raise TimeoutError("vLLM did not become ready within timeout")

    def is_running(self) -> bool:
        return self.process and self.process.poll() is None

    def stop(self):
        if self.pgid:
            try:
                os.killpg(self.pgid, signal.SIGTERM)
            except Exception:
                pass

    def wait(self):
        if self.process:
            self.process.wait()