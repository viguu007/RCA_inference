# src/launchers/vllm_launcher.py

import subprocess
import os
import signal


class VLLMLauncher:
    def __init__(self, model_name: str, host: str = "0.0.0.0", port: int = 8000):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.process = None
        self.pid = None
        self.pgid = None

    def launch(self):
        """
        Launch vLLM server as a subprocess and create a new process group.
        """

        cmd = [
            "vllm",
            "serve",
            self.model_name,
            "--port",
            str(self.port),
        ]

        self.process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,   # IMPORTANT: creates new process group
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.pid = self.process.pid
        self.pgid = os.getpgid(self.pid)

        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "process": self.process,
        }

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self):
        """
        Gracefully stop the entire process group.
        """

        if self.pgid is None:
            return

        try:
            # Kill entire process group
            os.killpg(self.pgid, signal.SIGTERM)
        except Exception:
            pass

    def wait(self):
        """
        Wait for process to finish.
        """
        if self.process:
            self.process.wait()