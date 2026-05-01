# src/utils/fault_injector.py

import multiprocessing as mp
import threading
import time
import numpy as np
import subprocess


class FaultInjector:
    def __init__(self):
        self.processes = []
        self.threads = []

    # -----------------------------------
    # CPU STRESS
    # -----------------------------------
    def start_cpu_stress(self, num_workers=2):
        def worker():
            while True:
                _ = np.random.rand(300, 300) @ np.random.rand(300, 300)

        for _ in range(num_workers):
            p = mp.Process(target=worker)
            p.start()
            self.processes.append(p)

        print(f"🔥 CPU stress started ({num_workers} workers)")

    # -----------------------------------
    # MEMORY STRESS
    # -----------------------------------
    def start_memory_stress(self, size_mb=500):
        def worker():
            data = []
            while True:
                data.append(bytearray(1024 * 1024))  # 1 MB
                if len(data) > size_mb:
                    data.pop(0)

        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        self.threads.append(t)

        print(f"🧠 Memory stress started (~{size_mb} MB)")

    # -----------------------------------
    # GPU STRESS (if CUDA available)
    # -----------------------------------
    def start_gpu_stress(self, duration=60):
        try:
            
            def worker():
                # Set power limit to 150W
                subprocess.run(["sudo", "nvidia-smi", "-i", "1", "-pl", "150"], check=True)
                print(f"🎮 GPU power limit set to 150W for {duration}s")
                
                # Wait for duration
                time.sleep(duration)
                
                # Reset power limit (optional - remove if you want persistent setting)
                subprocess.run(["sudo", "nvidia-smi", "-i", "1", "-pl", "300"], check=True)
                print("🎮 GPU power limit reset")
                
                t = threading.Thread(target=worker)
                t.start()
                self.threads.append(t)
        
        except Exception as e:
            print(f"⚠️ GPU power limit failed: {e}")

    # -----------------------------------
    # STOP ALL
    # -----------------------------------
    def stop_all(self):
        for p in self.processes:
            p.terminate()

        for p in self.processes:
            p.join()

        self.processes = []
        self.threads = []

        print("🛑 All faults stopped")