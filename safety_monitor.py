import os
import threading
import time
import torch
import psutil

class SafetyWatchdog:
    """
    A background watchdog thread that monitors System RAM and GPU VRAM.
    If memory usage exceeds critical safety thresholds, it intercepts and 
    terminates the process safely to prevent system freezes or hard OOM crashes.
    """
    def __init__(self, vram_limit_gb: float = 3.8, ram_percent_limit: float = 90.0, check_interval: float = 0.1):
        """
        :param vram_limit_gb: Maximum VRAM allocation allowed (in Gigabytes).
        :param ram_percent_limit: Maximum System RAM usage allowed (percentage 0-100).
        :param check_interval: How often to check the memory (in seconds).
        """
        self.vram_limit_gb = vram_limit_gb
        self.ram_percent_limit = ram_percent_limit
        self.check_interval = check_interval
        self.is_running = False
        self.thread = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[SafetyWatchdog] Active. Limits -> VRAM: {self.vram_limit_gb}GB | System RAM: {self.ram_percent_limit}%")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        print(f"[SafetyWatchdog] Deactivated.")

    def _monitor_loop(self):
        while self.is_running:
            try:
                # 1. Check System RAM using psutil
                ram_usage = psutil.virtual_memory()
                ram_percent = ram_usage.percent
                
                # 2. Check GPU VRAM using PyTorch Caching Allocator
                vram_allocated_gb = 0.0
                if torch.cuda.is_available():
                    vram_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                
                # 3. Evaluate Thresholds
                crash_reason = None
                
                if vram_allocated_gb > self.vram_limit_gb:
                    crash_reason = (
                        f"CRITICAL GPU VRAM OVERLOAD!\n"
                        f"  -> Limit   : {self.vram_limit_gb:.2f} GB\n"
                        f"  -> Current : {vram_allocated_gb:.2f} GB"
                    )
                elif ram_percent > self.ram_percent_limit:
                    crash_reason = (
                        f"CRITICAL SYSTEM RAM OVERLOAD!\n"
                        f"  -> Limit   : {self.ram_percent_limit}%\n"
                        f"  -> Current : {ram_percent}%\n"
                        f"  -> RAM Left: {ram_usage.available / (1024**3):.2f} GB"
                    )

                # 4. Terminate if dangerous
                if crash_reason:
                    print("\n" + "="*60)
                    print("🚨 SAFETY WATCHDOG TRIGGERED: ABORTING PROCESS 🚨")
                    print("="*60)
                    print(crash_reason)
                    print("="*60)
                    print("Action: Terminating process immediately to prevent hardware freeze/crash...")
                    print("="*60 + "\n")
                    # Force hard OS exit to instantly halt all PyTorch CUDA operations
                    os._exit(1)
                    
            except Exception as e:
                # Silently catch psutil/torch polling errors to prevent watchdog crashes
                pass
                
            time.sleep(self.check_interval)
