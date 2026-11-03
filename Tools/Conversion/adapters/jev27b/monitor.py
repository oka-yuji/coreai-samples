import _thread
import glob
import os
import re
import resource
import shutil
import signal
import subprocess
import threading
import time

SWAP_LIMIT_BYTES = 16e9
FREE_PERCENT = 8
STREAK = 3
INTERVAL_S = 5
KILL_AFTER_S = 120
COLUMNS = ("elapsed_s", "rss_bytes", "max_rss_bytes", "swap_used_bytes", "free_percent", "temporary_bytes")


def command_output(*command):
    return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout


def swap_used():
    match = re.search(r"used = ([0-9.]+)M", command_output("sysctl", "-n", "vm.swapusage"))
    return round(float(match[1]) * 2**20) if match else None


def free_percent():
    match = re.search(r"free percentage: (\d+)%", command_output("memory_pressure", "-Q"))
    return int(match[1]) if match else None


def lowest(*values):
    return min(value for value in values if value is not None)


def directory_bytes(patterns):
    folders = sorted({path for pattern in patterns for path in glob.glob(pattern) if os.path.isdir(path)})
    if not folders:
        return 0
    return sum(int(line.split()[0]) * 1024 for line in command_output("du", "-sk", *folders).splitlines() if line.strip())


class Monitor(threading.Thread):
    def __init__(self, path, volumes, patterns, on_abort):
        super().__init__(daemon=True)
        self.path, self.volumes, self.patterns, self.on_abort = path, [str(v) for v in volumes], patterns, on_abort
        self.halt, self.started, self.aborted_at = threading.Event(), time.perf_counter(), None
        self.reason, self.samples, self.rising, self.low, self.previous, self.first = None, 0, 0, 0, None, None
        self.peak = {"rss_bytes": 0, "max_rss_bytes": 0, "swap_used_bytes": 0, "temporary_bytes": 0}
        self.lowest = {"free_percent": None, "free_bytes": {volume: None for volume in self.volumes}}

    def sample(self):
        rss = int(command_output("ps", "-o", "rss=", "-p", str(os.getpid())).strip() or 0) * 1024
        row = {"elapsed_s": round(time.perf_counter() - self.started, 1), "rss_bytes": rss,
               "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "swap_used_bytes": swap_used(),
               "free_percent": free_percent(), "temporary_bytes": directory_bytes(self.patterns)}
        free = [shutil.disk_usage(volume).free for volume in self.volumes]
        for key in self.peak:
            if row[key] is not None:
                self.peak[key] = max(self.peak[key], row[key])
        if row["free_percent"] is not None:
            self.lowest["free_percent"] = lowest(self.lowest["free_percent"], row["free_percent"])
        for volume, value in zip(self.volumes, free):
            self.lowest["free_bytes"][volume] = lowest(self.lowest["free_bytes"][volume], value)
        swap = row["swap_used_bytes"]
        if self.first is None:
            self.first = {"swap_used_bytes": swap, "free_bytes": dict(zip(self.volumes, free))}
        rising = swap is not None and swap > SWAP_LIMIT_BYTES and self.previous is not None and swap > self.previous
        self.rising = self.rising + 1 if rising else 0
        self.low = self.low + 1 if row["free_percent"] is not None and row["free_percent"] < FREE_PERCENT else 0
        self.previous = swap
        self.samples += 1
        return [row[key] for key in COLUMNS] + free

    def verdict(self):
        if self.rising >= STREAK:
            return f"swap above {SWAP_LIMIT_BYTES:.0f} B rose on {self.rising} consecutive samples"
        if self.low >= STREAK:
            return f"memory free percentage below {FREE_PERCENT} on {self.low} consecutive samples"
        return None

    def run(self):
        with open(self.path, "w", encoding="utf-8") as log:
            log.write("\t".join(COLUMNS + tuple(f"free_bytes:{volume}" for volume in self.volumes)) + "\n")
            while True:
                log.write("\t".join("" if value is None else str(value) for value in self.sample()) + "\n")
                log.flush()
                if self.reason is None and self.verdict():
                    self.reason = self.verdict()
                    self.aborted_at = time.perf_counter()
                    self.on_abort(self.reason)
                    _thread.interrupt_main()
                if self.aborted_at is not None and time.perf_counter() - self.aborted_at > KILL_AFTER_S:
                    os.kill(os.getpid(), signal.SIGKILL)
                if self.halt.wait(INTERVAL_S):
                    return

    def finish(self):
        self.halt.set()
        self.join()
        return {"samples": self.samples, "interval_s": INTERVAL_S, "first": self.first, "peak": self.peak,
                "lowest": self.lowest, "abort": self.reason, "aborts": int(self.reason is not None),
                "limits": {"swap_used_bytes": SWAP_LIMIT_BYTES, "free_percent": FREE_PERCENT, "streak": STREAK}}
