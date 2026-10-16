import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group(pid, number):
    try:
        os.killpg(pid, number)
    except ProcessLookupError:
        pass


def stop_group(process):
    if group_exists(process.pid):
        signal_group(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while group_exists(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.1)
        if group_exists(process.pid):
            signal_group(process.pid, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while group_exists(process.pid) and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.1)
        if group_exists(process.pid):
            raise RuntimeError("A validation process group is still present; the lock is retained")
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wait", type=int, default=7200)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise ValueError("Missing validation command")
    deadline = time.monotonic() + args.wait
    while True:
        processes = subprocess.run(["pgrep", "-x", "-l", "coreai-chat|coreai-verify|CoreAISamples|embedding-verify|irodori-probe"], capture_output=True, text=True)
        if processes.returncode not in (0, 1):
            raise RuntimeError(processes.stderr)
        blocked = []
        for line in processes.stdout.splitlines():
            pid = line.split(maxsplit=1)[0]
            details = subprocess.run(["ps", "-p", pid, "-o", "command="], capture_output=True, text=True)
            if details.returncode == 1 and not details.stdout.strip():
                continue
            if details.returncode != 0:
                raise RuntimeError("Could not inspect process")
            if "/ProductCardsPreview.app/Contents/MacOS/CoreAISamples --product-preview" not in details.stdout:
                blocked.append(details.stdout.strip())
        if not blocked:
            try:
                args.lock.mkdir()
                break
            except FileExistsError:
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Shared GPU is still in use; no model was loaded")
        print("Waiting for shared model lock", flush=True)
        time.sleep(30)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    owner = f"embeddinggemma2 Swift native verification pid={os.getpid()} start={stamp} expected=20min\n"
    (args.lock / "owner.txt").write_text(owner)
    process = None
    try:
        with args.log.open("x") as log:
            process = subprocess.Popen(args.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result = process.wait(timeout=args.timeout)
        print("Validation exit:", result, flush=True)
    finally:
        if process is not None:
            stop_group(process)
        if (args.lock / "owner.txt").read_text() != owner:
            raise RuntimeError("Lock owner changed")
        args.lock.rename(args.lock.with_name("heavy-job-done-eg2-native-" + stamp))
    raise SystemExit(result)


if __name__ == "__main__":
    main()
