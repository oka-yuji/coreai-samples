import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wait", type=int, default=7200)
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
    try:
        with args.log.open("x") as log:
            result = subprocess.run(args.command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        print("Validation exit:", result.returncode, flush=True)
    finally:
        if (args.lock / "owner.txt").read_text() != owner:
            raise RuntimeError("Lock owner changed")
        args.lock.rename(args.lock.with_name("heavy-job-done-eg2-native-" + stamp))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
