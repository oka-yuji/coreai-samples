import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
pattern = re.compile(
    rb"[/]Users[/](?!runner\b)[^\s/`'\"]+"
    rb"|[/]home[/](?!runner\b)[^\s/`'\"]+"
    rb"|[/]Volumes[/][^\s/`'\"]+"
    rb"|[/]var[/]folders[/]"
)


def tracked():
    listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [name for name in listing.decode().split("\0") if name]


def findings():
    for name in tracked():
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        for line in sorted({data.count(b"\n", 0, match.start()) + 1 for match in pattern.finditer(data)}):
            yield f"{name}:{line}"


found = list(findings())
for item in found:
    print(item)
if found:
    print(f"{len(found)} lines contain an absolute local path. Use ~ or a path relative to the repository.", file=sys.stderr)
sys.exit(1 if found else 0)
