import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
pattern = re.compile(
    rb"[/]Users[/](?!runner[/]work[/])[^\s/`'\"]+"
    rb"|[-]Users[-][^\s/`'\"-]+[-]"
    rb"|[/]home[/][^\s/`'\"]+"
    rb"|[/]Volumes[/][^\s/`'\"]+"
    rb"|[/]var[/]folders[/]"
)


def git(*args, data=None):
    try:
        return subprocess.run(["git", "-C", str(root), *args], input=data, stdout=subprocess.PIPE, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        print("Could not read the tracked files with git.", file=sys.stderr)
        sys.exit(2)


def contents():
    entries = []
    for entry in git("ls-files", "-s", "-z").split(b"\0"):
        if entry:
            info, name = entry.split(b"\t", 1)
            mode, blob = info.split()[:2]
            if mode != b"160000":
                entries.append((os.fsdecode(name), blob))
    batch = git("cat-file", "--batch", data=b"".join(blob + b"\n" for _, blob in entries))
    offset = 0
    for name, _ in entries:
        end = batch.index(b"\n", offset)
        size = int(batch[offset:end].split()[2])
        staged = batch[end + 1:end + 1 + size]
        offset = end + 2 + size
        path = root / name
        if path.is_symlink():
            current = os.fsencode(os.readlink(path))
        elif path.is_file():
            current = path.read_bytes()
        else:
            current = staged
        yield name, current, staged


def lines(data):
    data = data.replace(b"\\/", b"/")
    return sorted({data.count(b"\n", 0, match.start()) + 1 for match in pattern.finditer(data)})


def commits(args):
    entries = []
    commit = b""
    for line in git("rev-list", "--objects", "--in-commit-order", "--reverse", *args).split(b"\n"):
        if line:
            sha, separator, name = line.partition(b" ")
            if not separator:
                commit = sha
            entries.append((sha, name, commit))
    batch = git("cat-file", "--batch", data=b"".join(sha + b"\n" for sha, _, _ in entries))
    found = []
    offset = 0
    for sha, name, commit in entries:
        end = batch.index(b"\n", offset)
        kind, size = batch[offset:end].split()[1:]
        data = batch[end + 1:end + 1 + int(size)]
        offset = end + 2 + int(size)
        if kind == b"commit":
            found += [f"{sha[:7].decode()}:{line} (message)" for line in lines(data.partition(b"\n\n")[2])]
        elif kind == b"blob":
            found += [f"{os.fsdecode(name)}:{line} ({commit[:7].decode()})" for line in lines(data)]
    return found


if sys.argv[1:2] == ["--commits"]:
    found = commits(sys.argv[2:] or ["origin/main..HEAD"])
else:
    found = []
    for name, current, staged in contents():
        found += [f"{name}:{line}" for line in lines(current)]
        if staged != current:
            found += [f"{name}:{line} (staged)" for line in lines(staged)]
for item in found:
    print(item)
if found:
    print(f"{len(found)} lines contain an absolute local path. Use ~ or a path relative to the repository.", file=sys.stderr)
sys.exit(1 if found else 0)
