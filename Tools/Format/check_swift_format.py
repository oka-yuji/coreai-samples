import argparse
from pathlib import Path
import subprocess
import sys


def git(*args):
    return subprocess.check_output(["git", *args]).decode("utf-8", "surrogateescape")


def main():
    parser = argparse.ArgumentParser(
        description="Check Swift working-tree files; reject divergent staged contents."
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--base", help="PR target ref; compare its merge-base with HEAD")
    scope.add_argument("--all", action="store_true", help="Audit all tracked and untracked Swift files")
    args = parser.parse_args()
    root = Path(git("rev-parse", "--show-toplevel").strip())
    if Path.cwd().resolve() != root.resolve():
        parser.error("run from the repository root")
    configuration = root / ".swift-format"
    if not configuration.is_file():
        parser.error(".swift-format is missing")

    staged = set(git("diff", "--cached", "--name-only", "-z", "HEAD", "--", "*.swift").split("\0"))
    unstaged = set(git("diff", "--name-only", "-z", "--", "*.swift").split("\0"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", "--", "*.swift")
    partial = (staged & (unstaged | set(untracked.split("\0")))) - {""}
    if partial:
        print(
            "Swift files have both staged and unstaged contents; inspect and align them before checking:",
            file=sys.stderr,
        )
        for name in sorted(partial):
            print(f"  {name}", file=sys.stderr)
        return 2

    if args.all:
        tracked = git("ls-files", "-z", "--", "*.swift")
    else:
        base = git("merge-base", "HEAD", args.base).strip()
        tracked = git("diff", "--name-only", "-z", "--diff-filter=ACMRT", base, "--", "*.swift")
    paths = sorted({name for name in (tracked + untracked).split("\0") if name})
    failed = False
    checked = 0
    for name in paths:
        path = root / name
        if path.is_symlink():
            print(f"Swift symlink requires an explicit scope decision: {name}", file=sys.stderr)
            failed = True
            continue
        if not path.is_file():
            continue
        checked += 1
        result = subprocess.run([
            "xcrun", "swift-format", "lint", "--strict", "--configuration",
            str(configuration), str(path),
        ])
        failed = failed or result.returncode != 0
    print(f"swift-format: checked {checked} Swift files; {'FAIL' if failed else 'PASS'}")
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"format check could not run: {error}", file=sys.stderr)
        sys.exit(2)
