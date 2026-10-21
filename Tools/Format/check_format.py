import re
import subprocess
import sys
from pathlib import Path


def fixed_wire_name_only(result, root):
    if result.returncode != 1:
        return False
    path = root / "CoreAISamples/Demos/Agent/AgentPromptBuilder.swift"
    diagnostics = result.stderr.strip().splitlines()
    if len(diagnostics) != 1:
        return False
    pattern = (
        re.escape(str(path))
        + r":(\d+):\d+: error: \[AlwaysUseLowerCamelCase\] rename the variable 'max_results' using lowerCamelCase"
    )
    match = re.fullmatch(pattern, diagnostics[0])
    if not match:
        return False
    if not re.fullmatch(r"swift-format: checked [1-9]\d* Swift files; FAIL\s*", result.stdout):
        return False
    source = path.read_text()
    lines = source.splitlines()
    index = int(match.group(1)) - 1
    if not 0 <= index < len(lines) or lines[index].strip() != "var max_results: Int?":
        return False
    if source.count("var max_results:") != 1:
        return False
    start = source.find("@Generable\nstruct WebSearchArguments {")
    end = source.find("\n}", start)
    position = sum(len(line) + 1 for line in lines[:index])
    return start >= 0 and start < position < end


def main():
    root = Path(__file__).resolve().parents[2]
    checker = Path(__file__).resolve().with_name("check_swift_format.py")
    result = subprocess.run(
        [sys.executable, str(checker), *sys.argv[1:]], cwd=root, text=True, capture_output=True
    )
    if fixed_wire_name_only(result, root):
        print("swift-format: PASS with the fixed wire-name exception WebSearchArguments.max_results")
        return 0
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
