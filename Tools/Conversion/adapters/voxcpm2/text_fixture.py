import argparse
import json
from pathlib import Path
import platform
import re
import unicodedata

SPACE = re.compile(r"\s")
RUN = re.compile(r"\s+")
SAMPLES = (
    ("empty", ""),
    ("one space", " "),
    ("spaces only", "   "),
    ("whitespace only", "\t\n\r　 \u0085  \u001c"),
    ("leading and trailing spaces", "  a b  "),
    ("leading and trailing newlines", "\n\na\n\n"),
    ("newline runs", "a\n\n\nb"),
    ("carriage return and newline", "a\r\nb\rc"),
    ("tab and space run", "a \t \t b"),
    ("mixed run", "a \t\n　 \u0085  \u001c\u001fb"),
    ("ideographic spaces", "こんにちは　　世界"),
    ("no-break space", "a b"),
    ("next line", "a\u0085b"),
    ("line and paragraph separators", "a b c"),
    ("zero width space", "a​b"),
    ("mongolian vowel separator", "a᠎b"),
    ("byte order mark", "a﻿b"),
    ("word joiner", "a⁠b"),
    ("combining mark after a space", "a ́b"),
    ("voice description", "(A calm voice)\nこんにちは。  hello"),
)


def normalize(text):
    if not text.strip():
        return None
    return RUN.sub(" ", text.replace("\n", " "))


def whitespace():
    return [code for code in range(0x110000) if SPACE.fullmatch(chr(code))]


def build():
    points = whitespace()
    samples = list(SAMPLES)
    samples += [(f"U+{code:04X} between letters", f"a{chr(code)}b") for code in points]
    samples += [(f"U+{code:04X} run at both ends", f"{chr(code)}{chr(code)}a{chr(code)}") for code in points]
    return {"format": 1, "python": platform.python_version(), "unicode": unicodedata.unidata_version,
            "pattern": r"\s", "isspaceMatches": points == [code for code in range(0x110000) if chr(code).isspace()],
            "whitespace": points,
            "cases": [{"name": name, "text": text, "normalized": normalize(text)} for name, text in samples]}


def write(path):
    with Path(path).open("x", encoding="ascii") as handle:
        handle.write(json.dumps(build(), indent=1, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    write(parser.parse_args().output)


if __name__ == "__main__":
    main()
