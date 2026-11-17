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
    ("whitespace only", "\t\n\r\u3000\u00a0\u0085\u2028\u2029\u001c"),
    ("leading and trailing spaces", "  a b  "),
    ("leading and trailing newlines", "\n\na\n\n"),
    ("newline runs", "a\n\n\nb"),
    ("carriage return and newline", "a\r\nb\rc"),
    ("tab and space run", "a \t \t b"),
    ("mixed run", "a \t\n\u3000\u00a0\u0085\u2028\u2029\u001c\u001fb"),
    ("ideographic spaces", "\u3053\u3093\u306b\u3061\u306f\u3000\u3000\u4e16\u754c"),
    ("no-break space", "a\u00a0b"),
    ("next line", "a\u0085b"),
    ("line and paragraph separators", "a\u2028b\u2029c"),
    ("zero width space", "a\u200bb"),
    ("mongolian vowel separator", "a\u180eb"),
    ("byte order mark", "a\ufeffb"),
    ("word joiner", "a\u2060b"),
    ("combining mark after a space", "a \u0301b"),
    ("voice description", "(A calm voice)\n\u3053\u3093\u306b\u3061\u306f\u3002  hello"),
    ("supplementary characters around spaces", "\U0001f600 \u3000\U0001f4ac\n\n\U00020000"),
    ("supplementary characters at both ends", " \U0001f600\t\U0001f600 "),
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
