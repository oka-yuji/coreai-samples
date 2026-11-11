import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def units(node):
    if "stringUnit" in node:
        yield node["stringUnit"]
    for variations in node.get("variations", {}).values():
        for variant in variations.values():
            yield from units(variant)
    for substitution in node.get("substitutions", {}).values():
        yield from units(substitution)


def placeholders(value):
    pattern = r"%(?:%|(?:(\d+)\$)?[-+#0'\d.]*(ll[diuoxX]|[hlLztj]?[diuoxXfFeEgGaAcCsSp]|@))"
    result = []
    for match in re.finditer(pattern, value):
        if match[2]:
            result.append((int(match[1]) if match[1] else len(result) + 1, match[2]))
    return collections.Counter(result)


root = Path(__file__).resolve().parents[2]
listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.xcstrings"], stdout=subprocess.PIPE, check=True).stdout
catalogs = sorted(os.fsdecode(name) for name in listing.split(b"\0") if name)
errors = []
counts = {}
for name in catalogs:
    data = json.loads((root / name).read_text())
    source = data["sourceLanguage"]
    languages = {source}
    for entry in data["strings"].values():
        languages.update(entry.get("localizations", {}))
    translated_count = 0
    for key, entry in data["strings"].items():
        if entry.get("shouldTranslate") is False:
            continue
        translated_count += 1
        source_units = list(units(entry.get("localizations", {}).get(source, {})))
        expected = placeholders(source_units[0]["value"] if source_units else key)
        for language in sorted(languages):
            localization = entry.get("localizations", {}).get(language)
            if localization is None and language == source:
                continue
            values = list(units(localization or {}))
            if not values:
                errors.append(f"{name}: {language}: missing translation: {key}")
            for unit in values:
                value = unit.get("value", "")
                if unit.get("state") != "translated" or (key and not value):
                    errors.append(f"{name}: {language}: incomplete translation: {key}")
                if placeholders(value) != expected:
                    errors.append(f"{name}: {language}: placeholder mismatch: {key}")
    counts[name] = {"keys": len(data["strings"]), "translatedKeys": translated_count, "languages": sorted(languages)}
print(json.dumps({"catalogs": counts, "errors": errors}, ensure_ascii=False, indent=2))
sys.exit(bool(errors) or not catalogs)
