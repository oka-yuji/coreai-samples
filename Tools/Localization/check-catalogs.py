import collections
import json
from pathlib import Path
import re
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
    pattern = r"%(?!%)(?:(\d+)\$)?[-+#0 '\d.]*(ll[diuoxX]|[hlLztj]?[diuoxXfFeEgGaAcCsSp]|@)"
    result = []
    for index, match in enumerate(re.finditer(pattern, value), 1):
        result.append((int(match[1]) if match[1] else index, match[2]))
    return collections.Counter(result)


root = Path(__file__).resolve().parents[2]
catalogs = sorted((root / "CoreAISamples/Resources").glob("*.xcstrings"))
errors = []
counts = {}
for path in catalogs:
    data = json.loads(path.read_text())
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
                errors.append(f"{path.name}: {language}: missing translation: {key}")
            for unit in values:
                value = unit.get("value", "")
                if unit.get("state") != "translated" or (key and not value):
                    errors.append(f"{path.name}: {language}: incomplete translation: {key}")
                if placeholders(value) != expected:
                    errors.append(f"{path.name}: {language}: placeholder mismatch: {key}")
    counts[path.name] = {"keys": len(data["strings"]), "translatedKeys": translated_count, "languages": sorted(languages)}
print(json.dumps({"catalogs": counts, "errors": errors}, ensure_ascii=False, indent=2))
sys.exit(bool(errors) or not catalogs)
