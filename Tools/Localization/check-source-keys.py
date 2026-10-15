import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
if len(sys.argv) not in (2, 3):
    sys.exit("Usage: check-source-keys.py DerivedData [CatalogFolder]")
derived = Path(sys.argv[1])
catalog_folder = Path(sys.argv[2]) if len(sys.argv) == 3 else root / "CoreAISamples/Resources"
outputs = sorted(derived.glob("Build/Intermediates.noindex/CoreAISamples.build/*/CoreAISamples.build/Objects-normal/*"))
if len(outputs) != 1:
    sys.exit(f"Expected one app build output in {derived}, found {len(outputs)}. Build one configuration and architecture.")
sources = {path.relative_to(root) for path in (root / "CoreAISamples").rglob("*.swift")}
covered = set()
stale = 0
errors = []
keys = {}
for path in sorted(outputs[0].glob("*.stringsdata")):
    data = json.loads(path.read_text())
    source = Path(data["source"])
    label = source.name
    if source.is_absolute():
        if not source.exists():
            stale += 1
            continue
        if not source.resolve().is_relative_to(root):
            errors.append(f"{path.name}: built from another checkout: {source}")
            continue
        label = source.resolve().relative_to(root)
        covered.add(label)
    for table, items in data["tables"].items():
        for item in items:
            keys.setdefault((table, item["key"]), f"{label}:{item.get('location', {}).get('startingLine', 0)}")
errors += [f"{path}: no .stringsdata from the build" for path in sorted(sources - covered)]
catalogs = {}
for (table, key), location in sorted(keys.items()):
    if table not in catalogs:
        path = catalog_folder / f"{table}.xcstrings"
        catalogs[table] = json.loads(path.read_text())["strings"] if path.exists() else None
    if catalogs[table] is None or key not in catalogs[table]:
        errors.append(f"{location}: missing from {table}.xcstrings: {json.dumps(key, ensure_ascii=False)}")
for error in errors:
    print(error)
print(f"Source keys: {len(keys)} from {len(covered)} files ({stale} stale files skipped); problems: {len(errors)}")
sys.exit(1 if errors else 0)
