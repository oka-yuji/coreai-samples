import json
import re

from Tools.LLMEvaluation.items import FILES, QUESTION, SETS, digest, item
from Tools.LLMEvaluation.synthetic.generate import generate

REVISION = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
PARAGRAPH = re.compile(r"(a\d+)p\d+")
TABLE = re.compile(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", re.MULTILINE)
VERDICTS = ["no_regression_found", "regression_found", "critical_failure", "incomplete"]
SCORERS = {"choice_digit", "reading_exact", "python_tests", "entity_sets", "json_exact", "final_integer",
           "format_rules", "unscored"}
PUBLIC = {"choice": "jcqa", "reading": "jsquad", "humaneval": "humaneval", "mbpp": "mbpp",
          "humanevalfix": "humanevalpack", "entities": "ner"}
KINDS = set(PUBLIC) | {"synthetic", "free"}
GENERATORS = {"extract", "long", "format"}
THRESHOLDS = [("max_probability_error",), ("band_mean_max_probability_error",), ("near_tie_probability_margin",),
              ("same_assets", "float16_ticks"), ("same_assets", "max_probability_error"), ("non_finite_values",),
              ("significant_top1_mismatches",), ("positions", "max_per_item"), ("items", "per_category")]
SOURCE_FIELDS = {"choice": {"q_id"}, "humaneval": {"task"}, "humanevalfix": {"task"}, "mbpp": {"task_id"},
                 "entities": {"index", "curid"}, "free": {"instruction"}}


def integer(value, low=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= low


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def lookup(value, path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def pins(contract, resources):
    found = []
    model = contract.get("model", {})
    if not REVISION.fullmatch(str(model.get("revision"))):
        found.append("model revision is not a 40-digit commit")
    for name in ("tokenizer.json", "chat_template.jinja"):
        pin = lookup(model, ("files", name)) or {}
        if not integer(pin.get("bytes"), 1) or not SHA256.fullmatch(str(pin.get("sha256"))):
            found.append(f"model file {name} is not pinned")
    for name, source in contract.get("sources", {}).items():
        if not REVISION.fullmatch(str(source.get("revision"))):
            found.append(f"source {name} revision is not a 40-digit commit")
        if not REVISION.fullmatch(str(source.get("git_blob_sha1"))) or not SHA256.fullmatch(str(source.get("sha256"))):
            found.append(f"source {name} file hash is not pinned")
        if not integer(source.get("bytes"), 1) or source.get("host") not in ("github", "huggingface"):
            found.append(f"source {name} size or host is missing")
        if not all(text(source.get(key)) for key in ("repository", "folder", "path", "license", "attribution", "url")):
            found.append(f"source {name} lacks provenance or license")
    for path, value in contract.get("resources", {}).items():
        if resources.get(path) != value:
            found.append(f"resource {path} differs from its pin")
    return found


def limits(contract):
    found = []
    layer1 = contract.get("layer1", {})
    for path in THRESHOLDS:
        if not number(lookup(layer1, path)):
            found.append(f"layer1 {'.'.join(path)} is missing")
    bands = layer1.get("bands")
    if not isinstance(bands, list) or not bands or bands[0][0] != 1 or any(
            not (integer(low, 1) and integer(high, low)) for low, high in bands) or any(
            bands[index + 1][0] != bands[index][1] + 1 for index in range(len(bands) - 1)):
        found.append("layer1 bands are missing or not contiguous")
    layer2 = contract.get("layer2", {})
    if layer2.get("verdicts") != VERDICTS or sorted(layer2.get("precedence", [])) != sorted(VERDICTS):
        found.append("layer2 verdicts differ from the four fixed labels")
    failures = layer2.get("critical_failures", {})
    eos = lookup(contract, ("generation", "eos_token_id"))
    if not isinstance(eos, list) or not eos or not all(integer(token) for token in eos):
        found.append("generation eos_token_id is missing")
    control = failures.get("control_token_ids")
    if not isinstance(control, list) or not control or set(control) & set(eos or []):
        found.append("control token ids are missing or include a stop token")
    if not all(integer(lookup(failures, ("repetition", key)), 1) for key in ("min_length", "min_consecutive")):
        found.append("repetition limits are missing")
    if not all(isinstance(failures.get(key), list) and failures[key] for key in ("candidate_only", "relative_to_baseline")):
        found.append("critical failure lists are missing")
    if not integer(layer2.get("min_baseline_correct"), 1):
        found.append("layer2 min_baseline_correct is missing")
    tests = lookup(contract, ("scorers", "python_tests")) or {}
    if not (number(tests.get("wall_clock_seconds")) and tests["wall_clock_seconds"] > 0):
        found.append("python_tests wall_clock_seconds is missing")
    if not integer(tests.get("timeout_reruns"), 1):
        found.append("python_tests timeout_reruns is missing")
    frame = lookup(contract, ("generation", "template")) or {}
    if not text(frame.get("prefix")) or not text(frame.get("suffix")):
        found.append("template frame is missing")
    return found


def categories(contract, prompts):
    found, seen = [], set()
    scorers = contract.get("scorers", {})
    for name in set(scorers) - SCORERS:
        found.append(f"unknown scorer kind in the contract: {name}")
    for category in contract.get("categories", []):
        name = category.get("id")
        if name in seen:
            found.append(f"duplicate category {name}")
        seen.add(name)
        kind = category.get("kind")
        if kind not in KINDS:
            found.append(f"{name}: unknown kind {kind}")
        if category.get("scorer") not in SCORERS or category.get("scorer") not in scorers:
            found.append(f"{name}: unknown scorer kind {category.get('scorer')}")
        if not (integer(category.get("primary"), 1) and integer(category.get("reserve"))):
            found.append(f"{name}: counts are missing")
        if not integer(category.get("max_new_tokens"), 1):
            found.append(f"{name}: max_new_tokens is missing")
        if not text(category.get("unit")) or not text(category.get("domain")):
            found.append(f"{name}: unit or domain is missing")
        if kind in PUBLIC and (category.get("source") != PUBLIC[kind] or PUBLIC[kind] not in contract.get("sources", {})):
            found.append(f"{name}: source is not pinned")
        if kind == "synthetic":
            generator = category.get("generator", {})
            if generator.get("kind") not in GENERATORS or generator.get("prompt") not in prompts:
                found.append(f"{name}: generator is missing")
            if generator.get("kind") == "long" and "token_range" not in category:
                found.append(f"{name}: token_range is missing")
        if "token_range" in category:
            bounds = category["token_range"]
            if not (isinstance(bounds, list) and len(bounds) == 2 and integer(bounds[0], 1) and integer(bounds[1], bounds[0])):
                found.append(f"{name}: token_range is malformed")
            elif bounds[1] > contract["layer1"]["bands"][-1][1]:
                found.append(f"{name}: token_range exceeds the last band")
            if kind == "reading" and not category.get("strata"):
                found.append(f"{name}: strata are missing")
    families = lookup(contract, ("layer1", "items", "families")) or {}
    for family in families.values():
        for name, count in family.items():
            match = [category for category in contract.get("categories", []) if category.get("id") == name]
            if not match or not integer(count, 1) or "token_range" not in match[0]:
                found.append(f"layer1 family member {name} is not a long-context category")
    for name in lookup(contract, ("layer2", "upper_bound", "categories")) or []:
        if name not in seen:
            found.append(f"upper bound category {name} is unknown")
    return found


def contract_problems(bundle):
    contract = bundle["contract"]
    found = []
    if not re.fullmatch(r"G31-QC-v\d+", str(contract.get("version"))) or bundle["prompts"].get("version") != contract.get("version"):
        found.append("contract or prompts version is missing or differs")
    if not integer(contract.get("seed")):
        found.append("seed is missing")
    return found + pins(contract, bundle["resources"]) + limits(contract) + categories(contract, bundle["prompts"])


def placement(source):
    keys = source.get("paragraphs")
    own = QUESTION.fullmatch(str(source.get("question")))
    if not isinstance(keys, list) or own is None:
        return None
    own = own.group(1) + own.group(2)
    if keys.count(own) != 1:
        return None
    expected = {"front": 0, "middle": (len(keys) - 1) // 2, "back": len(keys) - 1}.get(source.get("position"))
    return own if keys.index(own) == expected else None


def entry_problems(category, entry):
    found, source, kind = [], entry.get("source"), category["kind"]
    name = entry.get("id")
    expected = {"id", "category", "unit", "source", "item_sha256"} | ({"token_count"} if "token_range" in category else set())
    if set(entry) != expected:
        found.append(f"{name}: fields differ from the set schema")
    if entry.get("unit") != category["unit"] or not SHA256.fullmatch(str(entry.get("item_sha256"))):
        found.append(f"{name}: unit or item hash is malformed")
    if not re.fullmatch(re.escape(category["id"]) + r"-\d{4}", str(name)):
        found.append(f"{name}: id does not name its category")
    if "token_range" in category:
        low, high = category["token_range"]
        if not integer(entry.get("token_count")) or not low <= entry["token_count"] <= high:
            found.append(f"{name}: token_count is outside the range")
    if not isinstance(source, dict):
        return found + [f"{name}: source is missing"]
    if kind in SOURCE_FIELDS and set(source) != SOURCE_FIELDS[kind]:
        found.append(f"{name}: source fields differ")
    if kind in ("humaneval", "humanevalfix") and (not integer(source.get("task")) or source["task"] % 2 != category["parity"]):
        found.append(f"{name}: problem number parity differs")
    if kind == "mbpp" and not (integer(source.get("task_id")) and category["task_ids"][0] <= source["task_id"] <= category["task_ids"][1]):
        found.append(f"{name}: task id is outside the test split")
    if kind == "reading":
        fields = {"article", "question"} | ({"position", "paragraphs"} if "token_range" in category else set())
        if set(source) != fields or not str(source.get("question")).startswith(f"{source.get('article')}p"):
            found.append(f"{name}: reading source is malformed")
        elif "token_range" in category and (source["position"] not in category["strata"] or placement(source) is None):
            found.append(f"{name}: question paragraph position differs")
    if kind == "synthetic":
        fields = {"index", "template"} | ({"records"} if category["generator"]["kind"] == "long" else set())
        if set(source) != fields or not integer(source.get("index")):
            found.append(f"{name}: synthetic source is malformed")
    return found


def original(category, source):
    kind = category["kind"]
    if kind == "reading":
        return ("jsquad", source.get("article"))
    if kind in SOURCE_FIELDS:
        return (category["id"], json.dumps(source, sort_keys=True, ensure_ascii=False))
    return None


def set_problems(bundle):
    contract, found = bundle["contract"], []
    categories = {category["id"]: category for category in contract["categories"]}
    ids, originals, combinations, articles, fillers = set(), {}, {}, set(), []
    for name in SETS:
        document = bundle["sets"][name]
        if (document.get("contract_version"), document.get("set"), document.get("seed")) != (contract["version"], name, contract["seed"]):
            found.append(f"{name}: set header differs from the contract")
        counts = {}
        for entry in document.get("items", []):
            category = categories.get(entry.get("category"))
            if category is None:
                found.append(f"{entry.get('id')}: unknown category")
                continue
            counts[category["id"]] = counts.get(category["id"], 0) + 1
            if entry.get("id") in ids:
                found.append(f"duplicate item id {entry.get('id')}")
            ids.add(entry.get("id"))
            problems = entry_problems(category, entry)
            found += problems
            if problems:
                continue
            key = original(category, entry["source"])
            if key is not None:
                if key in originals:
                    found.append(f"{entry['id']}: same original as {originals[key]}")
                originals[key] = entry["id"]
            if category["kind"] == "reading":
                articles.add(entry["source"]["article"])
                for paragraph in entry["source"].get("paragraphs", []):
                    if paragraph != placement(entry["source"]):
                        fillers.append((entry["id"], PARAGRAPH.fullmatch(paragraph)))
            if category["kind"] in ("synthetic", "free"):
                value = item(category, entry, None, bundle)
                if digest(value) != entry["item_sha256"]:
                    found.append(f"{entry['id']}: item hash differs")
            if category["kind"] == "synthetic":
                made = generate(category, entry["source"]["index"], contract["seed"], bundle["prompts"],
                                bundle["vocabulary"], entry["source"].get("records"))
                key = (category["id"], json.dumps(made["combination"], ensure_ascii=False))
                if key in combinations:
                    found.append(f"{entry['id']}: same template combination as {combinations[key]}")
                combinations[key] = entry["id"]
        expected = {category["id"]: category[name] for category in contract["categories"] if category[name]}
        if counts != expected or document.get("summary") != counts:
            found.append(f"{name}: item counts differ from the contract")
    for name, match in fillers:
        if match is None or match.group(1) in articles:
            found.append(f"{name}: filler paragraph comes from a question article")
    return found


def document_problems(bundle):
    contract, document, found = bundle["contract"], bundle["document"], []
    if f"`{contract['version']}`" not in document or contract["frozen"] not in document:
        found.append("document does not state the contract version and freeze date")
    rows = TABLE.findall(document)
    table = dict(rows)
    if len(table) != len(rows):
        found.append("document repeats a file in the hash table")
    for path in FILES.values():
        recorded = table.get(f"Tools/LLMEvaluation/{path}")
        if recorded != bundle["hashes"][path]:
            found.append(f"document hash for {path} differs from the file")
    return found


def problems(bundle):
    found = contract_problems(bundle)
    if found:
        return found
    return set_problems(bundle) + document_problems(bundle)
