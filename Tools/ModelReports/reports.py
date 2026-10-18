import datetime
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
LANGS = ("en", "ja")
SUPPORT = {"verified": "✅", "pending": "△", "unavailable": "—"}
STATES = {"passed", "failed", "not_measured", "not_applicable"}
STATISTICS = {"single", "median", "mean", "p95", "min", "max", "aggregate"}
CHECKS = ("fidelity", "quality", "stability", "cancellation")
CONTEXT = ("date", "platform", "device", "os", "toolchain", "variant", "backend", "route",
           "source_revision", "artifact_revision", "code_revision", "workload", "batch_size",
           "concurrency", "warmups", "power", "thermal", "plan_sha256")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: need(False, f"Non-finite JSON value: {value}"))


def words(value):
    need(isinstance(value, dict) and set(value) == set(LANGS), "Expected en/ja text")
    need(all(isinstance(value[k], str) and value[k].strip() for k in LANGS), "Empty translation")


def inside(root, value):
    path = (root / value).resolve()
    need(not Path(value).is_absolute() and path.is_relative_to(root.resolve()), f"Unsafe path: {value}")
    need(not any(p in {".local", ".git", ".."} for p in Path(value).parts), f"Private path: {value}")
    return path


def anchor_map(path):
    text = re.sub(r"```.*?```", lambda m: re.sub(r"[^\n]", " ", m[0]), path.read_text(), flags=re.S)
    seen = {}
    result = {m[1]: m.start() for m in re.finditer(r'<a\s+(?:id|name)="([^"]+)"', text)}
    for match in re.finditer(r"^#{1,6} (.+)$", text, re.M):
        heading = match[1]
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading).replace("`", "")
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        result[slug + (f"-{count}" if count else "")] = match.start()
    return result


def anchor_set(path):
    return set(anchor_map(path))


def reference(root, value, evidence=False):
    need(isinstance(value, str) and value, "Missing evidence/reference")
    parsed = urlsplit(value)
    if parsed.scheme:
        need(parsed.scheme == "https" and parsed.netloc and not parsed.username and not parsed.password,
             f"Expected public HTTPS reference: {value}")
        return
    path = inside(root, unquote(parsed.path))
    need(path.is_file(), f"Missing reference: {value}")
    if parsed.fragment and path.suffix == ".md":
        need(unquote(parsed.fragment) in anchor_set(path), f"Missing anchor: {value}")
    if evidence:
        need(not path.is_relative_to((root / "docs/model-reports").resolve()), "Reporting metadata is not primary evidence")
        if path.suffix == ".md":
            need("<!-- model-catalog:start -->" not in path.read_text(), "Generated catalog is not primary evidence")
        if path.suffix == ".md" and "<!-- model-report:start -->" in path.read_text():
            text = path.read_text()
            position = anchor_map(path).get(unquote(parsed.fragment))
            need(position is not None and not (text.index("<!-- model-report:start -->") <= position <=
                                              text.index("<!-- model-report:end -->")),
                 "Generated summary is not primary evidence; cite an original section")


def number(value, metric, unit, spec):
    need(metric in spec["metrics"], f"Unknown metric: {metric}")
    definition = spec["metrics"][metric]
    need(unit in definition["units"], f"Wrong unit for {metric}: {unit}")
    need(type(value) in (int, float) and math.isfinite(value), f"Invalid numeric value: {metric}")
    for bound, compare in (("min", lambda a, b: a >= b), ("max", lambda a, b: a <= b)):
        need(bound not in definition or compare(value, definition[bound]), f"Out of range: {metric}")


def validate(root, spec, model):
    need(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", model["id"]), "Invalid model id")
    need(model["kind"] in spec["profiles"], "Unknown model kind")
    for key in ("name", "use", "overview", "release_note"):
        words(model[key])
    need(model["release"] in {"released", "in_progress", "planned"}, "Invalid release state")
    if model["download"]:
        reference(root, model["download"])
    if model["release"] == "released":
        need(model["download"] and model.get("public_access_evidence"), "Release needs public access evidence")
        reference(root, model["public_access_evidence"], evidence=True)
    for path in model["pages"].values():
        need(path.startswith("docs/") and path.endswith(".md"), "Report page must be under docs/")
        inside(root, path)
    need(set(model["pages"]) == set(LANGS), "Both report languages are required")
    for item in model["links"]:
        words(item["title"])
        reference(root, item["url"])
    for key, item in model["evidence"].items():
        need(re.fullmatch(r"[a-z0-9-]+", key), "Invalid evidence id")
        reference(root, item["url"], evidence=True)
        need(item["level"] in {"native_app", "diagnostic", "conversion", "plan"}, "Unknown evidence level")
        need(item["outcome"] in STATES and item["record_type"] in {"legacy", "standard"}, "Invalid evidence state")
        need(type(item["model_executed"]) is bool, "Record whether the model executed")
        need(set(CONTEXT) <= item.keys(), f"Missing context fields: {key}")
        for field in set(CONTEXT) - {"workload", "batch_size", "concurrency", "warmups"}:
            value = item[field]
            need(value is None or (isinstance(value, str) and value.strip()), f"Invalid context field: {field}")
        datetime.date.fromisoformat(item["date"])
        need(item["platform"] in {"macos", "ios", "other"}, "Invalid platform")
        words(item["workload"])
        if item["record_type"] == "standard":
            need(all(item[k] is not None for k in CONTEXT), f"Incomplete standard context: {key}")
            for field in ("source_revision", "artifact_revision", "code_revision"):
                need(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", item[field]), f"Unpinned {field}")
            need(re.fullmatch(r"[0-9a-f]{64}", item["plan_sha256"]), "Unpinned measurement plan")
        for field in ("batch_size", "concurrency", "warmups"):
            value = item[field]
            need(value is None or (type(value) is int and value >= (0 if field == "warmups" else 1)),
                 f"Invalid {field}")
    need(set(model["support"]) == {"macos", "ios"}, "Both platform states are required")
    for platform, item in model["support"].items():
        need(item["state"] in SUPPORT, "Unknown support state")
        words(item["note"])
        need(all(key in model["evidence"] for key in item["evidence"]), "Unknown support evidence")
        if item["state"] == "verified":
            need(item["evidence"] and item["variants"], "Verified support needs evidence and variants")
            for variant in item["variants"]:
                need(isinstance(variant, str) and variant.strip(), "Verified variant must be named")
                need(any(model["evidence"][key]["level"] == "native_app"
                         and model["evidence"][key]["outcome"] == "passed"
                         and model["evidence"][key]["model_executed"]
                         and model["evidence"][key]["platform"] == platform
                         and model["evidence"][key]["variant"] == variant
                         and all(isinstance(model["evidence"][key][field], str) and model["evidence"][key][field].strip()
                                 for field in ("device", "os", "route"))
                         and not re.search(r"simulator|preview|stub|fake", model["evidence"][key]["device"], re.I)
                         for key in item["evidence"]), "Verified demo requires matching native device evidence")
    for item in model["measurements"]:
        number(item["value"], item["metric"], item["unit"], spec)
        need(item["evidence"] in model["evidence"], "Unknown measurement evidence")
        source = model["evidence"][item["evidence"]]
        need(source["level"] != "plan", "A plan is not a measurement")
        if source["record_type"] == "standard":
            need(item["unit"] == spec["metrics"][item["metric"]]["units"][0], "New records require the canonical unit")
        need(item["statistic"] in STATISTICS, "Unknown statistic")
        n = item["n"]
        need((type(n) is int and n > 0) or (n is None and source["record_type"] == "legacy"), "Missing sample count")
        need(item["statistic"] != "p95" or (n is not None and n >= 20), "p95 needs at least 20 samples")
        need(item["statistic"] != "single" or n in (None, 1), "Single observation must have n=1")
        words(item["note"])
        if "source_value" in item:
            source_value = item["source_value"]
            reference(root, source_value["path"], evidence=True)
            value = read(inside(root, source_value["path"]))
            for component in source_value["pointer"].split("/")[1:]:
                value = value[int(component)] if isinstance(value, list) else value[component.replace("~1", "/").replace("~0", "~")]
            need(type(value) in (int, float) and value == item["value"], "Measurement differs from source JSON")
    for metric, item in model["omissions"].items():
        need(metric in spec["metrics"] and item["state"] in {"not_measured", "not_applicable"}, "Invalid omission")
        need(not any(row["metric"] == metric for row in model["measurements"]), "Measured metric also omitted")
        words(item["reason"])
    need(set(model["checks"]) == set(CHECKS), "All verification categories are required")
    for item in model["checks"].values():
        need(item["state"] in STATES, "Unknown check state")
        words(item["note"])
        need(all(key in model["evidence"] for key in item["evidence"]), "Unknown check evidence")
        if item["state"] in {"passed", "failed"}:
            need(item["evidence"], "A check result needs evidence")
            need(all(model["evidence"][key]["level"] != "plan" and model["evidence"][key]["model_executed"]
                     for key in item["evidence"]), "A plan or model-free check cannot establish model verification")
            need(all(model["evidence"][key]["outcome"] == item["state"] for key in item["evidence"]),
                 "Check outcome contradicts its evidence; record separate criteria separately")


def load(root):
    spec = read(root / "docs/model-reports/metrics.json")
    catalog = read(root / "docs/model-reports/catalog.json")
    need(spec["version"] == catalog["version"] == 1, "Unknown reporting version")
    datetime.date.fromisoformat(catalog["checked_on"])
    models = [read(inside(root, f"docs/model-reports/records/{key}.json")) for key in catalog["models"]]
    need([m["id"] for m in models] == catalog["models"], "Catalog/record identity mismatch")
    pages = [inside(root, p) for m in models for p in m["pages"].values()]
    need(len(set(catalog["models"])) == len(models) and len(set(pages)) == len(pages), "Duplicate model or page")
    for model in models:
        validate(root, spec, model)
    return spec, catalog, models


def cell(value):
    return str(value).replace("|", "&#124;").replace("\n", " ")


def link(root, page, target, title):
    if not urlsplit(target).scheme:
        path, _, fragment = target.partition("#")
        target = os.path.relpath(root / path, (root / page).parent).replace(os.sep, "/")
        if fragment:
            target += "#" + fragment
    return f"[{cell(title)}]({target})"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows])


def overview(root, spec, catalog, models, lang):
    ui = spec["text"][lang]
    page = "README.ja.md" if lang == "ja" else "README.md"
    rows = []
    for model in models:
        guide = link(root, page, model["pages"][lang], ui["guide"])
        supports = [link(root, page, model["pages"][lang] + "#model-report", SUPPORT[model["support"][p]["state"]]) for p in ("macos", "ios")]
        download = link(root, page, model["download"], ui["download"] if model["release"] == "released" else ui["restricted"]) if model["download"] else "—"
        rows.append([model["name"][lang], model["use"][lang], *supports,
                     ui[model["release"]] + " · " + model["release_note"][lang], download, guide])
    return ui["checked"] + " **" + catalog["checked_on"] + "**. " + ui["legend"] + "\n\n" + table(ui["catalog_headers"], rows)


def report(root, spec, model, lang):
    ui, page = spec["text"][lang], model["pages"][lang]
    evidence_ids = {key: f"E{index + 1}" for index, key in enumerate(model["evidence"])}
    def refs(keys):
        return " · ".join(link(root, page, model["evidence"][key]["url"], evidence_ids[key]) for key in keys) or "—"
    lines = ['<a id="model-report"></a>', "## " + ui["summary"],
             link(root, page, "docs/benchmarking" + (".ja" if lang == "ja" else "") + ".md", ui["protocol"])
             + " · " + link(root, page, f"docs/model-reports/records/{model['id']}.json", ui["record"]), ui["historical"]]
    rows = [[p, SUPPORT[item["state"]], item["note"][lang], refs(item["evidence"])] for p, item in model["support"].items()]
    lines += ["### " + ui["demo"], table(ui["support_headers"], rows), "### " + ui["performance"]]
    metrics = list(dict.fromkeys(spec["common"] + spec["profiles"][model["kind"]] + [m["metric"] for m in model["measurements"]]))
    rows = []
    for metric in metrics:
        values = [m for m in model["measurements"] if m["metric"] == metric]
        name = spec["metrics"][metric]["label"][lang]
        if not values:
            omission = model["omissions"].get(metric)
            rows.append([name, ui[omission["state"]] if omission else ui["not_recorded"], "—",
                         omission["reason"][lang] if omission else ui["missing"]])
        for value in values:
            count = str(value["n"]) if value["n"] is not None else ui["unrecorded"]
            result = f"{value['value']:,}" if type(value["value"]) is int else f"{value['value']:.9g}"
            rows.append([name, result + " " + value["unit"], value["statistic"] + "; n=" + count,
                         refs([value["evidence"]]) + " · " + value["note"][lang]])
    lines += [table(ui["metric_headers"], rows), "### " + ui["validation"]]
    rows = [[ui[key], ui[item["state"]], item["note"][lang], refs(item["evidence"])] for key, item in model["checks"].items()]
    lines += [table(ui["check_headers"], rows), "### " + ui["conditions"]]
    rows = []
    for key, item in model["evidence"].items():
        hardware = " / ".join(str(item[field]) if item[field] is not None else ui["unrecorded"] for field in ("device", "os", "toolchain"))
        limits = "; ".join(f"{field}={item[field]}" for field in ("variant", "backend", "route", "source_revision", "artifact_revision", "code_revision", "batch_size", "concurrency", "warmups", "power", "thermal") if item[field] is not None)
        missing = [field for field in CONTEXT if item[field] is None]
        if missing:
            limits += "; " + ui["unrecorded"] + ": " + ", ".join(missing)
        rows.append([refs([key]), item["date"] + " / " + item["record_type"] + " / " + item["level"], hardware,
                     item["workload"][lang] + " · " + limits])
    lines += [table(ui["evidence_headers"], rows) if rows else ui["missing"], "### " + ui["details"]]
    lines += ["- " + link(root, page, item["url"], item["title"][lang]) for item in model["links"]]
    return "\n\n".join(lines)


def replace_block(text, content, block):
    start, end = f"<!-- {block}:start -->", f"<!-- {block}:end -->"
    need(text.count(start) == text.count(end) == 1, f"Expected one {block} block")
    left, tail = text.split(start)
    _, right = tail.split(end)
    return left + start + "\n" + content + "\n" + end + right


def render(root, spec, catalog, models, write=False):
    outputs = {}
    for lang in LANGS:
        page = "README.ja.md" if lang == "ja" else "README.md"
        outputs[page] = replace_block((root / page).read_text(), overview(root, spec, catalog, models, lang), "model-catalog")
        for model in models:
            page = model["pages"][lang]
            outputs[page] = replace_block((root / page).read_text(), report(root, spec, model, lang), "model-report")
    if write:
        for path, content in outputs.items():
            inside(root, path).write_text(content)
    return outputs


def check(root, spec, catalog, models, complete=None):
    outputs = render(root, spec, catalog, models)
    for path, expected in outputs.items():
        need((root / path).read_text() == expected, f"Stale generated content: {path}; run render")
        for target in re.findall(r"\]\(([^)\s]+)\)", expected):
            if urlsplit(target).scheme:
                reference(root, target)
            else:
                filename, _, fragment = target.partition("#")
                destination = (root / path).parent / filename if filename else root / path
                need(destination.is_file(), f"Broken link in {path}: {target}")
                if fragment and destination.suffix == ".md":
                    need(unquote(fragment) in anchor_set(destination), f"Broken anchor in {path}: {target}")
    if complete:
        selected = [m for m in models if m["id"] == complete]
        need(selected, f"Unknown model: {complete}")
        complete_record(spec, selected[0])


def complete_record(spec, model):
    groups = {}
    fields = ("variant", "platform", "device", "os", "source_revision", "artifact_revision", "code_revision", "plan_sha256")
    for key, evidence in model["evidence"].items():
        if evidence["record_type"] == "standard":
            group = tuple(evidence[field] for field in fields)
            groups.setdefault(group, set()).add(key)
    required = set(spec["common"] + spec["profiles"][model["kind"]])
    waived = {key for key, value in model["omissions"].items() if value["state"] == "not_applicable"}
    for evidence_ids in groups.values():
        measured = {v["metric"] for v in model["measurements"] if v["evidence"] in evidence_ids}
        checks_pass = all(c["state"] == "not_applicable" or
                          (c["state"] == "passed" and set(c["evidence"]) & evidence_ids)
                          for c in model["checks"].values())
        if required <= measured | waived and checks_pass:
            return
    raise ValueError("Incomplete current-protocol report: no single pinned variant/device/plan covers metrics and checks")


def template(spec, kind, model_id):
    need(kind in spec["profiles"], "Unknown model kind")
    return {"protocol_version": 1, "model_id": model_id, "kind": kind,
            "context": {key: None for key in CONTEXT if key != "plan_sha256"},
            "fixture_revision": None, "timing_boundary": None, "acceptance_criteria": None,
            "warmup_runs": 1, "measured_runs": 20,
            "metrics": {key: spec["metrics"][key]["units"][0] for key in spec["common"] + spec["profiles"][kind]},
            "note": "Planning template only. Fill and freeze before measuring; not evidence of execution."}


def summarize(path, spec):
    rows = [json.loads(line, object_pairs_hook=unique) for line in path.read_text().splitlines() if line.strip()]
    need(rows, "No run records")
    identities = {json.dumps(row["context"], sort_keys=True) for row in rows}
    need(len(identities) == 1, "Mixed model/device/workload/cache conditions")
    context = rows[0]["context"]
    need(all(isinstance(context.get(k), str) and context[k].strip()
             for k in ("model", "variant", "device", "os", "workload", "cache", "plan_sha256")), "Incomplete run identity")
    need(re.fullmatch(r"[0-9a-f]{64}", context["plan_sha256"]), "Expected frozen plan SHA256")
    need(len({row["run_id"] for row in rows}) == len(rows), "Duplicate run id")
    need(all(row["status"] in {"passed", "failed", "warmup"} for row in rows), "Unknown run status")
    for row in rows:
        for metric, item in row["metrics"].items():
            number(item["value"], metric, item["unit"], spec)
            need(item["unit"] == spec["metrics"][metric]["units"][0], "New runs require the canonical unit")
    passed = [r for r in rows if r["status"] == "passed"]
    need(passed, "No successful measured runs")
    keys = set(passed[0]["metrics"])
    need(keys and all(set(row["metrics"]) == keys for row in passed), "Missing metrics in a successful run")
    result = {"protocol_version": 1, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "context": context, "counts": {state: sum(r["status"] == state for r in rows) for state in ("passed", "failed", "warmup")}, "metrics": {}}
    for metric in sorted(keys):
        values = [r["metrics"][metric]["value"] for r in passed]
        units = {r["metrics"][metric]["unit"] for r in passed}
        need(len(units) == 1, f"Mixed units: {metric}")
        unit = units.pop()
        for value in values:
            number(value, metric, unit, spec)
        ordered, count = sorted(values), len(values)
        result["metrics"][metric] = {"unit": unit, "n": count, "min": min(values), "median": statistics.median(values),
                                    "mean": statistics.mean(values), "p95": ordered[math.ceil(count * .95) - 1] if count >= 20 else None,
                                    "max": max(values)}
    return result
