import hashlib
import json
from pathlib import Path
import platform
import random
import statistics

from Tools.LLMEvaluation.items import SETS, Sources, digest, item
from Tools.LLMEvaluation.synthetic.generate import generate


class Counter:
    def __init__(self, contract, tokenizer_path, template_path):
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        import tokenizers

        files = contract["model"]["files"]
        for name, path in (("tokenizer.json", tokenizer_path), ("chat_template.jinja", template_path)):
            data = Path(path).read_bytes()
            if len(data) != files[name]["bytes"] or hashlib.sha256(data).hexdigest() != files[name]["sha256"]:
                raise ValueError(f"{name} differs from the contract")
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        added = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))["added_tokens"]
        control = sorted({token["id"] for token in added if token["special"]} - set(contract["generation"]["eos_token_id"]))
        if control != contract["layer2"]["critical_failures"]["control_token_ids"]:
            raise ValueError("Control token ids differ from the tokenizer")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                    extensions=["jinja2.ext.loopcontrols"])
        environment.globals["raise_exception"] = self.fail
        self.template = environment.from_string(Path(template_path).read_text(encoding="utf-8"))
        self.frame = contract["generation"]["template"]
        self.special = {token["id"] for token in added if token["special"]}
        self.expected = self.specials(self.tokenizer.encode(self.frame["prefix"] + "x" + self.frame["suffix"],
                                                            add_special_tokens=False).ids)
        self.versions = {"python": platform.python_version(), "tokenizers": tokenizers.__version__,
                         "jinja2": jinja2.__version__}

    @staticmethod
    def fail(message):
        raise ValueError(message)

    def specials(self, ids):
        return sorted(token for token in ids if token in self.special)

    def count(self, text):
        return len(self.tokenizer.encode(self.frame["prefix"] + text + self.frame["suffix"], add_special_tokens=False).ids)

    def verify(self, text):
        rendered = self.template.render(messages=[{"role": "user", "content": text}], **self.frame["arguments"])
        if rendered != self.frame["prefix"] + text + self.frame["suffix"]:
            raise ValueError("The rendered template differs from the recorded frame")
        ids = self.tokenizer.encode(rendered, add_special_tokens=False).ids
        if ids[:2] != self.frame["tokenization"]["first_ids"] or self.specials(ids) != self.expected:
            raise ValueError("The rendered input has special tokens outside the recorded frame")
        return len(ids)


def place(own, fillers, position):
    middle = {"front": 0, "middle": len(fillers) // 2, "back": len(fillers)}[position]
    return fillers[:middle] + [own] + fillers[middle:]


def assemble(category, source, entry_id, data, bundle, fillers, counter):
    own = data["jsquad"][source["article"]]["questions"][source["question"]]["paragraph"]
    order = list(fillers)
    random.Random(f"{bundle['contract']['seed']}:{category['id']}:{entry_id}").shuffle(order)
    low, high = category["token_range"]
    chosen = []
    for key in order:
        trial = {**source, "paragraphs": place(own, chosen + [key], source["position"])}
        tokens = counter.count(item(category, {"id": entry_id, "source": trial, "token_count": 0}, data,
                                    bundle)["messages"][0]["content"])
        if tokens > high:
            continue
        chosen.append(key)
        if tokens >= low:
            return trial
    raise ValueError(f"Could not reach the token range: {entry_id}")


def tune(category, index, bundle, counter):
    low, high = category["token_range"]
    records, seen = max(8, low // 14), set()
    while records not in seen:
        seen.add(records)
        made = generate(category, index, bundle["contract"]["seed"], bundle["prompts"], bundle["vocabulary"], records)
        tokens = counter.count(made["content"])
        if low <= tokens <= high:
            return records
        step = max(1, abs(tokens - (low + high) // 2) // 16) if len(seen) < 8 else 1
        records += step if tokens < low else -step
    raise ValueError(f"No record count reaches the token range: {category['id']} {index}")


def sample(category, data, bundle, rng, taken, used):
    total = category["primary"] + category["reserve"]
    kind = category["kind"]
    if kind == "choice":
        return [{"q_id": key} for key in rng.sample(sorted(data["jcqa"]), total)]
    if kind == "reading":
        names = rng.sample(sorted(set(data["jsquad"]) - taken), total)
        taken.update(names)
        return [{"article": name, "question": rng.choice(sorted(data["jsquad"][name]["questions"]))} for name in names]
    if kind in ("humaneval", "humanevalfix"):
        pool = [key for key in sorted(data[category["source"]]) if key % 2 == category["parity"]]
        return [{"task": key} for key in rng.sample(pool, total)]
    if kind == "mbpp":
        low, high = category["task_ids"]
        return [{"task_id": key} for key in rng.sample([key for key in sorted(data["mbpp"]) if low <= key <= high], total)]
    if kind == "entities":
        return [{"index": key, "curid": data["ner"][key]["curid"]} for key in rng.sample(sorted(data["ner"]), total)]
    if kind == "free":
        return [{"instruction": key} for key in rng.sample(sorted(row["id"] for row in bundle["prompts"]["free-ja"]), total)]
    if kind == "synthetic":
        picked, combinations, index = [], used.setdefault(category["generator"]["prompt"], set()), 0
        while len(picked) < total:
            made = generate(category, index, bundle["contract"]["seed"], bundle["prompts"], bundle["vocabulary"], 64)
            key = json.dumps(made["combination"], ensure_ascii=False)
            if key not in combinations:
                combinations.add(key)
                picked.append({"index": index, "template": made["template"]})
            index += 1
        return picked
    raise ValueError(f"Unknown category kind: {kind}")


def choose(bundle, datasets, counter):
    contract = bundle["contract"]
    data = Sources(contract, datasets)
    taken, used, rows, counts = set(), {}, {name: [] for name in SETS}, {}
    for category in contract["categories"]:
        picked = sample(category, data, bundle, random.Random(f"{contract['seed']}:{category['id']}"), taken, used)
        for number, source in enumerate(picked, start=1):
            name = "primary" if number <= category["primary"] else "reserve"
            position = number - 1 if name == "primary" else number - 1 - category["primary"]
            if "strata" in category:
                source["position"] = category["strata"][position % len(category["strata"])]
            rows[name].append({"id": f"{category['id']}-{number:04d}", "category": category["id"],
                               "unit": category["unit"], "source": source})
    fillers = sorted(key for name in set(data["jsquad"]) - taken for key in data["jsquad"][name]["paragraphs"])
    categories = {category["id"]: category for category in contract["categories"]}
    for name in SETS:
        for entry in rows[name]:
            category = categories[entry["category"]]
            if category["kind"] == "reading" and "token_range" in category:
                entry["source"] = assemble(category, entry["source"], entry["id"], data, bundle, fillers, counter)
            elif category["kind"] == "synthetic" and "token_range" in category:
                entry["source"]["records"] = tune(category, entry["source"]["index"], bundle, counter)
            value = item(category, {**entry, "token_count": 0}, data, bundle)
            tokens = counter.verify(value["messages"][0]["content"])
            counts.setdefault(category["id"], []).append(tokens)
            if "token_range" in category:
                low, high = category["token_range"]
                if not low <= tokens <= high:
                    raise ValueError(f"Token count outside the range: {entry['id']} {tokens}")
                entry["token_count"] = value["token_count"] = tokens
            entry["item_sha256"] = digest(value)
    return rows, counts


def documents(bundle, rows, counter):
    contract = bundle["contract"]
    environment = {**counter.versions, "tokenizer_sha256": contract["model"]["files"]["tokenizer.json"]["sha256"],
                   "chat_template_sha256": contract["model"]["files"]["chat_template.jinja"]["sha256"]}
    result = {}
    for name in SETS:
        summary = {}
        for entry in rows[name]:
            summary[entry["category"]] = summary.get(entry["category"], 0) + 1
        result[name] = {"contract_version": contract["version"], "set": name, "seed": contract["seed"],
                        "selected_with": environment, "summary": summary, "items": rows[name]}
    return result


def serialize(document):
    head = json.dumps({key: value for key, value in document.items() if key != "items"}, indent=2, sort_keys=True,
                      ensure_ascii=False)
    items = ",\n".join("    " + json.dumps(row, sort_keys=True, ensure_ascii=False) for row in document["items"])
    return head[:-2] + ',\n  "items": [\n' + items + "\n  ]\n}\n"


VERSIONS = ("python", "tokenizers", "jinja2")


def compare(document, committed):
    recorded = json.loads(committed).get("selected_with") or {}
    versions = {key: recorded.get(key) for key in VERSIONS}
    current = {key: document["selected_with"].get(key) for key in VERSIONS}
    same = serialize({**document, "selected_with": {**document["selected_with"], **versions}}) == committed
    return same, None if versions == current else {"recorded": versions, "current": current}


def statistics_of(counts):
    return {name: {"items": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}
            for name, values in counts.items()}
