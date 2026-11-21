from collections import Counter
from fractions import Fraction
import json
import re
import unicodedata

from Tools.LLMEvaluation.items import FILES, ROOT, canonical, read_json

CONTRACT = read_json(ROOT / FILES["contract"])
RULES = CONTRACT["scorers"]
LATER = {"unscored": "unscored", "python_tests": "pending_code"}


def nfkc(text):
    return unicodedata.normalize("NFKC", text)


def reading(text):
    rule = RULES["reading_exact"]
    text = nfkc(text).strip()
    if text.endswith(rule["trailing"]):
        text = text[:-len(rule["trailing"])].strip()
    if any(text.startswith(opening) and text.endswith(closing) for opening, closing in rule["enclosing"]):
        text = text[1:-1]
    return text.strip()


def overlap(output, answer):
    if not output or not answer:
        return Fraction(0)
    return Fraction(2 * sum((Counter(output) & Counter(answer)).values()), len(output) + len(answer))


def final_integer(text):
    marker = RULES["final_integer"]["marker"]
    text = nfkc(text)
    if marker not in text:
        return None, "no_marker"
    found = re.match(r"\s*(-?\d+)", text[text.rindex(marker) + len(marker):])
    return (int(found.group(1)), "scored") if found else (None, "no_integer")


def decoded(text, start):
    repeated = []

    def pairs(entries):
        if len({key for key, _ in entries}) != len(entries):
            repeated.append(entries)
        return dict(entries)

    value, _ = json.JSONDecoder(object_pairs_hook=pairs).raw_decode(text, start)
    return value, bool(repeated)


def first_object(text):
    for start in (index for index, character in enumerate(text) if character == "{"):
        try:
            value, repeated = decoded(text, start)
        except (json.JSONDecodeError, RecursionError):
            continue
        return (None, "duplicate_key") if repeated else (value, "scored")
    return None, "no_json"


def strings(value):
    return isinstance(value, list) and all(isinstance(entry, str) for entry in value)


def entities(value, answer):
    keys = CONTRACT["entity_keys"]
    predicted = set() if value is None else {(key, name) for key in keys if strings(value.get(key))
                                             for name in value[key]}
    expected = {(key, name) for key in keys for name in answer[key]}
    correct = (value is not None and set(value) == set(keys) and all(strings(value[key]) for key in keys)
               and predicted == expected)
    return correct, {"tp": len(predicted & expected), "fp": len(predicted - expected), "fn": len(expected - predicted)}


def formatted(text, rules):
    found = [line.strip() for line in text.splitlines() if line.strip()]
    texts = [line[len(rules["bullet"]):].strip() for line in found]
    return found, (len(found) == rules["items"] and all(line.startswith(rules["bullet"]) for line in found)
                   and all(len(value) <= rules["max_chars"] and value.endswith(tuple(rules["endings"]))
                           for value in texts)
                   and all(any(word in value for value in texts) for word in rules["required"]))


def score(item, text):
    kind, answer = item["scorer"]["kind"], item["answer"]
    if kind in LATER:
        return {"kind": kind, "correct": None, "state": LATER[kind], "parsed": None}
    extra, state = {}, "scored"
    if kind == "choice_digit":
        parsed = next((int(character) for character in nfkc(text) if character in RULES[kind]["digits"]), None)
        correct, state = parsed == answer, "scored" if parsed is not None else "no_digit"
    elif kind == "final_integer":
        parsed, state = final_integer(text)
        correct = parsed == answer
    elif kind == "reading_exact":
        parsed, answers = reading(text), [reading(value) for value in answer]
        correct, extra = parsed in answers, {"f1": max(overlap(parsed, value) for value in answers)}
    elif kind == "format_rules":
        parsed, correct = formatted(text, item["scorer"])
    elif kind == "json_exact":
        parsed, state = first_object(text)
        correct = parsed is not None and canonical(parsed) == canonical(answer)
    elif kind == "entity_sets":
        parsed, state = first_object(text)
        correct, extra = entities(parsed, answer)
    else:
        raise ValueError(f"Unknown scorer kind: {kind}")
    return {"kind": kind, "correct": correct, "state": state, "parsed": parsed, **extra}
