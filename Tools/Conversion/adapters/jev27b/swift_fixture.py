import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from ...storage import read_json
from .compare import tokens_sha256
from .fixtures import CARD_EXAMPLES, load_rows, serialize
from .prepare import distributed_tokenizer
from .typed_head import FIXED_OPTIONS, LINE_BREAKS, SLOTS, decide, render, tokenize

ROWS = SLOTS["choice"][1]
WIDTH = 4
FIELDS = ("id", "kind", "state", "question", "options")
NOUL, SCORE = (list(FIXED_OPTIONS[kind]) for kind in ("noul", "score"))
JAPANESE = (("JNLI", 3), ("JCommonsenseQA", 3), ("JSTS", 2))
REAL_HEAD_TASKS = ("JNLI", "JSTS")
STATE, QUESTION = "A state.", "A question?"
SYNTHETIC = (
    ("synthetic:choice-2", "choice", "The same order has two open support tickets.", "Which ticket should stay open?",
     ["the older ticket", "the newer ticket"]),
    ("synthetic:choice-16", "choice", "A routing table lists sixteen queues.", "Which queue should take the request?",
     ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett", "kilo", "lima",
      "mike", "november", "oscar", "papa"]),
    ("synthetic:quotes", "choice", "He wrote \"refund now\" and 'asap' in C:\\orders\\2026\\n.txt; also \\\\ and \\\"",
     "Is the note \"urgent\"?", ["say \"yes\"", "it's fine", "back\\slash"]),
    ("synthetic:tab", "noul", "column 1\tcolumn 2\t\tcolumn 4", "Is\tthe table complete?", NOUL),
    ("synthetic:emoji", "choice",
     "Box \U0001F4E6 arrived; dev \U0001F469\u200d\U0001F4BB replied; flag \U0001F1EF\U0001F1F5; "
     "thumbs \U0001F44D\U0001F3FD", "Which mood fits?", ["\U0001F600 happy", "\U0001F622 sad", "\U0001F610 neutral"]),
    ("synthetic:fullwidth", "choice",
     "\uff21\uff22\uff23 \uff11\uff12\uff13 \uff08note\uff09\uff01\uff1f\uff03\uff04\uff05\u3000end",
     "Which width is used?", ["\uff26\uff55\uff4c\uff4c", "half"]),
    ("synthetic:placeholder", "noul", "{state} {question} {options} {kind} {label}) {option} [decision]:",
     "Does {question} stay literal?", NOUL),
    ("synthetic:multiline", "score", "first line\nsecond line\n\nfourth line\n", "Line one?\nLine two?", SCORE),
    ("synthetic:empty-state", "noul", "", "Is the state empty?", NOUL),
    ("synthetic:empty-question", "choice", "The question is empty.", "", ["yes", "no"]),
    ("synthetic:whitespace", "choice", "  padded state  ", " padded question ", ["  left", "right  ", " both "]),
    ("synthetic:noul", "noul", "The invoice total matches the purchase order.", "Do the totals match?", NOUL),
    ("synthetic:score", "score", "The reply answered two of the three questions.", "Rate the completeness from 0 to 5.",
     SCORE),
    ("synthetic:scalar-distinct", "choice", "Two options are canonically equivalent but use different scalars.",
     "Which spelling is used?", ["caf\u00e9", "cafe\u0301"]),
    ("synthetic:empty-option", "choice", "One option is empty.", "Pick one.", ["", "non-empty"]),
)
REJECTED = (
    ("reject:choice-17", "choice", [f"option {index}" for index in range(1, 18)]),
    ("reject:choice-1", "choice", ["only"]),
    ("reject:choice-0", "choice", []),
    ("reject:duplicate", "choice", ["same", "other", "same"]),
    *((f"reject:line-break-u{ord(character):04x}", "choice", [f"before{character}after", "plain"])
      for character in LINE_BREAKS),
    ("reject:line-break-crlf", "choice", ["before\r\nafter", "plain"]),
    ("reject:noul-reversed", "noul", ["true", "false"]),
    ("reject:noul-case", "noul", ["False", "True"]),
    ("reject:score-five", "score", SCORE[:5]),
    ("reject:score-order", "score", SCORE[:4] + SCORE[:3:-1]),
)
WEIGHT = [[((7 * row + 3 * column) % 13 - 6) / 8 for column in range(WIDTH)] for row in range(ROWS)]
BIAS = [0.25, -0.5, 0.75, 0.0, -0.25, 0.5, -0.75, 1.0, -1.0, 1.5, 0.5, 1.5,
        -0.5, 0.25, 0.0, -0.25, 0.75, -0.75, 0.5, -1.0, 1.0, 0.25, -0.5, 0.0]
HEAD_CASES = (
    ("noul", "noul", 2, [0.5, -1.0, 1.5, 0.25]),
    ("score", "score", 6, [-0.75, 0.5, 1.0, -1.25]),
    ("choice-2", "choice", 2, [1.0, 1.0, -0.5, 0.75]),
    ("choice-5", "choice", 5, [-1.5, 0.25, 0.5, 2.0]),
    ("choice-16", "choice", 16, [0.75, -0.25, -1.0, 1.25]),
    ("tie", "choice", 4, [0.0, 0.0, 0.0, 0.0]),
)


def license_note(pins):
    return (f"Rows whose id starts with jglue: are derived from JGLUE {pins['tag']} (https://github.com/{pins['repository']}, "
            f"revision {pins['revision']}; Kurihara, Kawahara and Shibata, LREC 2022), licensed under {pins['license']} "
            "(https://creativecommons.org/licenses/by-sa/4.0/). The rows were placed into the bare-v1 template and "
            "tokenized, and they are shared under the same license. Rows whose id starts with card-example: use the "
            "examples of the autotrust/JEV-27B model card (Apache-2.0). The other rows are synthetic.")


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_rows():
    cards = [{"id": key, **{field: CARD_EXAMPLES[key][field] for field in FIELDS[1:]}} for key in CARD_EXAMPLES]
    return cards + [dict(zip(FIELDS, row)) for row in SYNTHETIC]


def japanese_rows(restored):
    rows = []
    for task, count in JAPANESE:
        rows += [row for row in restored if row["id"].startswith(f"jglue:{task}:")][:count]
    return [{field: row[field] for field in FIELDS} for row in rows]


def rendered(row, tokenizer):
    text = render(row["kind"], row["state"], row["question"], row["options"])
    ids = tokenize(tokenizer, text)
    if tokenizer.decode(ids[-1:]) != "]:":
        raise ValueError(f"The last token is not ]: in {row['id']}")
    return {**row, "text": text, "tokens": ids, "token_count": len(ids)}


def rejected_rows():
    rows = []
    for identifier, kind, options in REJECTED:
        try:
            render(kind, STATE, QUESTION, options)
        except ValueError:
            rows.append({"id": identifier, "kind": kind, "state": STATE, "question": QUESTION, "options": options,
                         "error": "invalidOptions"})
        else:
            raise ValueError(f"The reference renderer accepts {identifier}")
    return rows


def head_cases(temperatures):
    cases = []
    for identifier, kind, count, hidden in HEAD_CASES:
        logits = [math.fsum([w * h for w, h in zip(WEIGHT[row], hidden)] + [BIAS[row]]) for row in range(ROWS)]
        if any(float32(value) != value for value in logits + sum(WEIGHT, []) + BIAS + hidden):
            raise ValueError("Synthetic head values must be exact in float32")
        shipping, plain = decide(logits, kind, count, temperatures[kind])
        if shipping.count(max(shipping)) != (2 if identifier == "tie" else 1):
            raise ValueError(f"Unexpected tie structure in {identifier}")
        cases.append({"id": identifier, "kind": kind, "count": count, "hidden": hidden, "logits": logits,
                      "temperature": temperatures[kind], "plain": plain, "shipping": shipping,
                      "top": shipping.index(max(shipping))})
    return cases


def real_head_rows(rows, merged, hidden_path, head_path):
    from safetensors import safe_open

    with safe_open(str(hidden_path), "np") as handle:
        metadata = handle.metadata()
        ids = json.loads(metadata["ids"])
        hidden, mask = handle.get_tensor("hidden"), handle.get_tensor("mask")
    with safe_open(str(head_path), "np") as handle:
        weight, bias = handle.get_tensor("proj.weight"), handle.get_tensor("proj.bias")
    if ids != [row["id"] for row in merged["rows"]] or str(hidden.dtype) != "float32" \
            or metadata.get("merged_revision") != merged["merged_revision"]:
        raise ValueError("The hidden states do not follow the merged reference")
    result = []
    for row in rows:
        index = ids.index(row["id"])
        reference = merged["rows"][index]
        temperature = merged["temperatures"][row["kind"]]
        shipping, plain = decide(reference["logits1m"], row["kind"], len(row["options"]), temperature)
        if not mask[index] or reference["tokens"] != row["tokens"] \
                or (shipping, plain) != (reference["p1m_T"], reference["p1m"]):
            raise ValueError(f"The merged reference differs from the fixture: {row['id']}")
        logits = weight @ hidden[index] + bias
        difference = max(abs(float(value) - expected) for value, expected in zip(logits, reference["logits1m"]))
        print(f"{row['id']}: float32 head logits differ from logits1m by at most {difference:.3g}")
        if difference > 1e-4:
            raise ValueError(f"The hidden state does not reproduce logits1m: {row['id']}")
        result.append({"id": row["id"], "kind": row["kind"], "options": row["options"], "tokens": row["tokens"],
                       "logits1m": reference["logits1m"], "p1m": reference["p1m"], "p1m_T": reference["p1m_T"],
                       "hidden": [float(value) for value in hidden[index]]})
    return result


def inference_rows(rows, quantized):
    found = {row["id"]: row for row in quantized["rows"] if row["set"] == "F"}
    result = []
    for row in rows:
        reference = found.get(row["id"])
        if not reference or reference["rejected"] or reference["tokens"] != row["tokens"] \
                or reference["text_sha256"] != hashlib.sha256(row["text"].encode("utf-8")).hexdigest():
            raise ValueError(f"The quantized reference differs from the fixture: {row['id']}")
        result.append({**{field: row[field] for field in FIELDS}, "token_count": row["token_count"],
                       "tokens_sha256": tokens_sha256(row["tokens"]), "band": reference["band"],
                       **{key: reference[key] for key in ("logits2", "p2", "p2_T", "p1m")}})
    return result


def main():
    parser = argparse.ArgumentParser(description="Write the JSON fixtures of the Swift typed-decision tests.")
    for name in ("snapshot", "corpus", "jglue", "fixtures", "reference", "merged-reference", "hidden",
                 "quantized-reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    pins = read_json(args.fixtures / "jglue.json")
    tokenizer = distributed_tokenizer(args.snapshot)
    restored = load_rows(args.fixtures / "F.json", args.corpus, args.jglue)
    rows = [rendered(row, tokenizer) for row in synthetic_rows() + japanese_rows(restored)]
    references = {row["id"]: row for row in read_json(args.reference)["rows"]}
    for row in rows:
        reference = references.get(row["id"])
        if row["id"].startswith("synthetic:"):
            continue
        if not reference or reference["tokens"] != row["tokens"] \
                or reference["text_sha256"] != hashlib.sha256(row["text"].encode("utf-8")).hexdigest():
            raise ValueError(f"The reference differs from the fixture: {row['id']}")
    merged = read_json(args.merged_reference)
    quantized = read_json(args.quantized_reference)
    if quantized["merged_revision"] != merged["merged_revision"]:
        raise ValueError("The quantized reference uses another merged checkpoint")
    chosen = [row for row in rows if row["id"] in CARD_EXAMPLES]
    chosen += [next(row for row in rows if row["id"].startswith(f"jglue:{task}:")) for task in REAL_HEAD_TASKS]
    temperatures = read_json(args.snapshot / "calibration.json")["per_kind"]
    head_path = args.snapshot / "head.safetensors"
    documents = {
        "render": {"__license__": license_note(pins), "template": "bare-v1",
                   "tokenizer_sha256": sha256(args.snapshot / "tokenizer.json"), "rows": rows},
        "reject": {"template": "bare-v1", "rows": rejected_rows()},
        "head": {"width": WIDTH, "weight": WEIGHT, "bias": BIAS, "temperatures": temperatures,
                 "rows": head_cases(temperatures)},
        "real-head": {"__license__": license_note(pins), "head_sha256": sha256(head_path),
                      "merged_revision": merged["merged_revision"], "temperatures": merged["temperatures"],
                      "rows": real_head_rows(chosen, merged, args.hidden, head_path)},
        "inference": {"__license__": license_note(pins), "merged_revision": quantized["merged_revision"],
                      "temperatures": quantized["temperatures"],
                      "rows": inference_rows([row for row in rows if not row["id"].startswith("synthetic:")],
                                             quantized)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (args.output / f"{name}.json").write_text(serialize(document), encoding="utf-8")
    print(json.dumps({"render": len(rows), "japanese": sum(row["id"].startswith("jglue:") for row in rows),
                      "rejected": len(REJECTED), "head": len(HEAD_CASES), "real_head": len(chosen),
                      "inference": len(documents["inference"]["rows"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
