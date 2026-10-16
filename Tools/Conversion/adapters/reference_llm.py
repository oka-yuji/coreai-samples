import argparse
import json
from pathlib import Path
import re
import time

from pinned import load_contract


def encode_case(tokenizer, case):
    if "messages" in case:
        encoded = tokenizer.apply_chat_template(case["messages"], add_generation_prompt=True,
                                                enable_thinking=False, return_dict=True)
    else:
        encoded = tokenizer(case["text"], add_special_tokens=False)
    ids = encoded["input_ids"]
    if not isinstance(ids, list) or any(type(token) is not int for token in ids):
        raise ValueError("Tokenizer must return a flat input_ids list")
    return list(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract, snapshot = load_contract(args.contract, args.snapshot)
    cases = json.loads(args.fixture.read_text())
    if not isinstance(cases, list) or not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Fixtures require unique case IDs")
    for case in cases:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", case["id"]):
            raise ValueError("Invalid case ID")
        if type(case.get("generate", 0)) is not int or not 0 <= case.get("generate", 0) <= 32:
            raise ValueError("Fixture generation must be bounded to 0..32")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for case in cases:
            started = time.perf_counter()
            ids = encode_case(tokenizer, case)
            if len(ids) < 2 or len(ids) + case.get("generate", 0) > contract["context_length"]:
                raise ValueError("Fixture length is outside the exported context")
            prefix_length = len(ids)
            for _ in range(case.get("generate", 0)):
                logits = model(torch.tensor([ids]), use_cache=False).logits[0, -1]
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite reference generation")
                token = int(logits.argmax())
                ids.append(token)
                if token in eos:
                    break
            logits = model(torch.tensor([ids]), use_cache=False).logits[0, :-1].float().numpy()
            if not np.isfinite(logits).all():
                raise ValueError("Non-finite reference logits")
            name = case["id"]
            np.save(args.out / f"{name}.npy", logits, allow_pickle=False)
            (args.out / f"{name}.tokens.json").write_text(json.dumps({"tokens": ids}))
            rows.append({"id": name, "tokens": ids, "prefix_length": prefix_length,
                         "generated_eos": bool(len(ids) > prefix_length and ids[-1] in eos),
                         "shape": list(logits.shape), "seconds": time.perf_counter() - started})
    (args.out / "reference.json").write_text(json.dumps(
        {"contract": contract, "compute": "float32", "device": "cpu", "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
