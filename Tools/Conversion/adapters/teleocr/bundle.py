import argparse
import shutil
from pathlib import Path

from .contracts import (
    FUNCTIONS, IOS_CONTEXT, IOS_DOWNSCALE_SIDE, IOS_FUNCTIONS, IOS_MASKED, IOS_PREFILL, IOS_WINDOWS, MODEL_ID,
    PALETTE_BITS, PALETTE_GROUP, PROMPTS, REVISION, STOP_TOKENS, SYSTEM, TOKEN_LIMIT, UNIT_PATCHES, WINDOW_LIMIT,
    WINDOW_PATCHES, WINDOW_PIXELS, digest, read_json, write_json,
)

TOKENIZER = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
             "added_tokens.json", "special_tokens_map.json"]
TEMPLATE = ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "{prompt}<|im_end|>\n<|im_start|>assistant\n")
NOTICE = """TeleOCR Core AI bundle

This bundle contains model weights converted from {model}
(revision {revision}), distributed by XingChen-AGI under the Apache License 2.0.
The full license text is in LICENSE.

Changes from the original:
- The vision encoder and merger were rewritten as a dynamic window-batched graph that takes
  padded windows, a key validity mask and rotary tables prepared by the host, and were converted
  to Core AI with {vision_precision}.
- The token embedding table was exported unchanged (bfloat16) with a float32 output.
- The text decoder was exported as prefill and main functions with float16 key/value states.
- RMS normalization is computed in an equivalent rescaled form to stay within the float16 range.

Converted by Yuji Oka with the coreai-samples conversion recipe. This is not an official
release of the original authors.
"""
IOS_CHANGES = """
iOS static-shape variant:
- The vision graph is specialized to fixed window buckets and masks hidden keys with -40000.
- The token embedding takes a fixed 2048-token input.
- The decoder is specialized to fixed prefill lengths and a single-token decode function that
  share a fixed 4096-token float16 key/value state.
"""
PALETTE_CHANGES = """- The decoder's linear weights were compressed with 4-bit k-means palettization (group size 8).
  The token embedding, the tied LM head and the norms keep their original values.
"""


def build(args):
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name, source in (("vision.aimodel", args.vision / "vision.aimodel"), ("embed.aimodel", args.embed / "embed.aimodel"),
                         ("main.aimodel", args.decoder / "main.aimodel")):
        shutil.copytree(source, output / name)
    (output / "tokenizer").mkdir()
    for name in TOKENIZER:
        shutil.copyfile(args.snapshot / name, output / "tokenizer" / name)
    shutil.copyfile(args.license, output / "LICENSE")
    ios = args.platform == "ios"
    notice = NOTICE.format(model=MODEL_ID, revision=REVISION, vision_precision="float16 weights and activations")
    if ios:
        notice = notice.replace("\nConverted by", IOS_CHANGES + (PALETTE_CHANGES if args.precision == "pal4" else "")
                                + "\nConverted by")
    (output / "NOTICE").write_text(notice)
    config = read_json(args.snapshot / "config.json")
    text, vision = config["text_config"], config["vision_config"]
    image = read_json(args.snapshot / "preprocessor_config.json")
    generation = read_json(args.snapshot / "generation_config.json")
    tokens = read_json(args.snapshot / "added_tokens.json")
    if generation["eos_token_id"] != STOP_TOKENS or vision["window_size"] != WINDOW_PIXELS:
        raise ValueError("The source configuration differs from the verified contract")
    reference = read_json(args.reference / "reference.json")
    metadata = {
        "format": "coreai-samples-teleocr/1",
        "name": "teleocr",
        "precision": "fp16",
        "source": {"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0"},
        "assets": {"vision": "vision.aimodel", "embed": "embed.aimodel", "main": "main.aimodel"},
        "tokenizer": "tokenizer",
        "image": {"mean": image["image_mean"], "std": image["image_std"], "rescale": 1 / 255,
                  "min_pixels": image["min_pixels"], "max_pixels": image["max_pixels"], "resize_multiple": 28,
                  "resample": "bicubic", "patch_size": image["patch_size"],
                  "temporal_patch_size": image["temporal_patch_size"], "merge_size": image["merge_size"]},
        "vision": {"window_pixels": WINDOW_PIXELS, "window_patches": WINDOW_PATCHES, "unit_patches": UNIT_PATCHES,
                   "fullatt_block_indexes": vision["fullatt_block_indexes"], "depth": vision["depth"],
                   "rotary_theta": 10000.0, "rotary_dim": vision["hidden_size"] // vision["num_heads"] // 2,
                   "window_limit": WINDOW_LIMIT,
                   "layout": "Windows of 4x4 merge units in row-major window order; inside a window, merge units in "
                             "row-major order; edge windows keep empty unit slots in place; windows without a valid "
                             "unit are omitted.",
                   "padding": {"pixel_values": 0.0, "key_valid": 0.0, "cos": 1.0, "sin": 0.0},
                   "output": "One row per merge unit in input order, including padded units; the host removes padded "
                             "rows and restores the source unit order.",
                   "verified": reference["vision"]},
        "text": {"layers": text["num_hidden_layers"], "hidden_size": text["hidden_size"],
                 "heads": text["num_attention_heads"], "kv_heads": text["num_key_value_heads"],
                 "head_dim": text["head_dim"], "vocab_size": text["vocab_size"], "rope_theta": text["rope_theta"],
                 "mrope_section": text["rope_scaling"]["mrope_section"], "token_limit": TOKEN_LIMIT,
                 "kv_write": "prefill and main write the new keys and values at [P, P + S) with P = "
                             "mask.shape[3] - S and attend over [0, P + S). Rotary positions are supplied separately "
                             "through cos and sin, so the write position and the rotary position can differ.",
                 "mask": "Additive float16 mask [1, 1, S, P + S]: 0 for visible keys and -inf for hidden keys."},
        "tokens": {name.strip("<|>"): tokens[name] for name in
                   ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>")},
        "generation": {"greedy": True, "repetition_penalty": generation["repetition_penalty"],
                       "stop_token_ids": STOP_TOKENS, "pad_token_id": generation["pad_token_id"], "max_new_tokens": 4096},
        "prompt": {"system": SYSTEM, "template": TEMPLATE, "prompts": PROMPTS},
        "functions": FUNCTIONS,
    }
    if ios:
        metadata.update(platform="ios", precision=args.precision, functions=IOS_FUNCTIONS, static={
            "context": IOS_CONTEXT, "masked": IOS_MASKED, "vision_windows": list(IOS_WINDOWS),
            "vision_buckets": {str(count): sizes for count, sizes in IOS_WINDOWS.items()},
            "downscale_side": IOS_DOWNSCALE_SIDE, "embed_tokens": IOS_PREFILL[-1], "prefill_tokens": IOS_PREFILL,
            "functions": "vision windows_<windows>, embed main, decoder prefill_<tokens> and main (one token)",
            "padding": "Vision pads whole windows with pixel 0, key_valid 0, cos 1 and sin 0. Embedding pads token "
                       "ids with 0. Prefill pads rows with zero embeddings, cos 1 and sin 0 and hides every key of a "
                       "padding row.",
            "kv_write": "prefill always starts at position 0 and writes [0, S); logit_row selects the last real "
                        "row. main writes the one position marked by write_mask and attends over the whole state "
                        "with mask [1, 1, 1, context]. Zero-fill both states when allocating them.",
            "mask": "Additive float16 masks with 0 for visible keys and the masked value for hidden keys."})
        metadata["vision"]["window_limit"] = max(IOS_WINDOWS)
        metadata["text"]["token_limit"] = IOS_CONTEXT
        metadata["text"].pop("kv_write")
        metadata["text"].pop("mask")
        if args.precision == "pal4":
            metadata["quantization"] = {"method": "k-means palettization", "bits": PALETTE_BITS,
                                        "group_size": PALETTE_GROUP, "axis": 0,
                                        "scope": "decoder linear weights; embedding, tied LM head and norms unchanged"}
    compression = {}
    for kind, directory in (("vision", args.vision), ("decoder", args.decoder)):
        path = directory / "compression.json"
        if path.exists():
            compression[kind] = read_json(path)
    if compression:
        metadata["compression"] = compression
        metadata["precision"] = f"int{compression['decoder']['bits']}" if "decoder" in compression else "fp16"
        metadata["component_precision"] = {"vision": "int8" if "vision" in compression else "fp16",
                                            "decoder": metadata["precision"], "embedding": "bfloat16"}
        if "vision" in compression:
            (output / "NOTICE").write_text(NOTICE.format(
                model=MODEL_ID, revision=REVISION, vision_precision="INT8 Linear weights and float16 activations"))
        with (output / "NOTICE").open("a") as handle:
            for kind, record in compression.items():
                handle.write(f"\n- {kind}: INT{record['bits']} {record['scheme']} {record['granularity']} Linear weights. "
                             "Activations, normalization, embedding, tied LM head and KV are not quantized.\n")
    metadata["files"] = {str(path.relative_to(output)): digest(path)
                         for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser()
    for name in ("snapshot", "reference", "vision", "embed", "decoder", "license", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos")
    parser.add_argument("--precision", choices=["fp16", "pal4"], default="fp16")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
