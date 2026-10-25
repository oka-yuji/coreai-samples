import ctypes
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from Tools.Conversion.local import default_lock

from .score import blocks, load_set

ROOT = Path(__file__).resolve().parents[2]
MODEL = "XingChen-AGI/TeleOCR"
REVISION = "70a19ec6a3b036311af19ae5e762e650f41a5980"
SNAPSHOT_SHA256 = {
    "model.safetensors": "9817b18041bd96403f75e38a33b28ed0cd5fb2641f67eead673afabd3c408109",
    "modeling_naviocr.py": "7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922",
    "config.json": "92c5e15f1768dcb54bceae170043fbfefd248508159700b2179824b88c33ae75",
    "generation_config.json": "50867494bddf9ddc8e289e6214be5a2263e68f16c127a3652c6eb9f17990e5d9",
    "preprocessor_config.json": "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0",
    "chat_template.jinja": "a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb",
    "tokenizer.json": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
}
PROMPTS = {
    "text": "Please output the text content from the image.",
    "table": "This is the image of a table. Please output the table in OTSL format.",
    "layout": "Analyze the image layout.",
}
LAYOUT_SIDE = 1036
MAX_TIME_S = 600.0
MODEL_PROCESSES = "coreai-chat|coreai-verify|llm-runner|coreai-build|CoreAISamples|embedding-verify|irodori-probe"


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def snapshot():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))


def footprint():
    buf = ctypes.create_string_buffer(16 + 8 * 40)
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if libc.proc_pid_rusage(os.getpid(), 4, buf) != 0:
        return None, None
    vals = memoryview(buf.raw)[16:].cast("Q")
    return vals[7], vals[28]


def image_path(set_dir, gt):
    names = [gt["image"]] if "image" in gt else [f"{gt['id']}.png", f"{gt['id']}.jpg"]
    return next(set_dir / n for n in names if (set_dir / n).exists())


def run_one(set_dir, out, sid, prompt_id, crop, resize, name, device="mps", dtype="bf16"):
    started_at = now()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    gt = json.loads((set_dir / f"{sid}.json").read_text())
    path = image_path(set_dir, gt)
    image = Image.open(path).convert("RGB")
    if crop:
        image = image.crop(tuple(crop))
    if resize:
        image = image.resize((resize, resize), Image.Resampling.BICUBIC)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    source = snapshot()
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(source), trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(str(source), trust_remote_code=True, torch_dtype=torch_dtype).to(device).eval()
    if device == "mps":
        torch.mps.synchronize()
    load_s = time.perf_counter() - t0
    if model.generation_config.repetition_penalty != 1.05:
        raise RuntimeError("generation_config.repetition_penalty must be 1.05")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[prompt_id]}]},
    ]
    chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_prompt], images=[image], padding=True, return_tensors="pt").to(device=model.device, dtype=model.dtype)
    t1 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, use_cache=True, max_new_tokens=4096, do_sample=False, max_time=MAX_TIME_S)
    if device == "mps":
        torch.mps.synchronize()
    generate_s = time.perf_counter() - t1
    prompt_len = inputs.input_ids.shape[1]
    gen = output_ids.cpu().tolist()[0][prompt_len:]
    text = processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    raw = processor.batch_decode([gen], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    finished_by = ("eos" if gen and gen[-1] in (151645, 151643) else "max_new_tokens" if len(gen) >= 4096
                   else "max_time" if generate_s >= MAX_TIME_S else "other")
    outputs = out / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / f"{name}.txt").write_text(text + "\n")
    (outputs / f"{name}.raw.txt").write_text(raw)
    t, h, w = inputs.image_grid_thw[0].tolist()
    phys, phys_max = footprint()
    rec = {
        "fixture": sid, "prompt_id": prompt_id, "prompt": PROMPTS[prompt_id], "device": device, "dtype": dtype,
        "source_image": path.name, "crop": crop, "resize": resize,
        "image_px": list(image.size), "processor_px": [w * 14, h * 14], "image_grid_thw": [t, h, w],
        "image_tokens": t * (h // 2) * (w // 2), "prompt_tokens": prompt_len,
        "output_tokens": len(gen), "finished_by": finished_by,
        "load_s": round(load_s, 3), "generate_s": round(generate_s, 3),
        "tok_s": round(len(gen) / generate_s, 3) if generate_s > 0 else None,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "phys_footprint_bytes": phys, "lifetime_max_phys_footprint_bytes": phys_max,
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if device == "mps" else None,
        "attn_implementation": model.config._attn_implementation,
        "generation": {"do_sample": False, "max_new_tokens": 4096, "use_cache": True, "max_time_s": MAX_TIME_S,
                       "repetition_penalty_from_generation_config": model.generation_config.repetition_penalty},
        "output_path": f"outputs/{name}.txt", "started_at": started_at,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("fixture", "prompt_id", "image_tokens", "output_tokens", "finished_by", "generate_s")}))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def environment():
    source = snapshot()
    for name, expected in SNAPSHOT_SHA256.items():
        if sha256(source / name) != expected:
            raise RuntimeError(f"snapshot file {name} differs from the pinned SHA-256")
    return {
        "os": {"version": command("sw_vers", "-productVersion"), "build": command("sw_vers", "-buildVersion")},
        "machine": {"model": command("sysctl", "-n", "hw.model"), "cpu": command("sysctl", "-n", "machdep.cpu.brand_string"),
                    "memory_bytes": int(command("sysctl", "-n", "hw.memsize"))},
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ("torch", "torchvision", "transformers", "tokenizers", "pillow", "numpy")},
        "model": {"repo": MODEL, "revision": REVISION, "sha256": SNAPSHOT_SHA256},
        "repository": {"commit": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
                       "tracked_changes": bool(command("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"))},
    }


def model_processes():
    found = subprocess.run(["pgrep", "-x", "-l", MODEL_PROCESSES], capture_output=True, text=True)
    if found.returncode not in (0, 1):
        raise RuntimeError(found.stderr)
    return found.stdout.split("\n") if found.stdout.strip() else []


def acquire(lock, describe, wait_s, log):
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_s
    while True:
        if not model_processes():
            try:
                lock.mkdir()
                owner = describe()
                (lock / "owner.txt").write_text(owner)
                return owner
            except FileExistsError:
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError("the shared model lock stayed busy; no model was loaded")
        log("waiting for the shared model lock")
        time.sleep(60)


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


def child(set_dir, out, logs, sid, prompt_id, name, crop=None, resize=None):
    args = [sys.executable, "-m", "Tools.OCREval", "run-reference", "--one", "--set", str(set_dir), "--out", str(out),
            "--sample", sid, "--prompt", prompt_id, "--name", name]
    if crop:
        args += ["--crop", ",".join(map(str, crop))]
    if resize:
        args += ["--resize", str(resize)]
    with open(logs / f"{name}.log", "w") as log:
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=os.environ | {"HF_HUB_OFFLINE": "1"})
        try:
            code = process.wait(timeout=MAX_TIME_S + 300)
        except BaseException:
            stop(process)
            raise
    if code != 0:
        raise RuntimeError(f"{name} exited with {code}")


def run_sample(set_dir, out, logs, gt, dtype="bf16"):
    sid = gt["id"]
    for pipeline in gt["pipelines"]:
        if pipeline == "text":
            child(set_dir, out, logs, sid, "text", f"{sid}-text-{dtype}", crop=gt.get("crop_px"))
        elif pipeline == "table":
            child(set_dir, out, logs, sid, "table", f"{sid}-table-{dtype}", crop=gt.get("table_bbox") or gt["crop_px"])
        else:
            child(set_dir, out, logs, sid, "layout", f"{sid}-layout-{dtype}", resize=LAYOUT_SIDE)
            layout = (out / "outputs" / f"{sid}-layout-{dtype}.txt").read_text()
            for i, b in enumerate(blocks(layout, *gt["size_px"]), 1):
                child(set_dir, out, logs, sid, "text", f"{sid}-block{i:02d}-text-{dtype}", crop=b["crop_px"])


def run_plan(set_dir, out, logs, only, job, session, wait_minutes):
    gts = load_set(set_dir)
    ids = only or list(gts)
    missing = [s for s in ids if s not in gts]
    if missing:
        raise ValueError(f"unknown samples: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    def log(message):
        print(f"{now()} {message}", flush=True)

    env = environment()
    lock = default_lock()
    def describe():
        return (f"session: {session}\njob: {job}\nstarted: {now()}\npid: {os.getpid()}\n"
                f"expected: {len(ids)} samples, one process per run\n")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    waited = time.monotonic()
    owner = acquire(lock, describe, wait_minutes * 60, log)
    held, done, failed = time.monotonic(), [], []
    log(f"lock acquired after {held - waited:.0f}s")
    try:
        for sid in ids:
            try:
                run_sample(set_dir, out, logs, gts[sid])
                done.append(sid)
            except RuntimeError as error:
                failed.append({"sample": sid, "error": str(error)})
            log(f"{sid} done ({len(done)} ok, {len(failed)} failed)")
    finally:
        if (lock / "owner.txt").read_text() != owner:
            raise RuntimeError("lock owner changed; the lock was left in place")
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        lock.rename(lock.with_name(f"heavy-job-done-{job}-{stamp}"))
        record = env | {"job": job, "samples": ids, "failed": failed, "lock_wait_s": round(held - waited),
                        "lock_held_s": round(time.monotonic() - held), "finished_at": now()}
        with open(out / "invocations.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"lock released after {record['lock_held_s']}s")
    return failed
