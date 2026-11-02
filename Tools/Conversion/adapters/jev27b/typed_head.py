import math

LABELS = "ABCDEFGHIJKLMNOP"
SLOTS = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}
FIXED_OPTIONS = {"noul": ("false", "true"), "score": ("0", "1", "2", "3", "4", "5")}
BANDS = (("<=512", 512), ("513-2048", 2048), ("2049-4096", 4096), ("4097-8192", 8192))
LINE_BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85  "


def option_lines(kind, options):
    if not isinstance(options, (list, tuple)):
        raise ValueError("options must be a list")
    if kind in FIXED_OPTIONS:
        if tuple(options) != FIXED_OPTIONS[kind]:
            raise ValueError(f"{kind} accepts only {list(FIXED_OPTIONS[kind])}")
        return list(options)
    if kind != "choice":
        raise ValueError(f"Unknown kind: {kind!r}")
    if not 2 <= len(options) <= len(LABELS):
        raise ValueError("choice needs 2-16 options")
    if any(not isinstance(option, str) or any(c in LINE_BREAKS for c in option) for option in options):
        raise ValueError("choice options must be single-line strings")
    if len(set(options)) != len(options):
        raise ValueError("choice options must be distinct")
    return [f"{LABELS[i]}) {option}" for i, option in enumerate(options)]


def render(kind, state, question, options):
    lines = option_lines(kind, options)
    if not isinstance(state, str) or not isinstance(question, str):
        raise ValueError("state and question must be strings")
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def slot_range(kind, n):
    if kind not in SLOTS:
        raise ValueError(f"Unknown kind: {kind!r}")
    start, end = SLOTS[kind]
    if type(n) is not int or not (2 <= n <= end - start if kind == "choice" else n == end - start):
        raise ValueError(f"Invalid option count for {kind}: {n!r}")
    return start, start + n


def softmax(values):
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def decide(logits24, kind, n, temperature):
    values = [float(value) for value in logits24]
    if len(values) != SLOTS["choice"][1]:
        raise ValueError("Expected 24 head logits")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Head logits must be finite")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive finite number")
    start, end = slot_range(kind, n)
    selected = values[start:end]
    scaled = [value / temperature for value in selected]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("Head logits divided by the temperature must be finite")
    return softmax(scaled), softmax(selected)


def band(token_count):
    if type(token_count) is not int or token_count < 0:
        raise ValueError("Token count must be a non-negative integer")
    for label, limit in BANDS:
        if token_count <= limit:
            return label
    return None


def tokenize(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def last_hidden(model, ids):
    import torch

    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    with torch.no_grad():
        output = model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    return output.last_hidden_state[0, -1].float().cpu()
