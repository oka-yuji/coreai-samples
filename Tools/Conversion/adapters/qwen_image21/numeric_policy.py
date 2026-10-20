import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import tensor_metrics
from .evaluation import canonical_hash, require


def numeric_policy():
    path = Path(__file__).resolve().parents[2] / "recipes/qwen_image21/numeric-gates-v1.json"
    value = json.loads(path.read_text())
    require(value.get("format") == 1 and value.get("id") == "qmi-numeric-gates-v1", "Unsupported numeric policy")
    return value


def numeric_metrics(reference, actual, *, compute_dtype, boundary):
    policy = numeric_policy()
    require(compute_dtype in policy["component"], "An explicit supported compute dtype is required")
    require(boundary in ("component", "fixed-step-target", "final-latent", "final-rgba"), "Unknown numeric boundary")
    left, right = np.asarray(reference), np.asarray(actual)
    require(left.dtype == right.dtype == np.float32, "Comparison captures must be Float32")
    require(left.shape == right.shape and left.size > 0, "Comparison shapes differ or are empty")
    if boundary in ("fixed-step-target", "final-latent"):
        require(left.shape == (1, 1024, 64), "Expected exactly the target latent region")
    elif boundary == "final-rgba":
        require(left.shape == (1, 4, 1, 512, 512), "Expected the complete unclipped RGBA decoder output")
    if boundary in ("component", "fixed-step-target"):
        limits = policy["component"][compute_dtype]
        result = tensor_metrics(left, right, max_limit=limits["normalized_max"], rmse_limit=limits["normalized_rmse"])
    elif boundary == "final-latent":
        result = tensor_metrics(left, right, max_limit=np.inf, rmse_limit=policy["rollout"]["latent_normalized_rmse"])
    else:
        result = tensor_metrics(left, right, max_limit=np.inf, rmse_limit=np.inf)
        result["passed"] = result["finite"] and result["absolute_rmse"] <= policy["rollout"]["rgba_absolute_rmse"]
    source = Path(__file__).parent
    code = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("numeric_policy.py", "contracts.py", "evaluation.py")}
    return {"policy_id": policy["id"], "policy_sha256": canonical_hash(policy),
            "metric_code_sha256": code, "compute_dtype": compute_dtype, "boundary": boundary,
            "metrics": result, "scope": "One already-selected valid tensor region; not suite or profile verification"}
