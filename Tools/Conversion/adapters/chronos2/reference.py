import argparse
import os
from pathlib import Path

from .artifacts import new_output, nullable_context, validate_config, validate_predictions, vector_document, write_new
from .prepare import check_ready


def load_official(snapshot):
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(4)
    pipeline = Chronos2Pipeline.from_pretrained(str(snapshot), device_map="cpu", torch_dtype=torch.float32,
                                               local_files_only=True)
    model = pipeline.model.eval()
    validate_config(model.config.to_dict())
    if any(parameter.dtype != torch.float32 or parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("The reference model must be CPU FP32")
    return model


def context_tensor(context):
    import torch

    return torch.tensor([nullable_context(context)], dtype=torch.float32, device="cpu")


def prediction_values(prediction):
    import torch

    if prediction.dtype != torch.float32 or tuple(prediction.shape) != (1, 21, 16):
        raise ValueError("Expected Float32 predictions with shape [1, 21, 16]")
    result = prediction.detach().cpu().tolist()
    validate_predictions(result)
    return result


def run_reference(model, fixtures):
    import torch

    values = []
    with torch.no_grad():
        for case in fixtures["cases"]:
            prediction = model(context=context_tensor(case["context"]), num_output_patches=1).quantile_preds
            values.append(prediction_values(prediction))
    return values


def reference(model_path, fixture_path, snapshot, ready, out):
    out = new_output(out)
    _, fixtures, identity = check_ready(ready, model_path, fixture_path, snapshot)
    model = load_official(snapshot)
    document = vector_document("reference", identity, fixtures, run_reference(model, fixtures))
    write_new(out / "reference.json", document)
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference(args.model, args.fixtures, args.snapshot, args.ready, args.out)


if __name__ == "__main__":
    main()
