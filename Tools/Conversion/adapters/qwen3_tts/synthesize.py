import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import soundfile as sf
import torch

from ...storage import write_json
from .runtime import CoreAIComponents
from .source import add_source_arguments, array, generation_settings, model_from_arguments
from .synthesis import CoreAIBridge


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    specification = json.loads(args.fixtures.read_text())
    wrapper, contract = model_from_arguments(args)
    report = {"complete": False, "passed": False, "contract": contract,
              "specification": specification, "utterances": []}
    write_json(args.output / "generation.json", report)
    components = CoreAIComponents(args.export, contract)
    try:
        bridge = CoreAIBridge(wrapper, components)
        with ExitStack() as stack, torch.inference_mode():
            bridge.install(stack)
            for fixture in specification["fixtures"]:
                torch.manual_seed(specification["seed"])
                started = time.perf_counter()
                waves, sample_rate = wrapper.generate_custom_voice(text=fixture["text"], **generation_settings(specification))
                seconds = time.perf_counter() - started
                audio = np.asarray(waves[0])
                if sample_rate != contract["sample_rate"] or not audio.size or not np.isfinite(audio).all():
                    raise ValueError("Core AI synthesis produced invalid audio")
                sf.write(args.output / f"{fixture['id']}.wav", audio, sample_rate, subtype="FLOAT")
                np.save(args.output / f"{fixture['id']}-sequence.npy", array(bridge.last_sequence), allow_pickle=False)
                eos = wrapper.model.config.talker_config.codec_eos_token_id
                row = {"id": fixture["id"], "seconds": seconds, "audio_seconds": len(audio) / sample_rate,
                       "finite": True, "output": f"{fixture['id']}.wav", "tokens": bridge.last_sequence.shape[1],
                       "stop_reason": "eos" if int(bridge.last_sequence[0, -1]) == eos else "token_limit",
                       "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
                report["utterances"].append(row)
                write_json(args.output / "generation.json", report)
                print(json.dumps(row), flush=True)
        report["complete"] = True
        report["passed"] = bool(report["utterances"])
        report["components"] = components.metrics
        write_json(args.output / "generation.json", report)
    finally:
        components.close()


if __name__ == "__main__":
    main()
