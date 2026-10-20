import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from Tools.Conversion.adapters.qwen3_tts.inputs import EXECUTION_SHAPES
from Tools.Conversion.adapters.qwen3_tts.runtime import CoreAIComponents, validate_inputs
from Tools.Conversion.adapters.qwen3_tts.source import validate_source
from Tools.Conversion.storage import file_hash


def configuration(size, width):
    return {"model_type": "qwen3_tts", "tts_model_type": "custom_voice", "tts_model_size": size,
            "talker_config": {"hidden_size": width, "text_hidden_size": 2048,
                              "code_predictor_config": {"hidden_size": 1024}}}


class QwenTTSProfileTests(unittest.TestCase):
    def test_both_fixed_profiles_require_the_matching_source_architecture(self):
        for label, size, width in [("0.6B", "0b6", 1024), ("1.7B", "1b7", 2048)]:
            for actual_size, actual_width, valid in [(size, width, True), ("base", width, False), (size, width * 2, False)]:
                with self.subTest(label=label, actual_size=actual_size, actual_width=actual_width), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / ("a" * 40)
                    root.mkdir()
                    config = root / "config.json"
                    config.write_text(json.dumps(configuration(actual_size, actual_width)))
                    contract = {"model_id": f"Qwen/Qwen3-TTS-12Hz-{label}-CustomVoice", "revision": root.name,
                                "files": {"config.json": file_hash(config)}}
                    if valid:
                        self.assertEqual(validate_source(root, contract), root.resolve())
                    else:
                        with self.assertRaises(ValueError):
                            validate_source(root, contract)

    def test_unknown_model_is_rejected_even_with_valid_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ("a" * 40)
            root.mkdir()
            config = root / "config.json"
            config.write_text(json.dumps(configuration("0b6", 1024)))
            contract = {"model_id": "unrecognized/model", "revision": root.name, "files": {"config.json": file_hash(config)}}
            with self.assertRaises(ValueError):
                validate_source(root, contract)

    def test_runtime_checks_raw_predictor_width_before_its_projection(self):
        for width in (1024, 2048):
            for kind in ("talker", "predictor"):
                with self.subTest(width=width, kind=kind):
                    extra = np.zeros((3, 1, 2), np.int32) if kind == "talker" else np.array([0], np.int32)
                    validate_inputs(kind, (np.zeros((1, 2, width), np.float32), extra), talker_width=width)
                    with self.assertRaises(ValueError):
                        validate_inputs(kind, (np.zeros((1, 2, 3072 - width), np.float32), extra), talker_width=width)

    def test_other_size_export_is_rejected_before_model_load(self):
        expected = {"model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "execution_shapes": EXECUTION_SHAPES}
        other = {**expected, "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "export.json").write_text(json.dumps({"complete": True, "kind": "predictor", "contract": other}))
            with patch.object(CoreAIComponents, "_load", new_callable=AsyncMock) as load:
                with self.assertRaisesRegex(ValueError, "contracts do not match"):
                    CoreAIComponents([root], expected)
                load.assert_not_called()
