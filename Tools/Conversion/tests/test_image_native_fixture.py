import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Tools.Conversion.adapters.qwen_image21.contracts import PROFILE, REVISION
from Tools.Conversion.adapters.qwen_image21.fixtures import DEFINITION_SHA256
from Tools.Conversion.adapters.qwen_image21.native_fixture import run


class NativeFixtureTests(unittest.TestCase):
    def test_subset_source_and_snapshot_changes_fail_before_writing_goldens(self):
        for mutation, message in (
            ("subset", "All sixty suite runs"),
            ("reference", "different source or execution profile"),
            ("snapshot", "Unexpected source snapshot"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixtures, reference = root / "fixtures", root / "reference"
                fixtures.mkdir()
                reference.mkdir()
                profile = copy.deepcopy(PROFILE)
                if mutation == "reference":
                    profile["revision"] = "0" * 40
                (reference / "reference.json").write_text(json.dumps({
                    "complete": True, "precision": "fp32", "cache": "off", "target_profile": profile,
                }))
                (fixtures / "fixtures.json").write_text(json.dumps({
                    "format": 1, "id": "qmi-quality-30x2-v1", "source_revision": REVISION,
                    "definition_sha256": DEFINITION_SHA256,
                    "runs": [{"fixture_id": name} for name in ("g02", "e06")],
                }))
                snapshot = root / ("0" * 40 if mutation == "snapshot" else REVISION)
                output = root / "output"
                with self.assertRaisesRegex(ValueError, message):
                    run(SimpleNamespace(snapshot=snapshot, reference=reference, fixtures=fixtures, output=output))
                self.assertFalse(output.exists())
