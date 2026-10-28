import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import diffusers
    import transformers.vision_utils
except ImportError as error:
    raise unittest.SkipTest(f"Qwen image dependencies unavailable: {error}") from error

from Tools.Conversion.adapters.qwen_image21.fixtures import file_record
from Tools.Conversion.adapters.qwen_image21.mobile_aot_pack import pack
from Tools.Conversion.adapters.qwen_image21.mobile_artifacts import inventory
from Tools.Conversion.adapters.qwen_image21.mobile_profile import make_profile


class ImageMobileAOTPackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.make_fixture(Path(self.temporary.name))
        lock = patch("Tools.Conversion.adapters.qwen_image21.mobile_aot_pack.require_heavy_job")
        lock.start()
        self.addCleanup(lock.stop)

    def make_fixture(self, root):
        self.bundle, self.compiled = root / "raw", root / "aot"
        self.bundle.mkdir()
        self.compiled.mkdir()
        profile = make_profile("w4-dit-g16-blocks16-23-f32", "a" * 64, {"fixture.py": "b" * 64})
        for kind, spec in profile["components"].items():
            path = self.bundle / spec["asset"] / "main.mlirb"
            path.parent.mkdir(parents=True)
            path.write_bytes((kind + " model fixture").encode())
        manifest = {"format": 2, "profile": profile, "assets": inventory(self.bundle),
                    "verified": True, "numeric_verified": True, "quality_verified": True}
        (self.bundle / "bundle.json").write_text(json.dumps(manifest))
        self.original = (self.bundle / "bundle.json").read_bytes()
        source_hash = file_record(self.bundle / profile["components"]["denoiser"]["asset"] / "main.mlirb")["sha256"]
        (self.compiled / "metadata.json").write_text(json.dumps({"sourceHash": source_hash.upper()}))
        (self.compiled / "resources.bin").write_bytes(b"compiled model fixture")
        self.args = SimpleNamespace(bundle=self.bundle, manifest_sha256=file_record(self.bundle / "bundle.json")["sha256"],
                                    compiled=[("denoiser", self.compiled)], architecture="h18p", output=root / "result")

    def test_pack_preserves_raw_files_and_binds_compiled_inventory_without_device_acceptance(self):
        pack(self.args)
        result = self.args.output / "bundle"
        manifest = json.loads((result / "bundle.json").read_text())
        receipt = json.loads((self.args.output / "receipt.json").read_text())
        self.assertEqual((self.bundle / "bundle.json").read_bytes(), self.original)
        self.assertEqual(inventory(result), {**manifest["assets"], "bundle.json": receipt["manifest"]})
        self.assertEqual(manifest["compiled_assets"]["denoiser"]["architecture"], "h18p")
        self.assertEqual(manifest["compiled_verification"], "pending")
        self.assertEqual(manifest["device_verification"], "pending")
        self.assertFalse(receipt["device_verified"])
        self.assertFalse((result / ".packing").exists())
        with self.assertRaises(FileExistsError):
            pack(self.args)

    def test_bad_pins_sources_links_architectures_and_overlaps_are_rejected(self):
        for defect in ["pin", "source", "raw", "link", "architecture", "overlap", "duplicate", "unknown"]:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                self.make_fixture(Path(folder))
                if defect == "pin":
                    self.args.manifest_sha256 = "0" * 64
                elif defect == "source":
                    (self.compiled / "metadata.json").write_text(json.dumps({"sourceHash": "0" * 64}))
                elif defect == "raw":
                    (self.bundle / "denoiser/denoiser.aimodel/main.mlirb").write_bytes(b"changed")
                elif defect == "link":
                    (self.compiled / "linked").symlink_to(self.compiled / "resources.bin")
                elif defect == "architecture":
                    self.args.architecture = "../h18p"
                elif defect == "overlap":
                    self.args.output = self.bundle / "result"
                elif defect == "duplicate":
                    self.args.compiled.append(("denoiser", self.compiled))
                else:
                    self.args.compiled = [("unknown", self.compiled)]
                with self.assertRaises((ValueError, FileExistsError)):
                    pack(self.args)
                self.assertFalse((self.args.output / "receipt.json").exists())

    def test_the_parsed_manifest_and_metadata_must_match_the_inspected_bytes(self):
        original_read = Path.read_bytes
        for selected in [self.bundle / "bundle.json", self.compiled / "metadata.json"]:
            def changed_read(path):
                data = original_read(path)
                return data + b"\n" if path == selected else data
            with self.subTest(selected=selected), patch.object(Path, "read_bytes", changed_read):
                with self.assertRaises(ValueError):
                    pack(self.args)
            self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
