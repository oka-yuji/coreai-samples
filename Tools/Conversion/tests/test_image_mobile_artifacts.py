import json
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.qwen_image21.evaluation import canonical_hash
from Tools.Conversion.adapters.qwen_image21.fixtures import file_record, save_json
from Tools.Conversion.adapters.qwen_image21.mobile_artifacts import inventory, read_component
from Tools.Conversion.adapters.qwen_image21.mobile_convert import new_output, write_new_json
from Tools.Conversion.adapters.qwen_image21.mobile_profile import make_profile


class ImageMobileArtifactTests(unittest.TestCase):
    def test_existing_outputs_and_dangling_links_are_preserved(self):
        for defect in ("existing-file", "dangling-link", "existing-directory", "root-link"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                parent = Path(folder)
                root = parent / "out"
                root.mkdir()
                target = root / "compression.json"
                if defect == "existing-file":
                    target.write_text("original")
                elif defect == "dangling-link":
                    target.symlink_to(parent / "outside.json")
                elif defect == "existing-directory":
                    target.mkdir()
                else:
                    link = parent / "linked-output"
                    link.symlink_to(root, target_is_directory=True)
                    root = link
                with self.assertRaises(FileExistsError):
                    new_output(root, ("compression.json",))
                self.assertFalse((parent / "outside.json").exists())
                if defect == "existing-file":
                    self.assertEqual(target.read_text(), "original")

    def test_json_creation_rejects_destinations_added_after_the_initial_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "profile.json"
            new_output(root, (target.name,))
            target.symlink_to(root / "outside.json")
            with self.assertRaises(FileExistsError):
                write_new_json(target, {"value": 1})
            self.assertFalse((root / "outside.json").exists())

    def fixture(self, root):
        profile = make_profile("w4", "a" * 64, {"conversion.py": "b" * 64})
        asset = root / "text.aimodel"
        asset.mkdir()
        (asset / "weights.bin").write_bytes(b"fixture weights")
        compression = {"format": 1, "kind": "text", "profile_sha256": canonical_hash(profile),
                       "compression": {"kind": "linear-integer", "layers": []}}
        save_json(root / "compression.json", compression)
        record = {"format": 2, "kind": "text", "profile": profile, "profile_sha256": canonical_hash(profile),
                  "conversion_complete": True, "asset": "text.aimodel", "inputs": profile["components"]["text"]["inputs"],
                  "outputs": profile["components"]["text"]["outputs"], "compression_file": file_record(root / "compression.json"),
                  "asset_inventory": inventory(asset), "numeric_verification": "pending"}
        save_json(root / "component.json", record)
        return profile

    def test_converted_component_remains_pending_numeric_and_device_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile = self.fixture(root)
            record, _ = read_component(root, profile, "text")
            self.assertEqual(record["numeric_verification"], "pending")
            self.assertNotIn("verified", record)

    def test_different_profiles_and_modified_reference_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile = self.fixture(root)
            other = make_profile("w8", "a" * 64, {"conversion.py": "b" * 64})
            with self.assertRaises(ValueError):
                read_component(root, other, "text")
            path = root / "compression.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                read_component(root, profile, "text")

    def test_missing_extra_corrupted_and_linked_asset_entries_are_rejected(self):
        for defect in ("missing", "extra", "corrupt", "link"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                profile = self.fixture(root)
                asset = root / "text.aimodel"
                if defect == "missing":
                    (asset / "weights.bin").unlink()
                elif defect == "extra":
                    (asset / "extra.bin").write_bytes(b"extra")
                elif defect == "corrupt":
                    (asset / "weights.bin").write_bytes(b"changed weights")
                else:
                    (asset / "link.bin").symlink_to(asset / "weights.bin")
                with self.assertRaises(ValueError):
                    read_component(root, profile, "text")

    def test_profile_numeric_type_changes_do_not_reuse_an_old_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile = self.fixture(root)
            path = root / "component.json"
            record = json.loads(path.read_text())
            record["profile"]["components"]["text"]["weight_encoding"]["bits"] = 4.0
            save_json(path, record)
            with self.assertRaises(ValueError):
                read_component(root, profile, "text")


if __name__ == "__main__":
    unittest.main()
