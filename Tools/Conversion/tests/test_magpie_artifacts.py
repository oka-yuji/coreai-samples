import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from Tools.Conversion.adapters.magpie.artifacts import compare_tensor, extract_archive, verify_file
from Tools.Conversion.storage import file_hash


class MagpieArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def archive(self, extras):
        path = self.root / "source.nemo"
        with tarfile.open(path, "w") as archive:
            for name, data in [("./model_config.yaml", b"old"), ("model_weights.ckpt", b"weights"), *extras]:
                entry = tarfile.TarInfo(name)
                if data is None:
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = "../outside"
                    archive.addfile(entry)
                else:
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
        return path

    def test_last_duplicate_configuration_is_used(self):
        source = self.archive([("model_config.yaml", b"new")])
        report = extract_archive(source, self.root / "extracted")
        self.assertEqual((self.root / "extracted/model_config.yaml").read_bytes(), b"new")
        self.assertEqual(report["duplicates"], {"model_config.yaml": 2})

    def test_unsafe_archives_leave_no_output(self):
        for name, data in [("../outside", b"bad"), ("/absolute", b"bad"), ("link", None),
                           ("folder", b"file"), ("back\\slash", b"bad")]:
            with self.subTest(name=name):
                extras = [(name, data)]
                if name == "folder":
                    extras.append(("folder/child", b"bad"))
                source = self.archive(extras)
                with self.assertRaises(ValueError):
                    extract_archive(source, self.root / "output")
                self.assertFalse((self.root / "output").exists())

    def test_existing_output_is_not_replaced(self):
        source = self.archive([])
        output = self.root / "output"
        output.mkdir()
        (output / "existing").write_text("keep")
        with self.assertRaises(FileExistsError):
            extract_archive(source, output)
        self.assertEqual((output / "existing").read_text(), "keep")

    def test_source_identity_rejects_changed_bytes(self):
        source = self.archive([])
        identity = file_hash(source)
        self.assertEqual(verify_file(source, identity), identity)
        source.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            verify_file(source, identity)


class MagpieNumericalTests(unittest.TestCase):
    def test_non_finite_candidate_cannot_pass(self):
        self.assertFalse(compare_tensor([1.0, 2.0], [float("nan"), 2.0])["passed"])

    def test_reference_and_shape_errors_are_rejected(self):
        for expected, actual in [([], []), ([1], [1, 2]), ([float("inf")], [1])]:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                compare_tensor(expected, actual)

    def test_fixed_threshold_measures_relative_maximum(self):
        self.assertTrue(compare_tensor([10.0, 1.0], [10.005, 1.0])["passed"])
        self.assertFalse(compare_tensor([10.0, 1.0], [10.02, 1.0])["passed"])


if __name__ == "__main__":
    unittest.main()
