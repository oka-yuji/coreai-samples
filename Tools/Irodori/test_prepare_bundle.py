import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("prepare_bundle", Path(__file__).with_name("prepare-bundle.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PrepareBundleTests(unittest.TestCase):
    def test_complete_bundle_and_existing_output_are_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts = root / "source"
            asset = artifacts / "example.aimodel"
            asset.mkdir(parents=True)
            (asset / "main.mlirb").write_bytes(b"model")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            rules = root / "rules.json"
            rules.write_text(json.dumps([{"stem":"example","files":{"main.mlirb":{"bytes":5,"sha256":hashlib.sha256(b"model").hexdigest()}}}]))
            output = root / "bundle"
            with patch.object(module, "RULES", rules), patch.object(module, "TOKENIZER_SHA", hashlib.sha256(b"tokenizer").hexdigest()):
                module.prepare(artifacts, tokenizer, output)
                self.assertEqual((output / "artifacts/example.aimodel/main.mlirb").read_bytes(), b"model")
                self.assertEqual((output / "tokenizer.json").read_bytes(), b"tokenizer")
                self.assertEqual(json.loads((output / "irodori.json").read_text())["precision"], "fp32")
                with self.assertRaises(FileExistsError): module.prepare(artifacts, tokenizer, output)
                self.assertEqual((output / "tokenizer.json").read_bytes(), b"tokenizer")
                self.assertEqual(list(root.glob(".irodori-*")), [])

    def test_asset_symlinks_are_materialized_as_regular_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts = root / "source"
            asset = artifacts / "example.aimodel"
            asset.mkdir(parents=True)
            real = root / "weights"
            real.write_bytes(b"model")
            (asset / "main.mlirb").symlink_to(real)
            (asset / "main.hash").symlink_to("../../weights")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            rules = root / "rules.json"
            item = {"bytes":5,"sha256":hashlib.sha256(b"model").hexdigest()}
            rules.write_text(json.dumps([{"stem":"example","files":{"main.mlirb":item,"main.hash":item}}]))
            output = root / "bundle"
            with patch.object(module, "RULES", rules), patch.object(module, "TOKENIZER_SHA", hashlib.sha256(b"tokenizer").hexdigest()):
                module.prepare(artifacts, tokenizer, output)
            for filename in ["main.mlirb", "main.hash"]:
                file = output / "artifacts/example.aimodel" / filename
                self.assertFalse(file.is_symlink())
                self.assertTrue(file.is_file())
                self.assertEqual(file.read_bytes(), b"model")

    def test_publication_race_does_not_replace_a_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "staging"
            source.mkdir()
            (source / "owned").write_bytes(b"new")
            destination = root / "bundle"
            destination.mkdir()
            with self.assertRaises(FileExistsError): module.rename_exclusive(source, destination)
            self.assertTrue(source.exists())
            self.assertEqual(list(destination.iterdir()), [])

if __name__ == "__main__": unittest.main()
