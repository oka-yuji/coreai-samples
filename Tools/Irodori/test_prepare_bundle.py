import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("prepare_bundle", Path(__file__).with_name("prepare-bundle.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PrepareBundleTests(unittest.TestCase):
    def test_legacy_manifest_retains_its_pinned_identity(self):
        data = module.make_manifest()
        self.assertEqual(len(data), 132)
        self.assertEqual(hashlib.sha256(data).hexdigest(), "ce41c61dd2249cab61aa69017316627c54329177fbfb19a1d1df4d775d65dbc7")
        self.assertNotIn("assetInventorySHA256", json.loads(data))

    def test_reference_manifest_changes_identity_when_the_inventory_changes(self):
        first = module.make_manifest(b"first inventory")
        second = module.make_manifest(b"second inventory")
        self.assertNotEqual(first, second)
        self.assertEqual(first, module.make_manifest(b"first inventory"))
        self.assertEqual(json.loads(first)["format"], "irodori-coreai-v2")
        self.assertEqual(json.loads(first)["modelID"], "Aratako/Irodori-TTS-v4.1-Small-MF")

    def test_reference_bundle_combines_sources_without_changing_the_original(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "base"
            reference = root / "reference"
            rules = []
            for folder, stem, data in [(base, "text", b"base"), (reference, "voice", b"voice")]:
                path = folder / (stem + ".aimodel")
                path.mkdir(parents=True)
                (path / "main.mlirb").write_bytes(data)
                rules.append({"stem":stem,"files":{"main.mlirb":{"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}}})
            rule_path = root / "rules.json"
            rule_path.write_text(json.dumps(rules))
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer")
            with patch.object(module, "REFERENCE_RULES", rule_path), patch.object(module, "TOKENIZER_SHA", hashlib.sha256(b"tokenizer").hexdigest()):
                output = module.prepare(base, tokenizer, root / "bundle", reference)
                self.assertEqual(json.loads((output / "irodori.json").read_text())["format"], "irodori-coreai-v2")
                self.assertEqual((output / "artifacts/voice.aimodel/main.mlirb").read_bytes(), b"voice")
                self.assertEqual((base / "text.aimodel/main.mlirb").read_bytes(), b"base")
                self.assertEqual((reference / "voice.aimodel/main.mlirb").read_bytes(), b"voice")

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
