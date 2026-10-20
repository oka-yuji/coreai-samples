import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.contracts import KINDS, PROFILE, validate_bundle

try:
    import diffusers
    import torch
    import transformers
    from PIL import Image
except ImportError as error:
    DEPENDENCY_ERROR = str(error)
else:
    DEPENDENCY_ERROR = None

if DEPENDENCY_ERROR is None:
    from Tools.Conversion.adapters.qwen_image21 import generate


SIGNAL_CHILD = """
import os
from pathlib import Path
import signal
from Tools.Conversion.adapters.qwen_image21 import generate

def wait_for_signal(*args, **kwargs):
    Path(os.environ["IMAGE_LIFECYCLE_READY"]).write_text(os.environ["COREAI_HEAVY_JOB_LOCK"])
    signal.pause()
    raise RuntimeError("The termination handler did not interrupt generation")

generate.generate = wait_for_signal
generate.main()
"""


class BundleFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="image-lifecycle-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        for kind in KINDS:
            directory = self.bundle / kind
            asset = directory / f"{kind}.aimodel"
            asset.mkdir(parents=True)
            (asset / "main.mlirb").write_bytes(b"synthetic-asset-v1\n")
            (directory / "component.json").write_text(json.dumps({
                "profile": PROFILE, "kind": kind, "passed": True,
                "asset": f"{kind}.aimodel", "inputs": [], "output_names": ["result"],
            }))
            (directory / "comparison.json").write_text('{"passed":true}\n')
        for relative in (
            "processor/tokenizer_config.json", "text_encoder/config.json",
            "transformer/config.json", "vae/config.json", "scheduler/scheduler_config.json",
        ):
            path = self.bundle / "config" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{"value":1}\n')
        (self.bundle / "NOTICE.txt").write_text("Synthetic lifecycle fixture without model weights.\n")
        self.manifest = {
            "profile": PROFILE, "component_verified": True, "end_to_end_verified": True,
            "components": list(KINDS), "assets": {
                path.relative_to(self.bundle).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in self.bundle.rglob("*") if path.is_file()
            },
        }
        self.write_manifest(self.manifest)
        self.assertEqual(validate_bundle(self.bundle), self.manifest)

    def write_manifest(self, manifest):
        (self.bundle / "bundle.json").write_text(json.dumps(manifest))


class ImageBundleLifecycleTests(BundleFixtureTests):
    def test_bundle_requires_both_verification_flags_and_all_five_components(self):
        mutations = (
            {"component_verified": False},
            {"end_to_end_verified": False},
            {"components": list(KINDS[:-1])},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                manifest = copy.deepcopy(self.manifest)
                manifest.update(mutation)
                self.write_manifest(manifest)
                with self.assertRaises(ValueError):
                    validate_bundle(self.bundle)
        self.write_manifest(self.manifest)

    def test_full_inventory_rejects_both_unlisted_and_missing_files(self):
        extra = self.bundle / "config" / "unexpected.json"
        extra.write_text("{}")
        with self.assertRaises(ValueError):
            validate_bundle(self.bundle)
        extra.unlink()
        self.assertEqual(validate_bundle(self.bundle), self.manifest)
        (self.bundle / "config" / "processor" / "tokenizer_config.json").unlink()
        with self.assertRaises(ValueError):
            validate_bundle(self.bundle)

    def test_same_size_configuration_and_asset_replacements_are_rejected(self):
        paths = (
            self.bundle / "config" / "scheduler" / "scheduler_config.json",
            self.bundle / "denoiser" / "denoiser.aimodel" / "main.mlirb",
        )
        for path in paths:
            with self.subTest(path=path.relative_to(self.bundle)):
                original = path.read_bytes()
                replacement = original.replace(b"1", b"2")
                self.assertEqual(len(replacement), len(original))
                self.assertNotEqual(replacement, original)
                path.write_bytes(replacement)
                try:
                    with self.assertRaises(ValueError):
                        validate_bundle(self.bundle)
                finally:
                    path.write_bytes(original)


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Qwen image generation dependencies unavailable: {DEPENDENCY_ERROR}")
class ImageGenerationLifecycleTests(BundleFixtureTests):
    def test_bundle_hashing_holds_the_same_lock_as_model_execution(self):
        lock = self.root / "locks" / "measured-job"
        def validate_locked_bundle(bundle):
            self.assertTrue(lock.is_dir())
            self.assertEqual(json.loads((lock / "owner.txt").read_text())["pid"], os.getpid())
            return validate_bundle(bundle)
        arguments = ["generate", "--bundle", str(self.bundle), "--output", str(self.root / "output"),
                     "--prompt", "fixture", "--lock-dir", str(lock)]
        with patch.object(sys, "argv", arguments), \
                patch.object(generate, "validate_bundle", side_effect=validate_locked_bundle), \
                patch.object(generate, "generate") as run:
            generate.main()
        run.assert_called_once()
        self.assertFalse(lock.exists())

    def test_partial_output_is_preserved_and_rejected_before_model_preparation(self):
        components = {kind: self.bundle / kind for kind in KINDS}
        for name in ("prompt.txt", "initial-noise.npy"):
            with self.subTest(name=name):
                output = self.root / name.replace(".", "-")
                output.mkdir()
                marker = output / name
                marker.write_bytes(b"previous incomplete generation\n")
                with patch.object(generate, "prompt_inputs", side_effect=AssertionError("Prompt preparation ran")) as prompt:
                    with patch.object(generate, "Component", side_effect=AssertionError("Model loading ran")) as component:
                        with self.assertRaises(FileExistsError):
                            generate.generate(components, self.bundle / "config", output, "A red cube")
                prompt.assert_not_called()
                component.assert_not_called()
                self.assertEqual(marker.read_bytes(), b"previous incomplete generation\n")
                self.assertEqual(list(output.iterdir()), [marker])

    @unittest.skipUnless(hasattr(signal, "pthread_sigmask"), "The shared directory lock requires POSIX signals")
    def test_sigterm_after_real_lock_acquisition_releases_the_isolated_lock(self):
        lock = self.root / "locks" / "heavy-job"
        ready = self.root / "ready.txt"
        home = self.root / "home"
        home.mkdir()
        environment = dict(os.environ)
        environment.update({
            "HOME": str(home), "HF_HOME": str(home / "huggingface"),
            "XDG_CACHE_HOME": str(home / "cache"), "TORCH_HOME": str(home / "torch"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "COREAI_HEAVY_JOB_LOCK": str(self.root / "unused-lock"),
            "IMAGE_LIFECYCLE_READY": str(ready),
        })
        repository = Path(__file__).resolve().parents[3]
        environment["PYTHONPATH"] = str(repository)
        command = [
            sys.executable, "-B", "-u", "-c", SIGNAL_CHILD,
            "--bundle", str(self.bundle), "--output", str(self.root / "output"),
            "--prompt", "A red cube", "--lock-dir", str(lock),
        ]
        log_path = self.root / "child.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(
                command, cwd=repository, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 30
                while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists(), log_path.read_text())
                self.assertIsNone(child.poll(), log_path.read_text())
                self.assertEqual(ready.read_text(), str(lock))
                owner = json.loads((lock / "owner.txt").read_text())
                self.assertEqual(owner["pid"], child.pid)
                self.assertEqual(owner["job"], "generate")
                child.send_signal(signal.SIGTERM)
                self.assertNotEqual(child.wait(timeout=10), 0)
                self.assertIn("KeyboardInterrupt", log_path.read_text())
                self.assertFalse(lock.exists())
                self.assertFalse((self.root / "unused-lock").exists())
                self.assertFalse((home / "Library/Caches/coreai-samples/heavy-job").exists())
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
