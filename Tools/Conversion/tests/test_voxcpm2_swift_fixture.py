import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Tools.Conversion.adapters.voxcpm2 import swift_fixture

SAMPLES = 4


def write_case(root, name="tiny", patches=2, finish="stop", seed=1708, change=None):
    folder = root / "swift_reference" / name
    folder.mkdir(parents=True)
    ids = np.array([5, 6, 101], np.int32)
    padded = np.zeros((1, 1, 8), np.int32)
    padded[0, 0, :3] = ids
    calls = {
        "tslm.embed.in.ids": padded,
        "tslm.embed.out.embeds": np.arange(16, dtype=np.float32).reshape(1, 1, 8, 2),
        "proj.heads.in.lm_hidden": np.full((patches, 1, 3), 0.5, np.float32),
        "proj.heads.out.stop_logits": np.array([[[1.0, 0.0]]] * patches, np.float32),
        "tslm.step.in.position": np.array([[3]] * (patches - 1), np.int32),
        "tslm.step.out.hidden": np.ones((patches - 1, 1, 3), np.float32),
    }
    order = [["tslm.embed", 0]]
    for index in range(patches):
        order.append(["proj.heads", index])
        if index < patches - 1:
            order.append(["tslm.step", index])
    patch_arrays = {key: np.full((patches, 1, 2), number, np.float32)
                    for number, key in enumerate(swift_fixture.PATCH_ARRAYS)}
    audio = np.linspace(-1, 1, patches * SAMPLES, dtype=np.float32)
    if change:
        change(calls, order, patch_arrays)
    np.save(folder / "ids.npy", ids)
    np.save(folder / "audio.npy", audio)
    np.savez(folder / "calls.npz", **calls)
    np.savez(folder / "patches.npz", **patch_arrays)
    (folder / "calls.json").write_text(json.dumps(order), encoding="utf-8")
    entry = {"seed": seed, "finish": finish, "patches": patches, "stopped_patch": patches - 1 if finish == "stop" else None,
             "files": sorted(swift_fixture.SOURCES)}
    report_path = root / "check.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() \
        else {"complete": True, "seed": 1708, "fixtures": {}}
    report["fixtures"][name] = {"swift_reference": entry}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return ids, calls, order, patch_arrays, audio


class SwiftFixtureTests(unittest.TestCase):
    def test_recorded_generations_become_raw_little_endian_arrays_with_an_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids, calls, order, patch_arrays, audio = write_case(root)
            cases = swift_fixture.convert(root, root / "out", SAMPLES)
            self.assertEqual(cases, [{"name": "tiny", "patches": 2, "calls": len(order)}])
            reference = json.loads((root / "out/reference.json").read_text(encoding="utf-8"))
            self.assertEqual((reference["format"], reference["seed"], reference["sentences"]), (1, 1708, ["tiny"]))
            case = json.loads((root / "out/tiny/case.json").read_text(encoding="utf-8"))
            self.assertEqual((case["ids"], case["calls"], case["patches"], case["stoppedPatch"]), (ids.tolist(), order, 2, 1))
            expected = {**calls, **{f"patch.{key}": value for key, value in patch_arrays.items()}, "audio": audio}
            self.assertEqual(sorted(case["arrays"]), sorted(expected))
            for key, value in expected.items():
                record = case["arrays"][key]
                data = (root / "out/tiny" / record["file"]).read_bytes()
                self.assertEqual(record["shape"], list(value.shape), key)
                self.assertEqual(record["type"], str(value.dtype), key)
                self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest(), key)
                stored = np.frombuffer(data, dtype=swift_fixture.TYPES[record["type"]]).reshape(value.shape)
                self.assertTrue(np.array_equal(stored, value), key)

    def test_an_existing_output_or_a_record_that_breaks_the_order_is_rejected(self):
        def gap(calls, order, patches):
            order[1][1] = 1

        def count(calls, order, patches):
            calls["tslm.step.out.hidden"] = np.ones((2, 1, 3), np.float32)

        def nonfinite(calls, order, patches):
            calls["proj.heads.in.lm_hidden"][0, 0, 0] = np.nan

        def wide(calls, order, patches):
            calls["tslm.step.out.hidden"] = np.ones((1, 1, 3), np.float64)

        def ids(calls, order, patches):
            calls["tslm.embed.in.ids"][0, 0, 0] = 7

        def missing(calls, order, patches):
            del patches["latent"]

        cases = [{"change": gap}, {"change": count}, {"change": nonfinite}, {"change": wide}, {"change": ids},
                 {"change": missing}, {"finish": "limit"}, {"seed": 1709}]
        for arguments in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_case(root, **arguments)
                with self.assertRaises(ValueError):
                    swift_fixture.convert(root, root / "out", SAMPLES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case(root)
            (root / "out").mkdir()
            with self.assertRaises(FileExistsError):
                swift_fixture.convert(root, root / "out", SAMPLES)


if __name__ == "__main__":
    unittest.main()
