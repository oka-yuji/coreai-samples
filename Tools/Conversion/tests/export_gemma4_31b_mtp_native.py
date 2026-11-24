import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from Tools.Conversion.adapters.gemma4_31b import assistant, drafter, mtp, mtp_native
from Tools.Conversion.storage import read_json
from Tools.Conversion.tests.test_gemma4_31b_mtp import (CONFIG, FIXED, FIXTURE_RECORDS, SHARD, write_assistant,
                                                         write_records, write_target)

ASSISTANT, TARGET = assistant.SYNTHETIC["assistant"], assistant.SYNTHETIC["target"]
SEED, STEPS = 11, 7
NUMPY = {"int32": "<i4", "float16": "<f2", "float32": "<f4", "float64": "<f8"}


class RealClassTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("SUBSET", FIXED), ("RECORDS", FIXTURE_RECORDS)):
            patcher = mock.patch.object(mtp, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        state = drafter.synthetic(SEED, 1).state_dict()
        self.records, self.inputs, self.outputs = write_records(self.folder)
        self.target, self.contract = write_target(self.folder, state["target_embed.weight"])
        self.assistant, self.assistant_contract = write_assistant(
            self.folder, ASSISTANT, {name: tensor for name, tensor in state.items() if name != "target_embed.weight"})
        self.output = self.folder / "inputs"
        self.record = mtp_native.write_inputs(self.contract, self.assistant_contract, self.target, self.assistant,
                                              self.records, self.inputs, self.outputs, STEPS, self.output)

    def row_arguments(self, number):
        folder, arguments = self.output / "rows" / f"{number:03d}", []
        for entry in json.loads((folder / "inputs.json").read_text())["inputs"]:
            values = np.fromfile(folder / entry["file"], dtype=NUMPY[entry["dtype"]]).reshape(entry["shape"])
            arguments.append(torch.from_numpy(values).to(torch.float32 if entry["dtype"] == "float16" else torch.int32))
        return arguments

    def test_the_names_and_order_of_the_inputs_are_those_of_the_class(self):
        self.assertEqual(mtp_native.INPUTS, drafter.INPUTS)
        self.assertEqual(list(mtp.expected_function(assistant.assistant_config(ASSISTANT, TARGET), STEPS)["inputs"]),
                         list(drafter.INPUTS))

    def test_the_inputs_of_every_row_match_the_descriptors_that_the_export_checks_the_assets_against(self):
        settings = assistant.assistant_config(ASSISTANT, TARGET)
        expected = mtp.expected_function(settings, STEPS)["inputs"]
        self.assertEqual(len(self.record["rows"]), 8)
        for number in range(len(self.record["rows"])):
            listed = json.loads((self.output / "rows" / f"{number:03d}" / "inputs.json").read_text())["inputs"]
            self.assertEqual([entry["name"] for entry in listed], list(expected))
            for entry in listed:
                wanted = expected[entry["name"]]
                with self.subTest(row=number, name=entry["name"]):
                    self.assertEqual(entry["dtype"], wanted["dtype"])
                    self.assertEqual(len(entry["shape"]), len(wanted["shape"]))
                    for dimension, fixed in zip(entry["shape"], wanted["shape"]):
                        self.assertTrue(dimension == fixed if fixed is not None else dimension >= 1)
            self.assertLessEqual(listed[3]["shape"][3], settings.sliding_window + 1)

    def test_the_arrays_are_what_the_real_class_computes_from_the_written_files(self):
        self.assertTrue(self.record["passed"], self.record["problems"])
        model = mtp.class_drafter(self.assistant, mtp.table(self.target / SHARD, CONFIG), CONFIG, STEPS)
        arrays = {name: np.fromfile(self.output / f"{name}.bin", dtype=NUMPY[entry["dtype"]]).reshape(
            entry["shape"]) for name, entry in self.record["arrays"].items()}
        for number in range(len(self.record["rows"])):
            arguments = self.row_arguments(number)
            with self.subTest(row=number), torch.no_grad():
                logits, _ = model.outputs(*arguments)
                self.assertTrue(np.array_equal(arrays["logits"][number], logits[0, 0].numpy()))
                self.assertEqual(arrays["drafts"][number].tolist(), model(*arguments)[0].tolist())
                token, hidden = arguments[0], arguments[1]
                for step in range(STEPS):
                    values, hidden = model.outputs(token, hidden, *arguments[2:])
                    ranked = torch.sort(values[0, 0].to(torch.float64), descending=True, stable=True).indices[:2].tolist()
                    self.assertEqual(arrays["top2_ids"][number, step].tolist(), ranked)
                    self.assertEqual(int(arrays["drafts"][number, step]), ranked[0])
                    token = values.argmax(dim=-1).to(torch.int32)

    def test_the_weights_come_from_the_checkpoint_of_the_assistant(self):
        other = drafter.synthetic(SEED + 1, 1).state_dict()
        folder = self.folder / "other"
        folder.mkdir()
        target, contract = write_target(folder, other["target_embed.weight"])
        assistant_folder, assistant_contract = write_assistant(
            folder, ASSISTANT, {name: tensor for name, tensor in other.items() if name != "target_embed.weight"})
        record = mtp_native.write_inputs(contract, assistant_contract, target, assistant_folder, self.records, self.inputs,
                                         self.outputs, STEPS, folder / "inputs")
        self.assertNotEqual(record["arrays"]["logits"]["sha256"], self.record["arrays"]["logits"]["sha256"])
        self.assertEqual(record["rows"][0]["files"], self.record["rows"][0]["files"])
        self.assertEqual(read_json(folder / "inputs/record.json"), record)
