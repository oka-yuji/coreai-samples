import copy
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from transformers import GenerationConfig
from transformers.generation.candidate_generator import SinglePositionMultiTokenCandidateGenerator
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
from transformers.models.gemma4_assistant.configuration_gemma4_assistant import Gemma4AssistantConfig
from transformers.models.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantForCausalLM

from Tools.Conversion.adapters.gemma4_31b import assistant, drafter, layers, mtp
from Tools.Conversion.adapters.gemma4_31b.prepare import load_contract
from Tools.Conversion.storage import read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_mtp import CONFIG, write_assistant, write_records, write_target

ASSISTANT, TARGET = assistant.SYNTHETIC["assistant"], assistant.SYNTHETIC["target"]
TEXT = ASSISTANT["text_config"]
VOCABULARY, BACKBONE = TARGET["vocab_size"], TARGET["hidden_size"]
WINDOW = TARGET["sliding_window"] + 1
SEED, LIMIT = 11, 1e-4
POSITIONS = {"one confirmed position": 1, "one position short of the window": 1024, "exactly the window": 1025,
             "one position beyond the window": 1026, "far beyond the window": 4000, "a large position": 16895}


def original(state):
    config = Gemma4AssistantConfig(**copy.deepcopy(ASSISTANT))
    config._attn_implementation = "eager"
    model = Gemma4AssistantForCausalLM(config).eval()
    missing, unexpected = model.load_state_dict({name: tensor for name, tensor in state.items()
                                                 if name != "target_embed.weight"}, strict=False)
    if missing != ["lm_head.weight"] or unexpected or model.lm_head.weight is not model.model.embed_tokens.weight:
        raise AssertionError(f"The synthetic weights do not fit the original drafter: {missing}, {unexpected}")
    embed = Gemma4TextScaledWordEmbedding(VOCABULARY, BACKBONE, 0, embed_scale=BACKBONE ** 0.5)
    embed.weight = torch.nn.Parameter(state["target_embed.weight"].clone(), requires_grad=False)
    embed.embed_scale = torch.tensor(layers.rounded_scale(BACKBONE))
    return model, embed


def case(position, seed):
    generator = torch.Generator().manual_seed(seed)
    token = torch.randint(0, VOCABULARY, (1, 1), generator=generator).to(torch.int32)
    hidden = torch.randn(1, 1, BACKBONE, generator=generator)
    sliding = [torch.randn(1, 1, TEXT["num_key_value_heads"], position, TEXT["head_dim"], generator=generator)
               for _ in range(2)]
    full = [torch.randn(1, 1, TEXT["num_global_key_value_heads"], position, TEXT["global_head_dim"],
                        generator=generator) for _ in range(2)]
    return SimpleNamespace(token=token, hidden=hidden, position=position, sliding=sliding, full=full)


def bordered(found):
    values = found.sliding[1].clone()
    values[..., found.position - WINDOW - 1:found.position - WINDOW + 1, :] *= 100
    return SimpleNamespace(**{**vars(found), "sliding": [found.sliding[0], values]})


def ours(found, window=WINDOW):
    keep = min(found.position, window)
    return (found.token, found.hidden, torch.tensor([[found.position]], dtype=torch.int32),
            *(tensor[..., -keep:, :] for tensor in found.sliding), *found.full)


def shared(found):
    return {"sliding_attention": tuple(tensor[0] for tensor in found.sliding),
            "full_attention": tuple(tensor[0] for tensor in found.full)}


def theirs(model, embed, found):
    with torch.no_grad():
        output = model(inputs_embeds=torch.cat([embed(found.token.long()), found.hidden], dim=-1),
                       attention_mask=torch.ones(1, found.position + 1, dtype=torch.long),
                       position_ids=torch.tensor([[found.position]]), shared_kv_states=shared(found), use_cache=False)
    return output.logits, output.last_hidden_state


def generated(model, embed, found, steps, stops):
    model.generation_config = GenerationConfig(num_assistant_tokens=steps, num_assistant_tokens_schedule="constant",
                                               eos_token_id=list(stops) or None, is_assistant=True)
    length = found.position + 1
    ids = torch.zeros(1, length, dtype=torch.long)
    ids[0, -1] = int(found.token)
    mask = torch.ones(1, length, dtype=torch.long)
    config = GenerationConfig(eos_token_id=list(stops) or None, max_length=length + steps + 1, pad_token_id=0)
    generator = SinglePositionMultiTokenCandidateGenerator(ids, model, embed, config, {"attention_mask": mask})
    outputs = SimpleNamespace(hidden_states=(found.hidden,), shared_kv_states=shared(found))
    with torch.no_grad():
        candidates, _ = generator.get_candidates(ids, {"attention_mask": mask}, outputs, False, 0)
    return candidates[0, length:].tolist()


def repeated(model, arguments, steps):
    token, hidden, rest, drafted = arguments[0], arguments[1], arguments[2:], []
    with torch.no_grad():
        for _ in range(steps):
            logits, hidden = model.outputs(token, hidden, *rest)
            token = logits.argmax(dim=-1).to(torch.int32)
            drafted.append(int(token))
    return drafted


def unrolled(model, arguments):
    with torch.no_grad():
        tokens = model(*arguments)
    if tokens.dtype != torch.int32 or tuple(tokens.shape) != (1, model.steps):
        raise AssertionError(f"The unrolled drafter returned {tokens.dtype} {list(tokens.shape)}")
    return tokens[0].tolist()


def largest(first, second):
    return float((first - second).abs().max())


class OneStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = drafter.synthetic(SEED, 1)
        cls.original, cls.embed = original(cls.model.state_dict())
        cls.cases = {label: case(position, number) for number, (label, position) in enumerate(POSITIONS.items())}
        cls.cases.update({f"{label} with large values at both sides of the window edge": bordered(found)
                          for label, found in list(cls.cases.items()) if found.position > WINDOW})

    def test_one_step_gives_the_logits_and_the_projection_of_the_original_drafter_up_to_a_large_position(self):
        for label, found in self.cases.items():
            with torch.no_grad():
                logits, projected = self.model.outputs(*ours(found))
            expected = theirs(self.original, self.embed, found)
            with self.subTest(case=label):
                self.assertEqual((tuple(logits.shape), tuple(projected.shape)), ((1, 1, VOCABULARY), (1, 1, BACKBONE)))
                self.assertLessEqual(largest(logits, expected[0]), LIMIT)
                self.assertLessEqual(largest(projected, expected[1]), LIMIT)
                self.assertEqual(int(logits.argmax()), int(expected[0].argmax()))

    def test_a_window_one_position_narrower_or_wider_than_the_original_window_changes_the_outputs(self):
        edges = [label for label in self.cases if label.endswith("edge")]
        self.assertEqual(len(edges), 3)
        for label in edges:
            found = self.cases[label]
            expected = theirs(self.original, self.embed, found)
            for width in (WINDOW - 1, WINDOW + 1):
                with torch.no_grad(), mock.patch.object(self.model, "check"):
                    logits, projected = self.model.outputs(*ours(found, width))
                with self.subTest(case=label, width=width):
                    self.assertGreater(max(largest(logits, expected[0]), largest(projected, expected[1])), LIMIT)

    def test_another_embedding_scale_than_the_rounded_one_changes_the_outputs(self):
        model = drafter.synthetic(SEED, 1)
        model.embed_scale = BACKBONE ** 0.5
        found = self.cases["far beyond the window"]
        with torch.no_grad():
            logits, _ = model.outputs(*ours(found))
        self.assertEqual(self.model.embed_scale, 6.9375)
        self.assertGreater(largest(logits, theirs(self.original, self.embed, found)[0]), LIMIT)

    def test_the_keys_and_values_of_the_other_layer_kind_cannot_be_read(self):
        model = drafter.synthetic(SEED, 1)
        for layer in model.model.layers:
            layer.self_attn.sliding = not layer.self_attn.sliding
        with self.assertRaises(RuntimeError), torch.no_grad():
            model.outputs(*ours(self.cases["far beyond the window"]))


class UnrolledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {steps: drafter.synthetic(SEED, steps) for steps in (1, 2, 7)}
        cls.original, cls.embed = original(cls.models[1].state_dict())
        cls.cases = {label: case(POSITIONS[label], 100 + number) for number, label in enumerate(
            ("one confirmed position", "exactly the window", "far beyond the window", "a large position"))}

    def test_the_unrolled_tokens_equal_one_step_repeated_and_the_drafts_of_the_original_generator(self):
        for steps, model in self.models.items():
            for label, found in self.cases.items():
                tokens = unrolled(model, ours(found))
                with self.subTest(steps=steps, case=label):
                    self.assertEqual(tokens, repeated(self.models[1], ours(found), steps))
                    self.assertEqual(tokens, generated(self.original, self.embed, found, steps, ()))

    def test_the_drafts_after_the_first_stop_token_are_cut_where_the_original_generator_stops(self):
        for label, found in self.cases.items():
            tokens = unrolled(self.models[7], ours(found))
            absent = next(token for token in range(VOCABULARY) if token not in tokens)
            for stops in ((tokens[0],), (tokens[3],), (absent, tokens[5]), (tokens[6], tokens[2]), (absent,)):
                with self.subTest(case=label, stops=stops):
                    self.assertEqual(mtp.cut(tokens, stops), generated(self.original, self.embed, found, 7, stops))

    def test_cuts_that_drop_the_stop_token_keep_later_drafts_or_stop_at_the_last_stop_token_differ_from_the_original(self):
        found = self.cases["far beyond the window"]
        tokens = unrolled(self.models[7], ours(found))
        first, later = tokens[1], next(token for token in tokens[2:] if token != tokens[1])
        stops = (first, later)
        expected = generated(self.original, self.embed, found, 7, stops)
        self.assertEqual(mtp.cut(tokens, stops), expected)
        variants = {"without the stop token": tokens[:tokens.index(first)], "every draft": tokens,
                    "at the last stop token": tokens[:max(i for i, token in enumerate(tokens) if token in stops) + 1]}
        for label, cut in variants.items():
            with self.subTest(variant=label):
                self.assertNotEqual(cut, expected)

    def test_unrolled_steps_that_advance_the_position_or_hand_on_other_values_differ_from_the_original(self):
        model = self.models[7]
        base, embed = model.step, model.embed

        def advancing():
            calls = itertools.count()
            return lambda embedded, hidden, position, *rest: base(embedded, hidden, position + next(calls), *rest)

        def anchored(found):
            return lambda token: embed(found.token)

        variants = {"a position advanced by one per step": lambda found: ("step", advancing()),
                    "the input hidden state at every step": lambda found: (
                        "step", lambda embedded, hidden, *rest: (base(embedded, hidden, *rest)[0], hidden)),
                    "the anchor token at every step": lambda found: ("embed", anchored(found))}
        for label, variant in variants.items():
            changed = 0
            for found in self.cases.values():
                name, replacement = variant(found)
                with mock.patch.object(model, name, replacement):
                    tokens = unrolled(model, ours(found))
                changed += tokens != generated(self.original, self.embed, found, 7, ())
            with self.subTest(variant=label):
                self.assertGreater(changed, 0)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = drafter.synthetic(SEED, 3)
        cls.found = case(1500, 7)

    def test_inputs_off_the_contract_are_refused_by_both_forms(self):
        good = list(ours(self.found))
        kv, heads = TEXT["num_key_value_heads"], TEXT["head_dim"]
        sliding, full = (3, 4), (5, 6)
        changes = {"anchor token of two positions": {0: good[0].repeat(1, 2)},
                   "anchor token as int64": {0: good[0].long()},
                   "hidden state of another width": {1: torch.zeros(1, 1, BACKBONE + 1)},
                   "hidden state in float16": {1: good[1].half()},
                   "position as int64": {2: good[2].long()},
                   "position of two values": {2: good[2].repeat(1, 2)},
                   "sliding keys and values wider than the window": dict.fromkeys(
                       sliding, torch.zeros(1, 1, kv, WINDOW + 1, heads)),
                   "sliding keys and values without a position": dict.fromkeys(sliding, torch.zeros(1, 1, kv, 0, heads)),
                   "sliding values shorter than the keys": {4: good[4][..., 1:, :]},
                   "sliding keys and values of the full-attention layers": dict(zip(sliding, good[5:7])),
                   "sliding keys and values with the heads of the full-attention layers": dict.fromkeys(
                       sliding, torch.zeros(1, 1, TEXT["num_global_key_value_heads"], WINDOW, heads)),
                   "full keys and values of the sliding-window layers": dict(zip(full, good[3:5])),
                   "full keys and values with another rank": {5: good[5][..., None], 6: good[6][..., None]},
                   "full keys and values without a position": dict.fromkeys(
                       full, torch.zeros(1, 1, 1, 0, TEXT["global_head_dim"])),
                   "full values in float16": {6: good[6].half()}}
        with torch.no_grad():
            self.model(*good)
            self.model.outputs(*good)
        for label, change in changes.items():
            arguments = [change.get(index, value) for index, value in enumerate(good)]
            for form in ("forward", "outputs"):
                with self.subTest(change=label, form=form), self.assertRaisesRegex(ValueError, "differ from the contract"):
                    getattr(self.model, form)(*arguments)

    def test_assistant_configurations_off_the_contract_or_the_target_are_refused(self):
        rope = ASSISTANT["text_config"]["rope_parameters"]
        text_changes = {"a layer type that is not known": {"layer_types": [assistant.SLIDING] * 3 + ["chunked_attention"]},
                        "no full-attention layer": {"layer_types": [assistant.SLIDING] * 4},
                        "fewer layer types than layers": {"layer_types": [assistant.SLIDING, assistant.FULL]},
                        "other key and value heads than the target": {"num_key_value_heads": 1},
                        "other full-attention heads than the target": {"num_global_key_value_heads": 2},
                        "query heads that do not divide into the key and value heads": {"num_attention_heads": 3},
                        "another head dimension than the target": {"head_dim": 8},
                        "another full-attention head dimension than the target": {"global_head_dim": 16},
                        "another window than the target": {"sliding_window": 512},
                        "another vocabulary than the target": {"vocab_size": 300},
                        "layers with their own keys and values": {"num_kv_shared_layers": 2},
                        "per-layer inputs": {"hidden_size_per_layer_input": 8},
                        "mixture of experts": {"enable_moe_block": True},
                        "double-wide MLP": {"use_double_wide_mlp": True},
                        "attention bias": {"attention_bias": True},
                        "another activation": {"hidden_activation": "gelu"},
                        "another RoPE": {"rope_parameters": {**rope, assistant.FULL: {**rope[assistant.FULL],
                                                                                    "partial_rotary_factor": 0.5}}},
                        "no head dimension": {"head_dim": None}}
        changes = {label: ({**ASSISTANT, "text_config": {**TEXT, **change}}, TARGET) for label, change in text_changes.items()}
        changes.update({"ordered embeddings with centroids": ({**ASSISTANT, "use_ordered_embeddings": True}, TARGET),
                        "untied embeddings": ({**ASSISTANT, "tie_word_embeddings": False}, TARGET),
                        "another backbone width than the target": ({**ASSISTANT, "backbone_hidden_size": 64}, TARGET),
                        "a target without its window": (ASSISTANT, {key: value for key, value in TARGET.items()
                                                                    if key != "sliding_window"}),
                        "no text configuration": ({key: value for key, value in ASSISTANT.items()
                                                   if key != "text_config"}, TARGET)})
        self.assertEqual(assistant.assistant_config(ASSISTANT, TARGET).layer_types, TEXT["layer_types"])
        for label, (document, target) in changes.items():
            with self.subTest(change=label), self.assertRaises(ValueError):
                assistant.assistant_config(document, target)

    def test_a_step_count_that_is_not_a_whole_number_from_one_is_refused(self):
        config = assistant.assistant_config(ASSISTANT, TARGET)
        for steps in (0, -1, True, 1.0, None, "7"):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "whole number of steps"):
                drafter.Gemma4AssistantDrafter(config, steps)


def recorded_setup(folder):
    state = drafter.synthetic(SEED, 1).state_dict()
    records, inputs, outputs = write_records(folder)
    target, contract = write_target(folder, state["target_embed.weight"])
    assistant, assistant_contract = write_assistant(folder, ASSISTANT, {name: tensor for name, tensor in state.items()
                                                                        if name != "target_embed.weight"})
    return (contract, assistant_contract, target, assistant, records, inputs, outputs)


def reference_records(arguments, steps):
    contract, _, target, assistant, records, inputs, outputs = arguments
    pinned = load_contract(contract)
    _, items = mtp.items_of(pinned, records, inputs, outputs, lambda record: record["items"])
    model = mtp.class_drafter(assistant, mtp.table(target / mtp.table_shard(target, pinned)[1], CONFIG), CONFIG, steps)
    totals = {}
    (records / "drafters").mkdir()
    for item in items:
        features, prompt, rows = mtp.loaded_features(item), len(item["input_ids"]), []
        for anchor in range(prompt, prompt + len(item["teacher"]) - 1):
            with torch.no_grad():
                tokens = mtp.cut(model(*mtp.row_inputs(item, features, anchor, WINDOW)["arguments"])[0].tolist(),
                                 CONFIG["eos_token_id"])
            accepted = mtp.matches(tokens, item["teacher"][anchor - prompt + 1:])
            rows.append({"a": anchor, "j": anchor - prompt, "fixed": {"accepted": accepted, "tokens": tokens,
                                                                     "margins": [0.5] * len(tokens)}})
        write_json(records / "drafters" / f"{item['id']}.json", {"id": item["id"], "P": prompt, "T": len(item["teacher"]),
                                                                  "mtp": rows})
        found = mtp.simulate([row["fixed"]["accepted"] for row in rows], len(item["teacher"]), steps)
        for name in (item["category"], "pooled"):
            totals[name] = [sum(pair) for pair in zip(totals.get(name, (0, 0)), found)]
    write_json(records / "trajectories.json", {"per_round_on_teacher_path": {"mtp_fixed": {str(steps): {
        name: {"rounds": rounds, "tokens": tokens, "tokens_per_round": tokens / rounds}
        for name, (rounds, tokens) in totals.items()}}}})
    return items


class RunTests(unittest.TestCase):
    def test_the_contract_of_the_records_holds_the_synthetic_target(self):
        self.assertEqual({key: CONFIG[key] for key in assistant.TARGET}, TARGET)

    def test_the_checkpoint_checked_before_loading_holds_the_tensors_of_both_drafters(self):
        expected = assistant.checkpoint_shapes(assistant.assistant_config(ASSISTANT, TARGET))
        state = drafter.synthetic(SEED, 1).state_dict()
        self.assertEqual({name: list(tensor.shape) for name, tensor in state.items() if name != "target_embed.weight"},
                         expected)
        model, _ = original({name: tensor for name, tensor in state.items()})
        self.assertEqual({name: list(tensor.shape) for name, tensor in model.state_dict().items()
                          if name != "lm_head.weight"}, expected)

    def test_runs_of_both_implementations_on_recorded_features_pass_the_comparison_unless_the_scale_is_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            arguments = recorded_setup(folder)
            original = mtp.run("original", *arguments, 7, folder / "original")
            model_class = mtp.run("class", *arguments, 7, folder / "class")
            self.assertEqual(original["rows"], model_class["rows"])
            self.assertEqual([(row["id"], row["anchor"], row["window"]) for row in original["rows"]],
                             [("alpha-0002", 20, 20), ("alpha-0002", 23, 23), ("alpha-0002", 26, 26),
                              ("alpha-0002", 28, 28), ("beta-0001", 1100, WINDOW), ("beta-0001", 1102, WINDOW),
                              ("beta-0001", 1104, WINDOW), ("beta-0001", 1105, WINDOW)])
            result = mtp.compare(folder / "original", folder / "class", arguments[0])
            self.assertTrue(result["passed"], result["problems"])
            self.assertLessEqual(result["summary"]["max_abs_logit"], LIMIT)
            self.assertEqual(result["summary"]["unrolled_equals_repeated"], 8)
            self.assertEqual(result["summary"]["cut_equals_original"], 8)
            with mock.patch.object(drafter, "bfloat16", lambda value: 1.0):
                mtp.run("class", *arguments, 7, folder / "unscaled")
            result = mtp.compare(folder / "original", folder / "unscaled", arguments[0])
            self.assertEqual(result["problems"], [])
            self.assertFalse(result["passed"])
            self.assertGreater(result["summary"]["max_abs_logit"], 1e-3)

    def test_the_acceptance_follows_the_teacher_path_and_counts_the_anchors_whose_records_differ(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            arguments = recorded_setup(folder)
            items = reference_records(arguments, 7)
            result = mtp.acceptance(*arguments, 7)
            self.assertEqual(result["problems"], [])
            pooled = result["categories"]["pooled"]
            self.assertEqual((pooled["items"], pooled["anchors"]), (3, sum(len(item["teacher"]) - 1 for item in items)))
            for name, entry in result["categories"].items():
                with self.subTest(category=name):
                    self.assertEqual((entry["rounds"], entry["tokens"]),
                                     (entry["reference"]["rounds"], entry["reference"]["tokens"]))
                    self.assertEqual(entry["anchors_with_other_acceptance"], 0)
            path = arguments[4] / "drafters/beta-0001.json"
            record = read_json(path)
            record["mtp"][2]["fixed"]["accepted"] += 1
            write_json(path, record)
            result = mtp.acceptance(*arguments, 7)
            self.assertEqual(len(result["problems"]), 2)
            self.assertEqual([result["categories"][name]["anchors_with_other_acceptance"] for name in ("alpha", "beta",
                                                                                                        "pooled")],
                             [0, 1, 1])


class ExportTests(unittest.TestCase):
    def test_the_one_step_and_the_unrolled_drafter_convert_with_their_names_shapes_and_dynamic_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "export"
            record = mtp.export_synthetic(output, 7, 16896, SEED)
            self.assertTrue(record["passed"], {name: entry["problems"] for name, entry in record["assets"].items()})
            self.assertEqual(read_json(output / "export.json"), record)
            self.assertEqual({name: entry["functions"] for name, entry in record["assets"].items()},
                             {"one_step": ["draft_1"], "unrolled": ["draft_7"], "both": ["draft_1", "draft_7"]})
            functions = record["assets"]["both"]["asset"]["functions"]
            self.assertEqual(functions["draft_7"]["outputs"], {"draft_tokens": {"dtype": "int32", "shape": [1, 7]}})
            self.assertEqual(functions["draft_1"]["outputs"], {"draft_tokens": {"dtype": "int32", "shape": [1, 1]}})
            for name in ("draft_1", "draft_7"):
                with self.subTest(function=name):
                    shapes = {item: entry["shape"] for item, entry in functions[name]["inputs"].items()}
                    self.assertEqual(shapes, {"input_ids": [1, 1], "hidden_states": [1, 1, BACKBONE], "position_ids": [1, 1],
                                              "k_sliding": [1, 1, 2, None, 16], "v_sliding": [1, 1, 2, None, 16],
                                              "k_full": [1, 1, 1, None, 32], "v_full": [1, 1, 1, None, 32]})
                    self.assertEqual(functions[name]["states"], {})
            operations = {name: entry["asset"]["operations"] for name, entry in record["assets"].items()}
            self.assertEqual((operations["one_step"]["argmax"], operations["unrolled"]["argmax"]), (1, 7))
            self.assertEqual(operations["unrolled"]["gather_nd"] - operations["one_step"]["gather_nd"], 6)
            self.assertTrue(all(entry["bytes"] > 0 for entry in record["assets"].values()))
            for steps, context in ((1, 16896), (7, WINDOW), (True, 16896)):
                with self.subTest(steps=steps, context=context), self.assertRaises(ValueError):
                    mtp.export_synthetic(Path(folder) / f"refused-{steps}-{context}", steps, context, SEED)
                self.assertFalse((Path(folder) / f"refused-{steps}-{context}").exists())
            with self.assertRaises(FileExistsError):
                mtp.export_synthetic(output, 7, 16896, SEED)


if __name__ == "__main__":
    unittest.main()
