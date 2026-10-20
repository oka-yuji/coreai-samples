import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from Tools.Conversion.adapters.qwen_image21.evaluation import (
    COMPONENTS, IMPLEMENTATION, MODEL, REVISION, SIGMAS, aggregate_runs, canonical_hash,
    live_memory_peak, measurement, raw_manifest_hash, validate_input_pair,
    validate_inventory, validate_manifest, validate_profile,
)
from Tools.Conversion.adapters.qwen_image21.profile import boundaries


def profile():
    return {
        "format": 2, "id": "fixture-fp32",
        "source": {"model": MODEL, "revision": REVISION, "implementation_revision": IMPLEMENTATION,
                   "dependency_lock_sha256": "a" * 64, "converter_sha256": {"convert.py": "b" * 64}},
        "workload": {"resolution": [512, 512], "reference_resolution": [512, 512], "batch": 1,
                     "maximum_references": 1, "processed_text_capacity": 512, "joint_capacity": 2304,
                     "operation": "generate-and-edit", "cfg": 1, "sigmas": list(SIGMAS),
                     "scheduler_compute_dtype": "float32", "scheduler_output_dtype": "float32",
                     "scheduler_delta_dtype": "model-output", "scheduler_product_dtype": "model-output",
                     "terminal_sigma": 0.0, "cast_back": "every-step-to-model-output",
                     "rounding": "nearest-ties-to-even", "denoiser_fp16_clip": [-65504, 65504],
                     "zero_centered_norm_dtype": "float32",
                     "prefix_kv_cache": False, "noise_shape": [1, 1024, 64], "noise_dtype": "float32",
                     "rng_version": "fixture-v1"},
        "components": {name: {"asset": name + ".aimodel", "function": "main", "state_names": [],
                              "compute_dtype": "float32", "weight_encoding": {"kind": "none"},
                              **boundaries()[name]}
                       for name in sorted(COMPONENTS)},
    }


def allocation(identity, size, begin, end, category="weights"):
    return {"storage_id": identity, "backing_bytes": size, "begin_ns": begin, "end_ns": end, "category": category}


def conditions(**changes):
    value = {"profile_sha256": "a" * 64, "workload_sha256": "b" * 64, "device": "fixture",
             "os_build": "fixture-os", "route": "native", "operation": "generate", "cache_state": "warm",
             "diagnostics": "off", "power": "unplugged", "thermal": "nominal"}
    return value | changes


class ProfileTests(unittest.TestCase):
    def test_valid_profile_has_a_stable_canonical_identity(self):
        value = profile()
        self.assertEqual(validate_profile(value), canonical_hash(value))
        reordered = dict(reversed(list(value.items())))
        self.assertEqual(validate_profile(value), validate_profile(reordered))

    def test_physical_shape_dtype_source_and_schedule_change_identity_or_fail(self):
        original = profile()
        changed = copy.deepcopy(original)
        changed["components"]["text"]["inputs"][1]["dtype"] = "float16"
        self.assertNotEqual(validate_profile(original), validate_profile(changed))
        for key, value in [("scheduler_compute_dtype", "float16"), ("scheduler_product_dtype", "float32"),
                           ("scheduler_delta_dtype", "float32"), ("sigmas", [1.0, 0.5]),
                           ("prefix_kv_cache", True), ("processed_text_capacity", True)]:
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed["workload"][key] = value
                with self.assertRaises(ValueError):
                    validate_profile(changed)
        changed = copy.deepcopy(original)
        changed["source"]["revision"] = "0" * 40
        with self.assertRaises(ValueError):
            validate_profile(changed)

    def test_generation_only_shape_does_not_claim_editing(self):
        value = profile()
        value["workload"]["joint_capacity"] = 1536
        with self.assertRaises(ValueError):
            validate_profile(value)
        value["workload"]["operation"] = "generate"
        value["components"]["denoiser"].update(boundaries(1536)["denoiser"])
        validate_profile(value)

    def test_all_five_components_and_valid_tensor_boundaries_are_required(self):
        for mutation in ["missing", "zero", "negative", "duplicate", "float64", "unsafe-path"]:
            with self.subTest(mutation=mutation):
                value = profile()
                component = value["components"]["vision"]
                if mutation == "missing":
                    del value["components"]["vision"]
                elif mutation in ("zero", "negative"):
                    component["inputs"][0]["shape"] = [0 if mutation == "zero" else -1]
                elif mutation == "duplicate":
                    component["inputs"] *= 2
                elif mutation == "float64":
                    component["inputs"][0]["dtype"] = "float64"
                else:
                    component["asset"] = "../vision.aimodel"
                with self.assertRaises(ValueError):
                    validate_profile(value)

    def test_raw_manifest_identity_preserves_saved_bytes(self):
        self.assertNotEqual(raw_manifest_hash(b'{"a":1}'), raw_manifest_hash(b'{ "a": 1 }'))
        with self.assertRaises(ValueError):
            raw_manifest_hash({"a": 1})

    def test_bundle_manifest_requires_all_assets_without_self_hash_or_path_escape(self):
        value = {"format": 2, "profile": profile(), "assets": {
            name + ".aimodel/main.mlirb": {"bytes": 100, "sha256": "a" * 64} for name in COMPONENTS}}
        self.assertEqual(validate_manifest(value), 500)
        for mutation in ("self", "missing", "overflow", "escape"):
            changed = copy.deepcopy(value)
            if mutation == "self":
                changed["assets"]["bundle.json"] = {"bytes": 10, "sha256": "b" * 64}
            elif mutation == "missing":
                del changed["assets"]["vision.aimodel/main.mlirb"]
            elif mutation == "overflow":
                changed["assets"]["vision.aimodel/main.mlirb"]["bytes"] = 2**63
            else:
                changed["assets"]["../escape"] = {"bytes": 10, "sha256": "b" * 64}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_comparisons_require_identical_frozen_inputs(self):
        value = {"fixture_id": "generation-01", "source_revision": REVISION, "prompt_sha256": "a" * 64,
                 "reference_sha256": None, "noise_sha256": "b" * 64, "processed_tokens_sha256": "c" * 64,
                 "layout_sha256": "d" * 64, "sigmas": list(SIGMAS), "operation": "generate"}
        validate_input_pair(value, copy.deepcopy(value))
        for field in ("fixture_id", "noise_sha256", "processed_tokens_sha256", "layout_sha256", "sigmas"):
            changed = copy.deepcopy(value)
            changed[field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_input_pair(value, changed)

    def test_fixture_content_not_only_size_is_checked_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "input.bin"
            path.write_bytes(b"ABCD")
            record = {"bytes": 4, "sha256": hashlib.sha256(b"ABCD").hexdigest()}
            validate_inventory(root, {"input.bin": record})
            path.write_bytes(b"EFGH")
            with self.assertRaises(ValueError):
                validate_inventory(root, {"input.bin": record})
            path.write_bytes(b"ABCD")
            (root / "alias.bin").symlink_to(path)
            for bad in ("../input.bin", "alias.bin", "/input.bin", "sub//input.bin"):
                with self.subTest(path=bad), self.assertRaises(ValueError):
                    validate_inventory(root, {bad: record})


class MemoryPlanTests(unittest.TestCase):
    def test_disjoint_lifetimes_do_not_add_model_sizes(self):
        result = live_memory_peak([allocation("text", 400, 0, 10), allocation("dit", 300, 10, 20)])
        self.assertEqual(result["planning_peak_bytes"], 400)

    def test_overlapping_loads_and_persistent_buffers_are_counted(self):
        result = live_memory_peak([allocation("text-load", 400, 0, 11), allocation("dit-load", 300, 10, 20),
                                   allocation("cache", 100, 0, 20, "cache"),
                                   allocation("output", 20, 10, 20, "images")])
        self.assertEqual(result["planning_peak_bytes"], 820)
        self.assertEqual(result["time_ns"], 10)

    def test_aliases_extend_lifetime_without_double_counting_the_backing_storage(self):
        result = live_memory_peak([allocation("shared", 400, 0, 10), allocation("shared", 400, 5, 15),
                                   allocation("next", 100, 10, 20)])
        self.assertEqual(result["planning_peak_bytes"], 500)
        self.assertEqual(result["timeline"][-1]["live_bytes"], 0)

    def test_same_content_is_not_the_same_allocation_and_unknown_workspace_is_not_zero(self):
        values = [allocation("load-one", 400, 0, 10), allocation("load-two", 400, 0, 10)]
        self.assertEqual(live_memory_peak(values)["planning_peak_bytes"], 800)
        result = live_memory_peak(values, unknown_categories=["workspace"])
        self.assertIsNone(result["planning_peak_bytes"])
        self.assertEqual(result["known_live_peak_bytes"], 800)
        self.assertEqual(result["unmeasured_categories"], ["workspace"])

    def test_invalid_and_inconsistent_aliases_are_rejected(self):
        for values in [[allocation("a", -1, 0, 2)], [allocation("a", True, 0, 2)],
                       [allocation("a", 4, 2, 2)], [allocation("a", 4, 0, 2), allocation("a", 8, 1, 3)]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                live_memory_peak(values)


class MeasurementTests(unittest.TestCase):
    def test_missing_observations_remain_null(self):
        self.assertEqual(measurement(None, "failed", "B", "snapshot")["value"], None)
        with self.assertRaises(ValueError):
            measurement(0, "failed", "B", "snapshot")
        with self.assertRaises(ValueError):
            measurement(float("nan"), "ok", "B", "snapshot")

    def test_warm_quantiles_failures_and_sample_counts_are_preserved(self):
        rows = [{"run_id": str(index), "conditions": conditions(), "status": "completed", "total_seconds": index}
                for index in range(1, 21)]
        rows.append({"run_id": "failure", "conditions": conditions(), "status": "failed", "total_seconds": None})
        result, = aggregate_runs(rows)
        self.assertEqual((result["p50_seconds"], result["p95_seconds"], result["maximum_seconds"]), (10, 19, 20))
        self.assertEqual(result["outcomes"], {"completed": 20, "failed": 1})
        self.assertTrue(result["sample_requirement_met"])

    def test_different_devices_caches_diagnostics_and_operations_are_not_pooled(self):
        groups = [conditions(), conditions(cache_state="cached-cold"), conditions(device="other"),
                  conditions(diagnostics="trace"), conditions(operation="edit"), conditions(thermal="serious")]
        rows = [{"run_id": str(index), "conditions": group, "status": "completed", "total_seconds": 1}
                for index, group in enumerate(groups)]
        output = aggregate_runs(rows)
        self.assertEqual(len(output), len(groups))
        self.assertTrue(all(not row["sample_requirement_met"] for row in output))

    def test_failed_only_group_has_no_fabricated_latency(self):
        result, = aggregate_runs([{"run_id": "one", "conditions": conditions(), "status": "cancelled", "total_seconds": None}])
        self.assertIsNone(result["p95_seconds"])
        self.assertEqual(result["completed_samples"], 0)
        with self.assertRaises(ValueError):
            aggregate_runs([{"run_id": "one", "conditions": conditions(), "status": "failed", "total_seconds": 0}])


if __name__ == "__main__":
    unittest.main()
