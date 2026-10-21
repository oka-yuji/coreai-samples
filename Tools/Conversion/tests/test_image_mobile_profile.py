import copy
import unittest

from Tools.Conversion.adapters.qwen_image21.mobile_profile import CANDIDATES, make_profile, validate_mobile_profile


class ImageMobileProfileTests(unittest.TestCase):
    def profile(self, candidate="w4"):
        return make_profile(candidate, "a" * 64, {"mobile_models.py": "b" * 64})

    def test_candidates_keep_distinct_identities_and_the_complete_five_component_abi(self):
        hashes = set()
        for candidate in CANDIDATES:
            profile = self.profile(candidate)
            hashes.add(validate_mobile_profile(profile))
            components = profile["components"]
            self.assertEqual(len(components), 5)
            self.assertEqual(components["text"]["outputs"][0]["name"], "hidden")
            self.assertEqual(components["denoiser"]["outputs"][0]["shape"], [1, 2304, 64])
            self.assertNotIn("output_index", [item["name"] for item in components["denoiser"]["inputs"]])
            self.assertEqual(profile["workload"]["scheduler_output_dtype"], components["denoiser"]["compute_dtype"])
            self.assertFalse(profile["workload"]["prefix_kv_cache"])
        self.assertEqual(len(hashes), len(CANDIDATES))

    def test_weight_storage_does_not_change_text_or_vae_compute_precision(self):
        for candidate in ("w8", "w4"):
            components = self.profile(candidate)["components"]
            self.assertEqual(components["text"]["compute_dtype"], "float32")
            self.assertEqual(components["text"]["weight_encoding"]["storage_dtype"], "float16")
            self.assertEqual(components["denoiser"]["compute_dtype"], "float16")
            self.assertEqual(components["vision"]["weight_encoding"]["bits"], 8)
            self.assertEqual(components["vision"]["weight_encoding"]["group_size"], 16)
            self.assertEqual(components["text"]["weight_encoding"]["group_size"], 64)
            self.assertEqual(components["denoiser"]["weight_encoding"]["time_frequency_dtype"], "float32")
            for kind in ("vae_encoder", "vae_decoder"):
                self.assertEqual(components[kind]["weight_encoding"], {"kind": "dense", "dtype": "float32"})
                self.assertEqual(components[kind]["compute_dtype"], "float32")

    def test_mixed_source_precision_and_unreviewed_contract_changes_are_rejected(self):
        mutations = [
            lambda p: p["components"]["text"].update(compute_dtype="float16"),
            lambda p: p["components"]["text"]["weight_encoding"].update(bits=8),
            lambda p: p["components"]["text"]["weight_encoding"].update(bits=4.0),
            lambda p: p["workload"].update(scheduler_output_dtype="float32"),
            lambda p: p["components"]["denoiser"]["outputs"][0].update(shape=[1, 1024, 64]),
            lambda p: p["components"]["text"]["inputs"][0].update(dtype="float32"),
            lambda p: p["source"].update(revision="a" * 40),
            lambda p: p["components"].pop("vae_encoder"),
            lambda p: p.update(id="qmi-mobile-unknown-v1"),
        ]
        for mutate in mutations:
            profile = copy.deepcopy(self.profile())
            mutate(profile)
            with self.assertRaises(ValueError):
                validate_mobile_profile(profile)


if __name__ == "__main__":
    unittest.main()
