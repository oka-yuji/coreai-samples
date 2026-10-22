import copy
import unittest

from Tools.Conversion.adapters.qwen_image21.mobile_profile import CANDIDATES, comparison_candidate, make_profile, validate_mobile_profile


class ImageMobileProfileTests(unittest.TestCase):
    def profile(self, candidate="w4"):
        return make_profile(candidate, "a" * 64, {"mobile_models.py": "b" * 64})

    def test_sensitive_block_candidates_bind_the_exact_dense_weight_set(self):
        for candidate, blocks in (("w4-blocks20-21-f32", {20, 21}), ("w4-blocks16-23-f32", set(range(16, 24)))):
            profile = self.profile(candidate)
            encoding = profile["components"]["denoiser"]["weight_encoding"]
            self.assertEqual(len(encoding["dense_modules"]), 7 * len(blocks))
            self.assertEqual({int(name.split(".")[1]) for name in encoding["dense_modules"]}, blocks)
            self.assertEqual(comparison_candidate(profile), "mixed-f32")
            self.assertEqual(profile["components"]["denoiser"]["outputs"][0]["shape"], [1, 2304, 64])
            for mutate in (lambda values: values.pop(), lambda values: values.append(values[0]), lambda values: values.reverse()):
                altered = copy.deepcopy(profile)
                mutate(altered["components"]["denoiser"]["weight_encoding"]["dense_modules"])
                with self.assertRaises(ValueError):
                    validate_mobile_profile(altered)

    def test_full_precision_denoiser_keeps_compressed_storage_and_has_a_matching_dense_counterpart(self):
        compressed = self.profile("w4-f32")
        dense = self.profile("mixed-f32")
        for profile in (compressed, dense):
            validate_mobile_profile(profile)
            self.assertEqual(comparison_candidate(profile), "mixed-f32")
            self.assertTrue(all(value["compute_dtype"] == "float32" for value in profile["components"].values()))
            self.assertEqual(profile["workload"]["scheduler_output_dtype"], "float32")
            self.assertEqual(profile["components"]["denoiser"]["weight_encoding"]["other_parameters_dtype"], "float32")
            self.assertEqual(profile["components"]["denoiser"]["weight_encoding"]["time_frequency_dtype"], "float32")
        encoding = compressed["components"]["denoiser"]["weight_encoding"]
        self.assertEqual((encoding["bits"], encoding["group_size"], encoding["storage_dtype"]), (4, 64, "float16"))
        self.assertEqual(dense["components"]["denoiser"]["weight_encoding"]["kind"], "dense-linear")
        self.assertEqual(comparison_candidate(self.profile("w4")), "mixed")
        self.assertNotEqual(validate_mobile_profile(compressed), validate_mobile_profile(self.profile("w4")))

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
