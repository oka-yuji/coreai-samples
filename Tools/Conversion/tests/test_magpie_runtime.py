import unittest

import numpy as np

from Tools.Conversion.adapters.magpie.runtime import validate_inputs


class MagpieRuntimeContractTests(unittest.TestCase):
    def test_encoder_rejects_out_of_vocabulary_ids(self):
        with self.assertRaises(ValueError):
            validate_inputs("encoder", (np.array([[0, 24]], dtype=np.int32), np.ones((1, 2), dtype=np.float32)), 24)

    def test_local_rejects_unknown_codebooks_and_excess_length(self):
        for length, codebook in [(1, -1), (1, 16), (17, 0)]:
            with self.subTest(length=length, codebook=codebook), self.assertRaises(ValueError):
                validate_inputs("local", (np.ones((2, length, 768), dtype=np.float32), np.array([codebook], dtype=np.int32)))

    def test_codec_rejects_a_length_beyond_the_latent_sequence(self):
        with self.assertRaises(ValueError):
            validate_inputs("codec", (np.ones((1, 32, 8), dtype=np.float32), np.array([9], dtype=np.int32)))

    def test_decoder_rejects_a_zero_attention_prior(self):
        inputs = (np.ones((2, 5, 768), dtype=np.float32), np.ones((2, 5), dtype=np.float32),
                  np.ones((2, 4, 768), dtype=np.float32), np.ones((2, 4), dtype=np.float32),
                  np.zeros((2, 1, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            validate_inputs("decoder", inputs)

    def test_runtime_rejects_non_finite_inputs_and_wrong_dtypes(self):
        for hidden, index in [(np.full((2, 1, 768), np.nan, dtype=np.float32), np.array([0], dtype=np.int32)),
                              (np.ones((2, 1, 768), dtype=np.float16), np.array([0], dtype=np.int32))]:
            with self.subTest(dtype=hidden.dtype), self.assertRaises(ValueError):
                validate_inputs("local", (hidden, index))

    def test_runtime_accepts_the_declared_local_boundaries(self):
        for length, codebook in [(1, 0), (16, 15)]:
            values = (np.ones((2, length, 768), dtype=np.float32), np.array([codebook], dtype=np.int32))
            self.assertEqual(validate_inputs("local", values)[0].shape, (2, length, 768))


if __name__ == "__main__":
    unittest.main()
