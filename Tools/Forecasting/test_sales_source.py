import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sales_source import LIMIT, acquire


class Response(io.BytesIO):
    status = 200


class SourceTests(unittest.TestCase):
    def test_unknown_bytes_never_create_an_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "sales.csv"
            with self.assertRaises(ValueError):
                acquire(output, lambda *args, **kwargs: Response(b"synthetic invalid source"))
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_oversized_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "sales.csv"
            with self.assertRaises(ValueError):
                acquire(output, lambda *args, **kwargs: Response(b"x" * (LIMIT + 1)))
            self.assertFalse(output.exists())

    def test_synthetic_verified_bytes_publish_once_and_keep_existing_files(self):
        data = b"synthetic verifier input, no actual sales"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "sales.csv"
            with patch("sales_source.BYTE_COUNT", len(data)), patch("sales_source.SHA256", hashlib.sha256(data).hexdigest()):
                summary = acquire(output, lambda *args, **kwargs: Response(data))
            self.assertEqual(summary["bytes"], len(data))
            self.assertEqual(output.read_bytes(), data)
            self.assertEqual(list(Path(folder).iterdir()), [output])
            with self.assertRaises(FileExistsError):
                acquire(output, lambda *args, **kwargs: self.fail("Existing output must not start a request"))
            self.assertEqual(output.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
