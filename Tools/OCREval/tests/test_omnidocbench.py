import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Tools.OCREval import omnidocbench
from Tools.OCREval.omnidocbench import crop_box, fetch, html_cells, select


def page(name, language="english", source="book", blocks=()):
    return {"page_info": {"image_path": name, "width": 100, "height": 200,
                          "page_attribute": {"language": language, "data_source": source}},
            "layout_dets": list(blocks)}


def text(anno_id, value, ignore=False):
    return {"category_type": "text_block", "anno_id": anno_id, "text": value, "ignore": ignore, "poly": [0, 0, 10, 0, 10, 5, 0, 5]}


def table(anno_id, html):
    return {"category_type": "table", "anno_id": anno_id, "html": html, "ignore": False, "poly": [0, 0, 10, 0, 10, 5, 0, 5]}


GRID = "<table><tr><td rowspan=2>a</td><td colspan=2>b</td></tr><tr><td>c</td><td>d</td></tr><tr><td>e</td><td>f</td><td>g</td></tr></table>"


class OmniDocBenchTests(unittest.TestCase):
    def test_spans_fill_every_covered_position(self):
        self.assertEqual(html_cells(GRID), [["a", "b", "b"], ["a", "c", "d"], ["e", "f", "g"]])

    def test_crop_box_adds_margin_and_clips(self):
        self.assertEqual(crop_box([1.5, 2.2, 98.1, 2.2, 98.1, 199.0, 1.5, 199.0], 100, 200), [0, 0, 100, 200])
        self.assertEqual(crop_box([20.5, 30.2, 40.1, 30.2, 40.1, 50.9, 20.5, 50.9], 100, 200), [16, 26, 45, 55])

    def test_selection_filters_orders_and_caps(self):
        long = "x" * 50
        pages = [
            page("p1.png", blocks=[text(1, long), text(2, long), table(3, GRID)]),
            page("p2.png", blocks=[text(1, long + "$"), text(2, "short"), text(3, long, ignore=True), text(4, long)]),
            page("p3.png", language="simplified_chinese", blocks=[text(1, long)]),
            page("p4.png", source="note", blocks=[text(1, long)]),
            page("p5.png", language="traditional_chinese", blocks=[text(1, long)]),
        ]
        rules = dict(omnidocbench.SPEC["rules"]) | {"text": dict(omnidocbench.SPEC["rules"]["text"], per_language=2)}
        with mock.patch.dict(omnidocbench.SPEC, {"rules": rules}):
            chosen = select(pages)
        picked = [(c["id"], c["info"]["image_path"], c["block"]["anno_id"]) for c in chosen]
        self.assertEqual([p[0] for p in picked], ["OD-table-en-01", "OD-text-en-01", "OD-text-en-02", "OD-text-zh-01"])
        self.assertEqual(picked[0][1:], ("p1.png", 3))
        self.assertEqual({p[1] for p in picked[1:3]}, {"p1.png", "p2.png"})
        self.assertEqual(next(p for p in picked if p[1] == "p2.png")[2], 4)
        self.assertEqual(chosen[0]["content"]["cells"], [["a", "b", "b"], ["a", "c", "d"], ["e", "f", "g"]])


class PinnedSubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "subset"
        self.pages = json.dumps([page("p1.png", blocks=[text(7, "x" * 50)])]).encode()
        self.image = b"image bytes"
        self.files = {"OmniDocBench.json": self.pages, "images/p1.png": self.image}
        self.spec = {
            "source": {"dataset": "owner/name", "revision": "rev", "annotation": "OmniDocBench.json",
                       "annotation_sha256": hashlib.sha256(self.pages).hexdigest()},
            "rules": omnidocbench.SPEC["rules"],
            "samples": [{"id": "OD-text-en-01", "image_path": "p1.png", "anno_id": 7,
                         "image_sha256": hashlib.sha256(self.image).hexdigest()}],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def urlopen(self, url, timeout=None):
        return io.BytesIO(self.files[url.split("/resolve/rev/", 1)[1]])

    def make(self, **changes):
        spec = json.loads(json.dumps(self.spec))
        for key, value in changes.items():
            target, field = key.split(".")
            (spec["source"] if target == "source" else spec["samples"][0])[field] = value
        with mock.patch.dict(omnidocbench.SPEC, spec, clear=True), \
                mock.patch.object(omnidocbench.urllib.request, "urlopen", self.urlopen):
            return omnidocbench.make(self.out)

    def test_matching_pins_write_the_local_set(self):
        manifest = self.make()
        gt = json.loads((self.out / "OD-text-en-01.json").read_text())
        self.assertIs(gt["redistribute"], False)
        self.assertEqual(gt["crop_px"], [0, 0, 14, 9])
        self.assertEqual(json.loads((self.out / "selection.json").read_text()), self.spec["samples"])
        self.assertEqual(json.loads((self.out / "manifest.json").read_text()), manifest)
        self.assertEqual([r["id"] for r in manifest["samples"]], ["OD-text-en-01"])

    def test_a_different_annotation_is_refused(self):
        with self.assertRaisesRegex(ValueError, "OmniDocBench.json: SHA-256"):
            self.make(**{"source.annotation_sha256": "0" * 64})
        self.assertEqual(list((self.out / "source").iterdir()), [])

    def test_a_different_selection_is_refused(self):
        with self.assertRaisesRegex(ValueError, "selection differs"):
            self.make(**{"sample.anno_id": 8})

    def test_a_different_image_is_refused(self):
        with self.assertRaisesRegex(ValueError, "images/p1.png: SHA-256"):
            self.make(**{"sample.image_sha256": "0" * 64})
        self.assertFalse((self.out / "p1.png").exists())

    def test_a_corrupt_local_copy_is_fetched_again(self):
        self.out.mkdir(parents=True)
        (self.out / "p1.png").write_bytes(b"corrupt")
        with mock.patch.dict(omnidocbench.SPEC, self.spec, clear=True), \
                mock.patch.object(omnidocbench.urllib.request, "urlopen", self.urlopen):
            digest = fetch("images/p1.png", self.out / "p1.png", self.spec["samples"][0]["image_sha256"])
        self.assertEqual((digest, (self.out / "p1.png").read_bytes()), (self.spec["samples"][0]["image_sha256"], self.image))


if __name__ == "__main__":
    unittest.main()
