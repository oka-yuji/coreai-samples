import unittest
from unittest import mock

from Tools.OCREval import omnidocbench
from Tools.OCREval.omnidocbench import crop_box, html_cells, select


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


if __name__ == "__main__":
    unittest.main()
