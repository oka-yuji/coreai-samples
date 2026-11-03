import unittest

from Tools.Conversion.adapters.jev27b.typed_head import band, render
from Tools.Conversion.adapters.jev27b.typed_input import input_rows


def encode(text):
    return list(text)


ROWS = [{"id": "a", "kind": "noul", "state": "s", "question": "q", "options": ["false", "true"], "target": [0, 1]},
        {"id": "b", "kind": "choice", "state": "", "question": "pick", "options": ["x", "y", "z"], "label": 2}]


def entries(rows=ROWS):
    counts = [len(render(row["kind"], row["state"], row["question"], row["options"])) for row in rows]
    return [{"id": row["id"], "tokens": count, "band": band(count)} for row, count in zip(rows, counts)]


class TypedInputTests(unittest.TestCase):
    def test_rows_keep_only_the_request_fields_in_fixture_order(self):
        self.assertEqual(input_rows(entries(), ROWS, encode),
                         [{key: row[key] for key in ("id", "kind", "state", "question", "options")} for row in ROWS])

    def test_rows_that_differ_from_the_fixture_are_refused(self):
        changes = [lambda items: items[0].update(tokens=items[0]["tokens"] + 1),
                   lambda items: items[1].update(band="513-2048"),
                   lambda items: items[1].update(id="c"),
                   lambda items: items.pop()]
        for change in changes:
            broken = entries()
            change(broken)
            with self.assertRaises(ValueError):
                input_rows(broken, ROWS, encode)


if __name__ == "__main__":
    unittest.main()
