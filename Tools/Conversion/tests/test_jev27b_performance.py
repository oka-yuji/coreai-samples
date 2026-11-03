import json
from pathlib import Path
import random
import tempfile
import unittest

from Tools.Conversion.adapters.jev27b.fixtures import Builder, Sources
from Tools.Conversion.adapters.jev27b.performance import (
    LENGTHS, ROUNDS, build, cancel_runs, latency_runs, load_runs, memory_run, schedule, used, write_lines,
)
from Tools.ModelReports.reports import ROOT as REPORTS_ROOT, load, summarize

KINDS = {"noul": ["false", "true"], "choice": ["a", "b", "c"], "score": ["0", "1", "2", "3", "4", "5"]}


def encode(text, add_special_tokens=False):
    return {"input_ids": list(text.encode("utf-8"))}


def corpus(folder, count=240):
    rng = random.Random(5)
    rows = []
    for index in range(count):
        kind = ("noul", "choice", "score")[index % 3]
        words = " ".join("".join(rng.choice("abcdefghij") for _ in range(5)) for _ in range(60))
        rows.append({"id": f"row{index:03d}", "kind": kind, "state": words, "question": f"question {index}",
                     "options": KINDS[kind], "target": [1 / len(KINDS[kind])] * len(KINDS[kind]),
                     "family": f"family{index % 4}", "source": "yuri_v1" if index == 7 else "corpus"})
    (folder / "test_set_30k.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixtures(folder):
    f = [{"id": "row000", "source": "test_set_30k", "parts": None},
         {"id": "synthetic-long:x:0", "source": "synthetic-long", "parts": ["row001", "row002"]}]
    (folder / "F.json").write_text(json.dumps({"rows": f}))
    (folder / "Q.json").write_text(json.dumps({"rows": [{"id": "row003", "source": "test_set_30k", "parts": None}]}))


class RowTests(unittest.TestCase):
    def test_p_rows_have_exact_lengths_and_share_no_part_with_f_q_or_each_other(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus(root)
            fixtures(root)
            excluded = used(root)
            self.assertEqual(excluded, {"row000", "row001", "row002", "row003"})
            sources = Sources(root)
            entries = build(Builder(sources, encode), sources, excluded)
            again = build(Builder(Sources(root), encode), Sources(root), excluded)
        self.assertEqual(entries, again)
        self.assertEqual([(entry["tokens"], entry["kind"]) for entry in entries], list(LENGTHS))
        parts = [part for entry in entries for part in entry["parts"]]
        self.assertEqual(len(parts), len(set(parts)))
        self.assertFalse(set(parts) & (excluded | {"row007"}))
        self.assertTrue(all(entry["id"] == f"performance:tokens{entry['tokens']}" for entry in entries))

    def test_the_schedule_interleaves_the_lengths_after_one_warmup_each(self):
        rows = [{"id": f"p{tokens}"} for tokens, _ in LENGTHS]
        ids = [row["id"] for row in schedule(rows)]
        self.assertEqual(ids[:4], ["p100@warmup", "p500@warmup", "p2000@warmup", "p8192@warmup"])
        self.assertEqual(ids[4:8], ["p100@1", "p500@1", "p2000@1", "p8192@1"])
        self.assertEqual(len(ids), len(rows) * (ROUNDS + 1))
        self.assertEqual(len(set(ids)), len(ids))


class RecordTests(unittest.TestCase):
    PLAN = {"model_id": "jev27b", "context": {"variant": "v", "device": "d", "os": "o", "power": "AC", "thermal": "t"}}
    SHA = "a" * 64

    def manifest(self):
        return {"rows": [{"id": f"performance:tokens{tokens}", "tokens": tokens, "kind": kind,
                          "tokens_sha256": f"{tokens:064d}"} for tokens, kind in LENGTHS]}

    def latency(self):
        rows = []
        for label in ["warmup"] + [str(index) for index in range(1, ROUNDS + 1)]:
            for tokens, _ in LENGTHS:
                rows.append({"id": f"performance:tokens{tokens}@{label}", "token_count": tokens,
                             "tokens_sha256": f"{tokens:064d}", "elapsed_seconds": tokens / 200 + int(label != "warmup"),
                             "resident_bytes_max": {"rss_bytes": 10 * tokens, "footprint_bytes": tokens, "samples": 4}})
        return {"rows": rows}

    def test_latency_runs_summarize_per_length_with_warmups_and_failures(self):
        document = self.latency()
        document["rows"][-1]["tokens_sha256"] = "0" * 64
        runs = latency_runs(self.PLAN, self.SHA, self.manifest(), document)
        self.assertEqual(sorted(runs), [tokens for tokens, _ in LENGTHS])
        self.assertEqual([row["status"] for row in runs[8192]], ["warmup", "passed", "passed", "failed"])
        self.assertEqual(runs[100][1]["run_id"], "latency:100:1")
        spec = load(REPORTS_ROOT)[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latency-100.jsonl"
            write_lines(path, runs[100])
            summary = summarize(path, spec)
        self.assertEqual(summary["counts"], {"passed": ROUNDS, "failed": 0, "warmup": 1})
        self.assertEqual(summary["metrics"]["latency_s"]["n"], ROUNDS)
        self.assertIsNone(summary["metrics"]["latency_s"]["p95"])
        self.assertEqual(summary["metrics"]["latency_s"]["median"], 1.5)

    def test_loads_pass_only_with_the_expected_specialization(self):
        documents = [{"bundle": {"specialization": state, "load_seconds": {"bundle": 1, "tokenizer": 0.5, "head": 0.25,
                                                                          "backbone": 20}}}
                     for state in ("fresh", "cached", "fresh")]
        runs = load_runs(self.PLAN, self.SHA, documents, "fresh", "first_load_s")
        self.assertEqual([row["status"] for row in runs], ["passed", "failed", "passed"])
        self.assertEqual(runs[0]["metrics"]["first_load_s"], {"value": 21.75, "unit": "s"})

    def test_memory_and_cancellation_records(self):
        runs = latency_runs(self.PLAN, self.SHA, self.manifest(), self.latency())
        text = "   12.5 real   1.0 user   2.0 sys\n  55000  maximum resident set size\n  777  peak memory footprint\n"
        memory = memory_run(self.PLAN, self.SHA, text, runs)[0]["metrics"]
        self.assertEqual({name: item["value"] for name, item in memory.items()},
                         {"peak_rss_bytes": 55000, "peak_footprint_bytes": 777, "resident_bytes": 81920})
        items = [{"attempt": 1, "tokens": 2000, "entered": True, "cancelled": True, "request_to_error_seconds": 8.0,
                  "start_to_error_seconds": 10.0},
                 {"attempt": 2, "tokens": 2000, "entered": True, "cancelled": False, "request_to_error_seconds": 7.0,
                  "start_to_error_seconds": 9.0}]
        cancels = cancel_runs(self.PLAN, self.SHA, {"cancellations": items})
        self.assertEqual([row["status"] for row in cancels], ["passed", "failed"])
        self.assertEqual(cancels[0]["metrics"]["cancel_s"], {"value": 8.0, "unit": "s"})


if __name__ == "__main__":
    unittest.main()
