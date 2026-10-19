import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Tools.ModelReports.reports import ROOT, check, complete_record, load, read, reference, render, summarize, template, validate


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.catalog, cls.models = load(ROOT)

    def setUp(self):
        self.model = copy.deepcopy(self.models[0])

    def test_repository_reports_are_current_and_linked(self):
        check(ROOT, self.spec, self.catalog, self.models)

    def test_verified_demo_requires_actual_matching_native_execution(self):
        for field, value in [('level', 'conversion'), ('level', 'diagnostic'), ('model_executed', False),
                             ('platform', 'ios'), ('outcome', 'not_measured'), ('device', None),
                             ('device', 'iPhone Simulator'), ('variant', 'another variant')]:
            with self.subTest(field=field, value=value):
                model = copy.deepcopy(self.model)
                model['evidence']['app'][field] = value
                with self.assertRaisesRegex(ValueError, 'Verified demo|Invalid context'):
                    validate(ROOT, self.spec, model)

    def test_empty_or_nontext_context_and_variant_cannot_be_verified(self):
        for field in ['variant', 'device', 'os', 'toolchain', 'backend', 'route', 'power', 'thermal']:
            for value in ['', ' ', True, 123]:
                with self.subTest(field=field, value=value):
                    model = copy.deepcopy(self.model)
                    model['evidence']['app'][field] = value
                    with self.assertRaisesRegex(ValueError, 'Invalid context'):
                        validate(ROOT, self.spec, model)

    def test_failed_evidence_cannot_establish_a_pass(self):
        self.model['checks']['fidelity']['state'] = 'passed'
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            validate(ROOT, self.spec, self.model)

    def test_generated_blocks_and_registry_cannot_be_primary_evidence(self):
        for path in ['docs/qwen3.8-27b.md#performance-and-quality',
                     'docs/qwen3.8-27b.md#model-report', 'docs/qwen3.8-27b.md',
                     'docs/model-reports/teleocr.md', 'docs/model-reports/records/qwen38-27b.json',
                     'README.md#models-and-roadmap', 'README.ja.md#モデルとロードマップ']:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'primary evidence'):
                reference(ROOT, path, evidence=True)
        reference(ROOT, 'docs/qwen3.8-27b.md#key-numbers', evidence=True)

    def test_numeric_contract_rejects_invalid_values_and_units(self):
        for field, value in [('value', float('nan')), ('value', float('inf')), ('value', True),
                             ('value', -1), ('unit', 's'), ('metric', 'invented_metric')]:
            with self.subTest(field=field, value=value):
                model = copy.deepcopy(self.model)
                model['measurements'][0][field] = value
                with self.assertRaises(ValueError):
                    validate(ROOT, self.spec, model)

    def test_probability_range_and_unbounded_character_error_rate_are_distinct(self):
        row = self.model['measurements'][0]
        row.update(metric='task_success', value=1.2, unit='ratio')
        with self.assertRaisesRegex(ValueError, 'Out of range'):
            validate(ROOT, self.spec, self.model)
        row['metric'] = 'cer'
        validate(ROOT, self.spec, self.model)

    def test_percentile_requires_sufficient_samples(self):
        self.model['measurements'][0].update(statistic='p95', n=19)
        with self.assertRaisesRegex(ValueError, 'p95'):
            validate(ROOT, self.spec, self.model)

    def test_unknown_measurement_evidence_is_rejected(self):
        self.model['measurements'][0]['evidence'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'Unknown measurement evidence'):
            validate(ROOT, self.spec, self.model)

    def test_legacy_missing_count_is_preserved_without_upgrading_context(self):
        self.model['measurements'][0]['n'] = None
        validate(ROOT, self.spec, self.model)
        self.model['evidence']['bundle']['record_type'] = 'standard'
        with self.assertRaisesRegex(ValueError, 'Incomplete standard context'):
            validate(ROOT, self.spec, self.model)

    def test_json_source_value_mismatch_is_rejected(self):
        model = copy.deepcopy(next(m for m in self.models if m['id'] == 'embeddinggemma2'))
        row = next(r for r in model['measurements'] if 'source_value' in r)
        row['value'] += 1
        with self.assertRaisesRegex(ValueError, 'differs from source JSON'):
            validate(ROOT, self.spec, model)

    def test_plan_cannot_be_promoted_to_a_model_check(self):
        model = copy.deepcopy(next(m for m in self.models if m['id'] == 'gliner25'))
        model['checks']['quality'].update(state='passed', evidence=['product'])
        with self.assertRaisesRegex(ValueError, 'model-free'):
            validate(ROOT, self.spec, model)

    def test_private_paths_and_non_https_references_are_rejected(self):
        for target in ['../private.md', '/tmp/private.md', '.local/result.json', 'file:///tmp/data',
                       'https://name:secret@example.com/result', 'docs/no-such-report.md',
                       'docs/benchmarking.md#no-such-heading']:
            with self.subTest(target=target), self.assertRaises(ValueError):
                reference(ROOT, target)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'record.json'
            path.write_text('{"value": 1, "value": 2}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                read(path)

    def test_aliases_cannot_overwrite_another_models_page(self):
        models = copy.deepcopy(self.models)
        models[1]['pages']['en'] = 'docs/./qwen3.8-27b.md'
        with patch('Tools.ModelReports.reports.read', side_effect=[self.spec, self.catalog, *models]):
            with self.assertRaisesRegex(ValueError, 'Duplicate model or page'):
                load(ROOT)

    def test_generation_preserves_narrative_and_check_detects_manual_table_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['README.md', 'README.ja.md', *self.model['pages'].values()]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                marker = 'model-catalog' if name.startswith('README') else 'model-report'
                path.write_text(f'Original setup\n<!-- {marker}:start -->\n<!-- {marker}:end -->\nOriginal limitations\n')
            render(root, self.spec, self.catalog, [self.model], write=True)
            for path in self.model['pages'].values():
                text = (root / path).read_text()
                self.assertTrue(text.startswith('Original setup\n'))
                self.assertTrue(text.endswith('Original limitations\n'))
            path = root / 'README.md'
            path.write_text(path.read_text().replace('<!-- model-catalog:end -->', 'Changed table\n<!-- model-catalog:end -->'))
            with self.assertRaisesRegex(ValueError, 'Stale generated content'):
                check(root, self.spec, self.catalog, [self.model])

    def test_incomplete_historical_report_is_not_current_protocol_completion(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete current-protocol'):
            complete_record(self.spec, self.model)

    def test_completion_cannot_mix_variants_to_fill_missing_metrics(self):
        model = copy.deepcopy(self.model)
        source = model['evidence']['app']
        source.update(record_type='standard', plan_sha256='a' * 64)
        model['evidence']['other'] = copy.deepcopy(source)
        model['evidence']['other']['variant'] = 'another variant'
        required = self.spec['common'] + self.spec['profiles'][model['kind']]
        model['measurements'] = [{'metric': metric, 'evidence': 'app'} for metric in required]
        for item in model['checks'].values():
            item.update(state='passed', evidence=['app'])
        complete_record(self.spec, model)
        model['measurements'][0]['evidence'] = 'other'
        with self.assertRaisesRegex(ValueError, 'single pinned variant'):
            complete_record(self.spec, model)

    def test_template_is_a_plan_with_all_selected_metric_units(self):
        result = template(self.spec, 'tts', 'new-voice')
        self.assertIn('rtf', result['metrics'])
        self.assertIsNone(result['context']['device'])
        self.assertNotIn('plan_sha256', result['context'])
        self.assertIn('not evidence', result['note'])


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read(ROOT / 'docs/model-reports/metrics.json')

    def rows(self, count=20):
        context = dict(model='test', variant='fp16', device='test device', os='test OS',
                       workload='fixed fixture', cache='prepared', plan_sha256='a' * 64)
        return [dict(run_id=str(i), context=copy.deepcopy(context), status='passed',
                     metrics={'latency_s': {'value': i + 1, 'unit': 's'}}) for i in range(count)]

    def aggregate(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runs.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            return summarize(path, self.spec)

    def test_nearest_rank_and_median_exclude_but_report_failed_and_warmup_runs(self):
        rows = self.rows()
        warmup, failed = copy.deepcopy(rows[0]), copy.deepcopy(rows[0])
        warmup.update(run_id='warmup', status='warmup')
        failed.update(run_id='failed', status='failed')
        warmup['metrics']['latency_s']['value'] = 1000
        failed['metrics'] = {}
        result = self.aggregate([warmup, *rows, failed])
        self.assertEqual(result['counts'], dict(passed=20, failed=1, warmup=1))
        self.assertEqual(result['metrics']['latency_s'], dict(unit='s', n=20, min=1, median=10.5, mean=10.5, p95=19, max=20))

    def test_small_sample_has_no_claimed_percentile(self):
        metric = self.aggregate(self.rows(3))['metrics']['latency_s']
        self.assertEqual(metric['median'], 2)
        self.assertIsNone(metric['p95'])

    def test_new_memory_runs_use_bytes_instead_of_rounded_legacy_units(self):
        rows = self.rows()
        for row in rows:
            row['metrics'] = {'peak_rss_bytes': {'value': 3.1, 'unit': 'GB'}}
        with self.assertRaisesRegex(ValueError, 'canonical unit'):
            self.aggregate(rows)

    def test_hardware_workload_cache_and_plan_must_match(self):
        for key in ['variant', 'device', 'os', 'workload', 'cache', 'plan_sha256']:
            rows = self.rows()
            rows[-1]['context'][key] += 'different'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Mixed model'):
                self.aggregate(rows)

    def test_context_fields_are_nonempty_strings(self):
        for field,value in [('device',123),('workload',True),('os',' ')]:
            rows = self.rows()
            for row in rows:
                row['context'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Incomplete run identity'):
                self.aggregate(rows)

    def test_success_cannot_silently_omit_a_slow_metric(self):
        rows = self.rows()
        rows[-1]['metrics'] = {}
        with self.assertRaisesRegex(ValueError, 'Missing metrics'):
            self.aggregate(rows)

    def test_nonfinite_values_are_rejected_even_in_failed_runs(self):
        rows = self.rows()
        rows[-1]['status'] = 'failed'
        rows[-1]['metrics']['latency_s']['value'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid numeric'):
            self.aggregate(rows)

    def test_missing_identity_duplicate_runs_and_empty_successes_are_rejected(self):
        rows = self.rows()
        rows[1]['run_id'] = rows[0]['run_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate run'):
            self.aggregate(rows)
        rows = self.rows()
        for row in rows:
            row['context'].pop('plan_sha256')
        with self.assertRaisesRegex(ValueError, 'Incomplete run identity'):
            self.aggregate(rows)
        rows = self.rows()
        for row in rows:
            row['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'No successful'):
            self.aggregate(rows)


if __name__ == '__main__':
    unittest.main()
