"""Offline tests for native adaptive-context benchmark (no Ollama/GPU required)."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = load_script('eval_sprint8_ollama_models')
CREATOR = load_script('create_sprint8_ollama_context_variants')

ROW = {
    'id': 'nli4ct:6', 'source': 'nli4ct',
    'state': 'Clinical trial evidence', 'question': 'Does the evidence support the statement?',
    'options': [
        {'key': 'entailment', 'description': 'Supported'},
        {'key': 'contradiction', 'description': 'Refuted'},
    ],
    'answer_key': 'entailment',
}


def failure(tokens, limit=2048, status=400):
    detail = json.dumps({'error': f'prompt 0 has {tokens} tokens; expected 1–{limit} (input is never truncated)'})
    return HTTPError('http://localhost/v1/systemone', status, 'failed', {}, io.BytesIO(detail.encode()))


def success():
    body = {'answers': {'decision': {'type': 'choice', 'choice': 'entailment',
                                     'confidence': 0.9, 'probabilities': {'entailment': 0.95}}}}
    return io.BytesIO(json.dumps(body).encode())


class EvaluationTests(unittest.TestCase):
    def make_caller(self, replies):
        names = []
        sequence = iter(replies)

        def request(req, timeout):
            data = json.loads(req.data)
            names.append(data['model'])
            item = next(sequence)
            if isinstance(item, Exception):
                raise item
            return item
        return names, request

    def test_short_prompt_uses_original_only(self):
        names, caller = self.make_caller([success()])
        with patch.object(EVAL.urllib.request, 'urlopen', side_effect=caller):
            row = EVAL.query('http://localhost', 'biojev-systemone:4b', ROW, 30, '15m')
        self.assertEqual(names, ['biojev-systemone:4b'])
        self.assertEqual(row['attempts'], 1)
        self.assertEqual(row['context_limit_tokens'], 2048)
        self.assertEqual(row['prediction'], 'entailment')
        self.assertEqual(row['error'], '')

    def test_oversized_prompt_retries_optional_tag(self):
        names, caller = self.make_caller([failure(3099), success()])
        with patch.object(EVAL.urllib.request, 'urlopen', side_effect=caller):
            row = EVAL.query('http://localhost', 'biojev-systemone:9b', ROW, 30, '15m')
        self.assertEqual(names, ['biojev-systemone:9b', 'biojev-systemone:9b-4k'])
        self.assertEqual(row['attempts'], 2)
        self.assertEqual(row['overflow_tokens'], 3099)
        self.assertEqual(row['context_limit_tokens'], 4096)
        self.assertEqual(row['error'], '')
        result = EVAL.summary('biojev-systemone:9b', 'ALL', [dict(row, gold_key='entailment')])
        self.assertEqual(result['n_extended'], 1)

    def test_too_long_avoids_futile_retry(self):
        names, caller = self.make_caller([failure(5000)])
        with patch.object(EVAL.urllib.request, 'urlopen', side_effect=caller):
            row = EVAL.query('http://localhost', 'biojev-systemone:0.8b', ROW, 30, '15m')
        self.assertEqual(names, ['biojev-systemone:0.8b'])
        self.assertEqual(row['attempts'], 1)
        self.assertIn('not retried', row['error'])

    def test_non_context_error_is_not_retried(self):
        bad = HTTPError('http://localhost/v1/systemone', 400, 'bad request', {}, io.BytesIO(b'{"error":"bad criteria"}'))
        names, caller = self.make_caller([bad])
        with patch.object(EVAL.urllib.request, 'urlopen', side_effect=caller):
            row = EVAL.query('http://localhost', 'biojev-systemone:4b', ROW, 30, '15m')
        self.assertEqual(names, ['biojev-systemone:4b'])
        self.assertIn('bad criteria', row['error'])

    def test_no_adaptive_context(self):
        names, caller = self.make_caller([failure(2173)])
        with patch.object(EVAL.urllib.request, 'urlopen', side_effect=caller):
            row = EVAL.query('http://localhost', 'biojev-systemone:4b', ROW, 30, '15m', adaptive_context=False)
        self.assertEqual(names, ['biojev-systemone:4b'])
        self.assertIn('2173 tokens', row['error'])

    def test_missing_variant_reports_error(self):
        missing = HTTPError('http://localhost/v1/systemone', 404, 'not found', {}, io.BytesIO(b'{"error":"model not found"}'))
        names, caller = self.make_caller([failure(2102), missing])
        with patch.object(EVAL.urllib.request, 'urlopen', side_effect=caller):
            row = EVAL.query('http://localhost', 'biojev-systemone:4b', ROW, 30, '15m')
        self.assertEqual(names, ['biojev-systemone:4b', 'biojev-systemone:4b-4k'])
        self.assertIn('create_sprint8_ollama_context_variants.py', row['error'])

    def test_context_parser(self):
        self.assertEqual(CREATOR.context_setting('FROM foo\nPARAMETER num_ctx 2048\n'), 2048)
        self.assertEqual(CREATOR.context_setting('PARAMETER num_ctx 2048\nPARAMETER num_ctx 4096\n'), 4096)
        self.assertIsNone(CREATOR.context_setting('FROM foo\n'))


if __name__ == '__main__':
    unittest.main()
