"""Input errors should fail before any model is loaded."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples.demo import public_case
from scripts.predict import preflight_case
from scripts.train_reasoning import read_rows

ROOT = Path(__file__).resolve().parents[1]


class CLIPreflightTests(unittest.TestCase):
    def test_generic_case_and_context_budget(self):
        preflight_case(public_case(), json.loads((ROOT/'configs/inference.json').read_text()))

    def test_private_data_mismatched_profile_and_unknown_config(self):
        case = public_case()
        for update, config in (({'labels': [1]}, {}), ({'profile_version': 'wrong'}, {}),
                               ({}, {'profile_version': 'wrong'}), ({}, {'max_segments': 0}),
                               ({}, {'unexpected': 1})):
            with self.subTest(update=update, config=config):
                with self.assertRaises((ValueError, TypeError)):
                    preflight_case({**case, **update}, config)

    def test_compact_requires_both_assets(self):
        with self.assertRaisesRegex(ValueError, 'both compact'):
            preflight_case(public_case(), {}, public_directory=Path('unavailable'))
        with self.assertRaisesRegex(ValueError, 'requires public_json_case'):
            preflight_case(public_case(), {'profile_version': 'agentic-compact-markdown-v2'})

    def test_compact_asset_validation_precedes_model_import(self):
        case = {**public_case(), 'public_json_case': {}}
        with patch('castfactory.agentic.full_agent_v4.attach_public_json_case'), \
             patch('scripts.load_canonical.compact_view.validate_compact_profile_assets',
                   side_effect=ValueError('missing reviewed assets')):
            with self.assertRaisesRegex(ValueError, 'missing reviewed assets'):
                preflight_case(case, {'profile_version': 'agentic-compact-markdown-v2'})

    def test_invalid_config_cli_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'bad.json'
            config.write_text('{"profile_version":"unknown"}')
            result = subprocess.run([sys.executable, str(ROOT/'scripts/predict.py'),
                '--case', str(ROOT/'examples/public_case.json'), '--model', str(root/'no-model'),
                '--output', str(root/'out.json'), '--config', str(config)],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Unsupported Agent profile', result.stderr)
            self.assertNotIn('transformers', result.stderr)
            self.assertNotIn('torch', result.stderr)
            self.assertFalse((root/'out.json').exists())

    def test_reasoning_rows_require_explicit_matching_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.jsonl'
            row = {'input': 'load', 'output': '100', 'metadata': {'split': 'train'}}
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(read_rows(path, 'train'), [row])
            with self.assertRaisesRegex(ValueError, 'metadata.split=validation'):
                read_rows(path, 'validation')
            del row['metadata']
            path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'metadata.split=train'):
                read_rows(path, 'train')
