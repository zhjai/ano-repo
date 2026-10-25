"""Saved EOS representation must preserve exactly one native termination ID."""
import json
from pathlib import Path
import tempfile
import unittest

from castfactory.training.full_agent_sft_v4 import read_export_receipt


class ReceiptEOSTests(unittest.TestCase):
    def test_native_eos_representation_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'fixture')
            receipt = {
                'status': 'full_parameter_sft_completed',
                'profile_version': 'merged-t1-answer-v1',
                'parameterization': 'full', 'all_parameters_trainable': True,
                'model_path': str(root),
                'probe_ids': {role: [2] for role in ('T1', 'T2', 'T3', 'T4')},
                'export_reload_max_abs_difference': {role: 0 for role in ('T1', 'T2', 'T3', 'T4')},
                'generation_eos': {'eos_token_id': 2, 'source': 'effective_chat_tokenizer'},
            }
            path = root / 'receipt.json'
            path.write_text(json.dumps(receipt))
            for saved in (2, [2]):
                with self.subTest(accepted=saved):
                    (root / 'generation_config.json').write_text(json.dumps({'eos_token_id': saved}))
                    self.assertEqual(read_export_receipt(path), receipt)
            for saved in (3, [3], [2, 3], [2, 2], [], [[2]], True, [True], 2.0, [2.0], None, '2', -1):
                with self.subTest(rejected=saved):
                    (root / 'generation_config.json').write_text(json.dumps({'eos_token_id': saved}))
                    with self.assertRaisesRegex(ValueError, 'generation EOS'):
                        read_export_receipt(path)
            # The receipt itself must continue to be an actual tokenizer ID.
            for invalid in (True, [2], 2.0, -1):
                with self.subTest(receipt_eos=invalid):
                    receipt['generation_eos']['eos_token_id'] = invalid
                    path.write_text(json.dumps(receipt))
                    (root / 'generation_config.json').write_text('{"eos_token_id": 2}')
                    with self.assertRaisesRegex(ValueError, 'generation EOS'):
                        read_export_receipt(path)

if __name__ == '__main__':
    unittest.main()
