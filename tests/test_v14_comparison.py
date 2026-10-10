"""Comparison safeguards: no test-set evaluation, no silent stale reports."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.language.compare_v10 import (
    compare_reports, run_comparison, validate_report,
)
from src.skygrid_ai.language.large_bpe import LargeByteBPE


METHOD = ('All next-token targets within each document; reset context at EOS; '
          'exclude first token of each document')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def report(*, step, checkpoint, ckpt_hash, val_hash, fingerprint, loss):
    return {
        'version': 'skygrid_v09d_full_documents_v1',
        'split': 'val', 'checkpoint': checkpoint, 'training_step': step,
        'checkpoint_sha256': ckpt_hash, 'dataset_sha256': val_hash,
        'tokenizer_fingerprint': fingerprint, 'evaluation_method': METHOD,
        'evaluated_targets': 200, 'scored_utf8_bytes_excluding_first_token_per_document': 120,
        'cross_entropy_nats_per_token': loss,
        'perplexity': 2.718281828459045 ** loss,
        'bits_per_scored_byte_including_eos_cost': 2.0,
    }


class TestV14Comparison(unittest.TestCase):
    def test_lower_loss_means_better_but_not_aviation_knowledge(self):
        a = report(step=5000, checkpoint='last', ckpt_hash='a', val_hash='v',
                   fingerprint='f', loss=2.4)
        b = report(step=1000, checkpoint='last', ckpt_hash='b', val_hash='v',
                   fingerprint='f', loss=2.2)
        result = compare_reports(a, b)
        self.assertAlmostEqual(result['loss_difference_v10_minus_v09c'], -0.2)
        self.assertIn('not a factual aviation', result['caveat'])

    def test_noncomparable_samples_are_rejected(self):
        a = report(step=5000, checkpoint='last', ckpt_hash='a', val_hash='v',
                   fingerprint='f', loss=2.4)
        b = dict(a, evaluated_targets=201)
        with self.assertRaisesRegex(ValueError, 'Not comparable'):
            compare_reports(a, b)
        b = dict(a, split='test')
        with self.assertRaisesRegex(ValueError, 'Not comparable'):
            compare_reports(a, b)

    def test_stale_report_fails_closed(self):
        x = report(step=5000, checkpoint='last', ckpt_hash='old', val_hash='v',
                   fingerprint='f', loss=2.4)
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            validate_report(x, checkpoint_hash='new', val_hash='v',
                            tokenizer_fingerprint='f', checkpoint_name='last',
                            training_step=5000)
        with self.assertRaisesRegex(ValueError, 'split'):
            validate_report(dict(x, split='test'), checkpoint_hash='old', val_hash='v',
                            tokenizer_fingerprint='f', checkpoint_name='last',
                            training_step=5000)

    def setup_files(self, root, *, include_report=True):
        data = root / 'data' / 'aviation' / 'tokenized_v09b'
        data.mkdir(parents=True)
        # Use the real tokenizer implementation and its actual fingerprint.
        tokenizer = LargeByteBPE()
        tokenizer.save(data / 'tokenizer.json')
        for split in ('train', 'val'):
            (data / (split + '.bin')).write_bytes(b'actual dataset ' + split.encode())
        hashes = {s: sha((data / (s + '.bin')).read_bytes()) for s in ('train', 'val')}
        fp = tokenizer.fingerprint()
        old = root / 'checkpoints' / 'v09c' / 'last.pt'
        new = root / 'checkpoints' / 'v10' / 'last.pt'
        old.parent.mkdir(parents=True)
        new.parent.mkdir(parents=True)
        for filename, step in ((old, 5000), (new, 20)):
            torch.save({'format_version': 'skygrid_v09c_1', 'step': step,
                        'vocab_size': tokenizer.vocab_size,
                        'tokenizer_fingerprint': fp, 'data_digests': hashes}, filename)
        baseline = report(step=5000, checkpoint='last', ckpt_hash=sha(old.read_bytes()),
                          val_hash=hashes['val'], fingerprint=fp, loss=2.4)
        reference = root / 'checkpoints' / 'v09d' / 'val_last_step5000.json'
        reference.parent.mkdir(parents=True)
        reference.write_text(json.dumps(baseline), encoding='utf-8')
        contender = report(step=20, checkpoint='last', ckpt_hash=sha(new.read_bytes()),
                           val_hash=hashes['val'], fingerprint=fp, loss=4.9)
        result_path = root / 'checkpoints' / 'v10_evaluation' / 'val_last_step20.json'
        if include_report:
            result_path.parent.mkdir(parents=True)
            result_path.write_text(json.dumps(contender), encoding='utf-8')
        return contender, result_path, data

    def test_reuses_untampered_report_without_rerunning(self):
        with tempfile.TemporaryDirectory() as folder:
            self.setup_files(Path(folder))
            def fail_evaluator(*a, **kw):
                raise AssertionError('Should reuse the existing report')
            result = run_comparison(root=folder, evaluator=fail_evaluator)
            self.assertTrue(result['cached_report_reused'])
            self.assertAlmostEqual(result['loss_difference_v10_minus_v09c'], 2.5)

    def test_only_validation_is_evaluated_and_report_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            contender, expected, _ = self.setup_files(Path(folder), include_report=False)
            def fake_evaluator(split, name, **kw):
                self.assertEqual(split, 'val')
                self.assertEqual(name, 'last')
                self.assertEqual(kw['training_output'], Path(folder) / 'checkpoints' / 'v10')
                expected.parent.mkdir(parents=True)
                expected.write_text(json.dumps(contender), encoding='utf-8')
                return contender, expected
            result = run_comparison(root=folder, evaluator=fake_evaluator)
            self.assertFalse(result['cached_report_reused'])
            self.assertTrue(expected.exists())

    def test_changed_model_rejects_stale_cached_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            self.setup_files(Path(folder))
            ckpt = Path(folder) / 'checkpoints' / 'v10' / 'last.pt'
            payload = torch.load(ckpt, map_location='cpu', weights_only=True)
            payload['step'] = 21
            torch.save(payload, ckpt)
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                # cached step20 is not consulted; no step21 report is allowed
                def malformed_eval(*a, **kw):
                    return {}, Path(folder) / 'unexpected.json'
                run_comparison(root=folder, evaluator=malformed_eval)

    def test_mismatched_data_disables_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, data = self.setup_files(Path(folder))
            (data / 'val.bin').write_bytes(b'changed validation data')
            with self.assertRaisesRegex(ValueError, 'data no longer matches'):
                run_comparison(root=folder)


if __name__ == '__main__':
    unittest.main()
