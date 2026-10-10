import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from src.skygrid_ai.language.large_bpe import LargeByteBPE, EOS_ID
from src.skygrid_ai.language.modern_transformer import SkyGridTransformerV08
from src.skygrid_ai.language.qa_pilot import (
    load_protocol, encode_supervised, prepare_examples, select_evidence,
    digest_files,
)
from src.skygrid_ai.language.train_qa import collate, evaluate_answers, masked_loss, train_qa
from src.skygrid_ai.language.evaluate_qa import load_qa_checkpoint


class TestSupervisedAviationPilot(unittest.TestCase):
    def setUp(self):
        self.tok = LargeByteBPE()
        self.config = {'context_size': 192, 'd_model': 16, 'n_heads': 2,
                       'n_layers': 1, 'hidden_width': 32}
        self.rows = {
            'train': [{'id': 'training', 'kind': 'grounded',
                       'question': 'What does the heartbeat do?',
                       'evidence': 'The heartbeat advertises the existence of a system.',
                       'answer': 'It announces a system on the network.'},
                      {'id': 'refusal', 'kind': 'unsupported',
                       'question': 'Does heartbeat guarantee safe flight?',
                       'evidence': 'No supporting aviation source is available.',
                       'answer': 'The available evidence does not establish that.'}],
            'val': [{'id': 'val', 'kind': 'grounded',
                     'question': 'How does PX4 allocate controls?',
                     'evidence': 'Control allocation translates thrust and torque into actuator commands.',
                     'answer': 'It converts commands for motors or servos.'}],
            'test': [{'id': 'test', 'kind': 'unsupported',
                      'question': 'What is the universal safe voltage?',
                      'evidence': 'No supporting aviation source is available.',
                      'answer': 'The available evidence does not establish that.'}],
        }
        torch.manual_seed(0)
        model = SkyGridTransformerV08(vocab_size=self.tok.vocab_size, **self.config)
        self.start = {'model_state': model.state_dict(), 'model_config': self.config,
                      'data_digests': {'train':'a','val':'b'},
                      'format_version': 'skygrid_v09c_1', 'step': 10000}

    def test_document_disjoint_protocol(self):
        rows = load_protocol()
        docs = [(x['source'], x['path']) for split in rows.values()
                for x in split if x['kind'] == 'grounded']
        self.assertEqual(len(docs), len(set(docs)))
        self.assertEqual([len(rows[k]) for k in ('train', 'val', 'test')], [16, 4, 4])

    def test_reject_duplicate_protocol_question(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orig = json.loads(Path('data/qa/aviation_v15_protocol.json').read_text())
            orig['splits']['val'].append(orig['splits']['train'][0])
            p = root/'bad.json';p.write_text(json.dumps(orig))
            with self.assertRaisesRegex(ValueError, 'more than one split'):
                load_protocol(p)

    def test_reject_review_metadata_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            raw = json.loads(Path('data/qa/aviation_v15_protocol.json').read_text())
            raw['review_status'] = 'automatically_approved'
            p=Path(td)/'bad.json';p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, 'review'):
                load_protocol(p)

    def test_mask_excludes_prompt_and_padding(self):
        x, y = encode_supervised(self.rows['train'][0], self.tok, 192)
        self.assertGreater(y.count(-100), 10)
        self.assertEqual(y[-1], EOS_ID)
        self.assertEqual(len(x), len(y))
        xx, yy = collate(self.rows['train'], self.tok, 192)
        self.assertEqual(xx.shape, yy.shape)
        self.assertTrue(bool((yy == -100).any().item()))
        self.assertTrue(torch.isfinite(masked_loss(
            SkyGridTransformerV08(vocab_size=self.tok.vocab_size, **self.config), xx, yy)))

    def test_answer_is_never_clipped(self):
        long = dict(self.rows['train'][0], answer='very long answer ' * 100)
        with self.assertRaisesRegex(ValueError, 'Reference answer'):
            encode_supervised(long, self.tok, 192)

    def test_invalid_source_sentence_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no complete prose'):
            select_evidence('heartbeat', 'hello', '# only heading')

    def test_resume_matches_continuous_training_and_protects_weights(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td)/'continuous', Path(td)/'resume'
            train_qa(self.rows, self.tok, self.start, 'source_sha', 'dataset_sha',
                     output=a, steps=4, batch_size=1, eval_every=2)
            train_qa(self.rows, self.tok, self.start, 'source_sha', 'dataset_sha',
                     output=b, steps=2, batch_size=1, eval_every=2)
            train_qa(self.rows, self.tok, self.start, 'source_sha', 'dataset_sha',
                     output=b, steps=4, batch_size=1, eval_every=2, resume=True)
            left = torch.load(a/'last.pt', weights_only=True)
            right = torch.load(b/'last.pt', weights_only=True)
            self.assertEqual(left['step'], 4)
            for key in left['model_state']:
                self.assertTrue(torch.equal(left['model_state'][key], right['model_state'][key]))
            self.assertEqual(left['best_qa_val_loss'], right['best_qa_val_loss'])
            with self.assertRaisesRegex(FileExistsError, 'already exists'):
                train_qa(self.rows, self.tok, self.start, 'source_sha', 'dataset_sha',
                         output=a, steps=5)
            with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                train_qa(self.rows, self.tok, self.start, 'source_sha', 'changed',
                         output=b, steps=5, batch_size=1, eval_every=2, resume=True)

    def test_qa_checkpoint_provenance_guard(self):
        with tempfile.TemporaryDirectory() as td:
            train_qa(self.rows, self.tok, self.start, 'sha1', 'sha2',
                     output=td, steps=1, batch_size=1, eval_every=1)
            path=Path(td)/'best.pt'
            cp=load_qa_checkpoint(path,self.tok,self.start,'sha1','sha2')
            self.assertEqual(cp['format_version'], 'skygrid_qa_v15_1')
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                load_qa_checkpoint(path,self.tok,self.start,'WRONG','sha2')

    def test_data_file_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'protocol.json';p.write_text('hello')
            before=digest_files([p]);p.write_text('goodbye')
            self.assertNotEqual(before,digest_files([p]))

    def test_validation_only_scores_reference_targets(self):
        m=SkyGridTransformerV08(vocab_size=self.tok.vocab_size,**self.config)
        loss, n=evaluate_answers(m,self.rows['val'],self.tok,192)
        _, labels=encode_supervised(self.rows['val'][0],self.tok,192)
        self.assertEqual(n,sum(y!=-100 for y in labels))
        self.assertGreater(loss,0)

if __name__ == '__main__':
    unittest.main()
