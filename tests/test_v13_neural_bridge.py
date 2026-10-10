"""Unit and real-small-model checks for the experimental neural bridge."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from src.skygrid_ai.knowledge.neural_bridge import (
    answer_with_model, load_local_model_assets, make_prompt,
)
from src.skygrid_ai.knowledge.retrieval import SearchIndex
from src.skygrid_ai.language.large_bpe import LargeByteBPE, write_token_ids
from src.skygrid_ai.language.modern_transformer import SkyGridTransformerV08
from src.skygrid_ai.language.train_large_transformer import sha256_file


class ToyTokenizer:
    def encode(self, content):
        return list(content.encode('utf-8'))


class TestNeuralBridge(unittest.TestCase):
    def prepare(self, directory):
        folder = Path(directory)
        (folder / 'docs').mkdir(parents=True, exist_ok=True)
        docs = [
            ('mavlink', 'en/services/heartbeat.md', 'hb',
             'The heartbeat protocol is used to advertise the existence of a system on the MAVLink network, along with component identifier and flight mode.'),
            ('px4', 'en/concept/control_allocation.md', 'px',
             'PX4 takes desired torque and thrust commands from the core controllers and translates them to actuator commands which control motors or servos.'),
        ]
        entries, manifest = [], []
        for source, path, file, sentence in docs:
            text = sentence + '\n'
            (folder / 'docs' / (file + '.txt')).write_text(text, encoding='utf-8')
            url = 'https://example.org/' + source + '/' + file
            manifest.append({'source': source, 'path': path, 'text_file': 'docs/' + file + '.txt',
                             'url': url,
                             'sha256_cleaned': hashlib.sha256(' '.join(text.split()).casefold().encode()).hexdigest()})
            entries.append({'id': source + ':0', 'source': source, 'path': path,
                            'url': url, 'text': sentence})
        (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return SearchIndex(entries)

    def test_prompt_fits_small_context_without_half_characters(self):
        tok = ToyTokenizer()
        question = 'How does MAVLink HEARTBEAT work with telemetry systems and drones?'
        excerpt = ('The heartbeat protocol advertises systems and components through regular messages. ' * 50)
        out = make_prompt(question, excerpt, tok, 128)
        self.assertLessEqual(len(tok.encode(out)), 127)
        self.assertIn('Question:', out)
        self.assertIn('Evidence:', out)
        self.assertTrue(out.endswith('\nAnswer:'))

    def test_verified_source_and_unverified_model_continuation_are_separate(self):
        with tempfile.TemporaryDirectory() as td:
            index = self.prepare(td)
            loader = Mock(return_value=(ToyTokenizer(), {'model_config': {'context_size': 256}, 'step': 5000}))
            generator = Mock(side_effect=lambda prompt, **kwargs: prompt + ' Predicted experimental continuation.')
            out = answer_with_model(index, 'What does MAVLink HEARTBEAT do?', corpus=td,
                                    model_loader=loader, generator=generator, max_new_tokens=12)
            self.assertEqual(out['evidence_status'], 'excerpt_found')
            self.assertIn('advertise the existence', out['source_excerpt'])
            self.assertEqual(out['citations'][0]['url'], 'https://example.org/mavlink/hb')
            self.assertEqual(out['neural_draft_status'], 'generated_unverified')
            self.assertEqual(out['neural_draft'], 'Predicted experimental continuation.')
            self.assertNotIn('experimental', out['source_excerpt'].lower())
            self.assertEqual(out['model']['training_step'], 5000)
            self.assertEqual(generator.call_args.kwargs['max_new_tokens'], 12)

    def test_refusal_disables_model_even_if_checkpoint_absent(self):
        with tempfile.TemporaryDirectory() as td:
            index = self.prepare(td)
            loader = Mock(side_effect=AssertionError('model must never load'))
            generator = Mock(side_effect=AssertionError('must not generate'))
            for question, status in [('How does MAVLink perform control allocation?', 'source_mismatch'),
                                     ('unknown nebula orbit sensor banana', 'abstained')]:
                result = answer_with_model(index, question, corpus=td,
                                           model_loader=loader, generator=generator)
                self.assertEqual(result['evidence_status'], status)
                self.assertEqual(result['neural_draft_status'], 'not_generated')
                self.assertIsNone(result['neural_draft'])
            loader.assert_not_called()
            generator.assert_not_called()

    def test_fingerprint_tampering_blocks_generation(self):
        with tempfile.TemporaryDirectory() as td:
            index = self.prepare(td)
            (Path(td) / 'docs/hb.txt').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                answer_with_model(index, 'What does MAVLink HEARTBEAT do?', corpus=td,
                                  model_loader=Mock(side_effect=AssertionError('must not load')))

    def test_reject_malformed_generator_return_and_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            index = self.prepare(td)
            loader = lambda **k: (ToyTokenizer(), {'model_config': {'context_size': 256}, 'step': 4})
            with self.assertRaisesRegex(ValueError, 'does not preserve'):
                answer_with_model(index, 'What does MAVLink HEARTBEAT do?', corpus=td,
                                  model_loader=loader, generator=lambda prompt, **kw: 'wrong')
            with self.assertRaisesRegex(ValueError, 'sampling'):
                answer_with_model(index, 'What does MAVLink HEARTBEAT do?', corpus=td,
                                  max_new_tokens=0)
            with self.assertRaisesRegex(ValueError, 'model'):
                answer_with_model(index, 'What does MAVLink HEARTBEAT do?', corpus=td,
                                  model_version='nonexistent')

    def test_real_tiny_checkpoint_works_and_detects_changed_data(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            data = base / 'data' / 'aviation' / 'tokenized_v09b'
            data.mkdir(parents=True)
            tok = LargeByteBPE()
            tok.save(data / 'tokenizer.json')
            write_token_ids(data / 'train.bin', [10, 11, 12, 257] * 50)
            write_token_ids(data / 'val.bin', [10, 13, 14, 257] * 50)
            config = {'context_size': 64, 'd_model': 16, 'n_heads': 4,
                      'n_layers': 1, 'hidden_width': 32}
            model = SkyGridTransformerV08(vocab_size=tok.vocab_size, **config)
            ckpt_dir = base / 'checkpoints' / 'v09c'
            ckpt_dir.mkdir(parents=True)
            checkpoint = {
                'format_version': 'skygrid_v09c_1', 'step': 2,
                'vocab_size': tok.vocab_size,
                'tokenizer_fingerprint': tok.fingerprint(),
                'data_digests': {name: sha256_file(data / (name + '.bin')) for name in ('train','val')},
                'model_config': config, 'model_state': model.state_dict(),
            }
            torch.save(checkpoint, ckpt_dir / 'last.pt')
            loaded_tok, loaded_ckpt = load_local_model_assets(root=base)
            self.assertEqual(loaded_ckpt['step'], 2)
            self.assertEqual(loaded_tok.vocab_size, tok.vocab_size)
            index = self.prepare(base / 'source')
            out = answer_with_model(index, 'What does MAVLink HEARTBEAT do?',
                                    corpus=base / 'source', root=base,
                                    max_new_tokens=4, top_k_tokens=3)
            self.assertEqual(out['evidence_status'], 'excerpt_found')
            self.assertIn(out['neural_draft_status'], ('empty_continuation', 'generated_unverified'))
            write_token_ids(data / 'val.bin', [9, 257] * 50)
            with self.assertRaisesRegex(ValueError, 'dataset mismatch'):
                load_local_model_assets(root=base)

    def test_missing_checkpoint_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, 'Missing v10 checkpoint'):
                load_local_model_assets(root=td, model_version='v10')


if __name__ == '__main__':
    unittest.main()
