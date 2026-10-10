import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skygrid_ai.language import auto_qa_v18 as mod


class TestAutoQA(unittest.TestCase):
    def test_content_question_not_filename_question(self):
        q = mod.question_from_sentence('The heartbeat protocol is used to advertise a MAVLink system.', 'mavlink')
        self.assertEqual(q, 'In MAVLINK, what is heartbeat protocol used for?')
        self.assertIsNone(mod.question_from_sentence('It provides all settings.', 'px4'))
        self.assertIsNone(mod.question_from_sentence('A random statement.', 'px4'))

    def test_one_source_document_one_weak_example(self):
        docs = [dict(source='mavlink', path='en/heartbeat.md',
                     sha256_cleaned='s', url='https://example.com')]
        def fake_doc(*args, **kwargs):
            return ('The heartbeat protocol is used to advertise a MAVLink system.\n\n'
                    'The heartbeat protocol allows a station to detect communication.')
        def fake_answer(index, question, corpus=None):
            return {'status': 'excerpt_found',
                    'answer': 'The heartbeat protocol is used to advertise a MAVLink system.',
                    'citations': [{'source':'mavlink', 'path':'en/heartbeat.md', 'url':'https://example.com'}]}
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/'manifest.json').write_text(json.dumps(docs), encoding='utf-8')
            with patch.object(mod, 'load_verified_document', side_effect=fake_doc), patch.object(mod, 'answer_from_evidence', side_effect=fake_answer), patch.object(mod, 'sha256_file', return_value='hash'):
                data = mod.create_weak_dataset(maximum=40, corpus=folder, index=object())
        self.assertEqual(sum(map(len, data['splits'].values())), 1)
        row = next(r for group in data['splits'].values() for r in group)
        self.assertEqual(row['evidence'], row['answer'])
        self.assertEqual(row['kind'], 'weak_extractive')
        self.assertNotIn('approved', row)

    def test_question_only_retrieval_rejects_wrong_doc(self):
        docs = [dict(source='mavlink', path='en/heartbeat.md', sha256_cleaned='s', url='https://example.com')]
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/'manifest.json').write_text(json.dumps(docs), encoding='utf-8')
            with patch.object(mod, 'load_verified_document', return_value='The heartbeat protocol is used to advertise a MAVLink system.'), patch.object(mod, 'answer_from_evidence', return_value={'status':'excerpt_found', 'answer':'Some other source sentence.', 'citations':[{'source':'px4','path':'other.md'}]}), patch.object(mod, 'sha256_file', return_value='hash'):
                data = mod.create_weak_dataset(maximum=40, corpus=folder, index=object())
        self.assertEqual(sum(map(len, data['splits'].values())), 0)

    def test_refuses_training_with_too_few_weak_examples(self):
        with patch.object(mod, 'load_dataset', return_value={'train': [], 'val': [], 'test': []}):
            with self.assertRaisesRegex(ValueError, 'Not enough automatic'):
                mod.run_training('missing.json')

    def test_no_automatic_human_review_approval(self):
        self.assertNotIn('review_template', mod.main.__code__.co_names)
        self.assertIn('accept-unreviewed-weak-labels', Path(mod.__file__).read_text())


if __name__ == '__main__':
    unittest.main()
