import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skygrid_ai.language.qa_v16 import VERSION, hash_json, review_template
from src.skygrid_ai.language.review_qa_v16 import (
    counts, eligible_evidence, load_queue, record_decision, run_review,
    save_reviews,
)


def item(identifier='one', split='train', kind='grounded'):
    return {'id': identifier, 'split': split, 'kind': kind,
            'source': 'mavlink', 'path': 'en/services/heartbeat.md',
            'url': 'https://example.org/original',
            'question': 'What does MAVLink HEARTBEAT do?',
            'answer': 'It advertises a component on the network.'}


class FakeIndex:
    pass


class TestV17Review(unittest.TestCase):
    def prepare(self, folder, rows=None):
        if rows is None:
            rows = [item()]
        bundle = {'version': VERSION, 'candidates': rows}
        reviews = review_template(bundle)
        candidate_path, review_path = Path(folder) / 'c.json', Path(folder) / 'r.json'
        candidate_path.write_text(json.dumps(bundle), encoding='utf-8')
        review_path.write_text(json.dumps(reviews), encoding='utf-8')
        return candidate_path, review_path, bundle, reviews

    def test_empty_template_is_pending(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, _, _ = self.prepare(d, [item(), item('two', 'val')])
            bundle, record = load_queue(c, r)
            self.assertEqual(counts(bundle, record)['approved_total'], 0)
            self.assertEqual(counts(bundle, record)['by_split']['val']['pending'], 1)

    def test_candidate_modification_blocks_review(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, bundle, _ = self.prepare(d)
            bundle['candidates'][0]['answer'] = 'modified'
            c.write_text(json.dumps(bundle), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_queue(c, r)

    def test_explicit_approval_and_no_silent_reapproval(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, _, _ = self.prepare(d)
            bundle, record = load_queue(c, r)
            with self.assertRaisesRegex(ValueError, 'requires'):
                record_decision(record, 'one', 'approved', 'Reviewer')
            record_decision(record, 'one', 'approved', 'Reviewer', checked=True)
            self.assertEqual(counts(bundle, record)['approved_total'], 1)
            with self.assertRaisesRegex(ValueError, 'already reviewed'):
                record_decision(record, 'one', 'rejected', 'Reviewer')

    def test_reject_persists_and_resuming_skips_decided(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, _, _ = self.prepare(d)
            bundle, record = load_queue(c, r)
            original = r.read_bytes()
            record_decision(record, 'one', 'rejected', 'Reviewer', reason='Wrong claim')
            save_reviews(r, record, previous_bytes=original)
            bundle, again = load_queue(c, r)
            self.assertEqual(counts(bundle, again)['by_split']['train']['rejected'], 1)
            self.assertTrue((Path(d) / 'r.json.bak').exists())

    def test_concurrent_change_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, _, _ = self.prepare(d)
            bundle, record = load_queue(c, r)
            previous = r.read_bytes()
            r.write_text('unexpected', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                save_reviews(r, record, previous_bytes=previous)
            self.assertEqual(r.read_text(encoding='utf-8'), 'unexpected')

    def test_source_mismatch_preflight(self):
        with patch('src.skygrid_ai.language.review_qa_v16.answer_from_evidence',
                   return_value={'status': 'source_mismatch', 'citations': []}):
            excerpt, reason = eligible_evidence(item(), FakeIndex())
            self.assertIsNone(excerpt)
            self.assertIn('did not produce', reason)
        with patch('src.skygrid_ai.language.review_qa_v16.answer_from_evidence',
                   return_value={'status': 'excerpt_found', 'answer': 'wrong doc',
                                 'citations': [{'source': 'px4', 'path': 'elsewhere.md'}]}):
            excerpt, reason = eligible_evidence(item(), FakeIndex())
            self.assertIsNone(excerpt)
            self.assertIn('another document', reason)

    def test_unsupported_without_live_evidence_is_skipped(self):
        with patch('src.skygrid_ai.language.review_qa_v16.answer_from_evidence',
                   return_value={'status': 'abstained'}):
            excerpt, reason = eligible_evidence(item(kind='unsupported'), FakeIndex())
            self.assertIsNone(excerpt)
            self.assertIn('already refuses', reason)

    def test_no_autoapproval_when_confirmations_not_all_yes(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, _, _ = self.prepare(d)
            bundle, record = load_queue(c, r)
            answers = iter(['a', 'yes', 'yes', 'no'])
            with patch('src.skygrid_ai.language.review_qa_v16.eligible_evidence',
                       return_value=('Relevant excerpt.', None)), \
                 patch('src.skygrid_ai.language.review_qa_v16.load_verified_document',
                       return_value='source body'):
                count = run_review(bundle, record, reviews_path=r, reviewer='Reviewer',
                                   input_fn=lambda _: next(answers), print_fn=lambda _: None,
                                   index=FakeIndex())
            self.assertEqual(count, 0)
            self.assertEqual(counts(bundle, record)['approved_total'], 0)
            self.assertEqual(load_queue(c, r)[1]['reviews']['one']['approved'], False)

    def test_explicit_interactive_approval_saves_same_review_schema(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, _, _ = self.prepare(d)
            bundle, record = load_queue(c, r)
            answers = iter(['a', 'yes', 'yes', 'yes'])
            with patch('src.skygrid_ai.language.review_qa_v16.eligible_evidence',
                       return_value=('Relevant excerpt.', None)), \
                 patch('src.skygrid_ai.language.review_qa_v16.load_verified_document',
                       return_value='source body'):
                count = run_review(bundle, record, reviews_path=r, reviewer='Reviewer',
                                   input_fn=lambda _: next(answers), print_fn=lambda _: None,
                                   index=FakeIndex())
            self.assertEqual(count, 1)
            approved = load_queue(c, r)[1]['reviews']['one']
            self.assertEqual(approved['decision'], 'approved')
            self.assertTrue(all(approved[x] is True for x in
                                ('approved','source_checked','answer_supported','question_checked')))

    def test_existing_review_schema_not_modified_except_selected_record(self):
        with tempfile.TemporaryDirectory() as d:
            c, r, _, _ = self.prepare(d, [item(), item('two', 'val')])
            bundle, record = load_queue(c, r)
            untouched = dict(record['reviews']['two'])
            record_decision(record, 'one', 'approved', 'Reviewer', checked=True)
            self.assertEqual(record['reviews']['two'], untouched)
            self.assertEqual(record['candidate_sha256'], hash_json(bundle))


if __name__ == '__main__':
    unittest.main()
