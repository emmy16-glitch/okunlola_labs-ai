import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from src.skygrid_ai.knowledge.retrieval import (
    SearchIndex, build_passages, save_index, load_index, split_passages,
)
from src.skygrid_ai.knowledge.benchmark import (
    load_questions, evaluate_retrieval, evaluate_candidate_answers,
)


class TestV10Knowledge(unittest.TestCase):
    def sample(self):
        return SearchIndex([
            {'id': 'a', 'source': 'mavlink', 'path': 'en/services/heartbeat.md',
             'url': 'https://github.com/a', 'text': 'MAVLink heartbeat advertises system presence and detects disconnected components.'},
            {'id': 'b', 'source': 'px4', 'path': 'en/concept/control_allocation.md',
             'url': 'https://github.com/b', 'text': 'PX4 control allocation translates thrust and torque into actuator motor commands.'},
        ])

    def test_citation_and_deterministic_ranking(self):
        idx = self.sample()
        a = idx.search('How does heartbeat advertise connected system?')
        self.assertEqual(a, idx.search('How does heartbeat advertise connected system?'))
        self.assertEqual(a[0]['id'], 'a')
        self.assertTrue(a[0]['url'].startswith('https://'))

    def test_topic_filename_relevance_is_general_not_a_source_override(self):
        # Distractors repeat generic query terms, but the authoritative topic
        # is identified by its own filename and has real matching text.
        passages = [
            {'id': f'd{i}', 'source': 'docs', 'path': f'other_{i}.md',
             'url': f'https://example.org/other_{i}',
             'text': 'aircraft tracking navigation system data ' * 5}
            for i in range(7)
        ]
        passages.append({
            'id': 'topic', 'source': 'docs', 'path': 'manual/signal.md',
            'url': 'https://example.org/signal',
            'text': 'Signal configuration is documented here for aircraft.',
        })
        matches = SearchIndex(passages).search('aircraft tracking signal navigation system', limit=3)
        self.assertIn('topic', [item['id'] for item in matches])

    def test_document_diversity_and_body_evidence_required(self):
        rows = [
            {'id': f'a:{i}', 'source': 'docs', 'path': 'guide/topic.md',
             'url': 'https://example.org/topic', 'text': 'topic sensor calibration details ' * 3}
            for i in range(6)
        ]
        rows.append({'id': 'b:0', 'source': 'docs', 'path': 'guide/sensor.md',
                     'url': 'https://example.org/sensor', 'text': 'sensor calibration explanation'})
        rows.append({'id': 'c:0', 'source': 'docs', 'path': 'guide/other.md',
                     'url': 'https://example.org/other', 'text': 'entirely unrelated material'})
        matches = SearchIndex(rows).search('topic sensor calibration', limit=5)
        self.assertLessEqual(sum(x['path'] == 'guide/topic.md' for x in matches), 2)
        self.assertIn('b:0', [x['id'] for x in matches])
        self.assertNotIn('c:0', [x['id'] for x in matches])

    def test_no_evidence_abstains(self):
        self.assertEqual(self.sample().evidence('quasar pineapple orbit')['status'], 'abstained')

    def test_no_generated_answers(self):
        result = self.sample().evidence('PX4 thrust torque actuators')
        self.assertEqual(result['status'], 'evidence_found')
        self.assertIn('text', result['evidence'][0])
        self.assertNotIn('answer', result)

    def test_source_fingerprint_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'docs').mkdir()
            sample = 'MAVLink heartbeat advertises system presence.'
            (root / 'docs' / 'doc.txt').write_text(sample, encoding='utf-8')
            row = {'text_file': 'docs/doc.txt', 'source': 'mavlink', 'path': 'en/services/heartbeat.md',
                   'url': 'https://example.org', 'sha256_cleaned': hashlib.sha256(sample.casefold().encode()).hexdigest()}
            (root / 'manifest.json').write_text(json.dumps([row]), encoding='utf-8')
            rows, digest = build_passages(root)
            self.assertEqual(len(rows), 1)
            save_index(rows, digest, root / 'index.gz')
            self.assertEqual(len(load_index(root / 'index.gz', root).passages), 1)
            (root / 'docs' / 'doc.txt').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                build_passages(root)
            (root / 'manifest.json').write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                load_index(root / 'index.gz', root)

    def test_benchmark_source_recall(self):
        items = [{'id': 'hb', 'question': 'heartbeat system presence', 'source': 'mavlink',
                  'path': 'en/services/heartbeat.md', 'reference': 'presence', 'key_terms': ['heartbeat', 'presence']}]
        result = evaluate_retrieval(self.sample(), items)
        self.assertEqual(result['source_recall_at_k'], 1.0)
        self.assertEqual(result['mean_reference_term_coverage'], 1.0)

    def test_candidate_check_not_overclaim(self):
        item = [{'id': 'a', 'key_terms': ['heartbeat', 'component']}]
        result = evaluate_candidate_answers(item, {'a': 'heartbeat'})
        self.assertEqual(result['mean_reference_term_coverage'], 0.5)
        self.assertIn('not factual correctness', result['warning'])
        abstain = evaluate_candidate_answers(item, {'a': 'abstain'})
        self.assertEqual(abstain['abstention_rate'], 1.0)

    def test_benchmark_valid(self):
        rows = load_questions()
        self.assertGreaterEqual(len(rows), 8)
        self.assertEqual(len(rows), len({r['id'] for r in rows}))

    def test_chunk_overlap(self):
        chunks = split_passages(' '.join(str(x) for x in range(20)), size=8, overlap=2)
        self.assertEqual(chunks[0].split()[-2:], chunks[1].split()[:2])
        with self.assertRaises(ValueError):
            split_passages('test', size=5, overlap=5)


if __name__ == '__main__':
    unittest.main()
