import unittest

from src.skygrid_ai.knowledge.grounded import answer_from_evidence, queried_sources
from src.skygrid_ai.knowledge.retrieval import SearchIndex
from src.skygrid_ai.knowledge.benchmark import load_questions


class TestGroundedExtraction(unittest.TestCase):
    def index(self):
        return SearchIndex([
            {'id': 'hb:0', 'source': 'mavlink', 'path': 'services/heartbeat.md',
             'url': 'https://example.com/mavlink/heartbeat',
             'text': 'MAVLink HEARTBEAT announces component presence on a network and helps detect connection loss.'},
            {'id': 'px4:0', 'source': 'px4', 'path': 'concept/control_allocation.md',
             'url': 'https://example.com/px4/allocation',
             'text': 'PX4 control allocation translates desired thrust and torque into commands for motors and servos.'},
            {'id': 'px4:1', 'source': 'px4', 'path': 'data_links/telemetry.md',
             'url': 'https://example.com/px4/telemetry',
             'text': 'PX4 connects to QGroundControl through a MAVLink telemetry radio link.'},
        ])

    def test_returns_verbatim_sentence_and_exact_url(self):
        result = answer_from_evidence(self.index(), 'What does MAVLink HEARTBEAT announce?')
        self.assertEqual(result['status'], 'excerpt_found')
        self.assertEqual(result['answer'], self.index().passages[0]['text'])
        self.assertEqual(result['citations'][0]['url'], 'https://example.com/mavlink/heartbeat')
        self.assertIn('NOT independently checked', result['warning'])

    def test_unknown_question_abstains(self):
        result = answer_from_evidence(self.index(), 'quasar banana nebula')
        self.assertEqual(result['status'], 'abstained')
        self.assertIsNone(result['answer'])

    def test_wrong_system_is_not_silently_answered(self):
        result = answer_from_evidence(self.index(), 'How does MAVLink perform control allocation?')
        self.assertEqual(result['status'], 'source_mismatch')
        self.assertIsNone(result['answer'])
        self.assertIn('px4', result['reason'])

    def test_multiple_named_systems_do_not_force_one_source(self):
        result = answer_from_evidence(self.index(), 'How does PX4 connect to QGroundControl over MAVLink?')
        self.assertEqual(result['status'], 'excerpt_found')
        self.assertIn('telemetry radio', result['answer'])

    def test_subject_identification_is_boundary_safe(self):
        self.assertEqual(queried_sources('PX4 and MAVLink telemetry'), ['mavlink', 'px4'])
        self.assertEqual(queried_sources('mavlinker systems'), [])

    def test_reject_bad_inputs(self):
        for value in ['', '   ']:
            with self.assertRaises(ValueError):
                answer_from_evidence(self.index(), value)
        with self.assertRaises(ValueError):
            answer_from_evidence(self.index(), 'hello', top_k=0)

    def test_independent_reference_set_does_not_repeat_development_ids(self):
        original = {r['id'] for r in load_questions()}
        additional = load_questions('data/benchmarks/aviation_v11_independent.json')
        self.assertGreaterEqual(len(additional), 10)
        self.assertFalse(original.intersection({r['id'] for r in additional}))
        self.assertEqual(len(additional), len({r['path'] for r in additional}))


if __name__ == '__main__':
    unittest.main()
