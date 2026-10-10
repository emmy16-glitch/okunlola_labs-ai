import unittest
from src.skygrid_ai.language.qa_review import make_template, score_human_reviews
from src.skygrid_ai.language.evaluate_qa import generate_answer
from src.skygrid_ai.language.large_bpe import LargeByteBPE
from src.skygrid_ai.language.modern_transformer import SkyGridTransformerV08

class TestManualFactReview(unittest.TestCase):
    def make_report(self):
        return {'split':'val','qa_checkpoint_sha256':'abc123',
                'items':[{'id':'a','kind':'grounded'},{'id':'b','kind':'unsupported'}]}

    def test_template_is_not_prefilled_or_auto_scored(self):
        r=make_template(self.make_report())
        self.assertFalse(r['ratings']['a']['source_checked'])
        self.assertIsNone(r['ratings']['b']['qa'])

    def test_refuses_partial_or_unreviewed_grades(self):
        report=self.make_report();r=make_template(report)
        r['ratings']['a'].update(source_checked=True,baseline='incorrect',qa='correct')
        with self.assertRaisesRegex(ValueError,'Sources'):
            score_human_reviews(report,r)
        r['ratings']['b'].update(source_checked=True,baseline='incorrect',qa='invalid')
        with self.assertRaisesRegex(ValueError,'Complete factual grades'):
            score_human_reviews(report,r)

    def test_grades_are_factual_human_not_keyword(self):
        report=self.make_report();r=make_template(report)
        r['ratings']['a'].update(source_checked=True,baseline='incorrect',qa='correct')
        r['ratings']['b'].update(source_checked=True,baseline='abstained',qa='correct')
        score=score_human_reviews(report,r)
        self.assertEqual(score['correct_fraction']['qa'],1.0)
        self.assertEqual(score['correct_fraction']['baseline'],0.0)
        self.assertEqual(score['by_kind']['unsupported']['n'],1)

    def test_mismatched_report_rejected(self):
        report=self.make_report();r=make_template(report)
        r['source_report_sha256']='another checkpoint'
        with self.assertRaisesRegex(ValueError,'does not match'):
            score_human_reviews(report,r)

    def test_qa_preview_is_experimental(self):
        model=SkyGridTransformerV08(vocab_size=258,context_size=192)
        tok=LargeByteBPE()
        row={'question':'What does a heartbeat do?',
             'evidence':'It advertises vehicle network presence.',
             'answer':'It advertises existence.'}
        text=generate_answer(model,tok,row,192,limit=4)
        self.assertIsInstance(text,str)

if __name__=='__main__':unittest.main()
