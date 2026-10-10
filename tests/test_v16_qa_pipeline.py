import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.knowledge.retrieval import SearchIndex
from src.skygrid_ai.language.large_bpe import LargeByteBPE, EOS_ID
from src.skygrid_ai.language.modern_transformer import SkyGridTransformerV08
from src.skygrid_ai.language.qa_v16 import (
    VERSION, split_for, hash_json, review_template, load_reviewed,
    prompt_v16, encode_v16,
)
from src.skygrid_ai.language.train_qa_v16 import fit, collate_v16, evaluate_v16
from src.skygrid_ai.language.evaluate_qa_v16 import human_review_template, score_manual, refusal_audit


class TestQA16(unittest.TestCase):
    def setUp(self):
        self.tok = LargeByteBPE()
        self.config = {'context_size':192, 'd_model':16, 'n_heads':2,
                       'n_layers':1, 'hidden_width':32}
        self.rows = {
            'train':[{'id':'train','question':'What does MAVLink heartbeat advertise?',
                      'evidence':'MAVLink heartbeat advertises a system on a network.',
                      'answer':'It announces a connected system.'},
                     {'id':'train2','question':'What does a protocol advertise?',
                      'evidence':'A protocol advertises the existence of a system on the network.',
                      'answer':'It advertises system presence.'}],
            'val':[{'id':'val','question':'How does PX4 control allocation work?',
                    'evidence':'PX4 translates thrust into commands for motors.',
                    'answer':'It maps thrust to actuators.'}],
            'test':[{'id':'test','question':'What is sensor telemetry?',
                     'evidence':'Telemetry sends sensor values to a ground station.',
                     'answer':'It sends measurements.'}],
        }
        model = SkyGridTransformerV08(vocab_size=self.tok.vocab_size, **self.config)
        self.source = {'model_state': model.state_dict(), 'model_config':self.config,
                       'data_digests':{'train':'t', 'val':'v'}, 'step':10000}

    def test_prompt_is_identical_in_training_and_inference(self):
        row = self.rows['train'][0]
        prompt = prompt_v16(row['question'],row['evidence'],self.tok,192)
        x, y = encode_v16(row,self.tok,192)
        prefix = self.tok.encode(prompt)
        self.assertEqual(x[:len(prefix)], prefix)
        self.assertTrue(all(target == -100 for target in y[:len(prefix)-1]))
        self.assertEqual(y[-1], EOS_ID)
        self.assertEqual(len(x),len(y))

    def test_too_long_answer_rejected_instead_of_silently_clipped(self):
        row = dict(self.rows['train'][0],answer='something different '*200)
        with self.assertRaisesRegex(ValueError,'too long'):
            encode_v16(row,self.tok,192)

    def test_split_deterministic(self):
        for p in ('en/services/heartbeat.md','en/services/ping.md'):
            self.assertEqual(split_for('mavlink',p),split_for('mavlink',p))
            self.assertIn(split_for('mavlink',p), ('train','val','test'))

    def test_human_review_blocks_unapproved_and_changed_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            folder=root/'source';folder.mkdir()
            doc='The heartbeat advertises a system to its peers on the MAVLink network.\n\nConnected components can discover each other using heartbeat messages.'
            (folder/'hb.txt').write_text(doc)
            sha=hashlib.sha256(' '.join(doc.split()).casefold().encode()).hexdigest()
            manifest=[{'source':'mavlink','path':'en/services/heartbeat.md','url':'https://example.org/hb',
                       'text_file':'hb.txt','sha256_cleaned':sha}]
            (folder/'manifest.json').write_text(json.dumps(manifest))
            entry={'id':'a','source':'mavlink','path':'en/services/heartbeat.md',
                   'source_sha256':sha,'url':'https://example.org/hb',
                   'question':'What does MAVLink heartbeat advertise?',
                   'answer':'It announces system presence.',
                   'split':split_for('mavlink','en/services/heartbeat.md'),
                   'kind':'grounded'}
            bundle={'version':VERSION,'candidates':[entry]}
            path=root/'candidates.json';path.write_text(json.dumps(bundle))
            rating=review_template(bundle)
            rp=root/'reviews.json';rp.write_text(json.dumps(rating))
            idx=SearchIndex([{'id':'hb:0','source':'mavlink','path':'en/services/heartbeat.md',
                              'url':'https://example.org/hb','text':doc}])
            splits,errors=load_reviewed(candidates=path,reviews=rp,corpus=folder,index=idx,strict=False)
            self.assertEqual(sum(map(len,splits.values())),0)
            rating['reviews']['a']={'approved':True,'reviewer':'Test reviewer',
                                    'source_checked':True,'answer_supported':True,
                                    'question_checked':True}
            rp.write_text(json.dumps(rating))
            splits,errors=load_reviewed(candidates=path,reviews=rp,corpus=folder,index=idx,strict=False)
            self.assertEqual(sum(map(len,splits.values())),1)
            actual=next(iter([x for group in splits.values() for x in group]))
            self.assertIn('heartbeat',actual['evidence'].lower())
            self.assertIn('advertises',actual['evidence'].lower())
            entry['answer']='Changed after review'
            path.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError,'changed since human review'):
                load_reviewed(candidates=path,reviews=rp,corpus=folder,index=idx,strict=False)

    def test_low_sample_size_training_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'Insufficient'):
                fit(self.rows,self.tok,self.source,'s','d',output=td,min_train=20)

    def test_early_stop_preserves_best_and_resume_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            result=fit(self.rows,self.tok,self.source,'source','data',output=td,
                       steps=5,eval_every=1,patience=1,min_delta=1000,min_train=1,batch_size=1)
            self.assertTrue(result['stopped_early'])
            self.assertLess(result['last_step'],5)
            best=torch.load(Path(td)/'best.pt',weights_only=True)
            self.assertEqual(best['step'],1)
            with self.assertRaisesRegex(ValueError,'Early-stopped'):
                fit(self.rows,self.tok,self.source,'source','data',output=td,
                    steps=6,eval_every=1,patience=1,min_delta=1000,min_train=1,batch_size=1,resume=True)

    def test_unfinished_manual_review_cannot_be_scored(self):
        report={'model_sha256':'sha','split':'val','items':[{'id':'a'}]}
        form=human_review_template(report)
        with self.assertRaisesRegex(ValueError,'Source has not'):
            score_manual(report,form)
        form['grades']['a']['source_checked']=True
        for model in ('baseline','qa'):
            form['grades']['a'][model]={name:'pass' for name in
                                       ('factuality','source_support','relevance','refusal')}
        result=score_manual(report,form)
        self.assertEqual(result['examples_human_reviewed'],1)

    def test_refusal_gate_report_keeps_unproven_cases_for_review(self):
        class Index:
            def search(self, question, limit=8):
                return []
        cases=[{'id':'x','question':'quasar banana nebula','expected_gate':'refuse'},
               {'id':'y','question':'Does a safety guarantee exist?','expected_gate':'review'}]
        result=refusal_audit(index=Index(),challenges=cases)
        self.assertEqual(result['checks_passed'],1)
        self.assertIsNone(result['items'][1]['check_passed'])


if __name__ == '__main__':
    unittest.main()

class TestQA16FullRoundTrip(unittest.TestCase):
    def test_comparison_and_one_time_test_lock(self):
        from src.skygrid_ai.language.evaluate_qa_v16 import compare
        from src.skygrid_ai.language.train_qa_v16 import metadata_for, fit
        from src.skygrid_ai.language.train_large_transformer import sha256_file
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / 'data/aviation/tokenized_v09b'
            data.mkdir(parents=True)
            tok = LargeByteBPE()
            tok.save(data/'tokenizer.json')
            for split in ('train', 'val'):
                (data / (split + '.bin')).write_bytes(b'synthetic_' + split.encode())
            digest = {s: sha256_file(data/(s+'.bin')) for s in ('train','val')}
            config = {'context_size':192,'d_model':16,'n_heads':2,
                      'n_layers':1,'hidden_width':32}
            model = SkyGridTransformerV08(vocab_size=tok.vocab_size, **config)
            cp = {'format_version':'skygrid_v09c_1','model_config':config,
                  'model_state':model.state_dict(), 'data_digests':digest,
                  'tokenizer_fingerprint':tok.fingerprint(),
                  'vocab_size':tok.vocab_size,'step':10000}
            source=root/'checkpoints/v10/last.pt';source.parent.mkdir(parents=True)
            torch.save(cp,source)
            qa_folder=root/'checkpoints/qa_v16'
            evidence='PX4 sends useful actuator commands to motors.'
            example = lambda i: {'id':i, 'question':'How does PX4 send commands?',
                                 'evidence':evidence, 'answer':'PX4 commands motors.'}
            rows={'train':[example('t1'),example('t2')],
                  'val':[example('v1')], 'test':[example('heldout')]}
            fit(rows,tok,cp,sha256_file(source),'approved_hash',output=qa_folder,
                steps=2,eval_every=1,patience=2,min_train=2)
            report, path = compare(root=root, rows=rows,qa_hash='approved_hash',
                                   limit=2)
            self.assertTrue(path.exists())
            self.assertEqual(report['answer_targets']>0, True)
            self.assertEqual(len(report['items']),1)
            self.assertIn('qa_draft',report['items'][0])
            with self.assertRaisesRegex(FileExistsError,'already exists'):
                compare(root=root, rows=rows,qa_hash='approved_hash',limit=2)
            final,path=compare(root=root,rows=rows,qa_hash='approved_hash',
                               split='test',final_test=True,limit=2)
            self.assertEqual(final['split'],'test')
            with self.assertRaises(FileExistsError):
                compare(root=root,rows=rows,qa_hash='approved_hash',
                        split='test',final_test=True,limit=2)
