import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.language.large_bpe import LargeByteBPE
from src.skygrid_ai.language.modern_transformer import SkyGridTransformerV08
from src.skygrid_ai.language.train_large_transformer import sha256_file
from src.skygrid_ai.language.qa_pilot import digest_files
from src.skygrid_ai.language.train_qa import train_qa
from src.skygrid_ai.language.evaluate_qa import measure

class TestQAEndToEnd(unittest.TestCase):
    def test_own_source_weights_to_qa_metrics_and_test_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            token_dir=root/'data/aviation/tokenized_v09b'
            token_dir.mkdir(parents=True)
            tok=LargeByteBPE();tok.save(token_dir/'tokenizer.json')
            for split in ('train','val'):
                (token_dir/f'{split}.bin').write_bytes(b'local binary fixture'+split.encode())
            digests={s:sha256_file(token_dir/f'{s}.bin') for s in ('train','val')}
            config={'context_size':192,'d_model':16,'n_heads':2,
                    'n_layers':1,'hidden_width':32}
            model=SkyGridTransformerV08(vocab_size=tok.vocab_size, **config)
            original={'format_version':'skygrid_v09c_1','step':10000,
                      'model_config':config,'model_state':model.state_dict(),
                      'data_digests':digests,'vocab_size':tok.vocab_size,
                      'tokenizer_fingerprint':tok.fingerprint()}
            start=root/'checkpoints/v10/last.pt';start.parent.mkdir(parents=True)
            torch.save(original,start)
            for folder in ('qa','benchmarks'):
                (root/'data'/folder).mkdir(parents=True)
            fixture_root=Path.cwd()/'data'
            for name in ('aviation_v10.json','aviation_v11_independent.json'):
                shutil.copy2(fixture_root/'benchmarks'/name,root/'data/benchmarks'/name)
            shutil.copy2(fixture_root/'qa/aviation_v15_protocol.json',root/'data/qa/aviation_v15_protocol.json')
            protocol=root/'data/qa/aviation_v15_protocol.json'
            benchmarks=(root/'data/benchmarks/aviation_v10.json',
                        root/'data/benchmarks/aviation_v11_independent.json')
            data_hash=digest_files((protocol,*benchmarks))
            rows={
                'train':[{'id':'train', 'kind':'grounded','question':'What is a heartbeat?',
                          'answer':'It announces a vehicle.',
                          'evidence':'A MAVLink heartbeat advertises that a system exists.'}],
                'val':[{'id':'val','kind':'grounded','question':'What does PX4 allocation do?',
                        'answer':'It translates actuator requests.',
                        'evidence':'PX4 control allocation translates vehicle control commands.'}],
                'test':[{'id':'test','kind':'unsupported','question':'Is everything safe?',
                         'answer':'I cannot verify that claim.',
                         'evidence':'No supporting aviation source is available.'}],
            }
            train_qa(rows,tok,original,sha256_file(start),data_hash,
                     output=root/'checkpoints/qa_v15',steps=2,eval_every=1,batch_size=1)
            val,report=measure('val','best',root=root,examples=rows,sample_tokens=3)
            self.assertEqual(val['answer_targets']>0,True)
            self.assertEqual(len(val['items']),1)
            self.assertIn('baseline_draft_unverified',val['items'][0])
            self.assertTrue(report.exists())
            test,report=measure('test','best',root=root,examples=rows,sample_tokens=3,final_test=True)
            self.assertEqual(test['split'],'test')
            self.assertTrue(report.exists())
            with self.assertRaisesRegex(FileExistsError,'already run'):
                measure('test','best',root=root,examples=rows,final_test=True)

if __name__=='__main__':unittest.main()
