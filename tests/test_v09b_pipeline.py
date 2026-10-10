import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

from src.skygrid_ai.language.audit_large_corpus import (
    digest, grouped_splits, near_duplicates, write_audited, read_verified_documents,
)
from src.skygrid_ai.language.large_bpe import (
    LargeByteBPE, merge_pair, read_token_ids, write_token_ids,
)


class TestV09bPipeline(unittest.TestCase):
    def test_duplicate_families_never_cross_splits(self):
        base = ' '.join(f'aviation telemetry parameter{i}' for i in range(100))
        docs = [
            {'source': 'px4', 'path': 'one.md', 'split': 'train', 'text': base,
             'sha256_cleaned': digest(base)},
            {'source': 'px4', 'path': 'two.md', 'split': 'test', 'text': base + ' extra',
             'sha256_cleaned': digest(base + ' extra')},
        ]
        matches = near_duplicates(docs)
        self.assertTrue(matches)
        self.assertTrue(matches[0]['original_cross_split'])
        groups, families = grouped_splits(docs, matches)
        self.assertEqual(families, 1)
        self.assertEqual(sorted(len(g) for g in groups.values()), [0, 0, 2])

    def test_original_input_is_preserved(self):
        docs = [{'source': 'px4', 'path': 'topic.md', 'text': 'Aviation flight telemetry', 'split': 'val'}]
        original = docs[0].copy()
        groups, count = grouped_splits(docs, [])
        self.assertEqual(docs[0], original)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audited'
            result = write_audited(groups, [], count, path)
            self.assertEqual(sum(result['split_counts'].values()), 1)
            with self.assertRaises(FileExistsError):
                write_audited(groups, [], count, path)

    def test_bpe_unicode_roundtrip_and_save(self):
        text = 'MAVLink GPS telemetry! 🚁\n' * 40
        tokenizer = LargeByteBPE.train(text, vocab_size=320, max_unique_chunks=500)
        examples = ['MAVLink GPS', 'Unseen 飞行 ✈️', '', 'altitude: 120 m/s\n']
        for example in examples:
            encoded = tokenizer.encode(example, add_eos=True)
            self.assertEqual(tokenizer.decode(encoded), example)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tok.json'
            tokenizer.save(path)
            restored = LargeByteBPE.load(path)
            self.assertEqual(restored.encode('MAVLink 🚁'), tokenizer.encode('MAVLink 🚁'))
            self.assertEqual(restored.fingerprint(), tokenizer.fingerprint())

    def test_encoding_matches_rank_order_reference(self):
        text = 'telemetry telemetry telemetry GPS GPS GPS ' * 30
        tokenizer = LargeByteBPE.train(text, vocab_size=340)
        for chunk in ('telemetry', 'MAVLink', ' GPS ', '📡'):
            # Whitespace-aware tokenizer encodes whole contiguous chunks separately.
            reference = []
            from src.skygrid_ai.language.large_bpe import CHUNK_PATTERN, FIRST_MERGE_ID
            for item in CHUNK_PATTERN.findall(chunk):
                ids = tuple(item.encode('utf-8'))
                for new_id, pair in enumerate(tokenizer.merges, FIRST_MERGE_ID):
                    ids = merge_pair(ids, pair, new_id)
                reference.extend(ids)
            self.assertEqual(tokenizer.encode(chunk), reference)

    def test_source_revision_and_fingerprint_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'px4_user_guide' / 'en'
            source.mkdir(parents=True)
            (source / 'topic.md').write_text('Flight telemetry with GPS', encoding='utf-8')
            row = {
                'source': 'px4', 'path': 'en/topic.md',
                'revision': 'a' * 40,
                'sha256_cleaned': digest('Flight telemetry with GPS'),
                'split': 'train',
            }
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps([row]), encoding='utf-8')
            with patch('src.skygrid_ai.language.audit_large_corpus.subprocess.check_output', return_value='a' * 40):
                data = read_verified_documents(manifest, root)
                self.assertEqual(data[0]['text'], 'Flight telemetry with GPS')
            with patch('src.skygrid_ai.language.audit_large_corpus.subprocess.check_output', return_value='b' * 40):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    read_verified_documents(manifest, root)
            (source / 'topic.md').write_text('Different text', encoding='utf-8')
            with patch('src.skygrid_ai.language.audit_large_corpus.subprocess.check_output', return_value='a' * 40):
                with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                    read_verified_documents(manifest, root)

    def test_binary_pipeline_uses_per_document_eos(self):
        from src.skygrid_ai.language import prepare_large_tokens as prep
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / 'audited'
            corpus.mkdir()
            docs = corpus / 'docs'
            docs.mkdir()
            rows = []
            for split in ('train', 'val', 'test'):
                content = ('MAVLink telemetry GPS flights. ' * 20) if split == 'train' else 'Unseen aviation report.'
                if split == 'train':
                    contents = [content, 'Battery warning and position control. ' * 20]
                else:
                    contents = [content]
                for i, document in enumerate(contents):
                    name = f'docs/{split}{i}.txt'
                    (corpus / name).write_text(document, encoding='utf-8')
                    rows.append({'split': split, 'text_file': name})
                (corpus / f'{split}.txt').write_text('\n\n'.join(contents), encoding='utf-8')
            (corpus / 'manifest.json').write_text(json.dumps(rows), encoding='utf-8')
            with patch.object(prep, 'INPUT', corpus), patch.object(prep, 'OUTPUT', root / 'tokens'), patch('sys.argv', ['prepare_large_tokens', '--vocab-size', '320']):
                prep.main()
            ids = read_token_ids(root / 'tokens' / 'train.bin')
            self.assertEqual(ids.count(257), 2)
            self.assertEqual(read_token_ids(root / 'tokens' / 'test.bin').count(257), 1)

    def test_binary_file_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tokens.bin'
            write_token_ids(path, [0, 257, 511, 999999])
            self.assertEqual(read_token_ids(path), [0, 257, 511, 999999])
            with path.open('ab') as stream:
                stream.write(b'a')
            with self.assertRaises(ValueError):
                read_token_ids(path)


if __name__ == '__main__':
    unittest.main()
