"""Prepare local SkyGrid v0.9b BPE training tokens, never touch older runs.

Run: python -m src.skygrid_ai.language.prepare_large_tokens
"""
import argparse
import hashlib
import json
from pathlib import Path

from .audit_large_corpus import OUTPUT as INPUT
from .large_bpe import LargeByteBPE, write_token_ids

OUTPUT = INPUT.parent / 'tokenized_v09b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-size', type=int, default=768)
    args = parser.parse_args()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f'{OUTPUT} is not empty; existing tokenizer data preserved')
    texts = {}
    for split in ('train', 'val', 'test'):
        path = INPUT / f'{split}.txt'
        texts[split] = path.read_text(encoding='utf-8')
    manifest = json.loads((INPUT / 'manifest.json').read_text(encoding='utf-8'))
    tokenizer = LargeByteBPE.train(texts['train'], vocab_size=args.vocab_size)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUTPUT / 'tokenizer.json')
    report = {
        'vocabulary_size': tokenizer.vocab_size,
        'tokenizer_fingerprint': tokenizer.fingerprint(),
        'tokenizer_training_split': 'train',
        'format': 'uint32_le',
        'source': str(INPUT),
        'training_rights_review_required': True,
        'splits': {},
    }
    for split, text in texts.items():
        # Each document ends with EOS, not just the entire corpus split.
        ids = []
        source_bytes = bytearray()
        rows = [row for row in manifest if row['split'] == split]
        if not rows:
            raise ValueError(f'No documents in {split} split')
        for row in rows:
            document = (INPUT / row['text_file']).read_text(encoding='utf-8')
            doc_ids = tokenizer.encode(document, add_eos=True)
            if tokenizer.decode(doc_ids) != document:
                raise ValueError(f'Tokenization round-trip failed: {row["text_file"]}')
            ids.extend(doc_ids)
            source_bytes.extend(document.encode('utf-8'))
        write_token_ids(OUTPUT / f'{split}.bin', ids)
        source_bytes = bytes(source_bytes)
        report['splits'][split] = {
            'utf8_bytes': len(source_bytes),
            'documents': len(rows),
            'eos_tokens': len(rows),
            'token_count_with_eos': len(ids),
            'bits_per_token': 32,
            'bytes_per_token_excluding_eos': round(len(source_bytes) / max(len(ids) - len(rows), 1), 3),
            'sha256_utf8_text': hashlib.sha256(source_bytes).hexdigest(),
        }
        print(f'{split}: {len(source_bytes):,} bytes -> {len(ids):,} token IDs')
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Vocabulary:', tokenizer.vocab_size)
    print('Fingerprint:', tokenizer.fingerprint()[:16])
    print('Saved:', OUTPUT)
    print('Do not load these .bin files with the v0.8 JSON training script.')


if __name__ == '__main__':
    main()
