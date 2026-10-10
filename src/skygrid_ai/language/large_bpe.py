"""SkyGrid v0.9b: train and apply a scalable, original byte BPE tokenizer.

Merge rules are learned from training text only. No pretrained vocabularies.
The whitespace-aware chunks are a deliberate change from the v0.8 tokenizer.
"""

import hashlib
import json
import re
from array import array
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

PAD_ID = 256
EOS_ID = 257
FIRST_MERGE_ID = 258
CHUNK_PATTERN = re.compile(r'\S+|\s+', flags=re.UNICODE)


def merge_pair(tokens, pair, new_id):
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return tuple(result)


class LargeByteBPE:
    """UTF-8 tokenizer with rank-ordered byte pair merges within text chunks."""
    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merge_ranks = {}
        for token_id, pair in enumerate(self.merges, FIRST_MERGE_ID):
            if (len(pair) != 2 or
                    any(not isinstance(t, int) or t not in self.id_to_bytes for t in pair) or
                    pair in self.merge_ranks):
                raise ValueError(f'Invalid merge rule: {pair}')
            self.merge_ranks[pair] = token_id
            self.id_to_bytes[token_id] = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]

    @property
    def vocab_size(self):
        return FIRST_MERGE_ID + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size=768, max_unique_chunks=12000):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Training text must not be empty')
        if vocab_size < FIRST_MERGE_ID or max_unique_chunks < 1:
            raise ValueError('Invalid tokenizer settings')

        counts = Counter(
            chunk.encode('utf-8')
            for chunk in CHUNK_PATTERN.findall(text)
            if len(chunk.encode('utf-8')) <= 512
        )
        # Frequency cap avoids excessive CPU work with very large document sets.
        common = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_unique_chunks]
        patterns = {tuple(data): frequency for data, frequency in common}
        merges = []

        for new_id in range(FIRST_MERGE_ID, vocab_size):
            pairs = Counter()
            for seq, frequency in patterns.items():
                if len(seq) > 1:
                    local = Counter(zip(seq, seq[1:]))
                    for pair, occurrences in local.items():
                        pairs[pair] += frequency * occurrences
            if not pairs:
                break
            best_pair, frequency = max(
                pairs.items(), key=lambda row: (row[1], -row[0][0], -row[0][1])
            )
            if frequency < 2:
                break
            next_patterns = Counter()
            for seq, count in patterns.items():
                next_patterns[merge_pair(seq, best_pair, new_id)] += count
            patterns = next_patterns
            merges.append(best_pair)
        return cls(merges)

    @lru_cache(maxsize=16384)
    def _encode_chunk(self, chunk):
        ids = tuple(chunk.encode('utf-8'))
        while len(ids) > 1:
            valid = [
                (self.merge_ranks[(ids[i], ids[i + 1])], i)
                for i in range(len(ids) - 1)
                if (ids[i], ids[i + 1]) in self.merge_ranks
            ]
            if not valid:
                break
            rank, offset = min(valid)
            pair = (ids[offset], ids[offset + 1])
            ids = merge_pair(ids, pair, rank)
        return ids

    def encode(self, text, add_eos=False):
        if not isinstance(text, str):
            raise TypeError('Input must be text')
        result = []
        for chunk in CHUNK_PATTERN.findall(text):
            result.extend(self._encode_chunk(chunk))
        if add_eos:
            result.append(EOS_ID)
        return result

    def decode(self, ids, skip_special=True):
        pieces = []
        for value in ids:
            value = int(value)
            if value in (PAD_ID, EOS_ID):
                if not skip_special:
                    raise ValueError('Special tokens are not UTF-8 text')
                continue
            if value not in self.id_to_bytes:
                raise ValueError(f'Unknown token ID: {value}')
            pieces.append(self.id_to_bytes[value])
        return b''.join(pieces).decode('utf-8')

    def fingerprint(self):
        payload = json.dumps(
            {'kind': 'skygrid_large_byte_bpe_v1', 'merges': self.merges},
            separators=(',', ':')
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'type': 'skygrid_large_byte_bpe_v1',
            'merges': [list(pair) for pair in self.merges],
            'fingerprint': self.fingerprint(),
        }, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('type') != 'skygrid_large_byte_bpe_v1':
            raise ValueError('Wrong tokenizer format')
        tok = cls(data['merges'])
        if tok.fingerprint() != data.get('fingerprint'):
            raise ValueError('Tokenizer fingerprint mismatch')
        return tok


def write_token_ids(path, ids):
    """Store unsigned 32-bit IDs as portable little-endian binary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array('I', ids)
    if data.itemsize != 4:
        raise RuntimeError('Unsigned int must be 4 bytes')
    if sys.byteorder != 'little':
        data.byteswap()
    with path.open('wb') as stream:
        data.tofile(stream)


def read_token_ids(path):
    path = Path(path)
    if path.stat().st_size % 4:
        raise ValueError('Invalid token file byte length')
    ids = array('I')
    with path.open('rb') as stream:
        ids.fromfile(stream, path.stat().st_size // 4)
    if sys.byteorder != 'little':
        ids.byteswap()
    return list(ids)
