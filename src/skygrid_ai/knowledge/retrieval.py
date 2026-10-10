"""Offline, citation-first aviation-document retrieval. No pretrained model or API."""
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / 'data' / 'aviation' / 'processed_v09b'
INDEX = ROOT / 'data' / 'aviation' / 'index_v10' / 'passages.json.gz'
STOP = set('a an and are as at be by can do does for from how in is it of on or that the their there these this to use used what when where which why with'.split())


def words(text):
    return [w for w in re.findall(r'[a-z0-9_]+', text.casefold()) if w not in STOP]


def _sha(text):
    return hashlib.sha256(' '.join(text.split()).casefold().encode('utf-8')).hexdigest()


def split_passages(text, size=125, overlap=25):
    if not 0 <= overlap < size:
        raise ValueError('overlap must be smaller than size')
    items = text.split()
    result = []
    for start in range(0, len(items), size - overlap):
        chunk = items[start:start + size]
        if chunk:
            result.append(' '.join(chunk))
        if start + size >= len(items):
            break
    return result


def build_passages(corpus=CORPUS):
    """Validate text fingerprints and link every excerpt to its original document."""
    corpus = Path(corpus)
    manifest_path = corpus / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = []
    for row in manifest:
        if not {'text_file', 'source', 'path', 'url', 'sha256_cleaned'} <= row.keys():
            raise ValueError('Manifest missing required provenance fields')
        docpath = (corpus / row['text_file']).resolve()
        if not docpath.is_relative_to(corpus.resolve()):
            raise ValueError('Unsafe document path')
        document = docpath.read_text(encoding='utf-8')
        if _sha(document) != row['sha256_cleaned']:
            raise ValueError(f'Document fingerprint mismatch: {row["path"]}')
        for n, content in enumerate(split_passages(document)):
            entries.append({
                'id': f'{row["source"]}:{row["path"]}:{n}',
                'source': row['source'], 'path': row['path'],
                'url': row['url'], 'text': content,
            })
    return entries, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def save_index(passages, manifest_digest, destination=INDEX):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Index exists: {destination}')
    with gzip.open(destination, 'wt', encoding='utf-8') as stream:
        json.dump({'version': 1, 'manifest_sha256': manifest_digest,
                   'passages': passages}, stream, ensure_ascii=False)
    return destination


def load_index(path=INDEX, corpus=CORPUS):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        payload = json.load(stream)
    if payload.get('version') != 1:
        raise ValueError('Unsupported index format')
    actual = hashlib.sha256((Path(corpus) / 'manifest.json').read_bytes()).hexdigest()
    if payload['manifest_sha256'] != actual:
        raise ValueError('Index source manifest changed; rebuild the index')
    return SearchIndex(payload['passages'])


class SearchIndex:
    """Small standard-library BM25 inverted index, deterministic ranked citations."""
    def __init__(self, passages):
        self.passages = list(passages)
        self.postings = defaultdict(list)
        lengths = []
        for index, passage in enumerate(self.passages):
            counts = Counter(words(passage['text']))
            lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self.postings[token].append((index, count))
        self.avglen = sum(lengths) / max(len(lengths), 1)
        self.lengths = lengths

    def search(self, query, limit=5):
        if limit < 1:
            raise ValueError('limit must be positive')
        tokens = set(words(query))
        if not tokens or not self.passages:
            return []
        scores = defaultdict(float)
        N = len(self.passages)
        for term in tokens:
            matches = self.postings.get(term, [])
            if not matches:
                continue
            idf = math.log(1 + (N - len(matches) + 0.5) / (len(matches) + 0.5))
            for doc_id, frequency in matches:
                denominator = frequency + 1.5 * (0.25 + 0.75 * self.lengths[doc_id] / max(self.avglen, 1))
                scores[doc_id] += idf * frequency * 2.5 / denominator
        ranked = sorted(scores, key=lambda i: (-scores[i], self.passages[i]['id']))[:limit]
        return [{**self.passages[i], 'score': round(scores[i], 4)} for i in ranked]

    def evidence(self, question, limit=3, minimum_score=1.0):
        """Returns source excerpts, NOT an AI-authored factual answer."""
        hits = self.search(question, limit=limit)
        if not hits or hits[0]['score'] < minimum_score:
            return {'question': question, 'status': 'abstained', 'evidence': [],
                    'note': 'No sufficiently relevant indexed source passage found'}
        return {'question': question, 'status': 'evidence_found', 'evidence': hits,
                'note': 'Source excerpts only; verify before applying aviation guidance'}
