"""SkyGrid v0.9b: review existing aviation documents and group near-duplicates.

Run: python -m src.skygrid_ai.language.audit_large_corpus
No model weights or existing datasets are modified.
"""

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .build_large_corpus import OUTPUT as ORIGINAL, REPOS, SOURCES, clean_markdown

OUTPUT = ORIGINAL.parent / 'processed_v09b'


def digest(text):
    canonical = ' '.join(text.split()).casefold()
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def read_verified_documents(manifest_path=ORIGINAL / 'manifest.json', repos=REPOS):
    """Rebuild document text from the pinned Git checkout; reject changed inputs."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    source_map = {item['name']: item for item in SOURCES}
    revisions = {}
    documents = []
    for doc in manifest:
        name = doc['source']
        if name not in source_map:
            raise ValueError(f'Unknown source: {name}')
        repo = Path(repos) / source_map[name]['folder']
        if name not in revisions:
            revisions[name] = subprocess.check_output(
                ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True
            ).strip()
        if revisions[name] != doc['revision']:
            raise ValueError(f'Source checkout changed: {name}. Rebuild the original corpus.')
        path = (repo / doc['path']).resolve()
        if not path.is_relative_to(repo.resolve()):
            raise ValueError(f'Unsafe source path: {doc["path"]}')
        text = clean_markdown(path.read_text(encoding='utf-8-sig', errors='replace'))
        if digest(text) != doc['sha256_cleaned']:
            raise ValueError(f'Content fingerprint mismatch: {name}/{doc["path"]}')
        documents.append({**doc, 'text': text})
    return documents


def shingles(text, width=5):
    """Stable 5-word fingerprints for detecting highly similar documents."""
    words = re.findall(r'\w+', text.casefold(), flags=re.UNICODE)
    return {
        int.from_bytes(hashlib.blake2b(
            ' '.join(words[i:i + width]).encode('utf-8'), digest_size=8
        ).digest(), 'big')
        for i in range(max(0, len(words) - width + 1))
    }


def near_duplicates(documents, threshold=0.82, sketch_size=32):
    """Heuristic shingle matching; non-exhaustive, but verifies reported pairs exactly."""
    fingerprints = [shingles(row['text']) for row in documents]
    buckets = defaultdict(list)
    for index, values in enumerate(fingerprints):
        for signature in sorted(values)[:sketch_size]:
            buckets[signature].append(index)
    shared = Counter()
    for matches in buckets.values():
        if len(matches) > 80:
            continue  # boilerplate shingles generate too many candidates
        for offset, left in enumerate(matches):
            for right in matches[offset + 1:]:
                shared[left, right] += 1
    matches = []
    for (i, j), common_sketches in shared.items():
        if common_sketches < 2:
            continue
        a, b = fingerprints[i], fingerprints[j]
        if not a or not b:
            continue
        common = len(a & b)
        union = len(a) + len(b) - common
        similarity = common / union
        containment = common / min(len(a), len(b))
        if similarity >= threshold or (common >= 40 and containment >= 0.93):
            matches.append({
                'left': i, 'right': j,
                'jaccard': round(similarity, 5),
                'containment': round(containment, 5),
                'original_cross_split': documents[i]['split'] != documents[j]['split'],
            })
    return sorted(matches, key=lambda p: (p['left'], p['right']))


def grouped_splits(documents, pairs, seed=42):
    """Put every detected near-duplicate family in a single deterministic split."""
    parent = list(range(len(documents)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        a, b = root(pair['left']), root(pair['right'])
        if a != b:
            parent[max(a, b)] = min(a, b)
    families = defaultdict(list)
    for i in range(len(documents)):
        families[root(i)].append(i)

    groups = {'train': [], 'val': [], 'test': []}
    for indices in families.values():
        # Sort source identifiers so the assignment is independent of input order.
        names = sorted(f"{documents[i]['source']}:{documents[i]['path']}" for i in indices)
        identifier = f'{seed}|' + '|'.join(names)
        bucket = int(hashlib.sha256(identifier.encode('utf-8')).hexdigest(), 16) % 10
        split = 'test' if bucket == 0 else 'val' if bucket == 1 else 'train'
        for i in indices:
            groups[split].append(documents[i])
    for key in groups:
        groups[key].sort(key=lambda d: (d['source'], d['path']))
    return groups, len(families)


def write_audited(groups, pairs, family_count, output=OUTPUT):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty: {output}; preserve prior runs')
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    (output / 'docs').mkdir(exist_ok=True)
    report = {
        'original_cross_split_suspected_pairs': sum(p['original_cross_split'] for p in pairs),
        'suspected_near_duplicate_pairs': len(pairs),
        'document_families': family_count,
        'split_counts': {},
        'rights_review_required': True,
        'method_note': 'Candidate screening uses bottom-k shingles and is not an exhaustive plagiarism or leakage audit.',
    }
    for split, docs in groups.items():
        (output / f'{split}.txt').write_text(
            '\n\n'.join(doc['text'] for doc in docs), encoding='utf-8'
        )
        report['split_counts'][split] = len(docs)
        for doc in docs:
            identifier = hashlib.sha256(
                f"{doc['source']}:{doc['path']}".encode('utf-8')
            ).hexdigest()[:20]
            text_file = f'docs/{identifier}.txt'
            (output / text_file).write_text(doc['text'], encoding='utf-8')
            manifest.append({
                'split': split, 'text_file': text_file,
                **{k: v for k, v in doc.items() if k not in ('text', 'split')}
            })
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'audit.json').write_text(json.dumps({'report': report, 'pairs': pairs}, indent=2), encoding='utf-8')
    return report


def main():
    docs = read_verified_documents()
    pairs = near_duplicates(docs)
    groups, family_count = grouped_splits(docs, pairs)
    report = write_audited(groups, pairs, family_count)
    print('Documents inspected:', len(docs))
    print('Suspected near-duplicate pairs:', report['suspected_near_duplicate_pairs'])
    print('Original cross-split suspicious pairs:', report['original_cross_split_suspected_pairs'])
    print('Grouped families:', family_count)
    print('New document splits:', report['split_counts'])
    print('New corpus:', OUTPUT)
    print('Rights review remains required. Inspect audit.json before training.')


if __name__ == '__main__':
    main()
