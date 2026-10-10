"""Build a provenance-preserving aviation text corpus from local docs repos.

Run as: python -m src.skygrid_ai.language.build_large_corpus
"""

import hashlib
import html
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPOS = ROOT / "data" / "aviation" / "source"
OUTPUT = ROOT / "data" / "aviation" / "processed_v09_large"

# Only original project documentation; no source code, images or translations.
SOURCES = (
    {
        "name": "px4",
        "repo": "PX4/PX4-user_guide",
        "folder": "px4_user_guide",
        "directories": ("en",),
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    },
    {
        "name": "mavlink",
        "repo": "mavlink/mavlink-devguide",
        "folder": "mavlink_devguide",
        "directories": ("en",),
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    },
    {
        "name": "qgroundcontrol",
        "repo": "mavlink/qgroundcontrol",
        "folder": "qgroundcontrol",
        "directories": ("docs/en/qgc-user-guide", "docs/en/qgc-dev-guide"),
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    },
)

SKIP_NAMES = {"SUMMARY.md", "_sidebar.md", "_nav.md"}


def clean_markdown(text):
    """Basic text extraction; human quality review is still required."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", text, flags=re.S | re.I)
    text = re.sub(r"```[^\n]*\n.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^\n]*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\n]*?\)", r"\1", text)
    text = re.sub(r"<[^>\n]+>", " ", text)
    text = re.sub(r"^\s*:::[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\|[\s:\-|]+\|\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*_]\s*){3,}$", "", text, flags=re.M)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collect_documents(sources=SOURCES, repos=REPOS, min_chars=300):
    documents = []
    counts = Counter()
    seen = set()

    for source in sources:
        repo_dir = Path(repos) / source["folder"]
        if not (repo_dir / ".git").exists():
            raise FileNotFoundError(
                f"Missing {source['name']} clone: {repo_dir}"
            )
        revision = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        candidates = set()
        for relative_folder in source["directories"]:
            folder = repo_dir / relative_folder
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing documentation folder: {folder}")
            candidates.update(folder.rglob("*.md"))
        for path in sorted(candidates):
            if path.name in SKIP_NAMES:
                counts[(source["name"], "navigation_skip")] += 1
                continue
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
            cleaned = clean_markdown(raw)
            if len(cleaned) < min_chars or sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1) < 0.4:
                counts[(source["name"], "quality_skip")] += 1
                continue
            canonical = " ".join(cleaned.split()).casefold()
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if digest in seen:
                counts[(source["name"], "duplicate_skip")] += 1
                continue
            seen.add(digest)
            relative = path.relative_to(repo_dir).as_posix()
            documents.append({
                "source": source["name"],
                "path": relative,
                "url": f"https://github.com/{source['repo']}/blob/{revision}/{relative}",
                "revision": revision,
                "license": source["license"],
                "license_url": source["license_url"],
                "rights_review": "required",
                "sha256_cleaned": digest,
                "characters": len(cleaned),
                "text": cleaned,
            })
            counts[(source["name"], "accepted")] += 1
    return documents, counts


def split_by_source(documents, seed=42):
    grouped = {}
    for doc in documents:
        grouped.setdefault(doc["source"], []).append(doc)
    splits = {"train": [], "val": [], "test": []}
    for source in sorted(grouped):
        rows = sorted(grouped[source], key=lambda doc: doc["path"])
        rng = random.Random(f"{seed}:{source}")
        rng.shuffle(rows)
        if len(rows) < 3:
            raise ValueError(f"Need at least 3 eligible documents for {source}")
        n_holdout = max(1, int(len(rows) * 0.1))
        splits["test"].extend(rows[:n_holdout])
        splits["val"].extend(rows[n_holdout:2 * n_holdout])
        splits["train"].extend(rows[2 * n_holdout:])
    for name in splits:
        splits[name].sort(key=lambda doc: (doc["source"], doc["path"]))
    return splits


def write_corpus(splits, out=OUTPUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    manifest = []
    for split, docs in splits.items():
        destination = out / f"{split}.txt"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for index, doc in enumerate(docs):
                if index:
                    handle.write("\n\n")
                handle.write(doc["text"])
                manifest.append({
                    "split": split,
                    **{key: val for key, val in doc.items() if key != "text"},
                })
        summary[split] = {
            "documents": len(docs),
            "characters": sum(len(doc["text"]) for doc in docs),
            "by_source": dict(Counter(doc["source"] for doc in docs)),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    documents, counts = collect_documents()
    groups = split_by_source(documents)
    summary = write_corpus(groups)
    print("Documents accepted:", len(documents))
    for source in sorted({doc["source"] for doc in documents}):
        print(source, {status: count for (name, status), count in counts.items() if name == source})
    for split, data in summary.items():
        print(f"{split}: {data['documents']} documents; {data['characters']:,} characters")
    print("Output:", OUTPUT)
    print("Next: review licensing, cleanup quality, and cross-split near-duplicates before model training.")


if __name__ == "__main__":
    main()
