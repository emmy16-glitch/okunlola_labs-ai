"""Citation-backed extractive answers from intact, verified aviation documents.

The BM25 passage index locates candidate DOCUMENTS; answers are then selected
from original locally stored document text, not flattened 125-word chunks.
No neural model, external API or claim-verification engine is used.
"""
import json
import re
from pathlib import Path

from .retrieval import CORPUS, _sha, words

NAMED_SOURCES = ("mavlink", "px4", "qgroundcontrol")
ACTION_WORDS = set(
    "allows enables provides translates converts controls measures estimates "
    "determines uses sends broadcasts advertises detects synchronizes supports "
    "reports carries transfers exchanges connects calculates identifies".split()
)
ACTION_PHRASES = ("is used to", "are used to", "can be used", "is responsible for")
HISTORY_TERMS = ("replaces", "legacy", "earlier", "at the time of writing")


def queried_sources(question):
    """Identify explicitly named source families, not an assumed actor."""
    return [
        name for name in NAMED_SOURCES
        if re.search(r"\b" + name + r"\b", question.casefold())
    ]


def clean_inline_markup(text):
    """Remove formatting marks while retaining the original source words."""
    text = re.sub(r"\{#[a-zA-Z_][^}]*\}", "", text)
    text = re.sub(r"(?<!\w)[*`]+|[*`]+(?!\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def candidate_sentences(text):
    """Yield only complete prose sentences; reject headings, lists and fragments.

    The preprocessed source retains paragraph boundaries, unlike the BM25
    index where passages have their whitespace collapsed and may end midway
    through a sentence.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    for p_index, paragraph in enumerate(paragraphs):
        lines = []
        for original in paragraph.splitlines():
            line = original.strip()
            if not line or re.match(r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||<|::|```)", line):
                continue
            # Reject standalone Markdown section anchors and image remnants.
            if re.fullmatch(r"\{#[^}]+\}", line):
                continue
            lines.append(line)
        if not lines:
            continue
        content = " ".join(lines)
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`*(])", content)
        for sentence in pieces:
            sentence = clean_inline_markup(sentence)
            if (
                sentence.endswith((".", "!", "?"))
                and 8 <= len(sentence.split()) <= 85
                and 45 <= len(sentence) <= 600
                and not sentence.startswith(("#", "-", "|", "<", ">"))
                and not sentence.endswith("...")
            ):
                yield p_index, sentence


def load_verified_document(corpus, source, path, *, manifest=None):
    """Read an indexed source document safely; reject any fingerprint mismatch."""
    folder = Path(corpus).resolve()
    rows = manifest if manifest is not None else json.loads(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )
    matching = [r for r in rows if r["source"] == source and r["path"] == path]
    if len(matching) != 1:
        raise ValueError(f"Source document missing or ambiguous: {source}:{path}")
    row = matching[0]
    full = (folder / row["text_file"]).resolve()
    if not full.is_relative_to(folder):
        raise ValueError("Source document is outside the verified corpus")
    text = full.read_text(encoding="utf-8")
    if _sha(text) != row["sha256_cleaned"]:
        raise ValueError(f"Source document fingerprint mismatch: {source}:{path}")
    return text


def answer_from_evidence(index, question, *, top_k=8, corpus=None):
    """Return one source-derived explanatory sentence, or abstain.

    Passing 'corpus' enables full-document extraction (recommended).
    A corpus-free call can only use complete passage sentences. A matching
    keyword or retrieval score does not certify factual entailment.
    """
    if not isinstance(question, str) or not question.strip() or top_k < 1:
        raise ValueError("Provide a nonempty question and positive top_k")
    hits = index.search(question, limit=top_k)
    base = {"question": question, "method": "verified_document_extractive_sentence"}
    if not hits:
        return {**base, "status": "abstained", "answer": None, "citations": [],
                "reason": "No indexed passage matches the question"}
    named = queried_sources(question)
    eligible = hits
    if len(named) == 1:
        eligible = [p for p in hits if p["source"] == named[0]]
        if not eligible:
            return {**base, "status": "source_mismatch", "answer": None,
                    "citations": [],
                    "reason": f"The question names {named[0]}, but the retrieved documentation is about other source systems.",
                    "suggested_sources": list(dict.fromkeys(p["url"] for p in hits[:3]))}

    query_terms = set(words(question))
    if not query_terms:
        return {**base, "status": "abstained", "answer": None, "citations": [],
                "reason": "The question contains no searchable topic terms"}
    candidates = []
    manifest = None
    if corpus is not None:
        manifest = json.loads((Path(corpus) / "manifest.json").read_text(encoding="utf-8"))
    seen_docs = set()

    for rank, hit in enumerate(eligible):
        key = (hit["source"], hit["path"])
        if key in seen_docs:
            continue
        seen_docs.add(key)
        doc_title_terms = set(words(re.sub(r"[_-]", " ", Path(hit["path"]).stem)))
        topic_overlap = len(doc_title_terms & query_terms)
        source_text = (
            load_verified_document(corpus, *key, manifest=manifest)
            if corpus is not None else hit["text"]
        )
        for location, sentence in candidate_sentences(source_text):
            token_set = set(words(sentence))
            overlap = token_set & query_terms
            # Match meaningful topic wording, not just a named product.
            substantive = overlap - set(NAMED_SOURCES)
            if not substantive and topic_overlap < 1 and len(named) < 2:
                continue
            if len(overlap) < 2 and topic_overlap < 2:
                continue
            lower = sentence.casefold()
            action = any(word in token_set for word in ACTION_WORDS) or any(
                phrase in lower for phrase in ACTION_PHRASES
            )
            score = (
                2.0 * len(substantive)
                + 1.2 * topic_overlap
                + (2.5 if action else 0)
                - (1.3 if any(t in lower for t in HISTORY_TERMS) else 0)
                + (0.5 if location <= 2 else 0)
                - rank * 0.45
            )
            candidates.append((score, -location, sentence, hit))
    if not candidates:
        cross_source = len(named) == 1 and hits[0]["source"] != named[0]
        if cross_source:
            return {**base, "status": "source_mismatch", "answer": None,
                    "citations": [],
                    "reason": f"Top evidence is about {hits[0]['source']}, not {named[0]}; no complete supporting sentence was found.",
                    "suggested_sources": list(dict.fromkeys(p["url"] for p in hits[:3]))}
        return {**base, "status": "abstained", "answer": None, "citations": [],
                "reason": "No sufficiently relevant complete source sentence was found"}
    candidates.sort(key=lambda c: (-c[0], -c[1], c[3]["id"], c[2]))
    _, _, sentence, hit = candidates[0]
    return {**base, "status": "excerpt_found", "answer": sentence,
            "citations": [{"source": hit["source"], "path": hit["path"],
                           "url": hit["url"]}],
            "warning": "A lightly formatted excerpt from the cited document; NOT independently checked for factual accuracy, and not a neural-model answer"}
