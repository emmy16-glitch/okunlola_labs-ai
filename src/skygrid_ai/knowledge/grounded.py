"""Conservative, offline extractive answers with pinned document citations.

This does not use a language model, claim independent factual verification, or
invent text. Returned excerpts are copied from indexed documentation.
"""
import re

from .retrieval import words

NAMED_SOURCES = {"mavlink": "mavlink", "px4": "px4", "qgroundcontrol": "qgroundcontrol"}


def queried_sources(question):
    """Identify only explicitly named source families, never infer an actor."""
    return [name for name in NAMED_SOURCES if re.search(r"\b" + name + r"\b", question.casefold())]


def candidate_sentences(passage):
    """Find reasonably complete sentence excerpts from cleaned Markdown."""
    text = passage["text"]
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        # Do not output fragments, headings or unbounded tables as answers.
        if 35 <= len(sentence) <= 450 and not sentence.startswith(("|", "<", "```")):
            yield sentence


def answer_from_evidence(index, question, *, top_k=8):
    """Return the strongest verbatim sentence with its provenance, or abstain.

    Ranking is heuristic and is not a correctness or entailment check. A
    one-source question cannot silently be 'answered' with another product's
    description. Cross-system questions remain eligible for mixed evidence.
    """
    if not isinstance(question, str) or not question.strip() or top_k < 1:
        raise ValueError("Provide a nonempty question and positive top_k")
    hits = index.search(question, limit=top_k)
    base = {"question": question, "method": "extractive_indexed_document_sentence"}
    if not hits:
        return {**base, "status": "abstained", "answer": None, "citations": [],
                "reason": "No indexed passage matches the question"}
    named = queried_sources(question)
    eligible = hits
    if len(named) == 1:
        eligible = [p for p in hits if p["source"] == named[0]]
        if not eligible:
            return {**base, "status": "source_mismatch", "answer": None,
                    "citations": [], "reason":
                    f"Question names {named[0]}, but top retrieved sources are about other systems. Check the distinction before answering.",
                    "suggested_sources": [p["url"] for p in hits[:2]]}
    query_terms = set(words(question))
    candidates = []
    for hit in eligible:
        for sentence in candidate_sentences(hit):
            overlap = len(set(words(sentence)) & query_terms)
            # One broad matching word is not enough to imply support.
            required = 1 if len(query_terms) <= 2 else 2
            if overlap < required:
                continue
            candidates.append((overlap, overlap / max(1, len(query_terms)),
                               hit["score"], sentence, hit))
    if not candidates:
        cross_system = len(named) == 1 and hits[0]["source"] != named[0]
        if cross_system:
            return {**base, "status": "source_mismatch", "answer": None,
                    "citations": [], "reason":
                    f"The strongest matching documentation belongs to {hits[0]['source']}, not {named[0]}; no supporting sentence was found from {named[0]}.",
                    "suggested_sources": [p["url"] for p in hits[:2]]}
        return {**base, "status": "abstained", "answer": None, "citations": [],
                "reason": "No complete sentence has enough explicit term overlap for a cautious excerpt"}
    candidates.sort(key=lambda c: (-c[0], -c[1], -c[2], c[4]["id"], c[3]))
    _, _, _, sentence, hit = candidates[0]
    # Preserve exact wording from the indexed passage. We do not paraphrase it.
    assert sentence in hit["text"]
    return {**base, "status": "excerpt_found", "answer": sentence,
            "citations": [{"id": hit["id"], "source": hit["source"],
                           "path": hit["path"], "url": hit["url"]}],
            "warning": "Verbatim source excerpt, NOT independently checked factual accuracy or a neural-model answer"}
