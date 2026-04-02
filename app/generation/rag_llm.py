import json
import re
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.generation.hf_llm import hf_call
from app.retrieval.lc_rag_engine import LangChainRAGEngine

CORPUS_PATH = Path("data/data_processed/corpus.jsonl")
RETRIEVE_TOP_K = 12
MAX_CONTEXTS_FOR_ANSWER = 1
MIN_SIMILARITY_SCORE = 0.40
MIN_FINAL_CONFIDENCE = 0.55

QUERY_EXPANSIONS = {
    "disclosure": {"disclosures", "disclose", "reporting", "material"},
    "requirements": {"obligations", "compliance", "conditions", "provisions"},
    "listed": {"listing", "listed entity", "stock exchange"},
    "entity": {"company", "issuer", "listed entity"},
    "lodr": {"listing obligations", "disclosure requirements", "regulation"},
    "insider": {"pit", "unpublished price sensitive information", "upsi"},
}

INTENT_GROUPS = [
    ({"disclosure", "disclosures", "disclose"}, {"disclosure", "disclosures", "disclose", "material"}),
    ({"requirement", "requirements", "obligation", "obligations"}, {"requirement", "requirements", "obligation", "obligations"}),
    ({"listed", "listing"}, {"listed", "listing", "stock exchange"}),
]


@lru_cache(maxsize=1)
def get_rag_engine() -> LangChainRAGEngine:
    """Reuse the loaded retriever across requests."""
    return LangChainRAGEngine(top_k=RETRIEVE_TOP_K)


@lru_cache(maxsize=1)
def _load_corpus() -> List[Dict[str, str]]:
    if not CORPUS_PATH.exists():
        return []

    rows: List[Dict[str, str]] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
    return rows


@lru_cache(maxsize=1)
def _idf_map() -> Dict[str, float]:
    rows = _load_corpus()
    total_docs = max(1, len(rows))
    doc_freq: Dict[str, int] = {}

    for row in rows:
        tokens = _tokenize(str(row.get("text", "")))
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    idf: Dict[str, float] = {}
    for token, df in doc_freq.items():
        # Smoothed BM25-style IDF.
        idf[token] = math.log((1.0 + total_docs - df + 0.5) / (df + 0.5) + 1.0)

    return idf


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _normalize_for_scoring(text: str) -> str:
    cleaned = text
    cleaned = re.sub(
        r"listing\s+obligations\s+and\s+disclosure\s+requirements",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"securities\s+and\s+exchange\s+board\s+of\s+india",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(Inserted|Substituted|Omitted)\s+by\s+the\s+Securities\s+and\s+Exchange\s+Board\s+of\s+India.*?(?=\n|$)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"w\.e\.f\.\s*[0-9./-]+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bPrior to the (substitution|omission).*(?=\n|$)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _expand_query_tokens(question: str) -> set[str]:
    base = _tokenize(question)
    expanded = set(base)
    for token in list(base):
        for extra in QUERY_EXPANSIONS.get(token, set()):
            expanded.update(_tokenize(extra))
    return expanded


def _normalize_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _rerank_contexts(question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q_tokens_base = _tokenize(question)
    q_tokens_expanded = _expand_query_tokens(question)
    ranked: List[Dict[str, Any]] = []

    for row in contexts:
        text = str(row.get("text", ""))
        score_text = _normalize_for_scoring(text)
        t_tokens = _tokenize(score_text)

        overlap_base = 0.0
        if q_tokens_base:
            overlap_base = len(q_tokens_base.intersection(t_tokens)) / float(len(q_tokens_base))

        overlap_expanded = 0.0
        if q_tokens_expanded:
            overlap_expanded = len(q_tokens_expanded.intersection(t_tokens)) / float(len(q_tokens_expanded))

        overlap = (0.75 * overlap_base) + (0.25 * overlap_expanded)

        semantic = _normalize_score(row.get("score"))

        # Calibrate semantic score to a confidence-like range.
        calibrated_semantic = min(1.0, max(0.0, (semantic - 0.15) / 0.75))

        phrase_bonus = 0.0
        q_l = question.lower().strip()
        text_l = score_text.lower()
        if q_l and q_l in text_l:
            phrase_bonus = 0.1
        if (
            {"disclosure", "requirements"}.issubset(q_tokens_base)
            and re.search(r"disclosure.{0,80}(require|obligation)", text_l)
        ):
            phrase_bonus += 0.18

        footnote_hits = len(re.findall(r"(inserted by|substituted by|omitted by|w\.e\.f\.|prior to)", text.lower()))
        noise_penalty = min(0.2, footnote_hits * 0.04)

        digit_density = 0.0
        if text:
            digit_density = sum(ch.isdigit() for ch in text) / float(len(text))
        if digit_density > 0.08:
            noise_penalty += 0.12

        intent_penalty = 0.0
        for trigger_tokens, required_tokens in INTENT_GROUPS:
            if q_tokens_base.intersection(trigger_tokens) and not t_tokens.intersection(required_tokens):
                intent_penalty += 0.35

        final_score = (
            (0.5 * calibrated_semantic)
            + (0.4 * overlap)
            + phrase_bonus
            - noise_penalty
            - intent_penalty
        )
        final_score = min(1.0, max(0.0, final_score))

        item = dict(row)
        item["semantic_score"] = semantic
        item["semantic_score_calibrated"] = calibrated_semantic
        item["keyword_overlap"] = overlap
        item["score"] = final_score
        ranked.append(item)

    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    strong = [x for x in ranked if x.get("semantic_score", 0.0) >= MIN_SIMILARITY_SCORE]
    if strong:
        return strong
    return ranked


def _keyword_retrieve(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    corpus = _load_corpus()
    if not corpus:
        return []

    q_tokens_base = _tokenize(query)
    q_tokens_expanded = _expand_query_tokens(query)
    idf = _idf_map()
    q_base_weight = sum(idf.get(t, 1.0) for t in q_tokens_base) or 1.0
    q_exp_weight = sum(idf.get(t, 1.0) for t in q_tokens_expanded) or 1.0

    query_l = query.lower()
    contains_disclosure_phrase = "disclosure requirements" in query_l

    scored = []
    for row in corpus:
        text = row.get("text", "")
        score_text = _normalize_for_scoring(text)
        t_tokens = _tokenize(score_text)
        base_overlap = sum(idf.get(t, 1.0) for t in q_tokens_base.intersection(t_tokens)) / q_base_weight

        expanded_overlap = (
            sum(idf.get(t, 1.0) for t in q_tokens_expanded.intersection(t_tokens)) / q_exp_weight
        )

        phrase_bonus = 0.0
        text_l = score_text.lower()
        if contains_disclosure_phrase and "disclosure requirements" in text_l:
            phrase_bonus = 0.2
        elif contains_disclosure_phrase and "disclosure" in text_l and "require" in text_l:
            phrase_bonus = 0.12
        if contains_disclosure_phrase and re.search(r"disclosure.{0,80}(require|obligation)", text_l):
            phrase_bonus += 0.18

        footnote_penalty = min(
            0.2,
            len(re.findall(r"(inserted by|substituted by|omitted by|w\.e\.f\.|prior to)", text_l)) * 0.04,
        )
        digit_density = 0.0
        if text:
            digit_density = sum(ch.isdigit() for ch in text) / float(len(text))
        if digit_density > 0.08:
            footnote_penalty += 0.12

        score = (0.75 * base_overlap) + (0.25 * expanded_overlap) + phrase_bonus - footnote_penalty
        score = max(0.0, min(1.0, score))
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        # fallback to first few chunks so caller can still respond deterministically
        return corpus[:top_k]

    out = []
    for score, row in scored[:top_k]:
        item = dict(row)
        item["score"] = score
        out.append(item)
    return out


def _doc_key(row: Dict[str, Any]) -> str:
    row_id = row.get("id")
    if row_id:
        return str(row_id)
    return f"{row.get('source_file', 'unknown')}::{row.get('chunk_index', -1)}"


def _fuse_contexts(
    vector_rows: List[Dict[str, Any]],
    keyword_rows: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    combined: Dict[str, Dict[str, Any]] = {}

    # Reciprocal Rank Fusion, tuned for short legal corpora.
    for rank, row in enumerate(vector_rows, start=1):
        key = _doc_key(row)
        item = dict(row)
        item["_rrf"] = item.get("_rrf", 0.0) + (1.0 / (50.0 + rank))
        item["_vec_score"] = _normalize_score(row.get("score"))
        combined[key] = item

    for rank, row in enumerate(keyword_rows, start=1):
        key = _doc_key(row)
        if key not in combined:
            combined[key] = dict(row)
            combined[key]["_rrf"] = 0.0
            combined[key]["_vec_score"] = 0.0
        combined[key]["_rrf"] += 1.0 / (50.0 + rank)
        combined[key]["_kw_score"] = _normalize_score(row.get("score"))

    fused = list(combined.values())
    for row in fused:
        vec = _normalize_score(row.get("_vec_score", row.get("score")))
        kw = _normalize_score(row.get("_kw_score", 0.0))
        rrf = float(row.get("_rrf", 0.0))
        row["score"] = min(1.0, (0.45 * vec) + (0.35 * kw) + (0.20 * min(1.0, rrf * 60.0)))

    fused.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return fused[:top_k]


def _build_prompt(question: str, contexts: List[Dict[str, str]], history: List[Dict[str, str]]) -> str:
    history_text = ""
    if history:
        for turn in history:
            user = turn.get("user", "")
            assistant = turn.get("assistant", "")
            history_text += f"User: {user}\\nAssistant: {assistant}\\n\\n"

    context_block = "\\n\\n".join(
        f"[{c.get('source_file')} - chunk {c.get('chunk_index')}]:\\n{c.get('text')}"
        for c in contexts
    )

    return f"""
You are a SEBI compliance assistant.
Answer strictly from the retrieved SEBI context.
Use only the most relevant point from the provided context and do not include unrelated details.

If the answer is not clearly supported by the context, reply exactly:
"The referenced SEBI documents do not cover this information."

Conversation History:
{history_text}

User Question:
{question}

Retrieved SEBI Context:
{context_block}

Return 3-6 bullet points with precise compliance language and include one inline citation in this format:
[source_file - chunk chunk_index]
"""


def run_rag(question: str, history: Optional[List[Dict[str, str]]] = None):
    """Orchestrate retrieval + prompt creation + answer generation with fallbacks."""
    history = history or []

    try:
        rag = get_rag_engine()
        vector_contexts = rag.retrieve(question)
        keyword_contexts = _keyword_retrieve(question, top_k=RETRIEVE_TOP_K)
        raw_contexts = _fuse_contexts(vector_contexts, keyword_contexts, top_k=RETRIEVE_TOP_K)
        contexts = _rerank_contexts(question, raw_contexts)[:MAX_CONTEXTS_FOR_ANSWER]
        prompt = rag.build_prompt(question, contexts, history=history)
    except Exception as exc:
        print(f"[RAG] Embedding retrieval unavailable, using keyword fallback: {exc}")
        keyword_contexts = _keyword_retrieve(question, top_k=RETRIEVE_TOP_K)
        contexts = _rerank_contexts(question, keyword_contexts)[:MAX_CONTEXTS_FOR_ANSWER]
        prompt = _build_prompt(question, contexts, history)

    top_score = 0.0
    if contexts:
        top_score = float(contexts[0].get("score", 0.0))

    if not contexts or top_score < MIN_FINAL_CONFIDENCE:
        return {
            "question": question,
            "answer": "The referenced SEBI documents do not cover this information.",
            "evidence": contexts,
        }

    answer = hf_call(prompt)

    return {
        "question": question,
        "answer": answer,
        "evidence": contexts,
    }


if __name__ == "__main__":
    out = run_rag("What are the disclosure requirements for listed entities?")
    print("ANSWER:\n", out["answer"])
    print("\nEVIDENCE:\n", out["evidence"])
