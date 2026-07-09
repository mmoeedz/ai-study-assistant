"""
End-to-end verification suite for the AI Study Assistant.

Runs every test case from the implementation_plan.md verification plan.
Mimics Streamlit's UploadedFile API with a thin wrapper so we can call
StudyAssistant.ingest_pdfs() without launching the UI.
"""

from __future__ import annotations

import io
import os
import sys
import time
import shutil
import traceback
from pathlib import Path

# Make project root importable when running from tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from rag_pipeline import StudyAssistant, SimpleVectorStore  # noqa: E402


# ── ANSI helpers ──────────────────────────────────────────────────────
class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    M = "\033[95m"
    BOLD = "\033[1m"
    END = "\033[0m"


def section(title: str) -> None:
    print(f"\n{C.B}{C.BOLD}{'─' * 70}{C.END}")
    print(f"{C.B}{C.BOLD}  {title}{C.END}")
    print(f"{C.B}{C.BOLD}{'─' * 70}{C.END}")


def ok(msg: str) -> None:
    print(f"  {C.G}[PASS]{C.END} {msg}")


def fail(msg: str) -> None:
    print(f"  {C.R}[FAIL]{C.END} {msg}")


def info(msg: str) -> None:
    print(f"  {C.Y}[i]{C.END} {msg}")


def show(msg: str, value: str, max_chars: int = 600) -> None:
    snippet = value if len(value) <= max_chars else value[:max_chars] + " …[truncated]"
    print(f"  {C.M}{msg}:{C.END}\n{snippet}\n")


# ── Streamlit UploadedFile shim ───────────────────────────────────────
class FakeUploadedFile:
    """Minimal stand-in for Streamlit's UploadedFile (just .read() + .name)."""

    def __init__(self, path: Path):
        self._bytes = path.read_bytes()
        self.name = path.name

    def read(self) -> bytes:
        return self._bytes


# ── Locate test PDFs ──────────────────────────────────────────────────
SLIDES_DIR = Path(r"D:\Uni\NLP\Slides")
TEST_PDFS = [
    "Week 1 Intro to NLP.pdf",
    "Lecture 2 -Basic Text Processing_Final.pdf",
]


def get_test_files() -> list[FakeUploadedFile]:
    files = []
    for name in TEST_PDFS:
        p = SLIDES_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"Missing test PDF: {p}")
        files.append(FakeUploadedFile(p))
    return files


# ── Result tracker ────────────────────────────────────────────────────
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, note: str = "") -> None:
    RESULTS.append((name, passed, note))


# ── Tests ─────────────────────────────────────────────────────────────
def test_clean_state(assistant: StudyAssistant) -> None:
    section("TEST 0 — Clean State")
    assistant.clear_vectorstore()
    assert assistant.vectorstore is None
    assert assistant.total_chunks == 0
    assert assistant.indexed_files == []
    ok("Vector store cleared")
    record("clean_state", True)


def test_ingest(assistant: StudyAssistant) -> tuple[int, int]:
    section("TEST 1 — Upload & Ingest PDFs")
    files = get_test_files()
    info(f"Ingesting {len(files)} PDF(s): {[f.name for f in files]}")

    progress_log = []

    def cb(pct, msg):
        progress_log.append((pct, msg))

    t0 = time.time()
    num_docs, num_chunks = assistant.ingest_pdfs(files, progress_callback=cb)
    elapsed = time.time() - t0

    info(f"Elapsed: {elapsed:.1f}s | Docs: {num_docs} | Chunks: {num_chunks}")
    info(f"Progress callbacks fired: {len(progress_log)}")

    passed = (
        num_docs == len(files)
        and num_chunks > 0
        and assistant.vectorstore is not None
        and assistant.total_chunks == num_chunks
        and len(assistant.indexed_files) >= len(files)
    )
    if passed:
        ok(f"Indexed {num_docs} PDFs into {num_chunks} chunks")
    else:
        fail("Ingest result inconsistent")

    record("ingest", passed, f"{num_docs} docs, {num_chunks} chunks")
    return num_docs, num_chunks


def test_qa(assistant: StudyAssistant) -> None:
    section("TEST 2 — Q&A Mode")
    query = "What is Natural Language Processing?"
    info(f"Query: {query}")
    answer, docs = assistant.generate(query, mode="qa")
    show("Response", answer)
    info(f"Retrieved {len(docs)} source chunk(s)")

    required_markers = ["✅ Answer", "📌 Key Points", "📖 Source Insight"]
    found = [m for m in required_markers if m in answer]
    missing = [m for m in required_markers if m not in answer]

    passed = len(missing) == 0 and len(docs) > 0
    if passed:
        ok(f"All structural markers present: {required_markers}")
    else:
        fail(f"Missing markers: {missing}")

    record("qa_mode", passed, f"markers found={found}, sources={len(docs)}")


def test_summarize(assistant: StudyAssistant) -> None:
    section("TEST 3 — Summarize Mode")
    query = "Summarize the main concepts of NLP covered in week 1."
    info(f"Query: {query}")
    answer, docs = assistant.generate(query, mode="summarize")
    show("Response", answer)

    required = ["📝 Summary", "🎯 Main Themes", "📌 Key Concepts"]
    missing = [m for m in required if m not in answer]
    passed = len(missing) == 0 and len(docs) > 0
    if passed:
        ok("Summary structure complete")
    else:
        fail(f"Missing markers: {missing}")
    record("summarize_mode", passed, f"sources={len(docs)}")


def test_mcq(assistant: StudyAssistant) -> None:
    section("TEST 4 — MCQ Mode")
    query = "Generate MCQs about basic text processing."
    info(f"Query: {query}")
    answer, docs = assistant.generate(query, mode="mcq")
    show("Response", answer, max_chars=1500)

    # Count Q1.. markers + answer key
    q_markers = sum(1 for i in range(1, 8) if f"**Q{i}.**" in answer or f"Q{i}." in answer)
    has_key = "Answer Key" in answer or "answer key" in answer.lower()
    has_options = "A)" in answer and "B)" in answer and "C)" in answer and "D)" in answer

    passed = q_markers >= 3 and has_options and has_key and len(docs) > 0
    info(f"Detected questions: {q_markers} | options A-D: {has_options} | answer key: {has_key}")
    if passed:
        ok(f"{q_markers} MCQs with options & answer key")
    else:
        fail("MCQ structure incomplete")
    record("mcq_mode", passed, f"questions={q_markers}, key={has_key}")


def test_eli5(assistant: StudyAssistant) -> None:
    section("TEST 5 — Explain Simply (ELI5) Mode")
    query = "Explain tokenization simply."
    info(f"Query: {query}")
    answer, docs = assistant.generate(query, mode="eli5")
    show("Response", answer)

    required = ["🧒 Simple Explanation", "🔑 Remember These", "💡 Real-World Analogy"]
    missing = [m for m in required if m not in answer]
    passed = len(missing) == 0 and len(docs) > 0
    if passed:
        ok("ELI5 structure complete")
    else:
        fail(f"Missing markers: {missing}")
    record("eli5_mode", passed, f"sources={len(docs)}")


def test_hallucination_guard(assistant: StudyAssistant) -> None:
    section("TEST 6 — Hallucination Guard (off-topic question)")
    query = "What is the chemical formula for sulfuric acid and how is it manufactured?"
    info(f"Query: {query}")
    answer, docs = assistant.generate(query, mode="qa")
    show("Response", answer)

    # The model should refuse with the canonical fallback OR clearly
    # decline based on context. We accept either: explicit fallback string,
    # OR the answer says material doesn't cover this.
    fallback_phrases = [
        "answer is not available in the provided material",
        "not available in the provided",
        "not found in the context",
        "no information",
        "does not contain",
        "not covered",
        "cannot provide",
        "can't provide",
        "i cannot",
        "i can't",
        "unable to provide",
        "outside the scope",
        "not in the context",
    ]
    lowered = answer.lower()
    refused = any(p in lowered for p in fallback_phrases)
    # Anti-hallucination check: confirm no fabricated chemistry content
    hallucinated = any(
        term in lowered
        for term in ["h2so4", "sulfuric", "contact process", "lead chamber"]
    ) and not refused
    if hallucinated:
        refused = False

    if refused:
        ok("Model correctly refused off-topic question")
    else:
        fail("Model may have hallucinated (no refusal phrase detected)")
    record("hallucination_guard", refused, "off-topic refusal check")


def test_persistence() -> None:
    section("TEST 7 — Persistence (reload vector store from disk)")
    # Create a fresh assistant — it should auto-load from disk
    fresh = StudyAssistant()
    info(f"Reloaded indexed files: {fresh.indexed_files}")
    info(f"Reloaded chunks: {fresh.total_chunks}")

    passed = (
        fresh.vectorstore is not None
        and fresh.total_chunks > 0
        and len(fresh.indexed_files) > 0
    )
    if passed:
        ok(f"Vector store persisted & reloaded ({fresh.total_chunks} chunks)")
    else:
        fail("Vector store did not persist")

    # Sanity check: run a quick retrieval on the reloaded store
    if passed:
        docs = fresh.retrieve("natural language processing", k=2)
        if docs:
            ok(f"Retrieval works on reloaded store ({len(docs)} chunks)")
        else:
            fail("Reloaded store retrieval returned 0 results")
            passed = False

    record("persistence", passed, f"chunks={fresh.total_chunks}")


# ── Driver ────────────────────────────────────────────────────────────
def main() -> int:
    print(f"{C.BOLD}{C.M}╔══════════════════════════════════════════════════════════════════════╗{C.END}")
    print(f"{C.BOLD}{C.M}║    AI Study Assistant — Full Verification Suite                      ║{C.END}")
    print(f"{C.BOLD}{C.M}╚══════════════════════════════════════════════════════════════════════╝{C.END}")

    # Sanity: Ollama reachable?
    try:
        import requests
        r = requests.get(f"{config.LLM_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        info(f"Ollama models available: {models}")
        for needed in (config.LLM_MODEL, config.EMBEDDING_MODEL):
            if not any(needed in m for m in models):
                fail(f"Required model '{needed}' is NOT pulled in Ollama")
                return 2
    except Exception as e:
        fail(f"Ollama not reachable at {config.LLM_BASE_URL}: {e}")
        return 2

    assistant = StudyAssistant()

    try:
        test_clean_state(assistant)
        test_ingest(assistant)
        test_qa(assistant)
        test_summarize(assistant)
        test_mcq(assistant)
        test_eli5(assistant)
        test_hallucination_guard(assistant)
        test_persistence()
    except Exception as e:
        fail(f"Unhandled exception: {e}")
        traceback.print_exc()
        record("unhandled_exception", False, str(e))

    # ── Summary ──────────────────────────────────────────────────────
    section("SUMMARY")
    width = max(len(n) for n, _, _ in RESULTS) + 2
    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    for name, p, note in RESULTS:
        tag = f"{C.G}PASS{C.END}" if p else f"{C.R}FAIL{C.END}"
        print(f"  [{tag}] {name.ljust(width)} {note}")

    print(f"\n  {C.BOLD}Total: {passed}/{total} passed{C.END}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
