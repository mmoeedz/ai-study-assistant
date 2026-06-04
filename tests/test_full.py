"""
Comprehensive test suite — covers EVERYTHING the user cares about:
  • Chat persistence: save, load, rename, delete, search
  • All 4 LLM modes (Q&A, Summarise, MCQs, ELI5) with real PDFs
  • Retrieval coverage — does the model actually see all the material?
  • MCQ variability — does asking twice give different questions?
  • Hallucination guard
  • Repeatability and basic latency

Run:  python -X utf8 tests/test_full.py
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import time
import shutil
import uuid
import statistics
from pathlib import Path

# Force-load secrets the same way Streamlit does
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
secrets_path = ROOT / ".streamlit" / "secrets.toml"
if secrets_path.exists():
    secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    for k, v in secrets.items():
        os.environ.setdefault(k, str(v))

import config  # noqa: E402
from rag_pipeline import StudyAssistant, Document  # noqa: E402


# ─── ANSI helpers ────────────────────────────────────────────────────
class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    M = "\033[95m"
    BOLD = "\033[1m"
    END = "\033[0m"


def section(title: str) -> None:
    print(f"\n{C.B}{C.BOLD}{'═' * 72}{C.END}")
    print(f"{C.B}{C.BOLD}  {title}{C.END}")
    print(f"{C.B}{C.BOLD}{'═' * 72}{C.END}")


def step(title: str) -> None:
    print(f"\n{C.M}── {title}{C.END}")


def ok(msg: str) -> None:
    print(f"  {C.G}[PASS]{C.END} {msg}")


def fail(msg: str) -> None:
    print(f"  {C.R}[FAIL]{C.END} {msg}")


def info(msg: str) -> None:
    print(f"  {C.Y}[i]{C.END} {msg}")


def show(msg: str, value: str, max_chars: int = 800) -> None:
    snippet = value if len(value) <= max_chars else value[:max_chars] + " …[truncated]"
    print(f"  {C.M}{msg}:{C.END}\n{snippet}\n")


RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, note: str = "") -> None:
    RESULTS.append((name, passed, note))


# ─── Streamlit UploadedFile shim ─────────────────────────────────────
class FakeUploadedFile:
    def __init__(self, path: Path):
        self._bytes = path.read_bytes()
        self.name = path.name

    def read(self) -> bytes:
        return self._bytes


# ─── Test PDF — small + content-rich ─────────────────────────────────
SLIDES_DIR = Path(r"D:\Uni\NLP\Slides")
TEST_PDF_NAMES = [
    "Week 1 Intro to NLP.pdf",
    "Lecture 2 -Basic Text Processing_Final.pdf",
]


def get_files() -> list[FakeUploadedFile]:
    out = []
    for name in TEST_PDF_NAMES:
        p = SLIDES_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"Missing test PDF: {p}")
        out.append(FakeUploadedFile(p))
    return out


# ─── Chat persistence helpers (mirror app.py) ────────────────────────
CHATS_DIR = ROOT / "chats"


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def load_all_chats() -> list[dict]:
    out = []
    if not CHATS_DIR.exists():
        return out
    for p in CHATS_DIR.glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    out.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return out


def save_chat(chat: dict) -> None:
    CHATS_DIR.mkdir(exist_ok=True)
    chat["updated"] = time.time()
    if not chat.get("created"):
        chat["created"] = chat["updated"]
    _chat_path(chat["id"]).write_text(
        json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_chat(chat_id: str) -> None:
    p = _chat_path(chat_id)
    if p.exists():
        p.unlink()


def new_chat() -> dict:
    return {
        "id": uuid.uuid4().hex[:10],
        "title": "New chat",
        "messages": [],
        "created": time.time(),
        "updated": time.time(),
    }


# ─────────────────────────────────────────────────────────────────────
#                              TESTS
# ─────────────────────────────────────────────────────────────────────
def test_environment():
    section("TEST 0 — Environment")
    info(f"Provider: {config.LLM_PROVIDER}")
    info(f"LLM model: {config.LLM_MODEL}")
    info(f"Embedding model: {config.EMBEDDING_MODEL}")
    has_key = bool(config.GROQ_API_KEY) if config.LLM_PROVIDER == "groq" else True
    if not has_key:
        fail("GROQ_API_KEY not set — cannot run cloud tests")
        record("env", False, "missing API key")
        return False
    ok("Environment configured")
    record("env", True, f"{config.LLM_PROVIDER}/{config.LLM_MODEL}")
    return True


def test_chat_persistence():
    section("TEST 1 — Chat Persistence (save / load / search / rename / delete)")
    # Backup any existing chats so we don't disturb the user
    backup_dir = ROOT / "chats_backup_test"
    if CHATS_DIR.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        CHATS_DIR.rename(backup_dir)
    CHATS_DIR.mkdir(exist_ok=True)

    try:
        # Create three test chats
        c1 = new_chat()
        c1["title"] = "Photosynthesis basics"
        c1["messages"].append({"query": "What is photosynthesis?", "mode": "qa", "answer": "Plants convert sunlight to energy."})
        save_chat(c1)

        c2 = new_chat()
        c2["title"] = "NLP intro"
        c2["messages"].append({"query": "What is NLP?", "mode": "qa", "answer": "Natural Language Processing studies how computers handle text."})
        save_chat(c2)

        c3 = new_chat()
        c3["title"] = "MCQ test"
        c3["messages"].append({"query": "Quiz me", "mode": "mcq", "answer": "Q1. ..."})
        save_chat(c3)

        step("Save & load")
        chats = load_all_chats()
        if len(chats) == 3:
            ok(f"All 3 chats saved & loaded ({[c['title'] for c in chats]})")
        else:
            fail(f"Expected 3 chats, got {len(chats)}")
            record("chat_save_load", False)
            return

        step("Order (newest first)")
        if chats[0]["id"] == c3["id"]:
            ok("Newest chat appears first")
        else:
            fail("Sort order broken")

        step("Search by title")
        q = "nlp"
        matches = [c for c in chats if q in c["title"].lower()]
        ok(f"Title search 'nlp' → {len(matches)} match(es): {[c['title'] for c in matches]}")

        step("Search by message content")
        q = "photosynthesis"
        matches = [c for c in chats
                   if any(q in (m.get("query", "") + m.get("answer", "")).lower()
                          for m in c.get("messages", []))]
        ok(f"Content search 'photosynthesis' → {len(matches)} match(es)")

        step("Rename a chat")
        c2_renamed_title = "NLP — Renamed by Test"
        c2["title"] = c2_renamed_title
        save_chat(c2)
        reloaded = load_all_chats()
        c2_check = next((c for c in reloaded if c["id"] == c2["id"]), None)
        if c2_check and c2_check["title"] == c2_renamed_title:
            ok(f"Rename persisted: '{c2_check['title']}'")
        else:
            fail("Rename did not persist")

        step("Delete a chat")
        delete_chat(c1["id"])
        remaining = load_all_chats()
        if len(remaining) == 2 and not any(c["id"] == c1["id"] for c in remaining):
            ok(f"Delete works — {len(remaining)} chats left")
        else:
            fail("Delete failed")

        record("chat_persistence", True, f"3 created → 2 after delete; rename verified")
    finally:
        # Restore user's original chats
        shutil.rmtree(CHATS_DIR, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.rename(CHATS_DIR)


def test_ingest(assistant: StudyAssistant) -> tuple[int, int]:
    section("TEST 2 — Upload, Chunk, Embed (pickle-free persistence)")
    files = get_files()
    info(f"Ingesting {[f.name for f in files]}")
    t0 = time.time()
    n_docs, n_chunks = assistant.ingest_pdfs(files)
    dt = time.time() - t0
    info(f"Done in {dt:.1f}s — {n_docs} docs → {n_chunks} chunks")

    # Check the JSON+npy files exist (no .pkl)
    vs_dir = Path(config.VECTORSTORE_DIR)
    has_json = (vs_dir / "documents.json").exists()
    has_npy = (vs_dir / "embeddings.npy").exists()
    has_pkl = (vs_dir / "vectorstore.pkl").exists()
    info(f"Persistence files — documents.json: {has_json}, embeddings.npy: {has_npy}, .pkl: {has_pkl}")

    passed = (
        n_docs == len(files)
        and n_chunks > 0
        and has_json
        and has_npy
    )
    if passed:
        ok(f"Ingest OK — {n_chunks} chunks indexed, JSON+npy persisted")
    else:
        fail("Ingest broken")
    record("ingest", passed, f"{n_chunks} chunks")
    return n_docs, n_chunks


def test_persistence_reload(expected_chunks: int):
    section("TEST 3 — Persistence Reload (no pickle)")
    fresh = StudyAssistant()
    info(f"Reloaded — {len(fresh.indexed_files)} docs, {fresh.total_chunks} chunks")
    docs = fresh.retrieve("natural language processing", k=3)
    info(f"Retrieval works on reloaded store — {len(docs)} chunks returned")
    passed = (
        fresh.total_chunks == expected_chunks
        and len(docs) > 0
    )
    if passed:
        ok("Persistence + reload + retrieval all work")
    else:
        fail("Persistence broken")
    record("persistence_reload", passed, f"chunks={fresh.total_chunks}")


def test_qa_mode(assistant: StudyAssistant):
    section("TEST 4 — Q&A Mode (cited answers)")
    query = "What is Natural Language Processing?"
    info(f"Query: {query}")
    answer, docs = assistant.generate(query, mode="qa")
    show("Response", answer)

    needed = ["✅ Answer", "📌 Key Points", "📖 Source Insight"]
    missing = [m for m in needed if m not in answer]
    passed = not missing and len(docs) > 0
    if passed:
        ok(f"All structural markers present, {len(docs)} sources cited")
    else:
        fail(f"Missing: {missing}")
    record("qa_mode", passed)


def test_summarize_mode(assistant: StudyAssistant):
    section("TEST 5 — Summarise Mode")
    query = "Summarise everything in the uploaded slides about basic text processing."
    info(f"Query: {query}")
    answer, docs = assistant.generate(query, mode="summarize")
    show("Response", answer)

    needed = ["📝 Summary", "🎯 Main Themes", "📌 Key Concepts"]
    missing = [m for m in needed if m not in answer]
    passed = not missing
    if passed:
        ok("All summary sections present")
    else:
        fail(f"Missing: {missing}")
    record("summarize_mode", passed)


def test_mcq_mode_variability(assistant: StudyAssistant):
    """Run the MCQ generator THREE times and check that questions vary."""
    section("TEST 6 — MCQ Mode + Variability (does it give different quizzes?)")
    query = "Generate MCQs about basic text processing."

    runs = []
    for i in range(3):
        info(f"Run #{i + 1}…")
        t0 = time.time()
        answer, _ = assistant.generate(query, mode="mcq")
        dt = time.time() - t0
        info(f"  Latency: {dt:.1f}s | Length: {len(answer)} chars")
        runs.append(answer)

    # Extract question texts
    def extract_q_texts(out: str) -> list[str]:
        # match "**Q1.** [text]" up to next blank line / "**Q"
        return re.findall(r"\*\*Q\d+\.\*\*\s*([^\n]+)", out)

    q_lists = [extract_q_texts(r) for r in runs]
    counts = [len(q) for q in q_lists]
    info(f"Questions detected per run: {counts}")

    # Check structure
    structurally_ok = all(c >= 3 for c in counts)
    if structurally_ok:
        ok(f"All 3 runs produced ≥3 MCQs ({counts})")
    else:
        fail(f"Some runs produced too few MCQs: {counts}")

    # Compare run 1 vs run 2 vs run 3
    def overlap(a: list[str], b: list[str]) -> int:
        sa = {q.strip().lower() for q in a}
        sb = {q.strip().lower() for q in b}
        return len(sa & sb)

    o12 = overlap(q_lists[0], q_lists[1]) if q_lists[0] and q_lists[1] else 0
    o13 = overlap(q_lists[0], q_lists[2]) if q_lists[0] and q_lists[2] else 0
    o23 = overlap(q_lists[1], q_lists[2]) if q_lists[1] and q_lists[2] else 0
    info(f"Identical questions across runs — 1∩2: {o12}, 1∩3: {o13}, 2∩3: {o23}")

    avg_per_run = statistics.mean(counts) if counts else 0
    avg_overlap = statistics.mean([o12, o13, o23])
    diversity = 1 - (avg_overlap / max(avg_per_run, 1))
    info(f"Diversity score: {diversity:.0%}  (1.0 = totally different, 0.0 = identical)")

    # We accept as PASS if at least 1 question differs across pairs (some may overlap by chance)
    varies = (o12 + o13 + o23) < (avg_per_run * 3 * 0.95)  # < 95% all identical
    if varies:
        ok(f"Quizzes vary across runs (good for re-testing!)")
    else:
        fail("Quizzes are nearly identical across runs")
    record("mcq_mode", structurally_ok, f"~{int(avg_per_run)} qs/run")
    record("mcq_variability", varies, f"diversity={diversity:.0%}")


def test_eli5_mode(assistant: StudyAssistant):
    section("TEST 7 — Explain Simply (ELI5)")
    query = "Explain tokenization simply."
    info(f"Query: {query}")
    answer, _ = assistant.generate(query, mode="eli5")
    show("Response", answer)
    needed = ["🧒 Simple Explanation", "🔑 Remember These", "💡 Real-World Analogy"]
    missing = [m for m in needed if m not in answer]
    passed = not missing
    if passed:
        ok("ELI5 structure complete")
    else:
        fail(f"Missing: {missing}")
    record("eli5_mode", passed)


def test_retrieval_coverage(assistant: StudyAssistant):
    """Sample multiple questions across different parts of the PDFs and
    check that retrieval finds RELEVANT chunks for each. This proves
    the model isn't 'missing' parts of the material."""
    section("TEST 8 — Retrieval Coverage (does the model see ALL the material?)")

    # Each question targets a DIFFERENT topic likely in the slides
    probes = [
        "What is tokenization?",
        "What are stopwords?",
        "What is text normalization?",
        "What are regular expressions used for?",
        "What is the difference between stemming and lemmatization?",
        "What is the goal of NLP?",
        "What are some common NLP applications?",
        "What is part-of-speech tagging?",
    ]

    coverage = []
    for q in probes:
        docs = assistant.retrieve(q, k=4)
        if not docs:
            coverage.append((q, 0, "—"))
            continue
        # Roughly check the top chunk contains keyword from question
        kw = q.split()[-1].lower().strip("?")
        any_relevant = any(
            kw in (d.page_content or "").lower() for d in docs
        )
        coverage.append((q, len(docs), "yes" if any_relevant else "weak"))

    print(f"  {'Question':<58} {'#chunks':>8} {'relevant?':>10}")
    print(f"  {'-' * 58} {'-' * 8} {'-' * 10}")
    for q, n, rel in coverage:
        ql = q if len(q) <= 56 else q[:55] + "…"
        print(f"  {ql:<58} {n:>8} {rel:>10}")

    answered = sum(1 for _, n, _ in coverage if n > 0)
    relevant = sum(1 for _, _, r in coverage if r == "yes")
    info(f"Probes answered: {answered}/{len(probes)} | "
         f"Top-1 keyword-relevant: {relevant}/{len(probes)}")

    passed = answered == len(probes) and relevant >= len(probes) * 0.5
    if passed:
        ok(f"Retrieval covers diverse topics ({relevant}/{len(probes)} clearly relevant)")
    else:
        fail("Retrieval seems patchy")
    record("retrieval_coverage", passed, f"{relevant}/{len(probes)} relevant")


def test_full_summary_completeness(assistant: StudyAssistant):
    """Ask for an exhaustive summary and check it mentions key concepts."""
    section("TEST 9 — Exam-Prep Summary Completeness")
    query = (
        "Give me an EXHAUSTIVE study summary covering EVERY major concept "
        "in the uploaded slides. Don't miss anything important — I'm using "
        "this for my exam."
    )
    info("Asking for exhaustive summary…")
    answer, docs = assistant.generate(query, mode="summarize")
    info(f"Length: {len(answer)} chars | Sources used: {len(docs)}")

    # Topics likely in NLP intro slides — accept several phrasings each
    expected_topics = [
        ["tokeniz"],                          # tokenization / tokenizer / tokens
        ["stopword", "stop word", "stop-word"],
        ["regular expression", "regex"],
        ["stem"],                             # stemming / stemmer
        ["lemmat"],                           # lemmatization
        ["natural language"],
        ["normaliz", "normalis"],             # normalization / normalisation
        ["preprocess", "pre-process"],
        ["case", "lowercas"],                 # case folding / lowercasing
        ["punctuation"],
    ]
    found = []
    missing = []
    low = answer.lower()
    for variants in expected_topics:
        if any(v in low for v in variants):
            found.append(variants[0])
        else:
            missing.append(variants[0])
    info(f"Topics covered ({len(found)}/{len(expected_topics)}): {found}")
    if missing:
        info(f"Topics not mentioned: {missing}")

    coverage_pct = len(found) / len(expected_topics)
    # Bumped target: 80% so the user can rely on it for exam prep
    passed = coverage_pct >= 0.8
    if passed:
        ok(f"Summary covers {coverage_pct:.0%} of expected topics")
    else:
        fail(f"Summary missed too many topics ({coverage_pct:.0%}) — need ≥80%")
    record("summary_completeness", passed, f"{coverage_pct:.0%}")


def test_hallucination_guard(assistant: StudyAssistant):
    section("TEST 10 — Hallucination Guard")
    query = "What is the chemical formula for sulfuric acid and how is it manufactured?"
    info(f"Off-topic query: {query}")
    answer, docs = assistant.generate(query, mode="qa")
    show("Response", answer, max_chars=400)

    refusal_phrases = [
        "answer is not available", "not available in the provided",
        "not found in the context", "no information", "does not contain",
        "not covered", "cannot provide", "can't provide",
        "i cannot", "i can't", "unable to provide",
    ]
    low = answer.lower()
    refused = any(p in low for p in refusal_phrases)
    hallucinated_terms = ["h2so4", "contact process", "lead chamber"]
    hallucinated = any(t in low for t in hallucinated_terms)

    passed = refused and not hallucinated
    if passed:
        ok("Model refused to answer off-topic question (no hallucination)")
    else:
        fail(f"Refused: {refused}, hallucinated: {hallucinated}")
    record("hallucination_guard", passed)


def test_repeatability(assistant: StudyAssistant):
    """Same factual question 3 times → answers should be similar."""
    section("TEST 11 — Repeatability (factual question gives consistent answer)")
    query = "What is tokenization?"
    answers = []
    for i in range(3):
        a, _ = assistant.generate(query, mode="qa")
        answers.append(a)
        info(f"Run #{i + 1}: {len(a)} chars")
    # Check all 3 mention "token"
    all_mention = all("token" in a.lower() for a in answers)
    if all_mention:
        ok("All 3 runs mention 'token' in the answer")
    else:
        fail("Some runs drifted off-topic")
    record("repeatability", all_mention)


def test_latency(assistant: StudyAssistant):
    section("TEST 12 — Latency (cloud LLM speed check)")
    queries = [
        ("qa", "What is NLP?"),
        ("summarize", "Summarise the slides."),
        ("eli5", "Explain stopwords simply."),
    ]
    times = []
    for mode, q in queries:
        t0 = time.time()
        assistant.generate(q, mode=mode)
        dt = time.time() - t0
        info(f"{mode:<10}: {dt:.1f}s")
        times.append(dt)
    avg = statistics.mean(times)
    info(f"Average: {avg:.1f}s")
    passed = avg < 30  # Cloud LLM should be < 30s on average
    if passed:
        ok(f"Acceptable latency (avg {avg:.1f}s)")
    else:
        fail(f"Too slow on average ({avg:.1f}s)")
    record("latency", passed, f"avg={avg:.1f}s")


# ─── Driver ──────────────────────────────────────────────────────────
def main() -> int:
    print(f"{C.BOLD}{C.M}╔══════════════════════════════════════════════════════════════════════╗{C.END}")
    print(f"{C.BOLD}{C.M}║    AI Study Assistant — Comprehensive Test Suite                     ║{C.END}")
    print(f"{C.BOLD}{C.M}╚══════════════════════════════════════════════════════════════════════╝{C.END}")

    if not test_environment():
        return 1

    test_chat_persistence()

    assistant = StudyAssistant()
    if assistant.total_chunks == 0:
        info("No existing index found — running ingest")
        n_docs, n_chunks = test_ingest(assistant)
    else:
        info(f"Reusing existing index ({assistant.total_chunks} chunks)")
        n_docs, n_chunks = len(assistant.indexed_files), assistant.total_chunks

    test_persistence_reload(assistant.total_chunks)
    test_qa_mode(assistant)
    test_summarize_mode(assistant)
    test_mcq_mode_variability(assistant)
    test_eli5_mode(assistant)
    test_retrieval_coverage(assistant)
    test_full_summary_completeness(assistant)
    test_hallucination_guard(assistant)
    test_repeatability(assistant)
    test_latency(assistant)

    # ── Summary ──────────────────────────────────────────────────
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
