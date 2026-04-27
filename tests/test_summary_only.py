"""Run ONLY the summary-completeness test for fast iteration."""
import os, sys, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
secrets_path = ROOT / ".streamlit" / "secrets.toml"
if secrets_path.exists():
    secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    for k, v in secrets.items():
        os.environ.setdefault(k, str(v))

from rag_pipeline import StudyAssistant  # noqa: E402

a = StudyAssistant()
print(f"Indexed: {len(a.indexed_files)} docs / {a.total_chunks} chunks")

query = (
    "Give me an EXHAUSTIVE study summary covering EVERY major concept "
    "in the uploaded slides. Don't miss anything important — I'm using "
    "this for my exam."
)
print("\nGenerating exhaustive summary…")
answer, docs = a.generate(query, mode="summarize")
print(f"Length: {len(answer)} chars | Chunks used: {len(docs)}")

print("\n--- ANSWER ---")
print(answer)
print("\n--- END ---\n")

expected_topics = [
    ["tokeniz"],
    ["stopword", "stop word", "stop-word"],
    ["regular expression", "regex"],
    ["stem"],
    ["lemmat"],
    ["natural language"],
    ["normaliz", "normalis"],
    ["preprocess", "pre-process"],
    ["case", "lowercas"],
    ["punctuation"],
]
low = answer.lower()
found, missing = [], []
for variants in expected_topics:
    if any(v in low for v in variants):
        found.append(variants[0])
    else:
        missing.append(variants[0])
pct = len(found) / len(expected_topics)
print(f"Coverage: {pct:.0%}  ({len(found)}/{len(expected_topics)})")
print(f"Found:   {found}")
print(f"Missing: {missing}")
print("\nPASS" if pct >= 0.80 else "FAIL — need >= 80%")
