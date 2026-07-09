"""Quick test of all 4 RAG modes: Q&A, Summarize, MCQ, ELI5"""
import os, sys, json
from pathlib import Path

# Setup
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

import config
from rag_pipeline import StudyAssistant

print("="*70)
print("  ALL FUNCTIONS TEST — Q&A, Summarize, MCQ, ELI5")
print("="*70)

assistant = StudyAssistant()
print(f"\n✓ StudyAssistant initialized")
print(f"  • Indexed files: {len(assistant.indexed_files)}")
print(f"  • Total chunks: {assistant.total_chunks}")

if assistant.total_chunks == 0:
    print("\n⚠️  No documents indexed. Skipping tests.")
    sys.exit(1)

# Test each mode
modes = ["qa", "summarize", "mcq", "eli5"]
modes_display = {
    "qa": "❓ Q&A (Inquire)",
    "summarize": "📝 Summarize", 
    "mcq": "📋 MCQ (Quiz Me)",
    "eli5": "💡 ELI5 (Explain Simply)"
}

for mode in modes:
    print(f"\n{'─'*70}")
    print(f"Testing: {modes_display[mode]}")
    print(f"{'─'*70}")
    try:
        query = "What is the main topic covered?"
        answer, docs = assistant.generate(query, mode=mode)
        
        # Check response
        has_content = len(answer) > 50
        has_sources = len(docs) > 0
        
        status = "✅ PASS" if (has_content and has_sources) else "⚠️  PARTIAL"
        print(f"{status}")
        print(f"  • Response length: {len(answer)} chars")
        print(f"  • Sources retrieved: {len(docs)} chunks")
        
        # Show snippet
        snippet = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"  • Preview: {snippet[:100]}")
        
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {str(e)[:100]}")

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}\n")
