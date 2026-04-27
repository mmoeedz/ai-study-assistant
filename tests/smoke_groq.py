"""Quick smoke test of the Groq + fastembed pipeline (no PDF needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force-load secrets the same way Streamlit does
import os, tomllib
secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
if secrets_path.exists():
    secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    for k, v in secrets.items():
        os.environ.setdefault(k, str(v))

import config  # noqa: E402
from rag_pipeline import StudyAssistant, Document  # noqa: E402

print(f"Provider:        {config.LLM_PROVIDER}")
print(f"LLM model:       {config.LLM_MODEL}")
print(f"Embedding model: {config.EMBEDDING_MODEL}")
print(f"GROQ_API_KEY set: {bool(config.GROQ_API_KEY)}")

print("\n[1/3] Initialising assistant…")
a = StudyAssistant()

print("[2/3] Building tiny in-memory index…")
docs = [
    Document(page_content="Photosynthesis converts sunlight into chemical energy in plants.",
             metadata={"source": "bio.pdf", "page": 1}),
    Document(page_content="Mitochondria are the powerhouse of the cell, producing ATP.",
             metadata={"source": "bio.pdf", "page": 2}),
    Document(page_content="The cell membrane is selectively permeable.",
             metadata={"source": "bio.pdf", "page": 3}),
]
a.build_vectorstore(docs)
print(f"  -> {a.total_chunks} chunks indexed")

print("\n[3/3] Asking a question via Groq…")
answer, sources = a.generate("What do mitochondria do?", mode="qa")
print("\n--- ANSWER ---")
print(answer)
print("\n--- SOURCES ---")
for s in sources:
    print(f"  • {s.metadata.get('source')} p.{s.metadata.get('page')}: {s.page_content[:60]}...")

# Cleanup the test vectorstore so it doesn't pollute real usage
a.clear_vectorstore()
print("\n[OK] Smoke test passed; test vectorstore cleared.")
