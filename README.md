# 🎓 AI Study Assistant — RAG-Powered Learning Companion

An academic study assistant that lets you upload PDFs and converse with them
through four scholarly modes — **Q&A**, **Summarise**, **Quiz Me**, and
**Explain Simply** — all powered by retrieval-augmented generation (RAG).

> **Try the live demo:** _link will appear here after deployment_

---

## ✨ Features

| Mode | What it does |
|---|---|
| ❓ **Inquire** | Cited Q&A with structured Answer / Key Points / Source Insight |
| 📝 **Summarise** | Themes, key concepts, and important details |
| 📋 **Quiz Me** | Auto-generated 5 MCQs with answer key |
| 💡 **Explain Simply** | ELI5 explanation with real-world analogies |

The system is grounded in your uploaded material — if the answer isn't in the
PDFs, it will say so rather than hallucinate.

---

## 🏗️ Architecture

```
PDF upload → text extraction (pypdf) → 1000-char chunks (200 overlap)
         ↓
   Embeddings  ────────────►  In-process vector store (NumPy cosine similarity)
   (BGE-small / nomic)        Persisted to disk via pickle
         ↑
   Query  →  top-k retrieval  →  prompt template  →  LLM  →  formatted answer
                                                    ↑
                            Groq (cloud) or Ollama (local)
```

---

## 🚀 Two ways to run it

### Option A — Cloud (free, instant, no install)
👉 **[Click the demo link](#)** — works on any device, no setup.

### Option B — Run locally (privacy / offline)

**Prerequisites**
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (cloud mode)
  *or* [Ollama](https://ollama.com/download) running locally

**Setup**
```bash
git clone https://github.com/mmoeedz/ai-study-assistant.git
cd ai-study-assistant
pip install -r requirements.txt
```

**Cloud mode (recommended for friends):**
```bash
# Create .streamlit/secrets.toml with your Groq key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your gsk_... key
streamlit run app.py
```

**Local-only mode (no API needed):**
```bash
# Pull the models (first time only, ~5 GB total)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Set the provider to ollama
echo 'LLM_PROVIDER = "ollama"' > .streamlit/secrets.toml
streamlit run app.py
```

The app opens at <http://localhost:8501>.

---

## ⚙️ Configuration

| Secret / env var | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` (cloud) or `ollama` (local) |
| `GROQ_API_KEY` | _required for groq_ | Your `gsk_...` key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Any Groq-supported model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama LLM model tag |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |

Other tunables (chunk size, retrieval-k, temperature) live in `config.py`.

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # Ingest / embed / retrieve / generate
├── prompts.py              # Prompt templates for each mode
├── config.py               # Provider switch + tunables
├── requirements.txt
├── .streamlit/
│   ├── config.toml         # Theme + server defaults
│   └── secrets.toml        # (gitignored) API keys
├── tests/
│   ├── test_e2e.py         # Full ollama-mode verification
│   └── smoke_groq.py       # Quick groq-mode smoke test
└── vectorstore/            # (gitignored) persisted index
```

---

## 🧪 Testing

```bash
# Cloud-mode smoke test (Groq)
python -X utf8 tests/smoke_groq.py

# Full ollama-mode E2E suite (needs Ollama running + PDFs)
python -X utf8 tests/test_e2e.py
```

---

## 🛠️ Tech Stack

- **LLM** — Groq (`llama-3.1-8b-instant`) or Ollama (`llama3.1:8b`)
- **Embeddings** — `fastembed` (BGE-small ONNX) or Ollama `nomic-embed-text`
- **Vector store** — Pure-Python NumPy cosine similarity (~zero deps)
- **PDF parsing** — `pypdf`
- **UI** — Streamlit with a custom academic theme

---

## 📜 License

MIT
