# 🎓 AI Study Assistant — RAG-Powered Learning Companion

An advanced, academic-themed study companion that helps students interact with their learning material. Upload PDFs, Word documents, or text files, build a local vector index, and explore your materials through four sophisticated academic sections: **Document Q&A**, **Detailed Summary**, **Interactive Quiz**, and **Coding Mode**. 

All powered by local/cloud retrieval-augmented generation (RAG) using **LLaMA 3.1 (8B)**.

---

## ✨ Features

| Mode | Functionality & Characteristics |
|---|---|
| 📄 **Document Q&A** | Chat with your documents. Ask questions and get answers structured with direct responses, bulleted key points, and detailed source references (showing filename and page numbers). Supports automated welcoming summaries. |
| 📝 **Detailed Summary** | Generates an exhaustive, exam-ready comprehensive summary of **all** document text. Employs a robust **Map-Reduce** batching pipeline to safely process infinite chunks without hitting LLM payload/context boundaries. |
| 🏆 **Interactive Quiz** | Generates a custom, 10-question multiple-choice quiz from across 15 diverse material chunks. Includes an interactive graded scorecard, percentage metrics, pass/fail thresholds, and a breakdown of answer keys with reasoning. |
| 💻 **Code Mode** | An expert programming tutor & debugger. Automatically adapts to your format/style (snake_case, camelCase, naming styles) and supports **multimodal visual references** (pasting or dragging screenshots/compiler errors). |

---

## 🏗️ Architecture

```
                       ┌────────────────────────┐
                       │   Uploaded Materials   │
                       │ (PDF, DOCX, TXT, MD)   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Text Extraction &     │
                       │   Recursive Chunking   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    Embedding Model     │
                       │  (BGE-small / Nomic)   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   SimpleVectorStore    │
                       │   (JSON + NumPy .npy)  │
                       └───────────┬────────────┘
                                   │ ▲
                Similarity Search  │ │  Query Vectorization
               (Cosine dot product)│ │
                                   ▼ │
                       ┌────────────────────────┐
                       │   Retrieved Context    │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │     Prompt Mapping     │
                       │ (QA/Summary/Quiz/Code) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │      LLM Engine        │
                       │ (Groq Cloud / Ollama)  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Streamlit Academic UI  │
                       └────────────────────────┘
```

---

## 🚀 Two ways to run it

### Option A — Cloud (free, instant, no install)
👉 **[Click the demo link](#)** — works on any device, no local setup required (requires a Groq API Key).

### Option B — Run locally (privacy / offline)

**Prerequisites**
- Python 3.10+
- A free [Groq API Key](https://console.groq.com/keys) (for Cloud Mode)
  *or* [Ollama](https://ollama.com/download) running on your local machine (for Local Mode)

**Setup**
```bash
git clone https://github.com/mmoeedz/ai-study-assistant.git
cd ai-study-assistant
pip install -r requirements.txt
```

**Cloud Mode (Groq API):**
```bash
# Create .streamlit/secrets.toml with your Groq key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your gsk_... key
streamlit run app.py
```

**Local-only Mode (Ollama):**
```bash
# Pull the models (first time only, ~5 GB total)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Set the provider to ollama in .streamlit/secrets.toml
# LLM_PROVIDER = "ollama"
streamlit run app.py
```

The app opens automatically at <http://localhost:8501>.

---

## ⚙️ Configuration

All major settings are loaded dynamically from environment variables or `.streamlit/secrets.toml`:

| Secret / env var | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Choose `groq` (cloud) or `ollama` (local) |
| `GROQ_API_KEY` | _required for groq_ | Your `gsk_...` console key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Any Groq-supported LLM |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model tag for the LLM |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model tag |

Tunables such as chunk sizes, retrieval count, and model temperatures live inside `config.py`.

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit frontend with ChatGPT-style sidebar & live quiz form
├── rag_pipeline.py         # Pure Python vector database, text extraction, & generation
├── prompts.py              # Prompt definitions for QA, Summarize, Coding, and Quiz
├── config.py               # Central environment variable switches & tunable boundaries
├── requirements.txt        # Core Python requirements (Streamlit, NumPy, fastembed, pypdf)
├── .streamlit/
│   ├── config.toml         # Custom server and theme properties
│   └── secrets.toml        # (gitignored) Active provider keys
├── tests/
│   ├── smoke_groq.py       # Groq-pipeline smoke validator
│   ├── test_all_modes.py   # Full execution validation across QA, Summarize, Code, Quiz
│   ├── test_e2e.py         # End-to-end integration test suite
│   ├── test_full.py        # Exhaustive validation runner
│   └── test_summary_only.py# Summary-coverage metric tester
└── vectorstore/            # (gitignored) persisted index (documents.json + embeddings.npy)
```

---

## 🧪 Testing

The system includes pre-written testing runners to validate integration reliability:

```bash
# Cloud-mode smoke test (Groq)
python -X utf8 tests/smoke_groq.py

# Complete multi-mode pipeline execution test
python -X utf8 tests/test_all_modes.py

# Full integration test suite
python -X utf8 tests/test_e2e.py
```

---

## 🛠️ Tech Stack

- **LLM Engine**: Groq Cloud (`llama-3.1-8b-instant`) or Ollama (`llama3.1:8b`).
- **Embeddings**: Local `fastembed` (BGE-small ONNX) or Ollama (`nomic-embed-text`).
- **Vector Database**: Pure-Python `SimpleVectorStore` implementing fast vectorized cosine similarity with NumPy.
- **Data Serialization**: Secure and cross-platform native JSON combined with float32 `.npy` arrays (no arbitrary code execution vulnerabilities like standard Pickle).
- **Parsers**: Native PDF parsing via `pypdf`, Word document content XML parsing via `zipfile`/`xml.etree`, and raw text encoding recovery.
- **GUI Layout**: Dark-themed academic visual skin using custom CSS style sheet injection in Streamlit.

---

## 📜 License

MIT

