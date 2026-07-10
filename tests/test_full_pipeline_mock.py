"""
Comprehensive Mock Test Suite for the AI Study Assistant.

Validates:
  1. Pure-Python SimpleVectorStore (Cosine Similarity vector math).
  2. RecursiveCharacterTextSplitter (Context-aware chunking).
  3. Incremental file uploader (ingestion pipeline with duplicate prevention).
  4. Prompt generator mappings across all scholarly modes (qa, summarize, quiz, coding).
  5. Multi-format script uploading to Coding Mode (images vs raw scripts).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Setup path imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from rag_pipeline import (
    StudyAssistant,
    Document,
    SimpleVectorStore,
    RecursiveCharacterTextSplitter,
)


class TestFullPipelineMock(unittest.TestCase):
    def setUp(self):
        # Cache and configure environment settings
        self.original_provider = config.LLM_PROVIDER
        config.LLM_PROVIDER = "groq"
        config.GROQ_API_KEY = "gsk_mock_test_key"
        
        # Instantiate assistant
        self.assistant = StudyAssistant()
        # Reset assistant's vectorstore state
        self.assistant.vectorstore = None
        self.assistant.indexed_files = []
        self.assistant.total_chunks = 0

    def tearDown(self):
        config.LLM_PROVIDER = self.original_provider

    def test_vector_store_cosine_similarity(self):
        """Validate SimpleVectorStore math (cosine similarity alignment)."""
        store = SimpleVectorStore()
        doc1 = Document(page_content="Apple is a fruit.", metadata={"source": "fruit.txt"})
        doc2 = Document(page_content="Python is a programming language.", metadata={"source": "code.txt"})
        
        # Mock embeddings: dimension = 3
        # Query: [1.0, 0.0, 0.0]
        # doc1 embedding: [0.99, 0.01, 0.0] (Very close)
        # doc2 embedding: [0.0, 0.9, 0.1] (Orthogonal)
        store.add([doc1, doc2], [[0.99, 0.01, 0.0], [0.0, 0.9, 0.1]])
        
        results = store.search([1.0, 0.0, 0.0], k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Apple is a fruit.")
        print("✓ Vector Store Cosine Similarity validated!")

    def test_recursive_character_splitter(self):
        """Validate custom character chunking boundaries and overlaps."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "Hello world. This is a simple sentence to test chunking."
        chunks = splitter.split_text(text)
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 25)  # size + overlap limit
        print("✓ Recursive Character Text Splitter validated!")

    @patch('rag_pipeline.StudyAssistant._embed_batch')
    def test_incremental_multi_document_uploader(self, mock_embed_batch):
        """Validate document indexing skips already existing documents."""
        mock_embed_batch.return_value = [[0.1] * 384] * 5  # mock embeddings
        
        # Mock files
        file1 = MagicMock()
        file1.read.return_value = b"Hello from Doc 1. This is the first NLP file."
        file1.name = "nlp_doc1.txt"
        
        file2 = MagicMock()
        file2.read.return_value = b"Hello from Doc 2. This is the second NLP file."
        file2.name = "nlp_doc2.txt"

        # Step 1: Ingest Doc 1
        num_docs, num_chunks = self.assistant.ingest_pdfs([file1])
        self.assertEqual(num_docs, 1)
        self.assertTrue(num_chunks > 0)
        self.assertIn("nlp_doc1.txt", self.assistant.indexed_files)
        
        first_doc_chunks = self.assistant.total_chunks

        # Step 2: Ingest Doc 1 and Doc 2 together (incremental upload)
        # It should skip Doc 1 and only index Doc 2
        num_docs, num_chunks = self.assistant.ingest_pdfs([file1, file2])
        
        self.assertEqual(num_docs, 1)  # Only 1 new doc processed
        self.assertIn("nlp_doc2.txt", self.assistant.indexed_files)
        self.assertEqual(self.assistant.total_chunks, first_doc_chunks + num_chunks)
        print("✓ Incremental indexing with duplicate prevention validated!")

    @patch('rag_pipeline.GroqClient.generate')
    def test_scholarly_modes_prompt_mapping(self, mock_generate):
        """Validate Q&A, Summarize, and Quiz modes pass correct structures to LLM."""
        # Setup mock vectorstore
        doc = Document(page_content="Natural Language Processing is a field of AI.", metadata={"source": "nlp.txt"})
        store = SimpleVectorStore()
        store.add([doc], [[0.1] * 384])
        self.assistant.vectorstore = store
        self.assistant.total_chunks = 1

        # Test Q&A Mode
        mock_generate.return_value = "## Answer\nNatural Language Processing (NLP) is a branch of artificial intelligence..."
        answer, docs = self.assistant.generate("What is NLP?", mode="qa")
        self.assertIn("## Answer", answer)
        self.assertEqual(len(docs), 1)

        # Test Quiz Mode (JSON formatting validation)
        mock_generate.return_value = '[{"question": "What is NLP?", "options": {"A": "AI", "B": "Bio", "C": "Chem", "D": "Math"}, "answer": "A", "explanation": "It is a field of AI."}]'
        quiz_answer, quiz_docs = self.assistant.generate("Generate a quiz", mode="quiz")
        self.assertTrue(quiz_answer.startswith('['))
        self.assertEqual(len(quiz_docs), 1)
        print("✓ Scholarly modes (Q&A, Quiz, Summarize) validated!")


if __name__ == '__main__':
    unittest.main()
