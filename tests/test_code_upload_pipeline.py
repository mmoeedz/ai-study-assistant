"""Test suite verifying the Code Mode image and file uploading integration inside app.py and rag_pipeline.py."""
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Setup path imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from rag_pipeline import StudyAssistant, Document

class TestCodeUploadPipeline(unittest.TestCase):
    def setUp(self):
        # Temporarily mock LLM clients to avoid requiring API keys during test run
        self.original_llm_provider = config.LLM_PROVIDER
        config.LLM_PROVIDER = "groq"
        config.GROQ_API_KEY = "gsk_fake_test_key_for_pipeline_validation"

    def tearDown(self):
        config.LLM_PROVIDER = self.original_llm_provider

    @patch('rag_pipeline.GroqClient.generate')
    def test_coding_mode_with_code_file_content(self, mock_generate):
        # Configure mock to return a simulated response
        mock_generate.return_value = "### Corrected Code:\n```python\nprint('hello world')\n```\n- Fixed syntax"

        assistant = StudyAssistant()
        
        # Simulate how app.py builds the prompt with code files
        filename = "test_script.py"
        code_text = "def hello():\n    print('hello world')\n"
        user_prompt = "Optimize this hello function"
        
        prompt_to_llm = f"{user_prompt}\n\n---\n\n### 📄 Uploaded Code File: `{filename}`\n```py\n{code_text}\n```\n"
        
        # Run generation
        response, source_docs = assistant.generate(
            prompt_to_llm,
            mode="coding",
            image_base64=None,
            history=[]
        )
        
        # Verify the prompt is successfully passed to LLM client with expected format
        mock_generate.assert_called_once()
        called_args, called_kwargs = mock_generate.call_args
        
        self.assertIn("Optimize this hello function", called_kwargs['prompt'])
        self.assertIn("### 📄 Uploaded Code File: `test_script.py`", called_kwargs['prompt'])
        self.assertIn("def hello():", called_kwargs['prompt'])
        self.assertEqual(response, mock_generate.return_value)
        self.assertEqual(source_docs, [])
        print("✓ Coding mode with code file content successfully validated!")

    @patch('rag_pipeline.GroqClient.generate')
    def test_coding_mode_with_image(self, mock_generate):
        mock_generate.return_value = "This screenshot shows a NameError in line 5."

        assistant = StudyAssistant()
        user_prompt = "What is the error in this image?"
        fake_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        response, source_docs = assistant.generate(
            user_prompt,
            mode="coding",
            image_base64=fake_image_base64,
            history=[]
        )
        
        mock_generate.assert_called_once()
        called_args, called_kwargs = mock_generate.call_args
        
        self.assertEqual(called_kwargs['image_base64'], fake_image_base64)
        self.assertEqual(response, mock_generate.return_value)
        print("✓ Coding mode with image base64 successfully validated!")

if __name__ == '__main__':
    unittest.main()
