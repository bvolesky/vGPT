"""Contains all the unit tests for maintaining TDD for the chatbot supporting toolkit"""

import unittest
import torch
from unittest.mock import patch, Mock
from scripts.toolkit import (
    get_intro,
    print_intro,
    get_random_response,
    check_end_chat,
    initialize_model_and_tokenizer,
    get_user_input,
    encode_input,
    generate,
)


class TestChatbotFunctions(unittest.TestCase):
    def test_get_intro(self):
        intro = get_intro()
        self.assertIn(
            intro,
            [
                "Hey there, I'm your friendly AI buddy! What's up?",
                "What's going on? Ready for a chat?",
                "Hello there! How's your day going?",
                "Hi, how's it going? Let's chat about whatever you feel like.",
                "Greetings! How's life treating you?",
                "Good day! Up for a casual conversation?",
                "Hi there! What's on your mind, or shall we just chat for fun?",
                "Hello! Lovely to see you around. What's new in your world?",
                "Hey, how's your day going? Let's have a friendly chat.",
                "Hi! Ready to have a relaxed conversation?",
            ],
        )

    @patch("builtins.print")
    def test_print_intro(self, mock_print):
        print_intro()
        mock_print.assert_called()

    def test_get_random_response(self):
        response = get_random_response()
        self.assertIsInstance(response, str)

    def test_check_end_chat(self):
        self.assertTrue(check_end_chat("bye"))
        self.assertFalse(check_end_chat("hello"))

    @patch("scripts.toolkit.os.path.exists")
    @patch("scripts.toolkit.AutoTokenizer.from_pretrained")
    @patch("scripts.toolkit.AutoModelForSeq2SeqLM.from_pretrained")
    def test_initialize_model_and_tokenizer(
        self, mock_model, mock_tokenizer, mock_exists
    ):
        model_local_path = (
            "../assets/machine_learning/model/facebook_blenderbot-400M-distill"
        )
        tokenizer_local_path = (
            "../assets/machine_learning/tokenizer/facebook_blenderbot-400M-distill"
        )

        # Scenario 1: The model and tokenizer are not already downloaded
        mock_exists.return_value = False
        model, tokenizer = initialize_model_and_tokenizer()
        mock_model.assert_called_with("facebook/blenderbot-400M-distill")
        mock_tokenizer.assert_called_with(
            "facebook/blenderbot-400M-distill", padding_side="left"
        )
        mock_model.reset_mock()
        mock_tokenizer.reset_mock()

        # Scenario 2: The model and tokenizer are already downloaded
        mock_exists.return_value = True
        model, tokenizer = initialize_model_and_tokenizer()
        mock_model.assert_called_with(model_local_path)
        mock_tokenizer.assert_called_with(tokenizer_local_path)

    @patch("builtins.input", return_value="Test input")
    def test_get_user_input(self, mock_input):
        self.assertEqual(get_user_input(debug_mode=False), "Test input")
        self.assertEqual(get_user_input(debug_mode=True), "I just ate an apple!")

    @patch("scripts.toolkit.AutoTokenizer")
    def test_encode_input(self, mock_tokenizer):
        mock_encoded_output = torch.tensor([[0, 1, 2, 3, 4]])
        mock_tokenizer.return_value.encode.return_value = mock_encoded_output
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        encoded_input = encode_input("Hello", mock_tokenizer.return_value)
        self.assertIsInstance(encoded_input, torch.Tensor)
        self.assertTrue(torch.equal(encoded_input, mock_encoded_output))





if __name__ == "__main__":
    unittest.main()
