"""Provides supporting functions to create and execute the llm chatbot."""
import os
import random

import requests.exceptions
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def get_intro():
    """
    Selects a random introductory greeting from a predefined list.

    Returns:
        str: A randomly chosen greeting message.
    """
    intros = [
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
    ]
    return random.choice(intros)


def get_outro():
    """
    Selects a random goodbye message from a predefined list.

    Returns:
        str: A randomly chosen goodbye message.
    """
    goodbyes = [
        "Goodbye! Have a great day!",
        "Farewell! It was nice chatting with you.",
        "Take care! Until next time.",
        "Goodbye for now! Stay awesome!",
        "Adios! Catch you later!",
        "Bye bye! Enjoy your day!",
        "See you later! It's been a pleasure.",
        "So long! Until we meet again.",
        "Goodbye, my friend! Take care.",
        "Bye for now! Stay safe and happy!",
    ]
    return random.choice(goodbyes)


def print_intro():
    """
    Print the introductory message.
    """
    print(f"vGPT: {get_intro()}")


def get_random_response():
    """
    Selects a random funny response from a predefined list.

    Returns:
        str: A randomly chosen funny response.
    """
    responses = [
        "Did you just ask the meaning of life, the universe, and everything?",
        "Hmm... my circuits are tingling with curiosity!",
        "Great! Your question just made my day.",
        "Okay, but have you considered asking a cat for advice?",
        "Interesting! You must be a professional question-asker.",
        "Tell me more, and I'll tell you a joke in return!",
        "Well, you're certainly full of surprises! Or is it just random chance?",
        "I appreciate your input, even if it's stranger than fiction.",
        "I see what you mean, or at least I think I do! Can you clarify?",
        "That's a real head-scratcher, but don't worry, I won't scratch too hard!",
    ]
    return random.choice(responses)


def check_end_chat(text):
    """
    Checks if the given text indicates the end of a chat session.

    Args:
        text (str): The user's input text.

    Returns:
        bool: True if the text indicates the end of the chat, False otherwise.
    """
    return text.lower().strip() in [
        "bye",
        "quit",
        "exit",
        "goodbye",
        "stop",
        "terminate",
        "done",
        "finish",
    ]


def initialize_model_and_tokenizer(path=None):
    """
    Initialize and return the model and tokenizer from Hugging Face.
    The function checks if the model and tokenizer are saved locally. If not present
    locally, it downloads them from the Hugging Face repository and saves them in
    these directories. Subsequent calls to this function will use the
    local copies, avoiding the need to download them again.

    Returns:
        model (AutoModelForSeq2SeqLM): The model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
    """
    if path is None:
        path = "../../../.."

    model_name = "facebook/blenderbot-400M-distill"
    clean_model_name = model_name.replace("/", "_")
    model_path = f"{path}/assets/machine_learning/model/{clean_model_name}"
    tokenizer_path = f"{path}/assets/machine_learning/tokenizer/{clean_model_name}"

    # Ensure directories exist
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(tokenizer_path):
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    # Load or download model
    if os.path.exists(model_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(model_path)

    # Load or download tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.save_pretrained(tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_user_input():
    """
    Get user input
    """
    user_text = input("You: ")
    return user_text


def encode_input(text, tokenizer):
    """
    Encode the input text using the provided tokenizer and add an end-of-sequence token.

    Args:
        text (str): The input text to be encoded.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding.

    Returns:
        torch.Tensor: A PyTorch tensor containing the encoded input text with an
        end-of-sequence token.
    """
    eos_text = f"{text}{tokenizer.eos_token}"
    return tokenizer.encode(eos_text, return_tensors="pt")


def generate(model, tokenizer, instruction, knowledge, dialog):
    """
    Generate a response using a language model.

    This function takes a language model, a tokenizer, an instruction, optional
    knowledge, and a dialog history to generate a response. The dialog history is
    converted into a single text string, and if knowledge is provided, it is
    incorporated into the query. The function then encodes the query using the
    tokenizer, and uses the model to generate a response based on the encoded input.
    The response is decoded and returned.

    Args:
        model (PreTrainedModel): The language model to be used for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        instruction (str): The instruction or prompt to guide the model's response.
        knowledge (str): Optional additional knowledge to provide context for
        the generation.
        dialog (list of str): The dialog history, a list of string messages.

    Returns:
        str: The generated response from the model.
    """
    # Join dialog history into a single text string
    dialog_text = " EOS ".join(dialog)

    # Construct the query incorporating the instruction, dialog, and optional knowledge
    if knowledge:
        query = f"{instruction} [CONTEXT] {dialog_text} [KNOWLEDGE] {knowledge}"
    else:
        query = f"{instruction} [CONTEXT] {dialog_text}"

    # Encode the query using the tokenizer
    input_ids = tokenizer.encode(query, return_tensors="pt")

    # Generate a response from the model
    outputs = model.generate(
        input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True
    )

    # Decode the output to a human-readable format
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output


def process_input(user_text):
    """
    Main execution function to run the chatbot.
    """
    instruction = (
        f"You are friendly, helpful, and conversational. Respond to the following:\n"
    )
    knowledge = ""
    dialog = []
    model_path = "assets/machine_learning/model/facebook_blenderbot-400M-distill"
    tokenizer_path = (
        "assets/machine_learning/tokenizer/facebook_blenderbot-400M-distill"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dialog.append(user_text)
    if check_end_chat(user_text):
        return get_outro()
    response = generate(model, tokenizer, instruction, knowledge, dialog)
    dialog.append(response)
    if not response.strip():
        response = get_random_response()
    return response


if __name__ == "__main__":
    process_input("bye")
