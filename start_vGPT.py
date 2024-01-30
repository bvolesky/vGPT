import os
import subprocess
import webbrowser
from threading import Timer


def install_requirements():
    """Install required packages."""
    try:
        os.system('pip install -r requirements.txt')
    except subprocess.CalledProcessError:
        print("Failed to install requirements.")
        exit(1)


def download_model_and_tokenizer():
    """Download the model and tokenizer if they are not already downloaded."""
    if not os.listdir("assets/machine_learning/model") and not os.listdir(
        "assets/machine_learning/tokenizer"
    ):
        from vGPTWebApp.chatbot.app.scripts.toolkit import (
            initialize_model_and_tokenizer,
        )

        initialize_model_and_tokenizer(os.getcwd())


def start_django_server():
    """Start the Django server."""
    try:
        os.system("python vGPTWebApp/manage.py runserver")
    except subprocess.CalledProcessError:
        print("Failed to start Django server.")
        exit(1)


def open_browser_after_delay(delay):
    """Open a web browser after a delay."""
    try:
        Timer(delay, lambda: webbrowser.open("http://127.0.0.1:8000/")).start()
    except Exception as e:
        print(f"Failed to open browser: {e}")
        exit(1)


def main():
    install_requirements()
    download_model_and_tokenizer()
    open_browser_after_delay(3)
    start_django_server()


if __name__ == "__main__":
    main()
