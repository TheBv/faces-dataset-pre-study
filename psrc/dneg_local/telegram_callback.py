import json
import os
import requests
from transformers import TrainerCallback

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


# optional: without the credentials file all messages are skipped
try:
    with open(f"{BP}/data/MISC/telegram.json", "r") as f:
        telegram_credentials = json.load(f)
except FileNotFoundError:
    telegram_credentials = {}

TELEGRAM_BOT_TOKEN = telegram_credentials.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = telegram_credentials.get("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    """Send a message to a Telegram chat using the Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",  # Optional: for formatted text
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to send Telegram message: {e}")

class TelegramLoggingCallback(TrainerCallback):
    """Callback to send evaluation metrics to Telegram per epoch."""
    def on_evaluate(self, args, state, control, **kwargs):
        # Called after each evaluation (per epoch)
        epoch = state.epoch
        try:
            message = f"*Epoch {epoch} Evaluation*\nF1-Score: {kwargs['metrics']['eval_f1']}"
        except:
            message = f"*Epoch {epoch} Evaluation*\nEMPTY"
        send_telegram_message(message)