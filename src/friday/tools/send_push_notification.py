import os
from dotenv import load_dotenv
import requests

load_dotenv()


def send_push_notification(notification: str):
    """
    Send a notification to Rajeev Siewnath.

    Args:
        notification: The notification to send

    Returns:
        Whether the notification was sent successfully
    """
    pushover_user = os.getenv("PUSHOVER_USER")
    pushover_token = os.getenv("PUSHOVER_TOKEN")
    payload = {
        "user": pushover_user,
        "token": pushover_token,
        "message": notification,
        "sound": "incoming",
    }
    requests.post("https://api.pushover.net/1/messages.json", data=payload)
    return True
