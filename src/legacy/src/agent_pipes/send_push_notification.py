import os
from dotenv import load_dotenv
import requests
from pipelines.pipe import Pipe

load_dotenv()


class SendPushNotification(Pipe[str, bool]):
    def __init__(self):
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")

    def run(self, input):
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": input,
            "sound": "incoming",
        }
        requests.post("https://api.pushover.net/1/messages.json", data=payload)
        return True
