from dataclasses import dataclass


@dataclass
class UserContext:
    user: str
    user_context: str
