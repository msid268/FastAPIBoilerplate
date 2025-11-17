from contextvars import ContextVar
from typing import Optional

current_request_id: ContextVar[Optional[str]] = ContextVar("current_request_id", default=None)

def get_request_id() -> Optional[str]:
    return current_request_id.get()

def set_request_id(rid: str):
    current_request_id.set(rid)