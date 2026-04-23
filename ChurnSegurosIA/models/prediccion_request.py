from typing import Optional
from pydantic import BaseModel


class PrediccionRequest(BaseModel):
    rut: str


class ChatRequest(BaseModel):
    rut: Optional[str] = None
    mensaje: Optional[str] = None
