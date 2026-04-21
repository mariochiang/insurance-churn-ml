from pydantic import BaseModel

class PrediccionRequest(BaseModel):
    rut: str