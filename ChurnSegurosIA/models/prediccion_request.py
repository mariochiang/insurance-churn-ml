from pydantic import BaseModel

class PrediccionRequest(BaseModel):
    cliente_id: int
    producto_id: int