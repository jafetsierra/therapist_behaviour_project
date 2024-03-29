from pydantic import BaseModel

class ApiResponse(BaseModel):
    predictions: int

class ApiRequest(BaseModel):
    text: str