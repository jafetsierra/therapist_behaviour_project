from pydantic import BaseModel

class ApiResponse(BaseModel):
    predictions: str

class ApiRequest(BaseModel):
    text: str