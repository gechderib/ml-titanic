from pydantic import BaseModel

class PredictionRequest(BaseModel):
    age: float
    sex: str
    fare: float
    pclass: int


class PredictionResponse(BaseModel):
    survived: bool
    probability: float