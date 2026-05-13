from pydantic import BaseModel

class PredictionRequest(BaseModel):
    age: float
    sex: str
    fare: float
    pclass: int
    SibSp: int
    Parch: int

class PredictionResponse(BaseModel):
    survived: bool
    probability: float