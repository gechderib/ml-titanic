from pydantic import BaseModel

class PredictionRequest(BaseModel):
    age: float = 24
    sex: str = "male"
    fare: float = 0.0
    pclass: int = 3
    SibSp: int = 0
    Parch: int = 0
    Embarked: str = "S"
    FamilySize: int = 1
    IsAlone: bool = False
    Title: str = "Unknown"

class PredictionResponse(BaseModel):
    survived: bool
    probability: float
    
    
    