from pydantic import BaseModel

class PredictionRequest(BaseModel):
    age: float
    sex: str
    fare: float
    pclass: int
    SibSp: int
    Parch: int
    Embarked: str
    FamilySize: int
    IsAlone: bool
    Title: str

class PredictionResponse(BaseModel):
    survived: bool
    probability: float
    
    
    