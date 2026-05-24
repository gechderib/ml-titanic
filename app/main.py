from fastapi import FastAPI

from app.schemas import PredictionRequest, PredictionResponse
from app.model.predict import predict

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# swagger add with defautl value
@app.post("/prediction", response_model=PredictionResponse)
def predictResult(request: PredictionRequest):
    result = predict(request)
    return result