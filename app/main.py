from fastapi import FastAPI

from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Placeholder for actual prediction logic
    return PredictionResponse(survived=True, probability=0.8)