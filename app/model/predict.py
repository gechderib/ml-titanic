# load model pkl file and make prediction on new data
import os
import pickle
from pathlib import Path

import pandas as pd
from app.schemas import PredictionRequest, PredictionResponse
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH', './models')
MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'titanic_model.pkl')

def load_model(model_file: str):
    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def _build_features(input_data: PredictionRequest) -> pd.DataFrame:
    sex_value = input_data.sex.strip().lower()
    if sex_value not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'")
    
    sex_map = {
        "male":0,
        "female":1
    }
    sex_encoded = sex_map.get(sex_value, 0)
    
    embarked_value = input_data.Embarked.strip().upper()
    if embarked_value not in {"S", "C", "Q"}:
        raise ValueError("Embarked must be 'S', 'C', or 'Q'")
    embarked_map = {
        "S": 0,
        "C": 1,
        "Q": 2
    }
    embarked_encoded = embarked_map.get(embarked_value, 0)
    
    title_value = input_data.Title.strip().capitalize()
    title_map = {
        "Mr": 0,
        "Miss": 1,
        "Mrs": 2,
        "Master": 3,
        "Sir": 4,
        "Other": 5
    }
    title_encoded = title_map.get(title_value, title_map['Other'])
        
    return pd.DataFrame([
        {
            "Pclass": input_data.pclass,
            "Sex": sex_encoded,
            "Age": input_data.age,
            "SibSp": input_data.SibSp,
            "Parch": input_data.Parch,
            "Fare": input_data.fare,
            "Embarked": embarked_encoded,   
            "FamilySize": input_data.FamilySize,
            "IsAlone": int(input_data.IsAlone), 
            "Title": title_encoded          
        }
    ])

def predict(input_data: PredictionRequest) -> PredictionResponse:
    model_file = os.path.join(MODEL_PATH, MODEL_FILENAME)
    model = load_model(model_file)
    features = _build_features(input_data)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    return PredictionResponse(survived=bool(prediction[0]), probability=float(probability))