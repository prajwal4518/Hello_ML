from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Define the input data model
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

app = FastAPI(title="Titanic Survival Prediction API")

# Load model
MODEL_PATH = "model/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.post("/predict")
def predict(passenger: Passenger):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocessing
    # 1. Encode Sex
    sex_encoded = 0 if passenger.Sex.lower() == 'male' else 1
    
    # 2. Encode Embarked
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_mapping.get(passenger.Embarked.upper(), 0) # Default to S=0
    
    # 3. Create DataFrame for model input
    # Feature order must match training: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    features = pd.DataFrame([{
        'Pclass': passenger.Pclass,
        'Sex': sex_encoded,
        'Age': passenger.Age,
        'SibSp': passenger.SibSp,
        'Parch': passenger.Parch,
        'Fare': passenger.Fare,
        'Embarked': embarked_encoded
    }])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]
    
    result = {
        "prediction": int(prediction),
        "survival_probability": float(probability),
        "survived": bool(prediction == 1)
    }
    
    return result

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
