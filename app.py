from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# load model and training columns
model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")

# create FastAPI app
app = FastAPI()

# input schema
class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: Literal["M", "F"]

# health check endpoint
@app.get("/")
def home():
    return {"status": "API is running"}

# prediction endpoint
@app.post("/predict")
def predict(data: PatientData):

    # create dataframe
    input_data = pd.DataFrame([{
        "AGE": data.age,
        "Urea": data.urea,
        "Cr": data.cr,
        "HbA1c": data.hba1c,
        "Chol": data.chol,
        "TG": data.tg,
        "HDL": data.hdl,
        "LDL": data.ldl,
        "VLDL": data.vldl,
        "BMI": data.bmi,
        "Gender_M": 1 if data.gender.upper() == "M" else 0
    }])

    # ensure columns match training data
    input_data = input_data.reindex(columns=training_columns, fill_value=0)

    # prediction
    prediction = model.predict(input_data)[0]

    # result
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return {
        "prediction": result
    }