import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pycaret.classification import predict_model, load_model
from pydantic import BaseModel
from typing import Optional
from machine_learning.feature_engineering import FeatureEngineer
from machine_learning.model import LoadFraudDetectionModel
import pandas as pd
import pycaret

app = FastAPI(
    title="API for Revolut Banking Transactions Fraud Detection",
    description="""An API that utilises a Machine Learning model that detects if a revolut transaction is fraudulent 
    or not""",
    version="1.0.0", debug=True)


class FraudParameters(BaseModel):
    TRANSACTION_ID: str
    USER_ID: str
    HAS_EMAIL: int
    PHONE_COUNTRY: str
    USER_CREATED_DATE: str
    USER_STATUS: str
    COUNTRY: str
    BIRTH_YEAR: int
    KYC: str
    FAILED_SIGN_IN_ATTEMPTS: int
    CURRENCY: str
    AMOUNT: float
    TRANSACTION_STATUS: str
    TRANSACTION_CREATED_DATE: str
    MERCHANT_CATEGORY: Optional[str]
    MERCHANT_COUNTRY: Optional[str]
    ENTRY_METHOD: str
    TYPE: str
    SOURCE: str


# Declare the model variable at the module level
model = None


def perform_feature_engineering(data: FraudParameters):
    data_dict = {
        'TRANSACTION_ID': [data.TRANSACTION_ID],
        'USER_ID': [data.USER_ID],
        'HAS_EMAIL': [data.HAS_EMAIL],
        'PHONE_COUNTRY': [data.PHONE_COUNTRY],
        'USER_CREATED_DATE': [data.USER_CREATED_DATE],
        'USER_STATUS': [data.USER_STATUS],
        'COUNTRY': [data.COUNTRY],
        'BIRTH_YEAR': [data.BIRTH_YEAR],
        'KYC': [data.KYC],
        'FAILED_SIGN_IN_ATTEMPTS': [data.FAILED_SIGN_IN_ATTEMPTS],
        'CURRENCY': [data.CURRENCY],
        'AMOUNT': [data.AMOUNT],
        'TRANSACTION_STATUS': [data.TRANSACTION_STATUS],
        'TRANSACTION_CREATED_DATE': [data.TRANSACTION_CREATED_DATE],
        'MERCHANT_CATEGORY': [data.MERCHANT_CATEGORY],
        'MERCHANT_COUNTRY': [data.MERCHANT_COUNTRY],
        'ENTRY_METHOD': [data.ENTRY_METHOD],
        'TYPE': [data.TYPE],
        'SOURCE': [data.SOURCE]
    }
    df = pd.DataFrame.from_dict(data_dict)
    feature_engineering = FeatureEngineer()
    return feature_engineering.process_input_data(df)


@app.get("/", response_class=PlainTextResponse)
async def running():
    # Use the global model variable
    global model
    load_saved_model = LoadFraudDetectionModel(r"./resources/pycaret_models/saved_random_forest_model")
    model = load_saved_model.load()
    if model:
        model_load_message = "Loaded Successfully"
    else:
        model_load_message = "Error in Loading Model"

    note = f"""
        Revolut Fraud Detection API 🙌🏻
        
        Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
        
        PYCARET VERSION: {pycaret.__version__}
        
        MACHINE LEARNING MODEL STATUS: {model_load_message}
  """
    return note


@app.post('/detect')
async def predict(data: FraudParameters):
    # Use the global model variable
    global model
    if model is None:
        load_saved_model = LoadFraudDetectionModel(r"./resources/pycaret_models/saved_random_forest_model")
        model = load_saved_model.load()
    processed_data, shape = perform_feature_engineering(data=data)
    fraud_prediction = predict_model(model, processed_data)
    return str(fraud_prediction['prediction_label'].values[0])


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level="info")
