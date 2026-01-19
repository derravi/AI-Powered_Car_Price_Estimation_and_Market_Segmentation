from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from schema.pydentic_model import UserInput

#Impotr PKL file
with open('models/models.pkl', 'rb') as f:
    model = pickle.load(f)                    

encoders = model["encoders"]                  
scaler = model["Standard_scaler"]             
lr = model["LinearRegression"]                 
dtrm = model["Decision_tree"]                 
xgb = model["XGBoost"]                         

#main api app name
app = FastAPI(
    title="AI-Powered Car Price Estimation and Market Segmentation",
    description="Predict car prices using Linear Regression, Decision Tree, and XGBoost models."
)
#checking API
@app.get("/")
def demo():
    return {"message": "AI-Powered Car Price Estimation and Market Segmentation"}

#Prediction API Pipeline for ML Model
@app.post("/predict")
def predict_car_price(data: UserInput):

    df = pd.DataFrame([{
        "brand": data.brand,
        "unique_model_number": data.unique_model_number,
        "year": data.year,
        "fuel": data.fuel,
        "transmission": data.transmission,
        "km_driven": data.km_driven,
        "owner": data.owner,
        "seller_type": data.seller_type
    }])

    categorical_cols = [
        "brand",
        "unique_model_number",
        "fuel",
        "transmission",
        "owner",
        "seller_type"
    ]

    for col in categorical_cols:
        if df[col].iloc[0] in encoders[col].classes_:   
            df[col] = encoders[col].transform(df[col])  
        else:
            df[col] = -1                               

    scaled_data = scaler.transform(df)    

    predict_lr = float(lr.predict(scaled_data)[0])      
    predict_dtree = float(dtrm.predict(df)[0])           
    predict_xgboost = float(xgb.predict(df)[0])          

    return JSONResponse(
        status_code=200,
        content={
            "LinearRegression_Price": round(predict_lr, 2),
            "DecisionTree_Price": round(predict_dtree, 2),
            "XGBoost_Price": round(predict_xgboost, 2)
        }
    )
