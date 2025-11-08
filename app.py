from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from schema.pydentic_model import UserInput

with open('models/models.pkl','rb') as f:
    model = pickle.dump(model,f)

# label_encoder = model["Label_encoder"]
# scaler  = model["Standard_scaler"]
# linear_model  = model["LinearRegression"]
# decision_tree  = model["Decision_tree"]
# xgboost_model  = model["XGBoost"]

app =FastAPI(title="AI-Powered Car Price Estimation and Market Segmentation",description="Predict car prices using Linear Regression, Decision Tree, and XGBoost models.")

@app.get("/")
def demo():
    return {"message":"AI-Powered Car Price Estimation and Market Segmentation"}

@app.post("/predict")
def predict_car_price(data:UserInput):

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

    temp=["brand", "unique_model_number", "fuel", "transmission", "owner", "seller_type"]

    #Encode the Labeled Data
    for i in temp:
        df[i] = label_encoder.transform(df[i])

    #Resclae the Hall User Input data
    scaled_data = scaler.transform(df)

    #Predict the all model Values

    predict_lr = linear_model.predict(scaled_data)[0]
    predict_dtree = decision_tree.predict(scaled_data)[0]
    predict_xgboost = xgboost_model.predict(scaled_data)[0]

    return JSONResponse(status_code=200,content={
        "LinearRegression_Price":predict_lr,
        "DecisionTree_Price":predict_dtree,
        "XGBoost_Price":predict_xgboost
    })