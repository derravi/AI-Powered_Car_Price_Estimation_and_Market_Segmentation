import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,classification_report
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pickle
from xgboost import XGBRegressor
#Data load
print("Lets Load the Datasheet........\n")
df = pd.read_csv("Data Sheet/CAR DETAILS FROM CAR DEKHO.csv")

#Check The Rows And Columns of the Datasets.
def shape():
    return f"The Total Rows of the Datasets is {df.shape[0]} and the total Columns is {df.shape[1]}.\n"
shape_object = shape()
print(shape_object)


#Show some data of the Datasets

print("Lets see some datasets valus..........\n")
df.head()

#Cleaning Data
print("Lets Clean the datasets.......\n")

print("Lets see there is any Null Values are not......\n")

df.isnull().sum()

#Compares average prices between Petrol, Diesel, CNG, etc.

plt.figure(figsize=(9,5))
df.groupby('fuel')['selling_price'].mean().sort_values().plot(kind='bar',color="green",edgecolor="black")
plt.title("Compares average prices between Petrol, Diesel, CNG, etc.")
plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price (₹)")
plt.xticks(rotation=0)
plt.grid(True,color="grey")
plt.savefig("Graph images/Average_vs_fuel.jpg",dpi=400,bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,5))
df['transmission'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'blue'])
plt.title("Percentage of Transmission Types")
plt.ylabel("")
plt.savefig("Graph images/Transmission_types.jpg",dpi=400,bbox_inches='tight')
plt.show()

#Relationship Between KM Driven and Selling Price
print("Relationship Between KM Driven and Selling Price.....\n")
plt.figure(figsize=(9,5))
plt.scatter(df['km_driven'],df['selling_price'],color="red",edgecolor="black")
plt.title("Relationship between KM Driven and Selling Price")
plt.xlabel("KM Drive (KM)")
plt.ylabel("Selling Price (₹)")
plt.grid(True,color="grey")
plt.savefig("Graph images/KM_Driven_and_Selling_Price.jpg",dpi=400,bbox_inches='tight')
plt.show()

print("Line Chart: Average Car Price vs KM Driven...\n")

# Create bins for km driven

df['km_bin'] = pd.cut(df['km_driven'], bins=10)

avg_price_by_km = df.groupby('km_bin')['selling_price'].mean()

plt.figure(figsize=(12,6))
plt.plot(avg_price_by_km.index.astype(str), avg_price_by_km.values, marker='o', color='darkorange', linewidth=2)
plt.title("Average Car Selling Price vs KM Driven")
plt.xlabel("KM Driven Range")
plt.ylabel("Average Selling Price (₹)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Graph images/Average_Price_vs_KMDriven.png", dpi=400, bbox_inches='tight')
plt.show()

#Lets use some encoding technique to encode the Labeled data from the datasets.

print(df['fuel'].unique())
print(df['seller_type'].unique())
print(df['transmission'].unique())
print(df['owner'].unique())

print("Lets see the Full Dataset After this encoding.......\n")
df.head(10)

#Lets Make two new columns of this Name of the cars that is "Brand", "Unique Model"

print("Lets Make two new columns of this Name of the cars that is 'Brand', 'Unique Model'..........\n")

df['brand'] = df["name"].str.split(' ').str[:2]
df['unique_model_number'] = df["name"].str.split(' ').str[2::]

df['brand'] = df['brand'].apply(lambda x : ' '.join(x))
df['unique_model_number'] = df['unique_model_number'].apply(lambda y: ' '.join(y))

print("Lets see the Updated data sheet after the making this new columns.........\n")
df.head()

print("Lets Drope the Unwanted Name Column from the datasheets and see the updated datasets.......\n")

df.drop('name',axis=1,inplace=True)

df.head()

#Lets Rearrange the columns of this datasets for batter result

new_order = ['brand','unique_model_number','year','fuel','transmission','km_driven','owner','seller_type','selling_price']

df = df.reindex(columns=new_order)

print("Check the columns after the Giving New Indexing.............\n")
df.head()

#Lets Describe all the Datasets Columns.
print("Lets Describe all the datasets columns..........\n")
df.describe(include='all')

#Lets Encode the Requered Columns of the Datasets.
print("Lets Encode the requered Columns of the datasets..........\n")

temp1 = []
temp2 = []

for i in df.columns:
    if df[i].dtype == 'object':
        temp1.append(i)
    else:
        temp2.append(i)
print("Object Data Types Column list :",temp1)
print("Int or Float Data Types Column list :",temp2)

#Lets Encode the Labeled Data or Object data type columns.
print("Encoding Categorical Columns....\n")

categorical_cols = ['brand', 'unique_model_number', 'fuel',
                    'transmission', 'owner', 'seller_type']

encoders = {}
for i in categorical_cols:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
    encoders[i] = le

print("Encoded Sample:\n")
df.head()

print(shape_object)

#Lets check there is any outlieres or not.

print("Lets check there is any outlieres or not.........\n")

numeric_cols = ['year', 'km_driven', 'selling_price'] 

plt.figure(figsize=(15, 3 * len(numeric_cols)))
for i, col in enumerate(numeric_cols, 1):
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

#Lets Remove the Outliers From the Requerede Columns 
print("Lets Remove the Outliers From the Requerede Columns...............\n")
temp3 = ['km_driven','selling_price']

for i in temp3:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)

    iqr = q3-q1

    min_range = q1 - (1.5*iqr)
    max_range = q3 + (1.5*iqr)

    df = df[(df[i] >= min_range) & (df[i] <= max_range)]
print("Outlieres is Removed...............\n")
print("Now",shape_object)

print("Lets Check again there is any outliers is still pending or not.............\n")

plt.figure(figsize=(15, 3 * len(numeric_cols)))
for i, col in enumerate(numeric_cols, 1):
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

print("Now There is no any outlieres is present..........\n")

#Lets Devide this into the Saperate columns for the train test split data

x = df.iloc[:,:-1]
y = df['selling_price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Lets Rescale the All the data.
print("Lets Rescale the datasets using StandardScaler........\n")

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

#Lets Use first Linear Regression Model.......
print("Lets use the LinearRegression model for predict the Price..........\n")
lr = LinearRegression()

lr.fit(x_train_scaler,y_train)
y_predict_lr = lr.predict(x_test_scaler)

print("Linear Regression Model Performance:\n")
print("Mean Absolute Error (MAE) : ",round(mean_absolute_error(y_test,y_predict_lr),2))
print("Mean Square Error (MSE):",round(mean_squared_error(y_test,y_predict_lr),2))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,y_predict_lr)))
print("R2 Score : ",(r2_score(y_test,y_predict_lr))*100,"%")

#Lets see the Decision Tree Regressor Model.

print("Lets use the Decision Tree Regressor Model..........\n")

dtrm = DecisionTreeRegressor(max_depth=10,random_state=42)

dtrm.fit(x_train,y_train)
y_predict_dtrm = dtrm.predict(x_test)

print("Decision Tree Model Performance:\n")
print("Mean Absolute Error (MAE) : ",round(mean_absolute_error(y_test,y_predict_dtrm)),2)
print("Mean Square Error (MSE):",round(mean_squared_error(y_test,y_predict_dtrm)),2)
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,y_predict_dtrm)))
print("R2 Score : ",(r2_score(y_test,y_predict_dtrm))*100,"%")

# Lets use XGBoost Regressor

print("Lets use the XGBoost Regressor Model..........\n")

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

xgb.fit(x_train,y_train)
y_predict_xgb = xgb.predict(x_test)

print("XGBoost Model Performance:\n")
print("Mean Absolute Error (MAE) : ",round(mean_absolute_error(y_test,y_predict_xgb)),2)
print("Mean Square Error (MSE):",round(mean_squared_error(y_test,y_predict_xgb)),2)
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,y_predict_xgb)))
print("R2 Score : ",(r2_score(y_test,y_predict_xgb))*100,"%")

df.head()


#User Input.
print("Lets Get the User Input and predict the Car Price for each model............\n")

print("Enter the Car Details here.........\n")

fule_details = ['Petrol', 'Diesel', 'CNG', 'LPG' ,'Electric']
transmission_details = ['Manual' ,'Automatic']
owner_details = ['First Owner' ,'Second Owner' ,'Fourth & Above Owner' ,'Third Owner','Test Drive Car']
seller_type_details = ['Individual', 'Dealer' ,'Trustmark Dealer']
   
brand = input("Enter the Car Brand(e.g. Maruti Wagon):")
unique_model_number = input("Enter the Unique model number of the car(e.g. R LXI Minor):")
year=int(input("Enter the Year:"))
fuel = input(f"Enter the Fule Details from the {fule_details}:")
transmission = input(f"Enter the car Manuality from {transmission_details}")
km_driven = float(input("Enter the KM Driven(e.g. 10526.2):"))
owner = input(f"Enter the Owner type from {owner_details}:")
seller_type = input(f"Enter the Seller type from {seller_type_details}:")

user_data = {
    'brand' : brand,
    'unique_model_number' : unique_model_number,
    'year':year,
    'fuel' : fuel,
    'transmission' : transmission,
    'km_driven' : km_driven,
    'owner' : owner,
    'seller_type' : seller_type,
}

data = pd.DataFrame([user_data])

for i in categorical_cols:
    if data[i][0] in encoders[i].classes_:
        data[i] = encoders[i].transform(data[i])
    else:
        data[i] = -1

user_scaled = scaler.transform(data)

# Predict
pred_lr = lr.predict(user_scaled)[0]
pred_dtr = dtrm.predict(data)[0]
predicted_price_xgboost = xgb.predict(data)[0]



print(f"Using Linear Regression, Predicted Price: ₹{round(pred_lr,2)}\n")
print(f"Using Decision Tree Regressor, Predicted Price: ₹{round(pred_dtr,2)}\n")
print(f"Using XGBoost, Predicted Price: ₹{round(predicted_price_xgboost,2)}\n")


#make the pickle models for Fast APIs

print("Lets make the Pickle model for FAST APIs Endpoin.........\n")

model = {
    "encoders":encoders,
    "Label_encoder":le,
    "Standard_scaler":scaler,
    "LinearRegression":lr,
    "Decision_tree":dtrm,
    "XGBoost":xgb
}

with open("models/models.pkl",'wb') as f:
    pickle.dump(model,f)

#Shows overall distribution of car prices in the dataset.
print("Shows overall distribution of car prices in the dataset........\n")
plt.figure(figsize=(8,5))
plt.hist(df['selling_price'], bins=40, color='red', edgecolor='black')
plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price (in ₹)")
plt.ylabel("Number of Cars")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Graph images/Distribution_of_Car_Selling_Prices.jpg",dpi=400,bbox_inches='tight')
plt.show()

#Visualize how well each ML model performs (R² score).
print("Visualize how well each ML model performs (R² score).")

r2_scores = {
    "Linear Regression": r2_score(y_test, y_predict_lr),
    "Decision Tree": r2_score(y_test, y_predict_dtrm),
    "XGBoost": r2_score(y_test, y_predict_xgb)
}

plt.figure(figsize=(8,5))
plt.bar(r2_scores.keys(),[v*100 for v in r2_scores.values()],color=['skyblue','lightcoral','lightgreen'], edgecolor='black')
plt.title("Model Performance Comparison (R² Score %)")
plt.xlabel("R² Score")
plt.ylabel("Models")
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Graph images/R2_score_comparission.jpg",dpi=400,bbox_inches='tight')
plt.show()

