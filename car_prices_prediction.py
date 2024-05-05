# ML2-Used_car_prices_prediction-

#Reading Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('/content/drive/MyDrive/train-data.csv')
df.head()

#EDA & Some Cleaning

df.info()

# Check for null values
pd.DataFrame(df.isnull().mean()*100, columns=['% null values'])

# Drop some columns
# 'Unnamed: 0' (just an index)
# 'New_Price' (too many null values)

df.drop(['Unnamed: 0', 'New_Price'], axis=1, inplace=True)

# Change the float format
pd.options.display.float_format = '{:.2f}'.format
df.describe()

#We will handle these outliers later

df.describe(include='O')

# Drop the null values as they are less than 1% of the data
df.dropna(inplace=True)

# Check the 'Name' feature
print('Number of unique values in Name feature: ', df['Name'].nunique())
df.Name.sample(10)

#There is a pattern in the 'Name' column. The brand name appears first followed by the model name. We can extract them into separate columns.

df['Brand'] = df.Name.str.split(' ').str[0]
df['Model'] = df.Name.str.split(' ').str[1]

df.drop('Name', axis=1, inplace=True)

# Check the new features
print('Number of unique values in Brand feature: ', df['Brand'].nunique())
print('Number of unique values in Model feature: ', df['Model'].nunique())

# Check other categorical features
for col in ['Owner_Type', 'Seats', 'Brand', 'Location', 'Fuel_Type', 'Transmission', 'Mileage' , 'Engine' , 'Power']:
    print(df[col].value_counts())
    print('--------------------------------')

# Some Feature Engineering
# df.Owner_Type = df.Owner_Type.map({'First':1, 'Second':2, 'Third':3, 'Fourth & Above':4})
df.Engine = df.Engine.str.split(' ').str[0]
df.Power = df.Power.str.split(' ').str[0]

df.sample(5)

# Mileage has different units, so we need to convert them to the same unit
df.Mileage

# Convert the units to km/kg
def get_mileage(x):
    num = float(x.split(' ')[0])
    unit = x.split(' ')[1]
    if unit == 'km/kg':
        return num
    elif unit == 'kmpl':
        return num*1.61
    else:
        return np.nan

df.Mileage = df.Mileage.apply(get_mileage)

# Convert 'Engine' and 'Power' to numeric
df.Engine = pd.to_numeric(df.Engine, errors='coerce')
df.Power = pd.to_numeric(df.Power, errors='coerce')

# Now we have some null values
pd.DataFrame(df.isnull().mean()*100, columns=['% null values'])

#'Power' has very low null values, but we will impute them

df.describe()

df.describe(include='object')

#Now, Data Types are correct

#Let's check the distributions of the numerical features

# Distribution of the numerical features
# make grid plot
fig, axes = plt.subplots(3, 2, figsize=(20, 10))
fig.suptitle('Distribution of the numerical features')
sns.histplot(ax=axes[0, 0], data=df, x='Kilometers_Driven', kde=True)
sns.histplot(ax=axes[0, 1], data=df, x='Mileage', kde=True)
sns.histplot(ax=axes[1, 0], data=df, x='Engine', kde=True)
sns.histplot(ax=axes[1, 1], data=df, x='Power', kde=True)
sns.histplot(ax=axes[2, 0], data=df, x='Seats', kde=True)
sns.histplot(ax=axes[2, 1], data=df, x='Price', kde=True)

# Box plot for the numerical features
# make grid plot
fig, axes = plt.subplots(3, 2, figsize=(20, 10))
fig.suptitle('Distribution of the numerical features')
sns.boxplot(ax=axes[0, 0], data=df, x='Kilometers_Driven')
sns.boxplot(ax=axes[0, 1], data=df, x='Mileage')
sns.boxplot(ax=axes[1, 0], data=df, x='Engine')
sns.boxplot(ax=axes[1, 1], data=df, x='Power')
sns.boxplot(ax=axes[2, 0], data=df, x='Seats')
sns.boxplot(ax=axes[2, 1], data=df, x='Price')

df1=df.copy

df.drop(['B', 'C'], axis=1)

import pandas as pd
df1 = pd.DataFrame(df)
df1=df1.drop(['Location','Fuel_Type','Transmission','Owner_Type','Brand','Model'],axis='columns')

df1.quantile([0, 0.1, 0.25,  0.9, 0.95, 0.99, 1])

#There are clear outliers in the 'Kilometer_Driven', 'Mileage' and 'Price' columns. We will remove them Now.

# We would use quantiles to remove outliers

# Take a copy of the data first
df_copy = df.copy()

# Remove the outliers
df_copy = df_copy[df_copy.Kilometers_Driven < df_copy.Kilometers_Driven.quantile(0.99)]
df_copy = df_copy[df_copy.Mileage > 0]
df_copy = df_copy[df_copy.Seats > 0]
df_copy = df_copy[df_copy.Price < 100]

# Recheck the box plot

fig, axes = plt.subplots(3, 2, figsize=(20, 10))
fig.suptitle('Distribution of the numerical features')
sns.boxplot(ax=axes[0, 0], data=df_copy, x='Kilometers_Driven')
sns.boxplot(ax=axes[0, 1], data=df_copy, x='Mileage')
sns.boxplot(ax=axes[1, 0], data=df_copy, x='Engine')
sns.boxplot(ax=axes[1, 1], data=df_copy, x='Power')
sns.boxplot(ax=axes[2, 0], data=df_copy, x='Seats')
sns.boxplot(ax=axes[2, 1], data=df_copy, x='Price')

# Compare the two dataframes
print('Shape of the original dataframe: ', df.shape)
print('Shape of the new dataframe: ', df_copy.shape)
print('Percentage of data removed: {:.2f}%'.format((df.shape[0]-df_copy.shape[0])/df.shape[0]*100))

df1.quantile([0.8, 0.9, 0.95, 0.99, 1])
s
# Check the correlation between the numerical features
# Heatmap
sns.heatmap(df1.corr(), annot=True, cmap='RdYlGn', center=0)

# Correlation with the target
df1.corr()['Price'].sort_values(ascending=False)

# Reascending
asc_num_features = df1.corr()['Price'].apply(abs).sort_values(ascending=False).drop('Price')
sns.barplot(x=asc_num_features, y=asc_num_features.index)
plt.title('Absolute Correlation with the target')
plt.show()

#Data Splitting

X = df_copy.drop('Price', axis=1)
y = df_copy.Price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Splitting Columns

num_cols = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Year']
nom_cat_cols = ['Location', 'Fuel_Type', 'Transmission', 'Brand', 'Model']
ord_cat_cols = ['Owner_Type']

#Data Preprocessing

#pip install category_encoders

from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='median')),
                                         ('scaler', StandardScaler())])

# Preprocessing for categorical data
nominal_categorical_transformer = Pipeline(steps=[('binary', BinaryEncoder())])
ordinal_categorical_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder(categories=[['Fourth & Above', 'Third', 'Second', 'First']]))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[('num_prep', numerical_transformer, num_cols),
                                                ('nom_prep', nominal_categorical_transformer, nom_cat_cols),
                                                ('ord_prep', ordinal_categorical_transformer, ord_cat_cols)])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

X_train_preprocessed.shape, X_test_preprocessed.shape

# Model Building
from sklearn.linear_model import LinearRegression

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_preprocessed, y_train)

# Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Predictions
y_pred = lr.predict(X_test_preprocessed)

# Model Evaluation
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

# Training Accuracy
print('Training Accuracy:', lr.score(X_train_preprocessed, y_train))
y_train_pred = lr.predict(X_train_preprocessed)
print('Training Accuracy:', r2_score(y_train, y_train_pred))

# Testing Accuracy
print('Testing Accuracy:', lr.score(X_test_preprocessed, y_test))
y_test_pred = lr.predict(X_test_preprocessed)
print('Testing Accuracy:', r2_score(y_test, y_test_pred))

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_preprocessed)
X_test_poly = poly.transform(X_test_preprocessed)

lr_poly = LinearRegression()

lr_poly.fit(X_train_poly, y_train)

# Accuracy
print('Training Accuracy', lr_poly.score(X_train_poly, y_train))
print('Testing Accuracy', lr_poly.score(X_test_poly, y_test))

X_train_preprocessed.shape, X_train_poly.shape

# Ridge Regression
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train_preprocessed, y_train)

# Predictions
y_pred = ridge.predict(X_test_preprocessed)

# Model Evaluation
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

# Ridge Regression with polynomial features
from sklearn.linear_model import Ridge

ridge = Ridge(alpha= 100)
ridge.fit(X_train_poly, y_train)

# Predictions
y_pred = ridge.predict(X_test_poly)

# Model Evaluation
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

# Ridge Regression
from sklearn.linear_model import Lasso

lasso = Lasso(alpha= 1)
lasso.fit(X_train_preprocessed, y_train)

# Predictions
y_pred = lasso.predict(X_test_preprocessed)

# Model Evaluation
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

for alpha in [0.01, 0.1, 1, 10, 100]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_preprocessed, y_train)
    y_pred = ridge.predict(X_test_preprocessed)
    print('alpha:', alpha)
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2:', r2_score(y_test, y_pred))
    print('-----------------------------')

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly = make_pipeline(PolynomialFeatures(2), Ridge(alpha=100))
poly.fit(X_train_preprocessed, y_train)

# Predictions
y_pred = poly.predict(X_test_preprocessed)

# Model Evaluation
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))
print('R2:', poly.score(X_test_preprocessed, y_test))

for degree in [2, 3, 4]:
    poly = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=100))
    poly.fit(X_train_preprocessed, y_train)
    y_pred = poly.predict(X_test_preprocessed)
    print('degree:', degree)
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2:', r2_score(y_test, y_pred))
    print('-----------------------------')

# Log Transform for the target
def log_transform(x):
    return np.log1p(x)

# Log Transform
y_train_log = log_transform(y_train)
y_test_log = log_transform(y_test)

fig, axes = plt.subplots(1, 2, figsize=(20, 5))
sns.histplot(ax=axes[0], data=y_train, kde=True)
sns.histplot(ax=axes[1], data=y_train_log, kde=True)
axes[0].set_title('Original Target')
axes[1].set_title('Log Transformed Target')
plt.show()

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_preprocessed, y_train_log)

# Predictions
y_pred = lr.predict(X_test_preprocessed)

# Model Evaluation
print('Training Accuracy:', lr.score(X_train_preprocessed, y_train_log))
print('Testing Accuracy:', lr.score(X_test_preprocessed, y_test_log))
print('Training RMSE:', np.sqrt(mean_squared_error(y_train_log, lr.predict(X_train_preprocessed))))
print('Testing RMSE:', np.sqrt(mean_squared_error(y_test_log, y_pred)))

#Log Transformation of the target variable affects the model performance positively

poly_ridge = make_pipeline(PolynomialFeatures(2), Ridge(alpha=100))
poly_ridge.fit(X_train_preprocessed, y_train_log)

# Predictions
y_pred = poly.predict(X_test_preprocessed)

# Model Evaluation
print('Training Accuracy:', poly_ridge.score(X_train_preprocessed, y_train_log))
print('Testing Accuracy:', poly_ridge.score(X_test_preprocessed, y_test_log))
print('Training RMSE:', np.sqrt(mean_squared_error(y_train_log, poly_ridge.predict(X_train_preprocessed))))
print('Testing RMSE:', np.sqrt(mean_squared_error(y_test_log, y_pred)))

# Try Gradient Descent

from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(random_state=0, max_iter=10000, tol=1e-3, penalty='l2', alpha=10, eta0=0.01)
poly_sgd = make_pipeline(PolynomialFeatures(2), sgd)

poly_sgd.fit(X_train_preprocessed, y_train_log.ravel())

# Predictions
y_pred = poly_sgd.predict(X_test_preprocessed)

# Model Evaluation
print('Training Accuracy:', poly_sgd.score(X_train_preprocessed, y_train_log))
print('Testing Accuracy:', poly_sgd.score(X_test_preprocessed, y_test_log))
print('Training RMSE:', np.sqrt(mean_squared_error(y_train_log, poly_sgd.predict(X_train_preprocessed))))
print('Testing RMSE:', np.sqrt(mean_squared_error(y_test_log, y_pred)))

# Access the coefficients in the pipeline
print('Intercept:', poly_ridge.named_steps['ridge'].intercept_)
print('Coefficients:', poly_ridge.named_steps['ridge'].coef_)

# Access the features names in the pipeline
preprocessor.transformers_[1][1].named_steps['binary'].get_feature_names()

# Preprocessor Saving
import pickle
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))

# Model Saving
pickle.dump(poly_ridge, open('model.pkl', 'wb'))

# Inferencing

# Load Preprocessor
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing
new_data = X_test.sample(1)
new_data

y_test[new_data.index]

np.expm1(y_test[new_data.index])

# Preprocessing & Prediction
new_data_preprocessed = preprocessor.transform(new_data)
log_price = model.predict(new_data_preprocessed) # in log scale
price = np.expm1(log_price) # in original scale

print('Price:', price[0])

#import requests

# Where USD is the base currency you want to use
url = 'https://v6.exchangerate-api.com/v6/f9914a6c7128b4adf0623796/latest/USD'

# Making our request
response = requests.get(url)
data = response.json()

# Your JSON object
print (data)

usd_to_inr= data['conversion_rates']['INR']
inr_to_usd = 1/usd_to_inr
lakh_to_usd = inr_to_usd * 100000
lakh_to_usd

# API for Currency Conversion
import requests
response = requests.get('https://api.exchangerate-api.com/v4/latest/INR')
response.json()

# from INR to USD
inr_to_usd = response.json()['rates']['USD']
inr_to_usd

# from Lakh to USD
lakh_to_usd = inr_to_usd*100000
lakh_to_usd

# From Lakhs to USD
price_usd = price[0] * lakh_to_usd
print('Price in USD:', price_usd)

# Extract Data for Deployment
locations = df_copy.Location.unique().tolist()
years = df_copy.Year.unique().tolist()
Owner_Type = df_copy.Owner_Type.unique().tolist()
Min_Mileage, Max_Mileage = df_copy.Mileage.min(), df_copy.Mileage.max()
Min_Engine, Max_Engine = df_copy.Engine.min(), df_copy.Engine.max()
Min_Power, Max_Power = df_copy.Power.min(), df_copy.Power.max()

# Models for every brand
brands = {}         # key: brand, value: list of models
for brand in df_copy.Brand.unique().tolist():
    brands[brand] = df_copy[df_copy.Brand == brand].Model.unique().tolist()
# Seats for every model
seats = {}          # key: model, value: list of seats
for model in df_copy.Model.unique().tolist():
    seats[model] = df_copy[df_copy.Model == model].Seats.unique().tolist()

# Transmissions
transmissions = {}
for model in df_copy.Model.unique().tolist():
    transmissions[model] = df_copy[df_copy.Model == model].Transmission.unique().tolist()

# Fuel Types
fuel_types = {}
for model in df_copy.Model.unique().tolist():
    fuel_types[model] = df_copy[df_copy.Model == model].Fuel_Type.unique().tolist()

# Save the data
data = {'locations': locations,
        'years': years,
        'fuel_types': fuel_types,
        'transmissions': transmissions,
        'brands': brands,
        'Owner_Type': Owner_Type,
        'Min_Mileage': Min_Mileage,
        'Max_Mileage': Max_Mileage,
        'Min_Engine': Min_Engine,
        'Max_Engine': Max_Engine,
        'Min_Power': Min_Power,
        'Max_Power': Max_Power,
        'seats': seats,
        'lakh_to_usd': lakh_to_usd
        }

pickle.dump(data, open('data_for_deploy.pkl', 'wb'))
Streamlit

#%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Used Cars Price Prediction')

# Load Data for Deployment

#Writing app.py
