
# STEP 1 : IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")


# STEP 2 : LOAD DATASET


# Replace with your file path if local
df = pd.read_csv("Fish.csv")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# STEP 3 : CHECK MISSING VALUES

print("\nMissing Values:\n")
print(df.isnull().sum())


# STEP 4 : DATA VISUALIZATION

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df["Weight"], kde=True)
plt.title("Weight Distribution")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=df.select_dtypes(include=np.number))
plt.xticks(rotation=45)
plt.title("Boxplot of Numerical Features")
plt.show()


# STEP 5 : DEFINE FEATURES / TARGET

X = df.drop("Weight", axis=1)
y = 
# STEP 6 : IDENTIFY COLUMN TYPES

categorical_features = ["Species"]
numeric_features = [col for col in X.columns if col not in categorical_features]


# STEP 7 : PREPROCESSING


numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])




# STEP 8 : TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# MODEL 1 : LINEAR REGRESSION
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

y_pred_lr = linear_model.predict(X_test)

print("\n========== Linear Regression ==========")
print("MAE :", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2  :", r2_score(y_test, y_pred_lr))


# MODEL 2 : RIDGE REGRESSION
 
ridge_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

print("\n========== Ridge Regression ==========")
print("MAE :", mean_absolute_error(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("R2  :", r2_score(y_test, y_pred_ridge))

# MODEL 3 : RANDOM FOREST#
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n========== Random Forest ==========")
print("MAE :", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2  :", r2_score(y_test, y_pred_rf))

# CROSS VALIDATION
 
scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\nRandom Forest Cross Validation R2 Scores:")
print(scores)
print("Average CV Score:", scores.mean())


# ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Actual vs Predicted Weight (Random Forest)")
plt.show()


# SAMPLE PREDICTION

sample = pd.DataFrame({
    "Species": ["Bream"],
    "Category": [1],
    "Height": [12.0],
    "Width": [4.5],
    "Length1": [25.0],
    "Length2": [27.0],
    "Length3": [30.0]
})

prediction = rf_model.predict(sample)

print("\nPredicted Fish Weight:", prediction[0], "grams")
 
# SAVE MODE
import joblib

joblib.dump(rf_model, "fish_weight_model.pkl")

print("\nModel Saved as fish_weight_model.pkl")

