import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('train.csv')

print(df.head())

X = df.drop('price_range', axis=1)
y = df['price_range']

numeric_features = X.columns

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


with open("Mobile_Price_Classification.pkl", "wb") as f:
    pickle.dump(model_rf, f)

print("✅ Random Forest pipeline saved as Mobile Price Classification.pkl")
