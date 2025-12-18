import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("car_price_prediction.csv")

# -----------------------------
# 2. Drop unnecessary columns
# -----------------------------
df.drop(columns=["Levy"], errors="ignore", inplace=True)

# -----------------------------
# 3. Remove rare manufacturers
#    (less than 50 records)
# -----------------------------
common_makes = df["Manufacturer"].value_counts()
common_makes = common_makes[common_makes > 50].index
df = df[df["Manufacturer"].isin(common_makes)]

# -----------------------------
# 4. Handle missing values
# -----------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# -----------------------------
# 5. Encode categorical features
# -----------------------------
encoders = {}
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# 6. Split features & target
#    (log transform price)
# -----------------------------
X = df.drop("Price", axis=1)
y = np.log1p(df["Price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Train XGBoost model
# -----------------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R2 Score (log-scale): {r2}")

# -----------------------------
# 9. Save model & encoders
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ XGBoost model trained with filtered manufacturers and saved successfully")
