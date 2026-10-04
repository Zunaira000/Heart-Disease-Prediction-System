"""
Heart Disease Prediction - Training Script
------------------------------------------
Steps (same teaching flow as the Titanic project):
  1. Load data            2. Pre-process (drop rows with missing values)
  3. Feature extraction   (4 inputs + 1 output)
  4. NO label encoding    (all 4 inputs are already numbers)
  5. Train / test split   6. Train the model   7. Evaluate   8. Save the model

Run:  python train_model.py
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- Step 1: Load data -----------------------------------------------------
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
data = pd.read_csv("processed.cleveland.data", header=None, names=COLUMNS, na_values="?")
print("Rows loaded:", len(data))

# ---- Step 2: Pre-process ---------------------------------------------------
data = data.dropna()                       # 6 rows have missing 'ca' / 'thal'
print("Rows after dropping missing values:", len(data))

# ---- Step 3: Feature extraction (4 inputs, 1 output) -----------------------
FEATURES = ["ca", "oldpeak", "thalach", "exang"]
X = data[FEATURES]                         # 4 inputs  (already numeric -> no label encoding)
y = (data["num"] > 0).astype(int)          # 1 output: 0 = no disease, 1 = disease
print("\nInputs :", FEATURES)
print("Output : heart disease (0 = No, 1 = Yes)")

# Save a clean copy of the 5 columns used (handy for students to look at)
pd.concat([X, y.rename("target")], axis=1).to_csv("heart_data_4_features.csv", index=False)

# ---- Step 5: Train / test split -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Step 6: Train the model ----------------------------------------------
# The scaler is stored INSIDE the pipeline, so app.py needs nothing extra.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# ---- Step 7: Evaluate ------------------------------------------------------
pred = model.predict(X_test)
print("\nTest accuracy:", round(accuracy_score(y_test, pred), 3))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\n", classification_report(y_test, pred, target_names=["No disease", "Disease"]))

# ---- Step 8: Save the model ------------------------------------------------
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved -> heart_model.pkl")
