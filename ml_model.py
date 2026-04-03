import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("realistic_fraud_dataset.csv")

X = df[[
    "Amount",
    "Hour",
    "Frequency",
    "Receiver_Known",
    "Deviation",
    "Location_New",
    "Transactions_Last_1min"
]]

y = df["Class"]

# -------------------------------
# HANDLE IMBALANCE
# -------------------------------
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)

# -------------------------------
# SAVE MODEL
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ MODEL TRAINED")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# PREDICT
# -------------------------------
def predict_fraud(amount, hour, freq, receiver_known, deviation, location_new, transactions_last_1min):

    X_input = np.array([[
        amount,
        hour,
        freq,
        receiver_known,
        deviation,
        location_new,
        transactions_last_1min
    ]])

    probability = float(model.predict_proba(X_input)[0][1])
    probability = max(0.01, min(0.99, probability))

    prediction = 1 if probability > 0.4 else 0

    return prediction, probability


# -------------------------------
# EXPLAINABLE AI
# -------------------------------
def explain_prediction(amount, hour, freq, receiver_known, deviation, location_new, transactions_last_1min):

    feature_names = [
    "Amount",
    "Time of Transaction",
    "Transaction Frequency",
    "Receiver Known",
    "Behavior Deviation",
    "New Location",
    "Recent Transactions"
]

    importances = model.feature_importances_

    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)

    explanations = [f"{f} influenced decision" for f, _ in pairs[:3]]

    return explanations