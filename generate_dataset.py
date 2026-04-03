import pandas as pd
import random

NUM_SAMPLES = 10000
TARGET_FRAUD_RATE = 0.065

data = []

# -------------------------------
# GENERATE DATA
# -------------------------------
for _ in range(NUM_SAMPLES):

    amount = random.randint(100, 200000)
    hour = random.randint(0, 23)

    receiver_known = random.randint(0, 1)
    location_new = random.randint(0, 1)

    # ✅ MATCH APP LOGIC
    deviation = min((amount / 100000) * 100, 100)

    # ✅ velocity
    transactions_last_1min = random.randint(0, 5)

    # ✅ IMPORTANT: same as app
    freq = transactions_last_1min

    # -------------------------------
    # RISK SCORE
    # -------------------------------
    risk_score = 0

    if amount > 120000:
        risk_score += 2
    if receiver_known == 0:
        risk_score += 2
    if location_new == 1:
        risk_score += 1
    if hour < 5 or hour > 22:
        risk_score += 1
    if freq > 3:
        risk_score += 2
    if transactions_last_1min > 3:
        risk_score += 3
    if deviation > 70:
        risk_score += 2

    data.append([
        amount,
        hour,
        freq,
        receiver_known,
        deviation,
        location_new,
        transactions_last_1min,
        risk_score
    ])

# -------------------------------
# CREATE DATAFRAME
# -------------------------------
df = pd.DataFrame(data, columns=[
    "Amount",
    "Hour",
    "Frequency",
    "Receiver_Known",
    "Deviation",
    "Location_New",
    "Transactions_Last_1min",
    "Risk"
])

# -------------------------------
# REALISTIC FRAUD LABELING (IMPORTANT)
# -------------------------------
fraud_labels = []

for _, row in df.iterrows():
    risk = row["Risk"]

    # 🔥 STRONGER SIGNAL
    if risk >= 7:
        fraud_labels.append(1 if random.random() < 0.9 else 0)
    elif risk >= 5:
        fraud_labels.append(1 if random.random() < 0.6 else 0)
    elif risk >= 3:
        fraud_labels.append(1 if random.random() < 0.3 else 0)
    else:
        fraud_labels.append(0)  # VERY IMPORTANT (no random noise here)

df["Class"] = fraud_labels


# -------------------------------
# CONTROL FRAUD RATE (~6.5%)
# -------------------------------
df = df.sample(frac=1).reset_index(drop=True)

fraud_target = int(NUM_SAMPLES * TARGET_FRAUD_RATE)

fraud_df = df[df["Class"] == 1].head(fraud_target)
non_fraud_df = df[df["Class"] == 0].head(NUM_SAMPLES - fraud_target)

df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1).reset_index(drop=True)

# -------------------------------
# CLEANUP
# -------------------------------
df = df.drop(columns=["Risk"])

# -------------------------------
# SAVE
# -------------------------------
df.to_csv("realistic_fraud_dataset.csv", index=False)

print("✅ FINAL REALISTIC DATASET CREATED")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True) * 100)