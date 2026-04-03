import os
import json
import datetime
from flask import Flask, request, render_template
from ml_model import predict_fraud, explain_prediction

app = Flask(__name__)

DATA_FILE = "data.json"

# -------------------------------
# Ensure data file
# -------------------------------
def ensure_data_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump([], f)

# -------------------------------
# Home
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Main Check
# -------------------------------
@app.route("/check", methods=["POST"])
def check():
    ensure_data_file()

    user_id = request.form["user_id"]
    amount = float(request.form["amount"])
    receiver = request.form["receiver"]
    location = request.form["location"]

    with open(DATA_FILE, "r") as file:
        data = json.load(file)

    now = datetime.datetime.now()
    hour = now.hour

    # Velocity
    freq = 0
    for tx in data:
        if tx.get("user_id") == user_id:
            tx_time = datetime.datetime.fromisoformat(tx["timestamp"])
            if (now - tx_time).total_seconds() <= 120:
                freq += 1

    freq = min(freq, 10)

    receiver_known = 1 if receiver.lower() in ["known", "saved"] else 0
    location_new = 1 if location.lower() == "new" else 0

    deviation = min((amount / 100000) * 100, 100)
    transactions_last_1min = freq

    # ML Prediction
    ml_result, ml_prob = predict_fraud(
        amount, hour, freq,
        receiver_known, deviation,
        location_new, transactions_last_1min
    )

    ml_score = int(ml_prob * 100)

    # Explainable AI
    explanations = explain_prediction(
        amount, hour, freq,
        receiver_known, deviation,
        location_new, transactions_last_1min
    )

    reasons = explanations

    # Decision
    if ml_score > 80:
        status = "BLOCKED"
        risk_level = "HIGH"
        message = "High risk transaction detected"
    elif ml_score > 50:
        status = "WARNING"
        risk_level = "MEDIUM"
        message = "This transaction may be risky"
    else:
        status = "APPROVED"
        risk_level = "LOW"
        message = "Transaction approved"

    # Save data
    data.append({
        "user_id": user_id,
        "amount": amount,
        "status": status,
        "timestamp": now.isoformat()
    })

    with open(DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

    return render_template(
        "index.html",
        user_id=user_id,
        status=status,
        message=message,
        risk_score=ml_score,
        risk_level=risk_level,
        reasons=reasons,
        ml_score=ml_score
    )

# -------------------------------
# Dashboard (FIXED)
# -------------------------------
@app.route("/dashboard")
def dashboard():
    ensure_data_file()

    with open(DATA_FILE, "r") as file:
        data = json.load(file)

    total = len(data)

    # ✅ SAFE VERSION (NO ERROR)
    blocked = sum(1 for t in data if t.get("status") == "BLOCKED")
    warning = sum(1 for t in data if t.get("status") == "WARNING")
    approved = sum(1 for t in data if t.get("status") == "APPROVED")

    return render_template(
        "dashboard.html",
        total=total,
        blocked=blocked,
        warning=warning,
        approved=approved
    )

# -------------------------------
# History (FIXED POSITION)
# -------------------------------
@app.route("/history")
def history():
    ensure_data_file()

    user_id = request.args.get("user_id")

    with open(DATA_FILE, "r") as file:
        data = json.load(file)

    if user_id:
        data = [t for t in data if t.get("user_id") == user_id]

    data.reverse()

    return render_template("history.html", transactions=data)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)