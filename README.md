# Fintech-fraud-detection
Machine learning-based fraud detection system built with Flask and XGBoost featuring risk analysis, explainable AI, and transaction monitoring dashboard.


---

## Key Features

- Built a machine learning model using XGBoost for fraud prediction  
- Implemented explainable AI to provide reasoning behind predictions  
- Designed a rule-based risk engine for detecting high-risk patterns  
- Integrated behavioral analysis using typing patterns and user activity  
- Developed a dashboard for monitoring transaction statistics  
- Implemented transaction history tracking with risk analysis  

---

## Technical Implementation

- Trained a classification model on a synthetic dataset with realistic fraud distribution  
- Combined ML predictions with rule-based and behavioral scoring for final decision-making  
- Designed a hybrid risk scoring system to reduce false positives  
- Built a full-stack web application using Flask and HTML/CSS  
- Integrated Chart.js for data visualization in dashboard and history pages  

---

## Tech Stack

- Python  
- Flask  
- XGBoost  
- Scikit-learn  
- Pandas, NumPy  
- HTML, CSS  
- Chart.js  

---

## Project Structure

fintech-fraud-detection/

app.py  
ml_model.py  
generate_dataset.py  
behavior.py  
risk.py  
realistic_fraud_dataset.csv  

templates/  
  index.html  
  dashboard.html  
  history.html  

---

## How to Run

pip install -r requirements.txt  
python app.py  

Open in browser:  
http://127.0.0.1:5000/

---

## Impact

This project demonstrates the design of a hybrid fraud detection system similar to those used in real-world financial platforms, combining multiple detection techniques for improved accuracy and interpretability.

---

## Author

Sahana Loganathan
