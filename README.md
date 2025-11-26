# Heart-Stroke-Prediction-ML
Stroke prediction using Logistic Regression, Random Forest, XGBoost and ensemble models

An end-to-end Machine Learning project to predict the probability of heart stroke using clinical and lifestyle data.  
This project focuses on **model comparison, calibration reliability, ensemble learning, and interpretability** following industry-standard ML practices.

---

## 🔍 Problem Statement
Stroke is one of the leading causes of death and disability worldwide.  
Early prediction can help in preventive healthcare and risk monitoring.

This project aims to build a robust and interpretable ML system to predict stroke risk from patient medical records.

---

## 📊 Dataset
Dataset used: Kaggle Stroke Prediction Dataset  
🔗 https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  

Target variable: `stroke`  
Class imbalance handled properly using resampling techniques and evaluation strategies.

---

## ⚙️ Models Implemented

I implemented and compared:

- Logistic Regression
- Random Forest
- XGBoost
- Voting Ensemble
- Stacking Ensemble

---

## 🧪 Advanced Techniques Used

✔ Feature Engineering  
✔ Class Imbalance Handling  
✔ Hyperparameter Tuning (GridSearchCV)  
✔ 5-Fold Cross Validation  
✔ Ensemble Learning (Voting + Stacking)  
✔ Model Calibration & Probability Reliability (Brier Score + Calibration Curve)  
✔ Feature Importance (Tree-Based Models)  
✔ Misclassification Analysis  

---

## 📈 Key Evaluation Analysis

The following evaluation aspects were performed:

- ROC-AUC Curve
- Precision–Recall tradeoff
- Calibration Curve + Brier Score
- Misclassification Error Analysis
- Feature Importance Visualization

All evaluation visualizations are available inside the `/results` folder.

---

## 🧠 Model Interpretability

Feature importance was computed using tree-based models to analyze critical stroke indicators such as:

- Age  
- BMI  
- Hypertension  
- Glucose level  
- Smoking status  

This improves the trustworthiness of the system in real-world medical use cases.

---

## 📁 Project Structure

