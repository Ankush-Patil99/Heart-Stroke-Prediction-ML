![Python](https://img.shields.io/badge/Python-ML-blue)
![Status](https://img.shields.io/badge/Project-Completed-success)


# 🧠 Heart Stroke Prediction 

An end-to-end Machine Learning project to predict the probability of stroke using clinical and lifestyle data using Logistic Regression, Random forest,XGBoost and ensemble methods.
This project focuses heavily on **class imbalance handling, ensemble learning, calibration reliability, and model interpretability**, following industry-style ML practices.


## 🚩 Problem Statement

Stroke is one of the leading causes of mortality and long-term disability worldwide.  
In such healthcare problems, **missing stroke cases (false negatives)** can have severe real-world consequences.

The goal of this project is to:
- Predict stroke risk using patient medical and lifestyle features.
- Handle severe class imbalance.
- Compare baseline models vs ensemble strategies.
- Analyze model reliability and interpretability.

---

## 📊 Dataset

**Dataset:** Kaggle Stroke Prediction Dataset  
🔗 https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  

- Target variable: `stroke`  
- Highly imbalanced dataset (very few stroke cases).

> The dataset is not uploaded due to licensing.  
> Please download it directly from Kaggle and place it inside the `data/` folder if running locally.


## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ankush-Patil99/Heart-Stroke-Prediction-ML.git
   cd Heart-Stroke-Prediction-ML/heart-stroke-prediction

---

## 📂 Project Navigation

| Folder     | Description              | Link |
|-----------|--------------------------|------|
| `src/`    | Core ML pipeline code    | [Open](./heart-stroke-prediction/src) |
| `models/` | Saved trained models     | [Open](./heart-stroke-prediction/models) |
| `results/`| Plots & evaluation outputs | [Open](./heart-stroke-prediction/results) |
| `notebooks/` | Full Kaggle notebook | [Open](./heart-stroke-prediction/notebooks) |
| `data/`   | Dataset details          | [Open](./heart-stroke-prediction/data) |



---

## ⚙️ Models Implemented

- Logistic Regression  
- Random Forest  
- XGBoost  
- Voting Ensemble  
- Stacking Ensemble  

Models were tuned using **GridSearchCV** and evaluated using stratified splits and multiple metrics.

---

## 📈 Results

### 1. Model Performance (Imbalanced Setting)

| Model | Accuracy | ROC-AUC | Stroke Recall | Stroke Precision |
|-------|----------|---------|---------------|------------------|
| Logistic Regression | 0.95 | 0.81 | 0.00 | 0.00 |
| Random Forest | 0.95 | 0.83 | 0.00 | 0.00 |
| XGBoost | 0.95 | 0.84 | 0.00 | 0.00 |
| Voting Ensemble | 0.72 | **0.85** | **0.84** | 0.13 |
| Stacking Ensemble | 0.95 | 0.84 | 0.02 | 0.33 |

✅ Baseline models achieve high accuracy but completely fail on stroke detection.  
✅ Voting Ensemble significantly improves stroke recall.

---

### 2. Misclassification Analysis

| Model | TP | TN | FP | FN | Stroke Recall |
|------|----|----|----|----|--------------|
| Logistic Regression | 0 | 972 | 0 | 50 | 0.00 |
| Random Forest | 0 | 970 | 2 | 50 | 0.00 |
| XGBoost | 0 | 972 | 0 | 50 | 0.00 |
| Voting Ensemble | 42 | 691 | 281 | 8 | **0.84** |
| Stacking Ensemble | 1 | 970 | 2 | 49 | 0.02 |

**Key Insight:**  
The Voting model intentionally trades accuracy for **patient safety** by minimizing false negatives.

---

### 3. Calibration & Probability Reliability

Brier Scores (lower = better):

| Model | Brier Score |
|-------|------------|
| Logistic Regression | 0.0408 |
| Random Forest | 0.0468 |
| XGBoost | 0.0419 |
| Stacking Ensemble | 0.0417 |
| Voting Ensemble | 0.1566 |

- Baseline & stacking models have better calibrated probabilities.
- Voting model sacrifices calibration to improve stroke detection.

---

### 4. ROC Curve

ROC curve visualization available in `/results` for **Stacking Classifier**.

- Stacking ROC-AUC ≈ **0.84**
- Other ROC-AUC values:
  - Logistic: 0.81
  - RF: 0.83
  - XGBoost: 0.84
  - Voting: 0.85

---

### 5. Feature Importance

Top important features across Random Forest and XGBoost:

1. Age
2. Avg Glucose Level
3. BMI
4. Work Type (Private / Self-employed)
5. Smoking Status
6. Gender
7. Urban Residence

These align with real-world clinical risk factors, increasing trust in model behavior.

---

## 🧠 Final Conclusion

- Accuracy alone is misleading in medical datasets.
- Baseline models failed to detect any stroke patients due to class imbalance.
- The **Voting Ensemble** achieved:
  - Stroke Recall = **0.84**
  - Best ROC-AUC = **0.85**
- This makes it the most clinically suitable model despite lower accuracy.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Joblib

---

## 🚀 Future Improvements

- Add proper probability calibration (Platt / Isotonic).
- Deploy as web application (Streamlit or FastAPI).
- Add explainability using SHAP.
- Explore SMOTE / advanced imbalance handling.

---

## 👤 Author

**Ankush Patil**  
📧 Email: ankpatil1203@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/ankush-patil-48989739a  
🐙 GitHub: https://github.com/Ankush-Patil99  
