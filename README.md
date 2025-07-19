# 🧠 Disease Prediction Models

This project contains two distinct machine learning models for predicting the likelihood of specific medical conditions: **Diabetes** and **Heart Disease**. Both models are built in Python using the **scikit-learn** library.

---

## 📌 Features

✅ **Two Prediction Systems**  
- **Diabetes Prediction**: Based on diagnostic medical attributes.  
- **Heart Disease Prediction**: Based on health and physiological indicators.

✅ **Model Training & Evaluation**  
- Complete training workflows with train-test splitting and accuracy scoring.

✅ **Model Persistence**  
- Models are saved as `.sav` files using `pickle` for easy reuse.

✅ **Sample Prediction**  
- Scripts include a demonstration of predicting a new instance.

✅ **Web App Support**  
- Frontend integration using **Streamlit** and **streamlit-option-menu** for interactive model deployment.

---

## 🧪 Models & Datasets

### 1. 🩺 Diabetes Prediction
- **Objective**: Predict whether a patient has diabetes.  
- **Algorithm**: Support Vector Machine (SVM) with a linear kernel.  
- **Script**: `diabetes_prediction.py`  
- **Dataset**: `diabetes.csv`  
- **Target Variable**: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)  
- **Saved Model**: `trained_diabetes_model.sav`  

---

### 2. ❤️ Heart Disease Prediction
- **Objective**: Predict whether a patient has heart disease.  
- **Algorithm**: Logistic Regression  
- **Script**: `heart_disease_prediction.py`  
- **Dataset**: `heart_disease_data.csv`  
- **Target Variable**: `target` (0 = No Heart Disease, 1 = Has Heart Disease)  
- **Saved Model**: `trained_heart_model.sav`  

---

## 💻 Requirements

### Libraries Used

- `numpy`  
- `pandas`  
- `scikit-learn`  
- `pickle-mixin`  
- `streamlit`  
- `streamlit-option-menu`  

### Installation

Use the following command to install all required libraries:

```bash
pip install numpy pandas scikit-learn pickle-mixin streamlit streamlit-option-menu
