# Early Stroke Risk Prediction Using Machine Learning

This project aims to build a reliable **stroke risk prediction model** using patient health data. The solution uses machine learning to help doctors and medical professionals assess stroke risk based on key medical indicators.

---

## Objective

To develop a machine learning-based classification model that predicts whether a person is at risk of a stroke using features like age, hypertension, heart disease, BMI, glucose level, and lifestyle habits.

---

## Dataset

- **Source:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Rows:** 5110 patients  
- **Features:** 12 columns including both numerical and categorical attributes  
- **Target Variable:** `stroke` (0 = no stroke, 1 = stroke)

---

## Preprocessing & Feature Engineering

- Removed 1 record with gender = "Other"
- Dropped the `id` column
- Handled missing values in `bmi` using median imputation
- Label encoded binary categorical features: `gender`, `ever_married`, `Residence_type`
- One-hot encoded multi-class features: `work_type`, `smoking_status`
- Standardized numerical features: `age`, `avg_glucose_level`, `bmi` using `StandardScaler`
- Balanced the dataset using **SMOTE** to address class imbalance (original: 3.8% stroke cases)

---

## Model Building

We trained and evaluated the following models:

| Model                | Accuracy | ROC-AUC | Key Notes |
|---------------------|----------|---------|-----------|
| Logistic Regression | 0.77     | 0.83    | Good recall for minority class after SMOTE |
| Decision Tree       | 0.87     | 0.59    | High accuracy, but poor generalization |
| Random Forest       | 0.89     | 0.76    | Best balance of accuracy & feature interpretability |
| XGBoost             | 0.94     | 0.52    | Best accuracy, but lower recall for stroke class |

---

## Final Model

- **Selected Model:** Random Forest (after hyperparameter tuning using `GridSearchCV`)
- **Best Parameters:**  
  - `n_estimators`: 200  
  - `max_depth`: None  
  - `min_samples_split`: 2  
  - `class_weight`: 'balanced'

The model achieves:
- **Accuracy:** ~88.7%
- **ROC-AUC Score:** ~0.76

---

## Feature Importance

We plotted feature importances using the Random Forest model. Top contributing features:
- `age`
- `avg_glucose_level`
- `bmi`
- `hypertension`
- `heart_disease`

---

## Inference on New Data

The final trained model is saved as:
- `xgboost_stroke_model.pkl` (XGBoost)
- `random_forest_stroke_model.pkl` (Random Forest)
- `model_features.pkl` (list of expected input features)

A Python function `predict_stroke_risk()` can be used to predict stroke risk on new patient data.

> Note: This is a **local, on-demand prediction setup**. The model is not deployed to a server or API for real-time use.

---

## Disclaimer

This project is for **educational purposes only**. The dataset is synthetic and should not be used in real medical diagnosis or treatment planning.

---

## GitHub Repo

If you found this project useful, feel free to star the repository and check out my other work!
