# 🏡 House Price Prediction using Machine Learning
_A Machine Learning project predicting house prices with Regression & Ensemble Models._
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn%20%7C%20XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub stars](https://img.shields.io/github/stars/YourUsername/house-price-prediction-ml?style=social)


This project predicts **house prices** using machine learning techniques.  
It is based on the [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and demonstrates the **end-to-end ML pipeline**: data preprocessing, feature engineering, model training, evaluation, and saving models.

---
##  Table of Contents
- [Overview](#-project-overview)
- [Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Models](#-models-used)
- [Workflow](#-workflow)
- [Results](#-results)
- [Run Locally](#-how-to-run-locally)
- [Future Work](#-future-improvements)
- [Author](#-author)


##  Project Overview
The main goal of this project is to build a regression model that can predict the sale price of a house based on various features such as:
- Lot size
- Neighborhood
- Number of rooms
- Garage type
- Overall quality of the house  
and more.

This project is part of my AI/ML internship at **INLIGHN TECH**.

---

##  Project Structure
house-price-prediction-ml/
│── data/ # Dataset files (if small, else link to Kaggle)
│── notebook.ipynb # Google Colab notebook (main code)
│── house_price_model.pkl # Saved trained model (via joblib)
│── README.md # Project documentation
│── requirements.txt # Python dependencies


---

## ⚙ Tech Stack
- **Python 3.8+**
- **NumPy, Pandas** → Data manipulation
- **Matplotlib, Seaborn** → Exploratory Data Analysis (EDA)
- **Scikit-learn** → Preprocessing & ML models
- **Joblib** → Saving trained models
- **Google Colab** → Development environment

---

##  Models Used
- **Linear Regression** → Baseline model
- **Ridge & Lasso Regression** → Regularized linear models
- **Decision Tree Regressor** → Nonlinear, interpretable model
- **Random Forest Regressor** → Ensemble learning, strong performance
- **Gradient Boosting Regressor** → Boosted trees, advanced performance
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

---

##  Workflow
1. **Data Preprocessing**
   - Handle missing values (median for numeric, mode for categorical)
   - One-hot encode categorical features
   - Scale numerical features

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots
   - Correlation heatmaps
   - Feature importance analysis

3. **Model Training**
   - Train/test split
   - Train multiple models (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting)
   - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

4. **Model Evaluation**
   - Metrics: MAE, RMSE, R²
   - Compare different models
   - Select best performing model

5. **Model Saving**
   - Save trained model using `joblib`
   - Load model for predictions

---

##  Results
- **Best Model:** Random Forest Regressor
- **Performance:**  
  - RMSE ≈ (add your result here)  
  - R² ≈ (add your result here)
| Model          | CV RMSE | Test RMSE | R² Score |
|----------------|---------|-----------|----------|
| Linear Reg.    | 0.22    | 0.24      | 0.72     |
| Random Forest  | 0.12    | 0.13      | 0.88     |
| XGBoost        | 0.11    | 0.12      | 0.90     |

---

## 🛠 How to Run Locally
1. Clone this repository:
   ```bash

## 👨‍💻 Author
**Vaibhav Shakya**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/your-link/)  
[![GitHub](https://img.shields.io/badge/GitHub-black)](https://github.com/yourusername)
