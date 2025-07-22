# 🫀 Heart Disease Prediction - Machine Learning Project

This project focuses on predicting the presence of heart disease using various supervised machine learning models. The pipeline includes data preprocessing, feature engineering, model training, and performance evaluation to determine the most accurate prediction method.

## 📁 Dataset
The dataset contains clinical attributes such as:
- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Exercise Induced Angina
- ST Depression
- Thallium Test Result
- Number of Major Vessels
- Target (0: No Disease, 1: Disease)

## 🔧 Preprocessing & Cleaning
- **Missing Values** handled using:
  - KNN imputation (for numerical features)
  - Mode replacement (for categorical features)
- **Outliers** detected and handled using IQR
- **Encoding**:
  - Label Encoding
  - One-Hot Encoding (for multiclass categorical variables)
- **Feature Scaling** (where needed)

## 📊 Feature Selection
- Selected key features using:
  - Mutual Information via `SelectKBest`
  - Correlation heatmap
  - Box plots for outlier visualization

## 🧠 Models Trained
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Logistic Regression (with and without regularization)

## ✅ Evaluation Metrics
Each model was trained using a train/test split and evaluated based on:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Mean Squared Error

## 🏁 Results
Models were compared to identify which performed best for predicting heart disease. Features such as **Chest Pain Type**, **Thallium**, and **Number of Major Vessels Fluro** showed strong predictive power.

## 📦 Dependencies
- Python 3.8+
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
