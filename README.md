# Loan-Approval-Project
💳 CreditWise — Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be approved, trained on applicant financial and demographic data using multiple classifiers.


📌 Overview

CreditWise helps financial institutions automate preliminary loan screening by predicting approval likelihood based on applicant profiles. The project compares three classifiers — Logistic Regression, K-Nearest Neighbors, and Naive Bayes — with feature engineering steps to improve accuracy.


🚀 Features


Automated missing value imputation (mean for numerical, mode for categorical)
Exploratory data analysis: class balance, income distributions, outlier boxplots
Label encoding + One-Hot encoding for categorical features
Feature engineering: squared terms (DTI_Ratio², Credit_Score²) and log transform (Applicant_Income_log)
StandardScaler normalization before model training
Side-by-side comparison of three ML models



🗂️ Project Structure

CreditWise/
├── credit_wise.ipynb          # Main Jupyter Notebook
├── loan_approval_data.csv     # Dataset (required)
├── app.py                     # Gradio web application
└── README.md


📊 Dataset

The project uses loan_approval_data.csv with the following features:

FeatureTypeDescriptionApplicant_IncomeNumericalPrimary applicant's monthly incomeCoapplicant_IncomeNumericalCo-applicant's monthly incomeCredit_ScoreNumericalApplicant's credit scoreDTI_RatioNumericalDebt-to-income ratioSavingsNumericalSavings amountGenderCategoricalGender of applicantEducation_LevelCategoricalHighest education levelEmployment_StatusCategoricalEmployment typeMarital_StatusCategoricalMarital statusLoan_PurposeCategoricalPurpose of the loanProperty_AreaCategoricalUrban / Semi-urban / RuralEmployer_CategoryCategoricalType of employerLoan_ApprovedTargetWhether loan was approved (Yes/No)


Place the CSV file in the same directory as the notebook before running.




🛠️ Tech Stack


Python 3
pandas / numpy — data manipulation and feature engineering
scikit-learn — preprocessing, model training, and evaluation
seaborn / matplotlib — EDA visualizations
gradio — interactive web demo





🧠 Model Pipeline

Raw Data (loan_approval_data.csv)

   │
   ▼
Impute Missing Values (mean / mode)

   │
   ▼
EDA — class balance, distributions, boxplots, correlation heatmap

   │
   ▼
Drop Applicant_ID

   │
   ▼
Label Encode (Education_Level, Loan_Approved)
One-Hot Encode (Employment_Status, Marital_Status, Loan_Purpose,
                Property_Area, Gender, Employer_Category)
                
   │
   ▼
Feature Engineering
   ├── DTI_Ratio² · Credit_Score²
   └── log1p(Applicant_Income)
   
   │
   ▼
Train/Test Split (80/20, random_state=42)

   │
   ▼
StandardScaler

   │
   ▼
┌──────────────────────────────────────┐
│  Logistic Regression  │  KNN  │  NB  │
└──────────────────────────────────────┘

   │
   ▼
   
Evaluate: Accuracy · Precision · Recall · F1 · Confusion Matrix



📈 Models Compared

ModelNotesLogistic RegressionBaseline linear model; run twice (before & after feature engineering)K-Nearest Neighborsn_neighbors=5; distance-based classificationNaive BayesGaussian NB; probabilistic approach
