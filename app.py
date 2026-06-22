"""
CreditWise — Loan Approval Prediction
Gradio web app that replicates the full notebook pipeline:
  - imputation → encoding → feature engineering → scaling → prediction
Run: python app.py
"""

import numpy as np
import pandas as pd
import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load & preprocess dataset (same as notebook)
# ─────────────────────────────────────────────

def load_and_train(csv_path="loan_approval_data.csv"):
    df = pd.read_csv(csv_path)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols   = df.select_dtypes(include=["number"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # Drop ID
    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)

    # Label encode
    le_edu = LabelEncoder()
    le_target = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le_target.fit_transform(df["Loan_Approved"])

    # One-hot encode
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded    = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Feature engineering (final version from notebook)
    df["DTI_Ratio_sq"]         = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"]      = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
    }
    trained, metrics = {}, {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        trained[name] = model
        y_pred = model.predict(X_test_sc)
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
            "Precision": round(precision_score(y_test, y_pred) * 100, 2),
            "Recall":    round(recall_score(y_test, y_pred)    * 100, 2),
            "F1 Score":  round(f1_score(y_test, y_pred)        * 100, 2),
        }

    return trained, scaler, ohe, le_edu, ohe_cols, X.columns.tolist(), metrics, le_target

# ─────────────────────────────────────────────
# 2. Build Gradio UI
# ─────────────────────────────────────────────

def build_app(csv_path="loan_approval_data.csv"):
    print("⏳ Loading dataset and training models …")
    try:
        trained, scaler, ohe, le_edu, ohe_cols, feature_cols, metrics, le_target = load_and_train(csv_path)
        dataset_loaded = True
        print("✅ Models trained successfully.")
    except FileNotFoundError:
        dataset_loaded = False
        print(f"⚠️  '{csv_path}' not found — prediction disabled. Update csv_path in app.py.")
        trained = scaler = ohe = le_edu = ohe_cols = feature_cols = metrics = le_target = None

    def predict(
        applicant_income, coapplicant_income, credit_score, dti_ratio, savings,
        gender, education_level, employment_status, marital_status,
        loan_purpose, property_area, employer_category,
        model_choice
    ):
        if not dataset_loaded:
            return "❌ Dataset not loaded. Place loan_approval_data.csv beside app.py.", ""

        # Build a one-row dataframe matching raw input format
        row = pd.DataFrame([{
            "Applicant_Income":    applicant_income,
            "Coapplicant_Income":  coapplicant_income,
            "Credit_Score":        credit_score,
            "DTI_Ratio":           dti_ratio,
            "Savings":             savings,
            "Gender":              gender,
            "Education_Level":     education_level,
            "Employment_Status":   employment_status,
            "Marital_Status":      marital_status,
            "Loan_Purpose":        loan_purpose,
            "Property_Area":       property_area,
            "Employer_Category":   employer_category,
        }])

        # Label encode Education_Level
        row["Education_Level"] = le_edu.transform(row["Education_Level"])

        # One-hot encode categorical cols
        encoded    = ohe.transform(row[ohe_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
        row = pd.concat([row.drop(columns=ohe_cols).reset_index(drop=True), encoded_df], axis=1)

        # Feature engineering
        row["DTI_Ratio_sq"]         = row["DTI_Ratio"] ** 2
        row["Credit_Score_sq"]      = row["Credit_Score"] ** 2
        row["Applicant_Income_log"] = np.log1p(row["Applicant_Income"])
        row = row.drop(columns=["Credit_Score", "DTI_Ratio"])

        # Align columns with training set
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_cols]

        row_sc   = scaler.transform(row)
        model    = trained[model_choice]
        pred     = model.predict(row_sc)[0]
        prob     = model.predict_proba(row_sc)[0] if hasattr(model, "predict_proba") else None

        label = le_target.inverse_transform([pred])[0]
        if label in ["Y", "Yes", "1", 1]:
            verdict = "✅ Loan APPROVED"
            color   = "green"
        else:
            verdict = "❌ Loan REJECTED"
            color   = "red"

        confidence = f"Confidence: {round(max(prob) * 100, 1)}%" if prob is not None else ""

        m = metrics[model_choice]
        metrics_str = (
            f"**{model_choice} — Test Set Performance**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Accuracy  | {m['Accuracy']}% |\n"
            f"| Precision | {m['Precision']}% |\n"
            f"| Recall    | {m['Recall']}% |\n"
            f"| F1 Score  | {m['F1 Score']}% |"
        )

        return f"### {verdict}\n{confidence}", metrics_str

    # ── UI layout ──
    with gr.Blocks(title="CreditWise — Loan Approval", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 💳 CreditWise — Loan Approval Predictor
            Fill in the applicant details below and click **Predict** to see the outcome.
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💰 Financial Details")
                applicant_income    = gr.Number(label="Applicant Income (monthly)", value=5000)
                coapplicant_income  = gr.Number(label="Co-applicant Income (monthly)", value=0)
                credit_score        = gr.Slider(300, 850, value=650, step=1, label="Credit Score")
                dti_ratio           = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="DTI Ratio (0–1)")
                savings             = gr.Number(label="Savings", value=10000)

            with gr.Column():
                gr.Markdown("### 👤 Personal Details")
                gender              = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                education_level     = gr.Dropdown(
                    ["Graduate", "Not Graduate"],
                    label="Education Level", value="Graduate"
                )
                marital_status      = gr.Dropdown(
                    ["Married", "Unmarried", "Single"],
                    label="Marital Status", value="Married"
                )
                employment_status   = gr.Dropdown(
                    ["Salaried", "Self-Employed", "Unemployed"],
                    label="Employment Status", value="Salaried"
                )
                employer_category   = gr.Dropdown(
                    ["Government", "Private", "NGO"],
                    label="Employer Category", value="Private"
                )

            with gr.Column():
                gr.Markdown("### 🏠 Loan Details")
                loan_purpose        = gr.Dropdown(
                    ["Home", "Education", "Business", "Personal", "Medical"],
                    label="Loan Purpose", value="Home"
                )
                property_area       = gr.Dropdown(
                    ["Urban", "Semiurban", "Rural"],
                    label="Property Area", value="Urban"
                )
                model_choice        = gr.Dropdown(
                    list(trained.keys()) if dataset_loaded else ["Logistic Regression"],
                    label="Model", value="Logistic Regression"
                )
                predict_btn         = gr.Button("🔍 Predict", variant="primary", size="lg")

        with gr.Row():
            result_output  = gr.Markdown(label="Prediction")
            metrics_output = gr.Markdown(label="Model Metrics")

        predict_btn.click(
            fn=predict,
            inputs=[
                applicant_income, coapplicant_income, credit_score, dti_ratio, savings,
                gender, education_level, employment_status, marital_status,
                loan_purpose, property_area, employer_category,
                model_choice
            ],
            outputs=[result_output, metrics_output]
        )

        gr.Markdown(
            """
            ---
            **Models available:** Logistic Regression · KNN (k=5) · Naive Bayes  
            *This tool is for educational purposes only and should not be used for real lending decisions.*
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app(csv_path="loan_approval_data.csv")
    app.launch()