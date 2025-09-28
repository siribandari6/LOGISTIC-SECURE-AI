import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier # HistGradientBoostingClassifier is no longer needed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

def feature_engineer(df):
    """Create features described by user."""
    df = df.copy()
    for c in ["Company_License", "Planned_Route", "Actual_Route", "Planned_Timestamp",
              "Actual_Timestamp", "Delivered_Invoice_ID", "Generated_Invoice_ID",
              "Expected_Warehouse", "Actual_Warehouse", "Return_Flag", "Refund_Flag"]:
        if c not in df.columns: df[c] = np.nan
    df["license_len"] = df["Company_License"].astype(str).apply(len)
    df["license_has_digits"] = df["Company_License"].astype(str).str.contains(r"\d").astype(int)
    df["license_has_letters"] = df["Company_License"].astype(str).str.contains(r"[A-Za-z]").astype(int)
    df["route_dev_flag"] = (df["Planned_Route"].fillna("").astype(str) != df["Actual_Route"].fillna("").astype(str)).astype(int)
    def token_overlap(a, b):
        a_tokens = set(str(a).split())
        b_tokens = set(str(b).split())
        if not a_tokens and not b_tokens: return 1.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        return inter/union if union>0 else 1.0
    df["route_token_overlap"] = df.apply(lambda r: token_overlap(r["Planned_Route"], r["Actual_Route"]), axis=1)
    def to_datetime_safe(x):
        try: return pd.to_datetime(x)
        except Exception: return pd.NaT
    dt_planned = df["Planned_Timestamp"].apply(to_datetime_safe)
    dt_actual = df["Actual_Timestamp"].apply(to_datetime_safe)
    df["timestamp_diff_minutes"] = (dt_actual - dt_planned).dt.total_seconds() / 60.0
    df["timestamp_diff_minutes"] = df["timestamp_diff_minutes"].fillna(0.0)
    df["timestamp_mismatch_flag"] = (df["timestamp_diff_minutes"].abs() > 60).astype(int)
    df["invoice_mismatch_flag"] = (df["Delivered_Invoice_ID"].fillna("").astype(str) != df["Generated_Invoice_ID"].fillna("").astype(str)).astype(int)
    df["warehouse_mismatch_flag"] = (df["Expected_Warehouse"].fillna("").astype(str) != df["Actual_Warehouse"].fillna("").astype(str)).astype(int)
    if "Return_Flag" in df.columns: df["customer_complaint_flag"] = df["Return_Flag"].fillna(0).astype(int)
    elif "Refund_Flag" in df.columns: df["customer_complaint_flag"] = df["Refund_Flag"].fillna(0).astype(int)
    else:
        text_cols = df.select_dtypes(include="object").columns.tolist()
        complaint = np.zeros(len(df), dtype=int)
        for c in text_cols: complaint |= df[c].fillna("").astype(str).str.contains("return|refund|complain|complaint", case=False, na=False).astype(int)
        df["customer_complaint_flag"] = complaint
    drop_cols = ["Company_Name", "Shipment_ID", "Planned_Timestamp", "Actual_Timestamp", "Delivered_Invoice_ID", "Generated_Invoice_ID"]
    for c in drop_cols:
        if c in df.columns: df = df.drop(columns=[c])
    return df

def prepare_X_y(df, target_col="True_Label"):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target '{target_col}' not found in dataframe")
    
    df[target_col] = df[target_col].astype(str)
    
    if df[target_col].nunique() < 2:
        return None, None 

    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y

def build_and_evaluate(X, y): # model_name parameter is removed
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_transformer = SimpleImputer(strategy="median")
    cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols), ("cat", cat_transformer, cat_cols)], remainder="passthrough")
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
        
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    X_enc = X.copy()
    for c in cat_cols:
        le = LabelEncoder()
        X_enc[c] = le.fit_transform(X_enc[c].astype(str))
        
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    try:
        importances = pipe.named_steps['model'].feature_importances_
        feature_names = X_enc.columns
        feature_importance_dict = dict(zip(feature_names, importances))
    except AttributeError:
        feature_importance_dict = {}

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    return {
        "model": "RandomForest", # Hardcoded model name
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "report": [
            {"class": "authorized", **report_dict.get("0", {})},
            {"class": "fraudulent", **report_dict.get("1", {})}
        ],
        "features": feature_importance_dict
    }

if __name__ == "__main__":
    dataset_paths = {
        "all": "data/final_processed_fraud_dataset.csv",
        "fraud_15000": "data/synthetic_fraud_15000_refined.csv",
        "fraud_30000": "data/synthetic_fraud_30000_refined.csv",
        "fraud_60000": "data/synthetic_fraud_60000_refined.csv"
    }
    all_results = {}
    for name, path in dataset_paths.items():
        print(f"\n=== Processing Dataset: {name} ({path}) ===")
        if not os.path.exists(path):
            print(f"Warning: File not found, skipping.")
            continue
        df = pd.read_csv(path)
        df_fe = feature_engineer(df)
        X, y = prepare_X_y(df_fe, target_col="True_Label")
        if X is None:
            print("Skipping due to insufficient classes.")
            continue
        
        # --- MODIFICATION: Only run the single evaluation function ---
        result = build_and_evaluate(X, y)
        print(f"  RandomForest F1: {result['f1Score']:.4f}")
        
        result['size'] = len(df)
        all_results[name] = result
        # -----------------------------------------------------------

    results_filename = 'ml_evaluation_results.json'
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✅ All evaluation results saved to {results_filename}")