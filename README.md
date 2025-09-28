Smart Logistics Fraud Detection 🚚
A production-ready pipeline for detecting fraudulent logistics events from shipment and invoice data using a Random Forest classifier. This project focuses on engineering interpretable features based on route deviation, timestamp mismatches, and document discrepancies to create a robust and explainable fraud detection system.

A simple web UI scaffold (index.html) is also provided to present real-time insights and alerts.

✨ Key Features
End-to-End Python Workflow: Complete pipeline including feature engineering, stratified train/test split, modeling, evaluation, and artifact export (JSON metrics).

Interpretable Signals: Detection is driven by highly interpretable, model-agnostic features such as:

route_dev_flag and route_token_overlap (Jaccard similarity).

timestamp_diff_minutes and timestamp_mismatch_flag (delta > 60 minutes).

invoice_mismatch_flag and warehouse_mismatch_flag.

Benchmarked Results: Evaluation metrics (including Confusion Matrices and per-class F1 scores) are provided across multiple dataset sizes for reliable performance analysis.

Frontend Prototype: A foundational HTML (index.html) to visualize analytics such as Fraud Identification, Risk Analysis, and Live Alerts.

🚀 How to Run
1. Install Dependencies
The project is built using standard Python data science libraries. Python 3.10+ is recommended.

Bash

pip install pandas numpy scikit-learn
2. Prepare Data
Place your input CSV files under a local directory named data/. The core pipeline expects the following files to be available:

final_processed_fraud_dataset.csv (Example dataset)

Optional: Other datasets like synthetic_fraud_*.csv for extended benchmarking.

3. Execute the Pipeline
Run the main Python script to train the model, evaluate performance, and generate metrics.

Bash

python classification_model.py
On completion, all evaluation metrics and feature importances will be written to ml_evaluation_results.json.

📊 Metrics Snapshot
The model is a RandomForestClassifier (100 trees, max_depth=10, class_weight="balanced"), specifically tuned to mitigate class imbalance.

Dataset Name	Size (n)	Overall Accuracy	Weighted F1-Score	Fraudulent Class F1	Confusion Matrix (tn, fp, fn, tp)
all	3,000	0.8517	0.8603	0.6337	tn=314, fp=66, fn=23, tp=77
fraud_15000	15,000	0.8523	0.8541	0.6811	tn=2084, fp=245, fn=198, tp=473
fraud_30000	30,000	0.9178	0.9110	0.7780	tn=4643, fp=7, fn=486, tp=864

Export to Sheets
Feature Importance Highlights
Feature importances provide explainability and guide feature pruning:

Smaller Set: license_len, Delay_Hours, Expected_Warehouse, and Actual_Route are critical signals.

Larger Sets: Features derived from direct mismatch checks, such as timestamp_mismatch_flag and invoice_mismatch_flag, gain significant prominence.

⚙️ Project Structure
File/Directory	Description
classification_model.py	The core Python pipeline: feature engineering, model training, evaluation, and JSON export.
ml_evaluation_results.json	Saved output metrics (Confusion Matrices, F1 Scores, Feature Importances) from the pipeline execution.
final_processed_fraud_dataset.csv	Example dataset used for training/evaluation.
index.html	The scaffold for the web UI, including layout for alerts and analytics.
data/	Directory where input CSV files are expected to be located.
bg*.jpg	Background assets for the UI styling.

Export to Sheets
Data Expectations
The pipeline expects a CSV with at least the following columns. Missing columns are auto-created as NaN and safely handled.

Company_License

Planned_Route, Actual_Route

Planned_Timestamp, Actual_Timestamp

Delivered_Invoice_ID, Generated_Invoice_ID

Expected_Warehouse, Actual_Warehouse

Return_Flag, Refund_Flag

True_Label (Must contain two classes, e.g., 'authorized' vs 'fraudulent').

🛠 Feature Engineering Summary
classification_model.py generates the following robust, model-agnostic features:

Feature Category	Features Derived	Description
License	license_len, has_digits, has_letters	Checks on the Company_License string.
Route	route_dev_flag, route_token_overlap	Binary flag for deviation and Jaccard overlap of route tokens.
Time	timestamp_diff_minutes, timestamp_mismatch_flag	Time delta and a flag for discrepancies over 60 minutes.
Document	invoice_mismatch_flag, warehouse_mismatch_flag	Binary flags for discrepancies between planned/actual documents and locations.
Complaint Proxy	customer_complaint_flag	Derived from Return_Flag/Refund_Flag fields.

Export to Sheets
Note: Redundant identifiers (Company_Name, Shipment_ID, raw timestamps/invoice IDs) are intentionally dropped before modeling to prevent leakage and promote generalization.

⏭ Roadmap
Model Optimization: Implement threshold tuning and calibrated probabilities for cost-sensitive decision-making.

Model Diversification: Evaluate and integrate additional models (e.g., Gradient Boosting) and ensembling techniques.

Monitoring: Introduce temporal validation and drift monitoring, especially for timestamp-based features.

Full UI Integration: Integrate index.html with the ml_evaluation_results.json output via an API for live, interactive charts and dashboard visualization.

📄 License
[TBD: Add a suitable OSS license, e.g., MIT or Apache-2.0, before open-sourcing.]
