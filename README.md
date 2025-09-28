
# Smart Logistics Fraud Detection 🚚

A production-ready pipeline for detecting **fraudulent logistics events** from shipment and invoice data using a Random Forest classifier. This project focuses on engineering interpretable features based on **route deviation**, **timestamp mismatches**, and **document discrepancies** to create a robust and explainable fraud detection system.

A simple web UI scaffold (`index.html`) is also provided to present real-time insights and alerts.

## ✨ Key Features

* **End-to-End Python Workflow:** Complete pipeline including feature engineering, stratified train/test split, modeling, evaluation, and artifact export (`ml_evaluation_results.json`).
* **Interpretable Signals:** Detection is driven by highly interpretable features such as:
    * `route_dev_flag` and `route_token_overlap` (Jaccard similarity).
    * `timestamp_diff_minutes` and `timestamp_mismatch_flag` (delta > 60 minutes).
    * `invoice_mismatch_flag` and `warehouse_mismatch_flag`.
* **Benchmarked Results:** Evaluation metrics (including Confusion Matrices and per-class F1 scores) are provided across multiple dataset sizes for reliable performance analysis.
* **Frontend Prototype:** A foundational HTML (`index.html`) to visualize analytics such as Fraud Identification, Risk Analysis, and Live Alerts.

---

## 🚀 How to Run

### 1. Install Dependencies

The project requires standard Python data science libraries. Python 3.10+ is recommended.

```bash
pip install pandas numpy scikit-learn
````

### 2\. Prepare Data

Place your input CSV files under a local directory named `data/`. The pipeline expects:

  * `final_processed_fraud_dataset.csv` (Example dataset)
  * Optional: Other synthetic datasets for extended benchmarking.

### 3\. Execute the Pipeline

Run the main Python script to train the model, evaluate performance, and generate metrics.

```bash
python classification_model.py
```

On completion, metrics will be written to **`ml_evaluation_results.json`**.

-----

## 📊 Metrics Snapshot

The core model is a **RandomForestClassifier** (100 trees, `max_depth=10`, `class_weight="balanced"`).

| Dataset Name | Size (n) | Overall Accuracy | Weighted F1 | **Fraudulent Class F1** | CM (tn, fp, fn, tp) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **all** | 3,000 | 0.8517 | 0.8603 | **0.6337** | (314, 66, 23, 77) |
| **fraud\_15000** | 15,000 | 0.8523 | 0.8541 | **0.6811** | (2084, 245, 198, 473) |
| **fraud\_30000** | 30,000 | 0.9178 | 0.9110 | **0.7780** | (4643, 7, 486, 864) |

### Key Feature Importance Signals

  * **Smaller Set:** `license_len`, `Delay_Hours`, `Expected_Warehouse`, and `Actual_Route`.
  * **Larger Sets:** `Origin`, **`timestamp_mismatch_flag`**, and **`invoice_mismatch_flag`**.

-----

## ⚙️ Project Structure

| File/Directory | Description |
| :--- | :--- |
| `classification_model.py` | Core pipeline: feature engineering, model training, evaluation, and JSON export. |
| `ml_evaluation_results.json` | Saved output metrics for multiple datasets. |
| `final_processed_fraud_dataset.csv` | Example dataset used for training/evaluation. |
| `index.html` | UI scaffold for analytics and live alerts. |
| `data/` | Directory for all input CSV files. |

### Data Expectations

The pipeline requires a CSV with at least these columns. Missing ones are safely handled. `True_Label` must be label-encodable (e.g., 'authorized' vs 'fraudulent').

`Company_License`, `Planned_Route`, `Actual_Route`, `Planned_Timestamp`, `Actual_Timestamp`, `Delivered_Invoice_ID`, `Generated_Invoice_ID`, `Expected_Warehouse`, `Actual_Warehouse`, `Return_Flag`, `Refund_Flag`, **`True_Label`**.

-----

## ⏭ Roadmap

  * **Threshold Tuning:** Implement calibrated probabilities for cost-sensitive decisions.
  * **Model Expansion:** Explore Gradient Boosting (e.g., XGBoost) and ensembling.
  * **Monitoring:** Introduce temporal validation and drift detection.
  * **UI Integration:** Connect `index.html` to the `ml_evaluation_results.json` output for dynamic visualization.

## 📄 License

\[Placeholder for a suitable OSS license, e.g., MIT or Apache-2.0.]

```
```
