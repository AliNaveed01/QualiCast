---

# 🏭 Quality Prediction in Plastic Injection Moulding using Machine Learning

An end-to-end machine learning solution to predict plastic product quality from injection moulding parameters—featuring advanced model comparison, statistical validation, and a real-time prediction dashboard.

---

## 📌 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Model Performance](#model-performance)
* [Dashboard](#dashboard)
* [Project Structure](#project-structure)

---

## 📖 Overview

Plastic injection moulding is a widely used manufacturing process. However, variations in critical parameters (like temperature or pressure) can result in poor-quality products, increased scrap, and high production costs.

This project uses machine learning to classify product quality into:

* Waste
* Acceptable
* Target
* Inefficient

Through exploratory data analysis, statistical testing (ANOVA), and robust model comparison, we identify the most reliable predictive model and deploy it using an interactive Streamlit dashboard.

---

## 🚀 Features

* ✅ **Data Cleaning & Preprocessing**: Outlier removal (IQR), normalization, correlation & ANOVA testing.
* ✅ **Modeling**: Five machine learning models implemented and benchmarked.
* ✅ **Best Model**: Random Forest with 93.4% accuracy & 0.986 ROC-AUC.
* ✅ **Real-Time Predictions**: Interactive Streamlit dashboard to input parameters and predict quality.
* ✅ **Model Interpretation**: Feature importance, confusion matrix, and classification metrics.
* ✅ **Statistical Analysis**: ANOVA feature analysis to validate parameter relevance.

---

## 🛠 Tech Stack

| Component        | Tool / Library         |
| ---------------- | ---------------------- |
| Language         | Python                 |
| ML Models        | scikit-learn, LightGBM |
| Deep Learning    | TensorFlow (for ANN)   |
| Dashboard        | Streamlit              |
| Data Handling    | pandas, numpy          |
| Visualization    | matplotlib, seaborn    |
| Statistical Test | scipy.stats (ANOVA)    |

---

## 🧰 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/quality-prediction-ml.git
cd quality-prediction-ml
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Make sure `random_forest_model.pkl` and `test_data.csv` are in the same directory.

---

## 🧪 Usage

The dashboard allows users to:

* Input custom process parameters
* Get quality predictions instantly
* View model confidence
* Analyze feature importance
* Inspect confusion matrix & classification report
* Understand feature relevance via ANOVA

---

## 📊 Model Performance

| Model             | Accuracy  | ROC-AUC   | F1 Score  |
| ----------------- | --------- | --------- | --------- |
| 🎯 Random Forest  | **93.4%** | **0.986** | **0.933** |
| ANN               | 85.9%     | 0.968     | 0.86      |
| Gradient Boosting | 92.4%     | 0.979     | 0.92      |
| LightGBM          | \~91%     | \~0.97    | \~0.91    |
| SVM               | \~88%     | \~0.95    | \~0.88    |

---

## 📁 Project Structure

```
├── app.py                     # Streamlit dashboard
├── Code.ipynb                # Full analysis & model training
├── random_forest_model.pkl   # Best trained model
├── test_data.csv             # Saved test dataset for validation
├── requirements.txt          # All dependencies
└── README.md                 # Project readme
```
