<div align="center">

# 🫀 Heart Stroke Predictor

### *An AI-powered clinical risk assessment web app*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.1-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Joblib](https://img.shields.io/badge/Joblib-1.5.1-4CAF50?style=for-the-badge&logo=python&logoColor=white)](https://joblib.readthedocs.io/)

---

> **Predict heart disease risk in seconds** using a trained K-Nearest Neighbors model — just fill in your clinical parameters and get an instant risk assessment.

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Input Parameters](#-input-parameters)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [File Structure](#-file-structure)

---

## 🔬 About the Project

**Heart Stroke Predictor** is a machine learning web application built to assist in early-stage heart disease risk screening. By entering a few clinical and lifestyle parameters, the app uses a pre-trained **K-Nearest Neighbors (KNN)** classifier to predict whether a patient is at **high** or **low** risk of heart disease.

This project bridges the gap between clinical data and actionable health insights — powered by a clean, interactive **Streamlit** interface that requires no medical expertise to operate.

> ⚠️ *This tool is intended for educational and screening purposes only. It is not a substitute for professional medical advice.*

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🌐 **Frontend / UI** | Streamlit | Interactive web interface |
| 🤖 **ML Model** | Scikit-Learn (KNN) | Heart disease classification |
| 🔢 **Data Processing** | Pandas, NumPy | Feature engineering & input formatting |
| 📦 **Model Persistence** | Joblib | Loading pre-trained model & scaler |
| 📊 **Visualization** | Matplotlib, Seaborn | (Used during model training/EDA) |
| 🗺️ **Charting** | Altair, Pydeck | Extended Streamlit chart support |

---

## ✨ Features

- 🏥 **11 clinical input parameters** — age, sex, ECG, cholesterol, blood pressure, and more
- ⚡ **Instant prediction** — real-time inference via a pre-trained KNN model
- 📏 **Automatic feature scaling** — uses a fitted `StandardScaler` for consistent input
- 🔄 **One-hot encoded inputs** — handles categorical variables correctly before inference
- ✅ **Clear risk output** — success (low risk) or error (high risk) visual feedback
- 🖥️ **Zero setup for end users** — runs entirely in the browser via Streamlit

---

## 🏗️ Project Architecture

```
User Input (Streamlit UI)
        │
        ▼
┌─────────────────────────────┐
│     Input Collection        │  ← age, sex, chest pain, BP, cholesterol,
│     (11 Clinical Features)  │    fasting BS, ECG, max HR, angina, oldpeak, ST slope
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   One-Hot Encoding &        │  ← Converts categorical values (sex, chest pain type,
│   Column Alignment          │    ECG, angina, ST slope) into binary columns
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   StandardScaler            │  ← Loaded from heart_scaler.pkl
│   (Feature Normalization)   │    Normalizes numerical features to mean=0, std=1
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   KNN Classifier            │  ← Loaded from knn_heart_model.pkl
│   (Trained ML Model)        │    Predicts: 0 = Low Risk / 1 = High Risk
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Result Display            │  ← ✅ Low Risk  or  ⚠️ High Risk
│   (Streamlit Output)        │
└─────────────────────────────┘
```

---

## 📊 Input Parameters

| # | Parameter | Type | Description |
|---|-----------|------|-------------|
| 1 | **Age** | Slider (18–100) | Patient's age in years |
| 2 | **Sex** | Dropdown | M (Male) / F (Female) |
| 3 | **Chest Pain Type** | Dropdown | ATA / NAP / TA / ASY |
| 4 | **Resting Blood Pressure** | Number Input | In mm Hg (80–200) |
| 5 | **Cholesterol** | Number Input | In mg/dL (100–600) |
| 6 | **Fasting Blood Sugar** | Dropdown | 0 = Normal / 1 = > 120 mg/dL |
| 7 | **Resting ECG** | Dropdown | Normal / ST / LVH |
| 8 | **Max Heart Rate** | Slider (60–220) | Maximum heart rate achieved |
| 9 | **Exercise-Induced Angina** | Dropdown | Y (Yes) / N (No) |
| 10 | **Oldpeak** | Slider (0.0–6.0) | ST depression during exercise |
| 11 | **ST Slope** | Dropdown | Up / Flat / Down |

---

## ⚙️ How It Works

1. **Input Collection** — The user fills in 11 clinical parameters via Streamlit widgets.
2. **One-Hot Encoding** — Categorical inputs are encoded into binary columns matching the training data format (e.g., `Sex_M`, `ChestPainType_ATA`).
3. **Column Alignment** — Missing one-hot columns are filled with `0` using the saved `heart_columns.pkl` to ensure consistent feature shape.
4. **Scaling** — The input DataFrame is transformed using the pre-fitted `StandardScaler` from `heart_scaler.pkl`.
5. **Prediction** — The scaled input is passed into the KNN model loaded from `knn_heart_model.pkl`.
6. **Result Display** — The app shows `⚠️ High Risk` or `✅ Low Risk` based on the model output.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/rizwimohdaltamash/Heart-Stroke-Prediction.git
cd Heart-Stroke-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501` in your browser.

---

## 🖱️ Usage

1. Open the app in your browser after running the command above.
2. Use the sliders and dropdowns to fill in your clinical details.
3. Click the **"Predict"** button.
4. View your **risk assessment result** instantly on screen.

---

## 📁 File Structure

```
Heart-Stroke-Prediction/
│
├── app.py                  # Main Streamlit application
├── knn_heart_model.pkl     # Pre-trained KNN classifier
├── heart_scaler.pkl        # Fitted StandardScaler for feature normalization
├── heart_columns.pkl       # Expected column names after one-hot encoding
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

<div align="center">

Made with ❤️ by **Mohd. Altamash Rizwi**

[![GitHub](https://img.shields.io/badge/GitHub-rizwimohdaltamash-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rizwimohdaltamash)

</div>
