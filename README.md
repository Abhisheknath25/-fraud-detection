# 🛡️ Fraud Detection System — Credit Card Transactions

A machine learning system that detects fraudulent credit card transactions in
real time. It trains a classification model on the [Kaggle Credit Card Fraud
Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset and
exposes it through a **FastAPI** REST API with a sleek web interface.

> ### 🌐 Live Demo
>
> The model is **deployed and live** on Render:
>
> 🔗 **[https://fraud-detection-api-zttp.onrender.com](https://fraud-detection-api-zttp.onrender.com)**
>
> - **Web UI**: [https://fraud-detection-api-zttp.onrender.com](https://fraud-detection-api-zttp.onrender.com)
> - **Swagger Docs**: [https://fraud-detection-api-zttp.onrender.com/docs](https://fraud-detection-api-zttp.onrender.com/docs)
> - **Health Check**: [https://fraud-detection-api-zttp.onrender.com/health](https://fraud-detection-api-zttp.onrender.com/health)
>
> *Note: Free tier may take ~30s to wake up on first request after inactivity.*

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| **ML Pipeline** | Logistic Regression, Random Forest, and XGBoost — best model auto-selected |
| **Class Balancing** | SMOTE oversampling for the highly imbalanced dataset (0.17% fraud) |
| **REST API** | FastAPI with Swagger docs, health checks, and JSON predictions |
| **Web UI** | Dark-themed, responsive interface for interactive testing |
| **Risk Levels** | Predictions include LOW / MEDIUM / HIGH risk classification |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download `creditcard.csv` from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it
in the `data/` directory.

### 3. Train the model & start the API

```bash
python run.py --train
```

This will:
1. Load and preprocess the dataset
2. Train 3 models and select the best one
3. Save the model to `models/fraud_model.pkl`
4. Start the API server at `https://fraud-detection-api-zttp.onrender.com`

### 4. Open the web UI

Navigate to **https://fraud-detection-api-zttp.onrender.com** in your browser.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web testing interface |
| `/predict` | POST | Submit features → get fraud prediction |
| `/health` | GET | API & model health check |
| `/model-info` | GET | Training metrics for the loaded model |
| `/docs` | GET | Interactive Swagger documentation |

### Example Request (Live API)

```bash
curl -X POST https://fraud-detection-api-zttp.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0, -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, 149.62]}'
```

### Example Response

```json
{
  "prediction": "Legitimate",
  "fraud_probability": 0.0023,
  "risk_level": "LOW"
}
```

---

## 🗂️ Project Structure

```
├── data/                  Dataset directory
│   └── creditcard.csv     (download from Kaggle)
├── models/                Trained models & metrics
├── src/
│   ├── preprocess.py      Data loading, SMOTE, scaling
│   ├── train.py           Model training & evaluation
│   └── predict.py         Prediction utilities
├── api/
│   └── app.py             FastAPI REST API
├── templates/
│   └── index.html         Web testing interface
├── requirements.txt       Python dependencies
├── render.yaml            Render deployment config
├── run.py                 Entry point
└── README.md              This file
```

---

## 📊 Model Performance

After training, metrics and plots are saved in `models/`:

- `metrics.json` — Precision, Recall, F1, AUC-ROC for each model
- `plots/roc_curves.png` — ROC curve comparison
- `plots/pr_curves.png` — Precision-Recall curve comparison
- `plots/confusion_matrices.png` — Confusion matrices

---

## 🛠️ CLI Options

```
python run.py --help

Options:
  --train        Train the model before starting the API
  --train-only   Train the model and exit (don't start API)
  --host TEXT     API host (default: 127.0.0.1)
  --port INT      API port (default: 8000)
```

---

## ☁️ Deployment

This project is deployed on **[Render](https://render.com)** using the free tier.

| Detail | Value |
|--------|-------|
| **Platform** | Render |
| **URL** | [https://fraud-detection-api-zttp.onrender.com](https://fraud-detection-api-zttp.onrender.com) |
| **Runtime** | Python 3.10 |
| **Config** | `render.yaml` |
| **Repo** | [GitHub](https://github.com/Abhisheknath25/-fraud-detection) |

To redeploy, simply push to the `main` branch — Render auto-deploys on every push.

---

## 📜 License

This project is for educational purposes. The dataset is provided by the
[Machine Learning Group — ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
