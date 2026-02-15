# Multi-Disease Prediction System using AI

A Flask-based web application that predicts risk for **Diabetes**, **Heart Disease**, and **Parkinson's disease** using trained machine learning models. The system also includes prediction explainability (SHAP), report logging, trend analysis, and an AI chatbot for follow-up guidance.

> ⚠️ **Medical disclaimer:** This project is for educational and prototype purposes only. Predictions are not medical diagnoses.

## Features

- Multi-disease prediction from browser forms:
  - Diabetes
  - Heart Disease
  - Parkinson's
- Model explainability with SHAP summary plots
- SQLite-based report logging and history
- Report analysis with trend/suggestion generation
- Gemini-powered chatbot endpoint for contextual Q&A
- Model training pipeline and preprocessing scripts

## Project Structure

```text
.
├── app.py                      # Flask app, routes, prediction flow, SHAP, chatbot
├── database.py                 # SQLite schema + initialization
├── data_preprocessing.py       # Dataset cleaning + scaling
├── train_models.py             # Model training/evaluation + saving best models
├── data/                       # Source CSV datasets
├── models/                     # Saved model/scaler artifacts (.pkl)
├── templates/                  # HTML templates
├── static/                     # CSS/JS assets
└── user_reports.db             # SQLite database file
```

## Requirements

- Python 3.10+
- pip

Recommended Python packages:

- Flask
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- shap
- google-genai

## Installation

1. Clone and enter the repository:

   ```bash
   git clone <your-repo-url>
   cd Multi-DiseasePredictionSystem-usingAI
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   # .venv\Scripts\activate      # Windows PowerShell
   ```

3. Install dependencies:

   ```bash
   pip install flask pandas numpy scikit-learn joblib matplotlib shap google-genai
   ```

## Setup

### 1) Initialize database

```bash
python database.py
```

This creates `user_reports.db` with `users` and `reports` tables.

### 2) Train (or refresh) models

```bash
python train_models.py
```

This generates/updates model artifacts in `models/`:

- `diabetes_model.pkl`, `diabetes_scaler.pkl`
- `heart_model.pkl`, `heart_scaler.pkl`
- `parkinsons_model.pkl`, `parkinsons_scaler.pkl`

## Running the app

```bash
python app.py
```

Then open:

- `http://127.0.0.1:5000/`

## Optional: Enable Gemini chatbot

Set the Gemini API key before launching the app:

```bash
export GEMINI_API_KEY="your_api_key_here"      # Linux/macOS
# setx GEMINI_API_KEY "your_api_key_here"       # Windows
```

If the key is missing or invalid, the chatbot endpoint responds with a fallback message.

## Main Routes

- `GET /` – home page
- `GET /diabetes` – diabetes input form
- `GET /heart` – heart disease input form
- `GET /parkinsons` – parkinson's input form
- `POST /predict_diabetes` – diabetes prediction
- `POST /predict_heart` – heart prediction
- `POST /predict_parkinsons` – parkinson's prediction
- `GET /results/<disease>` – prediction + SHAP result page
- `GET /report_analysis` – history analysis and health suggestions
- `POST /chat` – chatbot response endpoint

## Notes

- Keep `models/` artifacts synchronized with your current training code and datasets.
- `app.py` uses an in-memory cache for SHAP image payloads between prediction and result rendering.
- Debug mode is enabled by default in the current startup block.

## Future Improvements

- Add `requirements.txt` / `pyproject.toml` for reproducible installs
- Improve authentication (currently uses a default placeholder user)
- Add unit/integration tests
- Add Docker support for deployment

---

If you want, I can also add a `requirements.txt` and a one-command setup script (`make setup && make run`) in a follow-up change.
