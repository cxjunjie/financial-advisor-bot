import joblib
from pathlib import Path

MODEL_PATH = Path("models/logreg_pipeline.pkl")

def load_model():
    """Load the trained pipeline (scaler + logistic regression)."""
    return joblib.load(MODEL_PATH)

def predict(pipeline, latest_row, feature_cols):
    """Run prediction on the latest row."""
    X_last = latest_row[feature_cols].values.reshape(1, -1)
    prob = pipeline.predict_proba(X_last)[0, 1]
    signal = "BUY" if prob >= 0.5 else "SELL"
    return prob, signal
