import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

model: Pipeline = joblib.load(MODEL_DIR / "model.joblib")

def predict_transaction(features):

    y_proba = model.predict_proba(features)

    return y_proba[:, 1]