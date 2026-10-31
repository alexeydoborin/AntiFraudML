from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from pathlib import Path
from antifraud.data import load_data, split_data

import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_DIR / "elliptic_txs_features.csv", DATA_DIR / "elliptic_txs_classes.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print(classification_report(y_val, y_pred))
    joblib.dump(model, MODEL_DIR / "model.joblib")

if __name__ == "__main__":
    main()