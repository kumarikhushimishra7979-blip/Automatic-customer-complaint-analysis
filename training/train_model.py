from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from complaint_analyzer.modeling import train_and_evaluate, save_model

# Put the private, human-verified training workbook here.
TRAINING_FILE = ROOT / "training" / "Consumer Complain.xlsx"
MODEL_FILE = ROOT / "models" / "complaint_model.joblib"

TEXT_COLUMN = "COMPLAINT_DETAILS"
LABEL_COLUMN = "COMPLAINT_SUBTYPE"


def main():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training workbook not found: {TRAINING_FILE}\n"
            "Place the verified training workbook in the training folder."
        )

    df = pd.read_excel(TRAINING_FILE)
    missing = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    training = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    training.columns = ["complaint", "category"]
    training["complaint"] = training["complaint"].fillna("").astype(str).str.strip()
    training["category"] = training["category"].fillna("").astype(str).str.strip()
    training = training[
        training["complaint"].ne("") & training["category"].ne("")
    ]

    print(f"Training rows: {len(training):,}")
    print(f"Distinct categories: {training['category'].nunique():,}")
    print("\nCategory distribution:")
    print(training["category"].value_counts().to_string())

    model, metrics = train_and_evaluate(
        training["complaint"].tolist(),
        training["category"].tolist(),
        test_size=0.20
    )
    save_model(model, MODEL_FILE)

    print("\nModel training completed.")
    print(f"Saved model: {MODEL_FILE}")
    print(metrics["evaluation_note"])

    if metrics["accuracy"] is not None:
        print(f"Holdout accuracy: {metrics['accuracy']:.4f}")
        print(f"Holdout macro F1: {metrics['macro_f1']:.4f}")
        print("\nPer-class evaluation:")
        print(pd.DataFrame(metrics["report"]).T.to_string())
    else:
        print(
            "A reliable stratified holdout score was not available. "
            "Review the dataset and gather more examples for rare categories."
        )


if __name__ == "__main__":
    main()
