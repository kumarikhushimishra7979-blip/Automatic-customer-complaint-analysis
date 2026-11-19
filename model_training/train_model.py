from pathlib import Path
import sys
import pandas as pd

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from complaint_analyzer.modeling import (
    train_and_evaluate,
    save_model
)

# Training dataset
DATA_PATH = ROOT / "training" / "Customer_Complaints_Preprocessed.xlsx"
# Final model destination
MODEL_PATH = ROOT / "models" / "complaint_model.joblib"


# ==========================================
# LOAD TRAINING DATA
# ==========================================

def load_training_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}\n"
            "Place Customer_Complaints_Preprocessed.xlsx "
            "in the project root directory."
        )

    df = pd.read_excel(
    DATA_PATH,
    sheet_name="TRAINING_DATA"
)

    required_columns = [
        "COMPLAINT_DETAILS",
        "COMPLAINT_SUBTYPE"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' is missing."
            )

    texts = df["COMPLAINT_DETAILS"].tolist()

    labels = df["COMPLAINT_SUBTYPE"].tolist()

    print("\nTraining dataset loaded successfully!")

    print("Total records:", len(df))

    print(
        "Unique categories:",
        df["COMPLAINT_SUBTYPE"].nunique()
    )

    return texts, labels


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("\n================================")
    print("CUSTOMER COMPLAINT MODEL TRAINING")
    print("================================")

    # Load data
    texts, labels = load_training_data()

    # Train and evaluate
    print("\nStarting model training...")

    model, metrics = train_and_evaluate(
        texts,
        labels,
        test_size=0.20
    )

    # ======================================
    # DISPLAY EVALUATION RESULTS
    # ======================================

    print("\n================================")
    print("MODEL EVALUATION RESULTS")
    print("================================")

    if metrics["accuracy"] is not None:

        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

        print(
            f"Macro F1 Score: {metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1 Score: {metrics['weighted_f1']:.4f}"
        )

    else:

        print("Evaluation metrics unavailable.")

    print("\nEvaluation Note:")
    print(metrics["evaluation_note"])

    print("\nCategories evaluated:")
    print(len(metrics["evaluated_categories"]))

    print("\nCategories excluded from evaluation:")

    for category, count in metrics["excluded_categories"].items():

        print(f"{category}: {count} records")

    # ======================================
    # SAVE FINAL MODEL
    # ======================================

    print("\nSaving final model...")

    save_model(model, MODEL_PATH)

    print("\n================================")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("================================")

    print("Model saved at:")
    print(MODEL_PATH)


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    main()