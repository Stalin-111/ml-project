"""Train a diabetes classification model and save it to disk."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV dataset into a DataFrame."""
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zeros in select columns with NaN, then impute with median values."""
    zero_to_nan_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    df = df.copy()
    df[zero_to_nan_cols] = df[zero_to_nan_cols].replace(0, np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df


def train_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Train a logistic regression model with standard scaling."""
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="liblinear"),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print model evaluation metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("\nModel Evaluation")
    print("================")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", matrix)
    print("Classification Report:\n", report)


def save_model(model: Pipeline, output_path: str) -> None:
    """Save the trained model to disk."""
    joblib.dump(model, output_path)
    print(f"\nModel saved to {output_path}")


def predict_diabetes(model: Pipeline, features: list[float]) -> str:
    """Predict diabetes outcome for a single patient."""
    prediction = model.predict([features])[0]
    return "Diabetic" if prediction == 1 else "Not Diabetic"


def main() -> None:
    """Run the full training workflow."""
    data_path = "diabetes.csv"
    model_path = "diabetes_model.joblib"

    df = load_data(data_path)

    print("Dataset Info")
    print("============")
    print(df.info())
    print("\nSummary Statistics")
    print("=================")
    print(df.describe())

    df = clean_data(df)

    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)

    example_features = X_test.iloc[0].tolist()
    outcome = predict_diabetes(model, example_features)
    print(f"\nSample prediction: {outcome}")


if __name__ == "__main__":
    main()
