"""
Train and compare multiple ML models for heart disease prediction.
Selects the best model by accuracy and saves it as model.pkl.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"


def load_data(csv_path: str = "heart.csv") -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the heart disease dataset."""
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def build_models() -> dict:
    """Define all candidate models."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", random_state=42)),
            ]
        ),
    }


def train_and_compare(
    X_train, X_test, y_train, y_test
) -> tuple[object, str, float, dict[str, float]]:
    """Train all models, compare accuracy, and return the best one."""
    models = build_models()
    results: dict[str, float] = {}

    print("=" * 50)
    print("Model Comparison (Accuracy Score)")
    print("=" * 50)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name:25s}: {accuracy:.4f}")

    best_name = max(results, key=results.get)
    best_accuracy = results[best_name]
    best_model = models[best_name]

    print("=" * 50)
    print(f"Best Model: {best_name} (Accuracy: {best_accuracy:.4f})")
    print("=" * 50)

    return best_model, best_name, best_accuracy, results


def save_model(
    model,
    model_name: str,
    accuracy: float,
    all_results: dict[str, float],
    output_path: str = "model.pkl",
) -> None:
    """Save the best model and metadata using Joblib."""
    artifact = {
        "model": model,
        "model_name": model_name,
        "accuracy": accuracy,
        "all_results": all_results,
        "features": FEATURE_COLUMNS,
    }
    joblib.dump(artifact, output_path)
    print(f"\nModel saved to {output_path}")


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model, best_name, best_accuracy, all_results = train_and_compare(
        X_train, X_test, y_train, y_test
    )
    save_model(best_model, best_name, best_accuracy, all_results)


if __name__ == "__main__":
    main()
