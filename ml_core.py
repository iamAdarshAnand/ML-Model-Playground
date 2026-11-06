"""Reusable model training and evaluation logic."""
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

DATASETS = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer}
MODELS = ("Logistic Regression", "Random Forest", "K-Nearest Neighbors")

@dataclass
class TrainingResult:
    accuracy: float
    macro_f1: float
    confusion: np.ndarray
    report: str
    truth: np.ndarray
    predictions: np.ndarray
    labels: list[str]

def load_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASETS[name](as_frame=True)

def make_model(name: str, seed: int = 42):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=150, random_state=seed)
    if name == "K-Nearest Neighbors":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    raise ValueError(f"Unknown model: {name}")

def train_evaluate(dataset_name: str, model_name: str, test_size: float = 0.25, seed: int = 42) -> TrainingResult:
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    dataset = load_dataset(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=test_size, random_state=seed, stratify=dataset.target
    )
    model = make_model(model_name, seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    classes = list(range(len(dataset.target_names)))
    return TrainingResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        confusion=confusion_matrix(y_test, predictions, labels=classes),
        report=classification_report(y_test, predictions, labels=classes, target_names=dataset.target_names, zero_division=0),
        truth=np.asarray(y_test), predictions=np.asarray(predictions),
        labels=[str(label) for label in dataset.target_names],
    )
