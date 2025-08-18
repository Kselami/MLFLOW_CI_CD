import argparse
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def train_and_log(experiment_name: str, registered_model_name: str,
                  C: float, max_iter: int, seed: int, test_size: float,
                  min_accuracy: float) -> float:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    with mlflow.start_run(run_name=f"lr_C{C}_iter{max_iter}_seed{seed}") as run:
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": C,
            "max_iter": max_iter,
            "seed": seed,
            "test_size": test_size
        })

        clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=None, random_state=seed)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
        mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])

        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        fig_path = os.path.join("outputs", "confusion_matrix.png")
        plt.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="plots")

        signature = infer_signature(X_train, clf.predict(X_train))
        input_example = X_train[:2]

        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        if acc < min_accuracy:
            raise SystemExit(f"Accuracy {acc:.3f} < seuil {min_accuracy:.3f} (échec CI)")

        return acc

def parse_args():
    p = argparse.ArgumentParser(description="Train and log a LogisticRegression with MLflow.")
    p.add_argument("--experiment-name", type=str, required=True, help="Nom de l'expérience MLflow.")
    p.add_argument("--registered-model-name", type=str, required=True, help="Nom du modèle dans la Model Registry.")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--min-accuracy", type=float, default=0.8, help="Seuil qualité minimal (échoue si en dessous).")
    return p.parse_args()

def main():
    args = parse_args()
    acc = train_and_log(
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        C=args.C,
        max_iter=args.max_iter,
        seed=args.seed,
        test_size=args.test_size,
        min_accuracy=args.min_accuracy
    )
    print(f"Final accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
