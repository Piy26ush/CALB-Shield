#!/usr/bin/env python3
"""
src/classifier.py
Cross-architecture backdoor detection classifier and LOPO evaluation (RQ1).

Trains binary classifiers (e.g. Linear SVM, Logistic Regression, Random Forest)
on normalized behavioral fingerprints. Implements Leave-One-Pretrained-Out (LOPO)
cross-architecture evaluation to measure whether detectors trained on seen model
families generalize to unseen model architectures.
"""

import os
import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import joblib

class CrossArchClassifier:
    """Trains and evaluates cross-architecture backdoor classifiers with LOPO cross-validation."""

    def __init__(self, model_type: str = "logistic_regression", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.clf = self._init_classifier()

    def _init_classifier(self):
        if self.model_type == "logistic_regression":
            return LogisticRegression(class_weight="balanced", random_state=self.random_state, max_iter=1000)
        elif self.model_type == "linear_svc":
            return LinearSVC(class_weight="balanced", random_state=self.random_state, max_iter=2000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit classifier on training feature matrix and labels."""
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)[:, 1]
        elif hasattr(self.clf, "decision_function"):
            df = self.clf.decision_function(X)
            # Sigmoid transform for pseudo-probabilities
            return 1.0 / (1.0 + np.exp(-df))
        else:
            return self.predict(X).astype(np.float64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive detection metrics."""
        y_pred = self.predict(X)
        y_scores = self.predict_proba(X)

        # Handle edge case where y has only one class in test set
        try:
            auc = float(roc_auc_score(y, y_scores))
        except Exception:
            auc = 0.5

        return {
            "roc_auc": auc,
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0))
        }

    @staticmethod
    def run_lopo_experiment(
        dataset: Dict[str, Tuple[np.ndarray, np.ndarray]],
        model_type: str = "logistic_regression",
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Execute Leave-One-Pretrained-Out (LOPO) cross-architecture evaluation.

        Args:
            dataset: Dict mapping architecture name (e.g. 'llama3', 'mistral')
                     to a tuple (X_arch, y_arch).
            model_type: Model type to train.
            seed: Random seed.

        Returns:
            Dict containing per-held-out-architecture results and global macro averages.
        """
        archs = list(dataset.keys())
        if len(archs) < 2:
            raise ValueError(f"LOPO evaluation requires at least 2 architectures, got {len(archs)}.")

        fold_results = {}
        for held_out_arch in archs:
            # Train on all other architectures
            train_X_list = []
            train_y_list = []
            for a in archs:
                if a != held_out_arch:
                    X_a, y_a = dataset[a]
                    train_X_list.append(X_a)
                    train_y_list.append(y_a)

            X_train = np.vstack(train_X_list)
            y_train = np.concatenate(train_y_list)

            X_test, y_test = dataset[held_out_arch]

            clf = CrossArchClassifier(model_type=model_type, random_state=seed)
            clf.fit(X_train, y_train)
            metrics = clf.evaluate(X_test, y_test)
            metrics["n_train"] = int(len(y_train))
            metrics["n_test"] = int(len(y_test))
            metrics["train_architectures"] = [a for a in archs if a != held_out_arch]
            metrics["held_out_architecture"] = held_out_arch

            fold_results[held_out_arch] = metrics

        # Macro average across folds
        metric_keys = ["roc_auc", "balanced_accuracy", "f1_score", "precision", "recall"]
        macro_summary = {
            f"macro_{k}": float(np.mean([fold_results[a][k] for a in archs]))
            for k in metric_keys
        }

        return {
            "lopo_folds": fold_results,
            "macro_summary": macro_summary
        }

    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.clf, filepath)

    def load(self, filepath: str):
        """Load model from disk."""
        self.clf = joblib.load(filepath)
