#!/usr/bin/env python3
"""
implementation_2/src/baseline_classifiers.py
Standardized baseline classifiers and evaluation utilities.

Provides unified interfaces for:
- Logistic Regression
- Random Forest
- Gradient Boosted Trees (HistGradientBoosting / GradientBoosting)
- Linear SVM
- Multi-Layer Perceptron (PyTorch & Scikit-learn)
"""

import os
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from .mlp_classifier import PyTorchMLPDetector


class BaselineSuite:
    """Factory and management suite for comparative detection algorithms."""

    @staticmethod
    def create_classifier(name: str, seed: int = 42, **kwargs) -> Any:
        """Instantiates a standardized detector."""
        name_lower = name.lower()
        if name_lower in ("logistic_regression", "logreg", "lr"):
            return LogisticRegression(
                C=kwargs.get("C", 1.0),
                class_weight="balanced",
                random_state=seed,
                max_iter=1000
            )
        elif name_lower in ("random_forest", "rf"):
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                class_weight="balanced",
                random_state=seed
            )
        elif name_lower in ("gradient_boosting", "gbdt", "hist_gbdt"):
            return HistGradientBoostingClassifier(
                max_iter=kwargs.get("max_iter", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=seed
            )
        elif name_lower in ("linear_svc", "linear_svm", "svm"):
            return LinearSVC(
                penalty="l2",
                C=kwargs.get("C", 1.0),
                class_weight="balanced",
                random_state=seed,
                max_iter=2000
            )
        elif name_lower in ("pytorch_mlp", "mlp"):
            return PyTorchMLPDetector(
                input_dim=kwargs.get("input_dim", 180),
                hidden_dims=kwargs.get("hidden_dims", [64, 32]),
                dropout=kwargs.get("dropout", 0.15),
                lr=kwargs.get("lr", 1e-3),
                epochs=kwargs.get("epochs", 100),
                seed=seed
            )
        elif name_lower in ("sklearn_mlp",):
            return MLPClassifier(
                hidden_layer_sizes=kwargs.get("hidden_dims", (64, 32)),
                max_iter=kwargs.get("epochs", 500),
                random_state=seed,
                early_stopping=True
            )
        elif name_lower in ("mbconv", "structured_mbconv"):
            from .efficientnet_feasibility import StructuredMBConvDetector
            return StructuredMBConvDetector(
                base_channels=kwargs.get("base_channels", 16),
                lr=kwargs.get("lr", 1e-3),
                epochs=kwargs.get("epochs", 100),
                seed=seed
            )
        else:
            raise ValueError(f"Unknown classifier name: {name}")

    @staticmethod
    def get_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
        """Extracts class 1 (POISONED) probability in a unified manner."""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.ndim == 2:
                return probs[:, 1]
            return probs.flatten()
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X)
            # Apply Sigmoid to decision margin
            return 1.0 / (1.0 + np.exp(-df))
        else:
            preds = model.predict(X)
            return preds.astype(np.float64)

    @classmethod
    def evaluate_detector(
        cls,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Computes comprehensive academic evaluation metrics:
        TP, FP, TN, FN, Accuracy, Precision, Recall (TPR), FPR, F1, and ROC-AUC.
        """
        probs = cls.get_probabilities(model, X_test)
        preds = (probs >= threshold).astype(int)

        n_samples = len(y_test)
        n_pos = int(np.sum(y_test == 1))
        n_neg = int(np.sum(y_test == 0))

        # Confusion Matrix components
        tp = int(np.sum((preds == 1) & (y_test == 1)))
        fp = int(np.sum((preds == 1) & (y_test == 0)))
        tn = int(np.sum((preds == 0) & (y_test == 0)))
        fn = int(np.sum((preds == 0) & (y_test == 1)))

        tpr = (tp / n_pos) if n_pos > 0 else 0.0
        fpr = (fp / n_neg) if n_neg > 0 else 0.0
        acc = (tp + tn) / n_samples if n_samples > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = tpr
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # ROC-AUC requires at least one positive and one negative sample
        if n_pos > 0 and n_neg > 0 and len(np.unique(probs)) > 1:
            try:
                auc = float(roc_auc_score(y_test, probs))
            except Exception:
                auc = None
        else:
            auc = None

        return {
            "n_samples": n_samples,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall_tpr": float(rec),
            "false_positive_rate_fpr": float(fpr),
            "f1_score": float(f1),
            "roc_auc": auc,
            "threshold": threshold,
            "predicted_labels": preds.tolist(),
            "predicted_scores": [float(p) for p in probs]
        }
