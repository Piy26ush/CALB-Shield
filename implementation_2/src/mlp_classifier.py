#!/usr/bin/env python3
"""
implementation_2/src/mlp_classifier.py
PyTorch-based Multi-Layer Perceptron (MLP) for 180-dim behavioral fingerprint classification.

Provides a clean, modular neural architecture with Dropout and LayerNorm,
wrapped in a scikit-learn compatible interface (fit, predict, predict_proba).
"""

import os
import random
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed: int = 42):
    """Ensure deterministic PyTorch and NumPy execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class BehavioralMLPNet(nn.Module):
    """
    Lightweight feedforward neural network for 180-dimensional tabular behavioral vectors.
    
    Architecture:
      Input (180) -> Linear(64) -> LayerNorm -> GELU -> Dropout(0.15)
                  -> Linear(32) -> LayerNorm -> GELU -> Dropout(0.10)
                  -> Linear(1)
    """

    def __init__(
        self,
        input_dim: int = 180,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.15,
        use_layernorm: bool = True
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            drop = dropout if i == 0 else max(0.05, dropout - 0.05)
            layers.append(nn.Dropout(drop))
            prev_dim = h_dim

        # Final projection to binary classification logit
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs raw logits (shape: [batch_size, 1])."""
        return self.network(x)


class PyTorchMLPDetector:
    """
    Scikit-learn compatible detector wrapper around BehavioralMLPNet.
    """

    def __init__(
        self,
        input_dim: int = 180,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 8,
        seed: int = 42,
        device: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        # Use CPU by default for ultra-fast, deterministic tabular execution
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        set_seed(self.seed)
        self.model = BehavioralMLPNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        self.loss_history: List[float] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        Train the MLP on feature matrix X and binary labels y.
        Handles class imbalance with weighted BCEWithLogitsLoss.
        """
        set_seed(self.seed)
        self.model.train()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Compute pos_weight to handle class imbalance
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = TensorDataset(X_t, y_t)
        batch_size = min(self.batch_size, len(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.loss_history = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            avg_loss = epoch_loss / len(X)
            self.loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"[MLP Epoch {epoch+1:03d}/{self.epochs}] Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of predicted probabilities of class 1 (POISONED)."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions (0 = CLEAN, 1 = POISONED)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
