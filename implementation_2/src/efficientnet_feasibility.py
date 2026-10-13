#!/usr/bin/env python3
"""
implementation_2/src/efficientnet_feasibility.py
Scientific feasibility assessment and structured MBConv (EfficientNet building block) implementation.

Addresses Experiment 2.4:
- Evaluates the validity of convolutional inductive biases on 180-dim tabular behavioral signals.
- Proposes and justifies a structured (30 probes x 6 metrics) matrix representation.
- Implements a lightweight, parameter-proportional Depthwise-Separable MBConv neural network.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .mlp_classifier import set_seed


class EfficientNetFeasibilityReport:
    """Scientific feasibility evaluation of EfficientNet applied to LLM behavioral fingerprints."""

    @classmethod
    def get_assessment(cls) -> Dict[str, Any]:
        return {
            "candidate_model": "EfficientNet (Standard Vision Architecture, e.g. B0)",
            "input_expectation": "3-channel 2D spatial RGB images (224 x 224 pixels)",
            "primary_inductive_bias": (
                "Translational equivariance, local 2D spatial correlation, and hierarchical "
                "visual feature extraction (edges -> textures -> object parts)."
            ),
            "scientific_critique_of_naive_image_conversion": [
                {
                    "issue": "Absence of Natural Spatial Metric",
                    "explanation": (
                        "Tabular behavioral metrics (entropy, logit gap, probability mass) "
                        "have no physical 2D spatial distance. Reordering columns changes "
                        "convolutional activations, which is invalid for permutation-invariant features."
                    )
                },
                {
                    "issue": "False Adjacency Artifacts",
                    "explanation": (
                        "Wrapping 180 features into an arbitrary grid (e.g. 14x14) forces features "
                        "at the end of one row to be vertical neighbors with features of another row, "
                        "introducing spurious spatial relationships."
                    )
                },
                {
                    "issue": "Severe Parameter Overcapacity",
                    "explanation": (
                        "EfficientNet-B0 has ~5.3 million parameters. Applying 5.3M parameters to "
                        "small model cohorts (N=4) causes catastrophic overparameterization and "
                        "memorization rather than generalizable representation learning."
                    )
                }
            ],
            "justified_alternative_representation": {
                "structure": "Structured 2D matrix of shape (30, 6) [30 probes x 6 behavioral metrics]",
                "row_semantics": "Rows 0-29 index probes grouped contiguously into 5 domains (6 probes each)",
                "column_semantics": "Columns 0-5 index fixed statistical metrics per probe",
                "convolution_design": (
                    "Anisotropic Depthwise Separable Convolutions (MBConv blocks) with kernels "
                    "spanning local probe sequences within domains, scaled to ~8k parameters."
                ),
                "verdict": (
                    "Standard ImageNet EfficientNet-B0 is scientifically unjustified. "
                    "However, a parameter-scaled MBConv network operating on the structured "
                    "(30 x 6) probe matrix is scientifically defensible and directly testable."
                )
            }
        }


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block (core component of EfficientNet MBConv)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(2, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.SiLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block with Depthwise Separable Convolutions.
    Adapted for structured (1, 30, 6) probe feature maps.
    """

    def __init__(self, in_c: int, out_c: int, expand_ratio: int = 2, kernel_size: int = 3):
        super().__init__()
        self.use_residual = (in_c == out_c)
        mid_c = in_c * expand_ratio

        # 1. Pointwise Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU()
        ) if expand_ratio > 1 else nn.Identity()

        # 2. Depthwise Convolution (groups = mid_c)
        pad = kernel_size // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU()
        )

        # 3. Squeeze and Excitation
        self.se = SqueezeExcitation(mid_c)

        # 4. Pointwise Projection
        self.project = nn.Sequential(
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + res
        return out


class StructuredMBConvDetectorNet(nn.Module):
    """
    EfficientNet-inspired MBConv network tailored for (Batch, 1, 30, 6) structured probe matrices.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16, num_classes: int = 1):
        super().__init__()
        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU()
        )

        # MBConv Stages
        self.stage1 = MBConvBlock(base_channels, base_channels, expand_ratio=2, kernel_size=3)
        self.stage2 = MBConvBlock(base_channels, base_channels * 2, expand_ratio=2, kernel_size=3)

        # Global Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, 1, 30, 6) or (B, 180)
        if x.ndim == 2:
            x = x.view(-1, 1, 30, 6)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        logits = self.head(feat)
        return logits


class StructuredMBConvDetector:
    """
    Scikit-learn compatible detector wrapper around StructuredMBConvDetectorNet.
    """

    def __init__(
        self,
        base_channels: int = 16,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 8,
        seed: int = 42,
        device: Optional[str] = None
    ):
        self.base_channels = base_channels
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        # Default to CPU for fast deterministic execution
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        set_seed(self.seed)
        self.model = StructuredMBConvDetectorNet(
            base_channels=self.base_channels
        ).to(self.device)

        self.loss_history: List[float] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        set_seed(self.seed)
        self.model.train()

        # Reshape to (N, 1, 30, 6)
        if X.ndim == 2:
            X_tensor = torch.tensor(X.reshape(-1, 1, 30, 6), dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)

        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        dataset = TensorDataset(X_tensor, y_tensor)
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
            self.loss_history.append(epoch_loss / len(X))

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if X.ndim == 2:
                X_tensor = torch.tensor(X.reshape(-1, 1, 30, 6), dtype=torch.float32).to(self.device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
