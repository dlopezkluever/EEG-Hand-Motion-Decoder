"""EEGNet — compact CNN for EEG-based BCI (Lawhern et al., 2018).

Architecture: temporal Conv2D -> BatchNorm + ELU -> depthwise Conv2D ->
AvgPool + Dropout -> separable Conv2D -> AvgPool + Dropout ->
Flatten + Dense (2 classes).
"""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CV_N_FOLDS,
    EEGNET_BATCH_SIZE,
    EEGNET_D,
    EEGNET_DROPOUT,
    EEGNET_F1,
    EEGNET_F2,
    EEGNET_LR,
    EEGNET_MAX_EPOCHS,
    EEGNET_PATIENCE,
    MODELS_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class EEGNet(nn.Module):
    """EEGNet architecture for 2-class EEG classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_timepoints : int
        Number of time samples per epoch.
    F1 : int
        Number of temporal filters.
    F2 : int
        Number of pointwise filters (separable conv).
    D : int
        Depth multiplier for depthwise conv.
    dropout : float
        Dropout rate.
    n_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_timepoints: int = 721,
        F1: int = EEGNET_F1,
        F2: int = EEGNET_F2,
        D: int = EEGNET_D,
        dropout: float = EEGNET_DROPOUT,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Block 1: Temporal convolution
        # Conv2D input: (batch, 1, n_channels, n_timepoints)
        self.conv1 = nn.Conv2d(
            1, F1,
            kernel_size=(1, 64),
            padding=(0, 32),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise convolution (spatial filter per temporal feature)
        self.depthwise = nn.Conv2d(
            F1, F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.separable_depth = nn.Conv2d(
            F1 * D, F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.separable_point = nn.Conv2d(
            F1 * D, F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size()
        self.fc = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self) -> int:
        """Compute the flattened feature size after conv blocks."""
        x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
        x = self._forward_features(x)
        return x.shape[1]

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction blocks."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 1, n_channels, n_timepoints)

        Returns
        -------
        Tensor of shape (batch, n_classes) — logits.
        """
        x = self._forward_features(x)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_eegnet(
    X: np.ndarray,
    y: np.ndarray,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    lr: float = EEGNET_LR,
    batch_size: int = EEGNET_BATCH_SIZE,
    max_epochs: int = EEGNET_MAX_EPOCHS,
    patience: int = EEGNET_PATIENCE,
) -> dict:
    """Train EEGNet on raw epoch data with early stopping.

    Parameters
    ----------
    X : ndarray of shape (n_epochs, n_channels, n_timepoints)
        Normalized raw epoch data.
    y : ndarray of shape (n_epochs,)
        Integer labels (0 or 1).
    val_X, val_y : optional validation set. If None, uses 80/20 split.

    Returns
    -------
    dict with trained model, training history, and best validation accuracy.
    """
    device = _get_device()
    logger.info("Training EEGNet on device: %s", device)

    # Split into train/val if no validation set provided
    if val_X is None or val_y is None:
        np.random.seed(RANDOM_SEED)
        n = len(X)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]
        train_X, train_y = X[train_idx], y[train_idx]
        val_X, val_y = X[val_idx], y[val_idx]
    else:
        train_X, train_y = X, y

    n_channels = train_X.shape[1]
    n_timepoints = train_X.shape[2]

    # Create model
    model = EEGNet(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
    ).to(device)

    n_params = count_parameters(model)
    logger.info("EEGNet parameters: %d", n_params)

    # Create data loaders — add channel dim: (N, C, T) -> (N, 1, C, T)
    train_tensor_X = torch.FloatTensor(train_X[:, np.newaxis, :, :])
    train_tensor_y = torch.LongTensor(train_y)
    val_tensor_X = torch.FloatTensor(val_X[:, np.newaxis, :, :])
    val_tensor_y = torch.LongTensor(val_y)

    train_dataset = TensorDataset(train_tensor_X, train_tensor_y)
    val_dataset = TensorDataset(val_tensor_X, val_tensor_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss_sum += loss.item()
                val_batches += 1
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss = val_loss_sum / val_batches
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(
                "  Epoch %3d/%d — train_loss: %.4f  val_loss: %.4f  val_acc: %.4f",
                epoch + 1, max_epochs, train_loss, val_loss, val_acc,
            )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d). Best val_loss: %.4f",
                epoch + 1, patience, best_val_loss,
            )
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    best_val_acc = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0

    return {
        "model": model,
        "history": history,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "n_params": n_params,
        "epochs_trained": len(history["train_loss"]),
    }


def train_eegnet_cv(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train EEGNet with stratified k-fold cross-validation.

    Uses the same fold splits as logistic regression for fair comparison.

    Parameters
    ----------
    X : ndarray of shape (n_epochs, n_channels, n_timepoints)
    y : ndarray of shape (n_epochs,)

    Returns
    -------
    dict with per-fold accuracies, mean/std, F1, AUC-ROC, and predictions.
    """
    device = _get_device()

    skf = StratifiedKFold(
        n_splits=CV_N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_accuracies = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    last_history = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info("EEGNet Fold %d/%d — train=%d, test=%d",
                     fold_idx + 1, CV_N_FOLDS, len(train_idx), len(test_idx))

        # Train with internal train/val split
        result = train_eegnet(X_train, y_train)
        last_history = result["history"]
        model = result["model"]

        # Evaluate on test fold
        model.eval()
        test_tensor = torch.FloatTensor(X_test[:, np.newaxis, :, :]).to(device)

        with torch.no_grad():
            outputs = model(test_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        y_pred = predicted.cpu().numpy()
        y_prob = probs[:, 1].cpu().numpy()

        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        logger.info("  Fold %2d/%d — accuracy: %.4f", fold_idx + 1, CV_N_FOLDS, fold_acc)

        # Save best model weights for this fold
        model_path = MODELS_DIR / f"eegnet_fold{fold_idx + 1:02d}.pt"
        torch.save(model.state_dict(), model_path)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro")
    auc_roc = roc_auc_score(all_y_true, all_y_prob)

    logger.info(
        "EEGNet — Accuracy: %.4f +/- %.4f  F1: %.4f  AUC: %.4f",
        mean_acc, std_acc, macro_f1, auc_roc,
    )

    # Print summary
    print(f"\n{'='*50}")
    print("EEGNet — Cross-Validation Results")
    print(f"{'='*50}")
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1:2d}: {acc:.4f}")
    print(f"{'-'*50}")
    print(f"  Mean accuracy:  {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Macro F1-score: {macro_f1:.4f}")
    print(f"  AUC-ROC:        {auc_roc:.4f}")
    print(f"{'='*50}\n")

    return {
        "model_name": "EEGNet",
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "macro_f1": float(macro_f1),
        "auc_roc": float(auc_roc),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
        "history": last_history,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    logger.info("=== EEGNet — Smoke Test ===")

    from src.data_loader import download_data, load_raw
    from src.features import extract_raw_features
    from src.preprocessing import apply_filters, extract_epochs

    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])
    filtered = apply_filters(raw)
    epochs = extract_epochs(filtered)
    X, y = extract_raw_features(epochs)

    model = EEGNet(n_channels=X.shape[1], n_timepoints=X.shape[2])
    n_params = count_parameters(model)
    print(f"EEGNet parameters: {n_params}")
    print(f"Input shape: {X.shape}")

    # Quick forward pass test
    sample = torch.FloatTensor(X[:2, np.newaxis, :, :])
    out = model(sample)
    print(f"Output shape: {out.shape}")

    logger.info("=== Smoke test complete ===")
