from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import Config, HORIZONS
from .splits import Split
from .windows import TimeSeriesWindowDataset, WindowedData


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        checkpoint_path: Path,
    ) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.mixed_precision and device.type == "cuda"))

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        best_val = float("inf")
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", best_val)

        for epoch in range(start_epoch, self.cfg.max_epochs):
            model.train()
            train_loss = 0.0
            opt.zero_grad(set_to_none=True)

            step_count = 0
            for step, (xb, yb, sb) in enumerate(train_loader):
                step_count += 1
                xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                with torch.cuda.amp.autocast(enabled=(self.cfg.mixed_precision and device.type == "cuda")):
                    logits = model(xb, sb)
                    loss = self._bce_loss(logits, yb) / self.cfg.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                train_loss += loss.item() * self.cfg.gradient_accumulation_steps

            # Flush leftover accumulated gradients for the last partial micro-batch group.
            if step_count > 0 and (step_count % self.cfg.gradient_accumulation_steps) != 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            val_loss = float("nan")
            if val_loader is not None and len(val_loader) > 0:
                model.eval()
                vtotal = 0.0
                with torch.no_grad():
                    for xb, yb, sb in val_loader:
                        xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                        logits = model(xb, sb)
                        vtotal += self._bce_loss(logits, yb).item()
                val_loss = vtotal / max(len(val_loader), 1)

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val,
                },
                checkpoint_path,
            )

            if not math.isnan(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), checkpoint_path.with_suffix(".best_model.pt"))

            print(f"epoch={epoch} train_loss={train_loss / max(len(train_loader), 1):.6f} val_loss={val_loss:.6f}")

        best_path = checkpoint_path.with_suffix(".best_model.pt")
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        return model


def split_and_scale(
    window_data: WindowedData,
    split: Split,
    symbol_to_idx: Dict[str, int],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    X_train, y_train, s_train = (
        window_data.X[split.train_idx],
        window_data.y[split.train_idx],
        window_data.symbols[split.train_idx],
    )
    X_test, y_test, s_test = (
        window_data.X[split.test_idx],
        window_data.y[split.test_idx],
        window_data.symbols[split.test_idx],
    )

    has_val = len(split.val_idx) > 0
    if has_val:
        X_val = window_data.X[split.val_idx]
        y_val = window_data.y[split.val_idx]
        s_val = window_data.symbols[split.val_idx]
    else:
        X_val = y_val = s_val = None

    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    scaler.fit(X_train.reshape(-1, n_features))

    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    if has_val and X_val is not None:
        X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape).astype(np.float32)

    train_ds = TimeSeriesWindowDataset(X_train, y_train, s_train, symbol_to_idx)
    test_ds = TimeSeriesWindowDataset(X_test, y_test, s_test, symbol_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_loader: Optional[DataLoader] = None
    if has_val and X_val is not None and y_val is not None and s_val is not None:
        val_ds = TimeSeriesWindowDataset(X_val, y_val, s_val, symbol_to_idx)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def infer_probabilities(model: nn.Module, loader: DataLoader) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for xb, _, sb in loader:
            xb, sb = xb.to(device), sb.to(device)
            outputs.append(torch.sigmoid(model(xb, sb)).cpu().numpy())

    if not outputs:
        return np.empty((0, len(HORIZONS)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)
