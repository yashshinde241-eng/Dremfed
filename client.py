"""
DermFed – client.py
Flower federated learning client representing one hospital.

Usage (Windows):
    python client.py --hospital_id 0
    python client.py --hospital_id 1
    python client.py --hospital_id 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common import NDArrays, Scalar

from utils import (
    DEVICE,
    build_model,
    evaluate,
    get_loaders,
    train_one_epoch,
)

# ── Config ───────────────────────────────────────────────────────────────────
PARTITIONS_DIR = Path("data/partitions")
LOCAL_EPOCHS   = 2          # Epochs of local training per FL round
LEARNING_RATE  = 1e-3
SERVER_ADDRESS = "127.0.0.1:8080"


# ── Flower Client ─────────────────────────────────────────────────────────────
class DermFedClient(fl.client.NumPyClient):
    """
    Each instance represents one hospital.
    It loads local data, trains locally, and returns weight deltas to the server.
    """

    def __init__(self, hospital_id: int) -> None:
        self.hospital_id = hospital_id
        silo_dir         = PARTITIONS_DIR / f"hospital_{hospital_id}"

        if not silo_dir.exists():
            raise RuntimeError(
                f"Silo directory '{silo_dir}' not found.\n"
                "Run `python partition_data.py` first."
            )

        print(f"[Hospital {hospital_id}] Loading data from {silo_dir} …")
        self.train_loader, self.val_loader = get_loaders(silo_dir)
        self.model     = build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(),
                                          lr=LEARNING_RATE)
        print(f"[Hospital {hospital_id}] Ready. Device: {DEVICE}")

    # ── Required Flower methods ───────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return current model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load server-aggregated weights into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        1. Receive global weights.
        2. Train locally for LOCAL_EPOCHS.
        3. Return updated weights + metrics.
        """
        self.set_parameters(parameters)

        current_round = int(config.get("server_round", 0))
        print(f"\n[Hospital {self.hospital_id}] ── Round {current_round} ──")

        all_losses, all_accs = [], []
        for epoch in range(LOCAL_EPOCHS):
            loss, acc = train_one_epoch(
                self.model, self.train_loader, self.optimiser, self.criterion
            )
            all_losses.append(loss)
            all_accs.append(acc)
            print(
                f"  Epoch {epoch + 1}/{LOCAL_EPOCHS}  "
                f"loss={loss:.4f}  acc={acc:.4f}"
            )

        num_samples = len(self.train_loader.dataset)
        return (
            self.get_parameters(config={}),
            num_samples,
            {
                "train_loss": float(all_losses[-1]),
                "train_acc":  float(all_accs[-1]),
            },
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate global weights on the local validation set."""
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.val_loader, self.criterion)
        num_samples = len(self.val_loader.dataset)

        print(
            f"[Hospital {self.hospital_id}] Eval → "
            f"loss={loss:.4f}  acc={acc:.4f}"
        )
        return float(loss), num_samples, {"accuracy": float(acc)}


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="DermFed FL Client")
    parser.add_argument(
        "--hospital_id", type=int, required=True,
        help="Index of the hospital silo (0-based)."
    )
    parser.add_argument(
        "--server", type=str, default=SERVER_ADDRESS,
        help=f"FL server address (default: {SERVER_ADDRESS})"
    )
    args = parser.parse_args()

    client = DermFedClient(hospital_id=args.hospital_id)
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
