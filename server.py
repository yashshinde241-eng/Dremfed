"""
DermFed – server.py
Flower federated learning server using FedAvg aggregation.

Usage (Windows):
    python server.py --rounds 10 --n_clients 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAvg

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_ADDRESS  = "0.0.0.0:8080"
RESULTS_DIR     = Path("results")
GLOBAL_MODEL_PATH = Path("models/global_model.pt")


# ── Metric aggregation helpers ────────────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics using sample-count weighting."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for metric_key in metrics[0][1].keys():
        aggregated[metric_key] = (
            sum(n * m[metric_key] for n, m in metrics) / total
        )
    return aggregated


# ── Custom FedAvg with logging ────────────────────────────────────────────────
class DermFedStrategy(FedAvg):
    """
    Extends FedAvg to:
      1. Log per-round global metrics to CSV (for the dashboard).
      2. Save the global model weights after each round.
    """

    def __init__(self, results_csv: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.results_csv = results_csv
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
        GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV header
        with open(self.results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "timestamp",
                             "train_loss", "train_acc",
                             "val_loss",   "val_acc"])

    # ── Per-round config: inject server_round into fit config ────────────────
    def configure_fit(self, server_round, parameters, client_manager):
        """Pass server_round number down to each client's fit() config dict."""
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        # client_instructions is List[Tuple[ClientProxy, FitIns]]
        # Re-wrap each FitIns with server_round added to its config
        updated = []
        for client_proxy, fit_ins in client_instructions:
            new_config = dict(fit_ins.config)
            new_config["server_round"] = server_round
            updated.append(
                (client_proxy, fl.common.FitIns(fit_ins.parameters, new_config))
            )
        return updated

    # ── Fit aggregation: save weights + log train metrics ─────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        agg_params, fit_metrics = super().aggregate_fit(server_round, results, failures)

        # Extract weighted-average train loss/acc from client results
        train_loss = fit_metrics.get("train_loss", "") if fit_metrics else ""
        train_acc  = fit_metrics.get("train_acc",  "") if fit_metrics else ""

        # Store for use in aggregate_evaluate (same round)
        self._last_train_loss = train_loss
        self._last_train_acc  = train_acc

        # Persist global weights to disk
        if agg_params is not None:
            self._save_weights(agg_params, server_round)

        return agg_params, fit_metrics

    # ── Evaluate aggregation: log full row to CSV ─────────────────────────────
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        loss_agg, metrics = super().aggregate_evaluate(server_round, results, failures)

        val_acc  = float(metrics.get("accuracy", 0.0)) if metrics else 0.0
        val_loss = float(loss_agg) if loss_agg is not None else 0.0

        train_loss = getattr(self, "_last_train_loss", "")
        train_acc  = getattr(self, "_last_train_acc",  "")

        print(
            f"\n{'─'*55}\n"
            f"  [Server] Round {server_round:>3d} complete\n"
            f"  Train loss={train_loss}  acc={train_acc}\n"
            f"  Val   loss={val_loss:.4f}  acc={val_acc:.4f}\n"
            f"{'─'*55}\n"
        )

        # Append row to CSV (dashboard reads this)
        with open(self.results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                server_round,
                datetime.now().isoformat(timespec="seconds"),
                round(float(train_loss), 4) if train_loss != "" else "",
                round(float(train_acc),  4) if train_acc  != "" else "",
                round(val_loss, 4),
                round(val_acc,  4),
            ])

        return loss_agg, metrics

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _save_weights(self, parameters: Parameters, server_round: int) -> None:
        """Reconstruct the MobileNetV2 from aggregated ndarrays and save."""
        import torch
        from collections import OrderedDict
        from utils import build_model

        ndarrays    = fl.common.parameters_to_ndarrays(parameters)
        model       = build_model()
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict  = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)
        torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
        print(f"  [Server] ✓ Global model saved → {GLOBAL_MODEL_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="DermFed FL Server")
    parser.add_argument("--rounds",    type=int, default=10,
                        help="Number of federated learning rounds (default: 10)")
    parser.add_argument("--n_clients", type=int, default=3,
                        help="Minimum clients per round (default: 3)")
    parser.add_argument("--address",   type=str, default=SERVER_ADDRESS,
                        help=f"Bind address (default: {SERVER_ADDRESS})")
    args = parser.parse_args()

    results_csv = RESULTS_DIR / "fl_metrics.csv"

    strategy = DermFedStrategy(
        results_csv                  = results_csv,
        fraction_fit                 = 1.0,
        fraction_evaluate            = 1.0,
        min_fit_clients              = args.n_clients,
        min_evaluate_clients         = args.n_clients,
        min_available_clients        = args.n_clients,
        evaluate_metrics_aggregation_fn = weighted_average,
        fit_metrics_aggregation_fn   = weighted_average,
    )

    print(f"\n{'='*55}")
    print(f"  DermFed FL Server")
    print(f"  Address  : {args.address}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Clients  : {args.n_clients} min")
    print(f"  Metrics  → {results_csv}")
    print(f"{'='*55}\n")

    fl.server.start_server(
        server_address = args.address,
        config         = fl.server.ServerConfig(num_rounds=args.rounds),
        strategy       = strategy,
    )


if __name__ == "__main__":
    main()
