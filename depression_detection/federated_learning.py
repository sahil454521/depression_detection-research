"""Federated Learning aggregation layer with differential privacy.

Architecture
------------
Each LocalNode:
  1. Receives current global model weights
  2. Trains for local_epochs on its private dataset
  3. Computes delta_weights = local_weights - global_weights
  4. Clips delta_weights to L2 norm <= clip_norm  (bounding sensitivity)
  5. Adds Gaussian noise calibrated for (epsilon, delta)-DP guarantee
  6. Participates in Secure Aggregation (pairwise random masking)

Server (FedAVGAggregator):
  1. Receives masked, noised delta_weights from all nodes
  2. Sums masked updates (masks cancel in the sum)
  3. Applies weighted FedAVG: w += lr_global * Sigma_i (n_i/N) * delta_i
  4. Optionally adds a second layer of global DP noise (epsilon-DP on aggregate)

Privacy Guarantees
------------------
- Gaussian mechanism: sigma = clip_norm * sqrt(2 * ln(1.25 / delta)) / epsilon
  gives (epsilon, delta)-DP per FL round.
- Secure Aggregation: pairwise masks -> server sees only the SUM of deltas,
  not individual node contributions.
- Zero data leaves the local nodes -- only gradient updates are uploaded.

Pipeline position
-----------------
Data -> Model -> XAI -> Output -> [FEDERATED LEARNING] -> Validation
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from prediction import PredictionConfig, compute_prediction_losses


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FLConfig:
    """Hyper-parameters for the Federated Learning layer."""

    # Training
    num_rounds: int = 5
    """Number of global FL rounds."""

    local_epochs: int = 2
    """Epochs each node trains locally per round."""

    local_lr: float = 1e-3
    """Learning rate for local Adam optimisers."""

    global_lr: float = 1.0
    """Scale applied to the aggregated delta before updating the global model.
    Effective global learning rate = global_lr * (weighted_avg_delta)."""

    min_nodes: int = 2
    """Minimum nodes required per round (round skipped otherwise)."""

    # Differential Privacy
    epsilon: float = 1.0
    """Privacy budget (epsilon) for (epsilon, delta)-DP guarantee."""

    delta: float = 1e-5
    """Failure probability delta for the Gaussian mechanism."""

    clip_norm: float = 1.0
    """L2 norm clipping bound for individual gradient updates."""

    # Aggregation
    aggregation: str = "fedavg"
    """Aggregation rule: 'fedavg' (weighted mean) or 'fedmedian' (coordinate median)."""

    # Secure Aggregation
    use_secure_aggregation: bool = True
    """Apply pairwise random masks before upload."""

    # DP on aggregated gradient (server-side, optional second layer)
    global_dp_noise: bool = False
    """If True, add a small global DP noise to the aggregate after FedAVG."""

    global_dp_scale: float = 0.01
    """Scale of global DP noise relative to sigma."""


# ---------------------------------------------------------------------------
# Differential Privacy Mechanism
# ---------------------------------------------------------------------------

class DifferentialPrivacyMechanism:
    """Gaussian mechanism for (epsilon, delta)-differential privacy.

    sigma is computed from the Gaussian mechanism formula:
        sigma = clip_norm * sqrt(2 * ln(1.25 / delta)) / epsilon

    Usage per node:
        1. Clip gradient update to L2 norm <= clip_norm
        2. Add Gaussian noise N(0, sigma^2)
    This gives (epsilon, delta)-DP per FL round for a single query.
    """

    def __init__(self, epsilon: float, delta: float, clip_norm: float):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        self.clip_norm  = clip_norm
        self.sigma      = clip_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        self.epsilon    = epsilon
        self.delta      = delta

    def clip_update(
        self, delta_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Clip the weight-update dictionary so its global L2 norm <= clip_norm."""
        total_norm_sq = sum(v.norm().pow(2).item() for v in delta_weights.values())
        total_norm    = math.sqrt(total_norm_sq + 1e-12)
        scale = min(1.0, self.clip_norm / total_norm)
        return {k: v * scale for k, v in delta_weights.items()}

    def add_noise(
        self, delta_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise calibrated for (epsilon, delta)-DP."""
        return {
            k: v + torch.randn_like(v) * self.sigma
            for k, v in delta_weights.items()
        }

    def privatise(
        self, delta_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Clip then add noise -- complete DP privatisation pipeline."""
        return self.add_noise(self.clip_update(delta_weights))

    def __repr__(self) -> str:
        return (
            "DifferentialPrivacyMechanism("
            "epsilon={:.2f}, delta={:.0e}, clip_norm={:.2f}, sigma={:.4f})".format(
                self.epsilon, self.delta, self.clip_norm, self.sigma)
        )


# ---------------------------------------------------------------------------
# Secure Aggregation
# ---------------------------------------------------------------------------

class SecureAggregation:
    """Pairwise random mask protocol (simplified Bonawitz et al., 2017).

    Each pair of nodes (i, j) with i < j agrees on a random tensor r_ij:
      - Node i adds  r_ij to its delta
      - Node j subtracts r_ij from its delta

    When the server sums all masked deltas, the mask terms cancel:
      sum_i masked_delta_i = sum_i delta_i

    The server therefore sees only the aggregate, not individual updates.
    (In production, r_ij would be generated via cryptographic key agreement.)
    """

    @staticmethod
    def apply_masks(
        delta_list: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Apply pairwise masks to a list of delta-weight dicts.

        Parameters
        ----------
        delta_list : list of {param_name: Tensor}
            One dict per node.  All dicts must share the same keys and shapes.

        Returns
        -------
        masked_list : list of {param_name: Tensor}
            Masked deltas ready for upload.  Sum equals sum of inputs.
        """
        n = len(delta_list)
        if n < 2:
            return delta_list

        # Deep copy so we don't mutate in place
        masked = [{k: v.clone() for k, v in d.items()} for d in delta_list]
        keys   = list(delta_list[0].keys())

        for i in range(n):
            for j in range(i + 1, n):
                for k in keys:
                    # Both nodes must have matching shapes (guaranteed by shared architecture)
                    mask = torch.randn_like(delta_list[0][k])
                    masked[i][k] = masked[i][k] + mask
                    masked[j][k] = masked[j][k] - mask

        return masked

    @staticmethod
    def verify_cancellation(
        original: List[Dict[str, torch.Tensor]],
        masked: List[Dict[str, torch.Tensor]],
        tol: float = 1e-3,
    ) -> bool:
        """Debug helper: verify that sum(masked) == sum(original)."""
        keys = list(original[0].keys())
        for k in keys:
            orig_sum   = sum(d[k] for d in original)
            masked_sum = sum(d[k] for d in masked)
            if not torch.allclose(orig_sum, masked_sum, atol=tol):
                return False
        return True


# ---------------------------------------------------------------------------
# Local Node
# ---------------------------------------------------------------------------

class LocalNode:
    """Simulates one federated node (hospital / research institution).

    In a real deployment each node would run on an isolated machine.
    Here we simulate it by restricting data access to a subset of the
    full dataset.

    Parameters
    ----------
    node_id : str
        Human-readable identifier (e.g., 'wu3d_node', 'reddit_node').
    dataset : Dataset
        The node's private local dataset.
    batch_size : int
        Mini-batch size for local training.
    collate_fn : callable
        Collate function matching the dataset format.
    device : torch.device
        Device to run local training on.
    """

    def __init__(
        self,
        node_id: str,
        dataset: Dataset,
        batch_size: int = 16,
        collate_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        self.node_id   = node_id
        self.dataset   = dataset
        self.n_samples = len(dataset)
        self.device    = device or torch.device("cpu")
        self.loader    = DataLoader(
            dataset,
            batch_size=max(1, batch_size),
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def _local_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        prediction_config: PredictionConfig,
    ) -> torch.Tensor:
        """One-step forward + loss computation (no fairness during FL)."""
        from prediction import compute_prediction_losses
        out = model(
            text_input=batch["text_emb"],
            eeg=batch["eeg"],
            wearable=batch["wearable"],
            audio=batch["audio"],
            video=batch["video"],
            clinical=batch["clinical"],
            mfcc=batch.get("mfcc"),
            labels=batch["label"],
        )
        return compute_prediction_losses(
            out,
            {
                "label":          batch["label"],
                "phq9_score":     batch["phq9_score"],
                "symptom_labels": batch["symptom_labels"],
            },
            prediction_config,
        )

    def train_and_upload(
        self,
        global_model: nn.Module,
        dp_mechanism: DifferentialPrivacyMechanism,
        prediction_config: PredictionConfig,
        local_epochs: int = 2,
        local_lr: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """Train a local copy and return a DP-privatised weight-update dict.

        Steps
        -----
        1. Deep-copy global model -> local model (no data shared with server)
        2. Train local model for local_epochs on private data
        3. Compute delta = local_weights - global_weights
        4. Apply DP: clip(delta) + Gaussian noise
        5. Return privatised delta (uploaded to server; data stays local)

        Returns
        -------
        privatised_delta : Dict[str, Tensor]
            {param_name: delta_tensor} for all trainable parameters.
        """
        # --- 1. Copy global model
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=local_lr)

        # Snapshot global weights (for computing delta later)
        global_weights = {
            name: param.data.clone()
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }

        # --- 2. Train locally
        for epoch in range(local_epochs):
            for batch in self.loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = self._local_forward(local_model, batch, prediction_config)
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
                optimizer.step()

        # --- 3. Compute delta
        delta: Dict[str, torch.Tensor] = {}
        for name, param in local_model.named_parameters():
            if name in global_weights:
                delta[name] = (param.data - global_weights[name]).cpu()

        # --- 4. DP privatisation (clip + noise)
        privatised_delta = dp_mechanism.privatise(delta)

        print(
            "  [{}] Local training complete. "
            "n={:,}  epochs={}  norm(delta)={:.4f}  sigma={:.4f}".format(
                self.node_id,
                self.n_samples,
                local_epochs,
                math.sqrt(sum(v.norm().pow(2).item() for v in delta.values())),
                dp_mechanism.sigma,
            )
        )
        return privatised_delta

    def __repr__(self) -> str:
        return "LocalNode(id='{}', n_samples={:,})".format(self.node_id, self.n_samples)


# ---------------------------------------------------------------------------
# FedAVG Aggregator
# ---------------------------------------------------------------------------

class FedAVGAggregator:
    """Federated Averaging (McMahan et al., 2017).

    Computes a weighted average of node delta-weight dicts and applies it to
    the global model:
        w_global += global_lr * sum_i (n_i / N) * delta_i

    For 'fedmedian', uses coordinate-wise median instead of weighted mean.
    """

    @staticmethod
    def aggregate(
        global_model: nn.Module,
        masked_deltas: List[Dict[str, torch.Tensor]],
        node_sizes: List[int],
        global_lr: float = 1.0,
        aggregation: str = "fedavg",
        global_dp_noise: bool = False,
        global_dp_scale: float = 0.0,
    ) -> None:
        """In-place update of global_model parameters.

        Parameters
        ----------
        global_model : nn.Module
            The shared global model to update.
        masked_deltas : list of dict
            Privatised and (optionally) masked delta-weight dicts per node.
        node_sizes : list of int
            Number of training samples per node (for weighting).
        global_lr : float
            Global learning rate scaling the aggregated delta.
        aggregation : str
            'fedavg' (default) or 'fedmedian'.
        global_dp_noise : bool
            If True, add a small Gaussian noise to the final aggregate.
        global_dp_scale : float
            Noise scale relative to the aggregate's average norm.
        """
        if not masked_deltas:
            return

        total_n = sum(node_sizes)
        weights = [n / max(total_n, 1) for n in node_sizes]
        keys    = list(masked_deltas[0].keys())

        aggregated: Dict[str, torch.Tensor] = {}
        for k in keys:
            stacked = torch.stack([d[k] for d in masked_deltas], dim=0)  # [num_nodes, ...]
            if aggregation == "fedmedian":
                aggregated[k] = stacked.median(dim=0).values
            else:  # fedavg
                w_tensor = torch.tensor(weights, dtype=torch.float32)
                while w_tensor.dim() < stacked.dim():
                    w_tensor = w_tensor.unsqueeze(-1)
                aggregated[k] = (stacked * w_tensor).sum(dim=0)

        # Optional global DP noise
        if global_dp_noise and global_dp_scale > 0:
            for k in keys:
                noise_scale = global_dp_scale * aggregated[k].norm().item()
                aggregated[k] = aggregated[k] + torch.randn_like(aggregated[k]) * noise_scale

        # Update global model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in aggregated and param.requires_grad:
                    param.data += global_lr * aggregated[name].to(param.device)


# ---------------------------------------------------------------------------
# Federated Learning Layer
# ---------------------------------------------------------------------------

class FederatedLearningLayer:
    """Orchestrates multiple FL rounds across a set of LocalNodes.

    Usage
    -----
    fl = FederatedLearningLayer(nodes=[node_wu3d, node_reddit], config=cfg)
    history = fl.run(model, prediction_config)

    After each round the global model is updated in-place.
    """

    def __init__(
        self,
        nodes: List[LocalNode],
        config: Optional[FLConfig] = None,
    ):
        if not nodes:
            raise ValueError("At least one LocalNode is required.")
        self.nodes   = nodes
        self.cfg     = config or FLConfig()
        self.dp_mech = DifferentialPrivacyMechanism(
            epsilon=self.cfg.epsilon,
            delta=self.cfg.delta,
            clip_norm=self.cfg.clip_norm,
        )
        print("FederatedLearningLayer initialised:")
        print("  Nodes  : {}".format([n.node_id for n in nodes]))
        print("  DP     : {}".format(self.dp_mech))
        print("  Config : rounds={}, local_epochs={}, local_lr={}, aggregation={}".format(
            self.cfg.num_rounds, self.cfg.local_epochs,
            self.cfg.local_lr, self.cfg.aggregation,
        ))

    # ------------------------------------------------------------------

    def run_one_round(
        self,
        model: nn.Module,
        prediction_config: PredictionConfig,
        round_num: int = 0,
    ) -> Dict[str, float]:
        """Execute one global FL round.

        Returns
        -------
        round_metrics : dict
            {'round': int, 'num_nodes': int, 'avg_delta_norm': float}
        """
        active_nodes = [n for n in self.nodes if n.n_samples > 0]
        if len(active_nodes) < self.cfg.min_nodes:
            print("  Round {} skipped: fewer than {} active nodes.".format(
                round_num, self.cfg.min_nodes))
            return {"round": round_num, "num_nodes": 0, "avg_delta_norm": 0.0}

        print("  --- FL Round {} / {} ---".format(round_num + 1, self.cfg.num_rounds))

        # 1. Each node trains locally and uploads privatised delta
        raw_deltas: List[Dict[str, torch.Tensor]] = []
        node_sizes: List[int] = []
        for node in active_nodes:
            delta = node.train_and_upload(
                global_model=model,
                dp_mechanism=self.dp_mech,
                prediction_config=prediction_config,
                local_epochs=self.cfg.local_epochs,
                local_lr=self.cfg.local_lr,
            )
            raw_deltas.append(delta)
            node_sizes.append(node.n_samples)

        # 2. Secure Aggregation (pairwise masking)
        if self.cfg.use_secure_aggregation and len(raw_deltas) >= 2:
            upload_deltas = SecureAggregation.apply_masks(raw_deltas)
        else:
            upload_deltas = raw_deltas

        # 3. FedAVG aggregation
        FedAVGAggregator.aggregate(
            global_model=model,
            masked_deltas=upload_deltas,
            node_sizes=node_sizes,
            global_lr=self.cfg.global_lr,
            aggregation=self.cfg.aggregation,
            global_dp_noise=self.cfg.global_dp_noise,
            global_dp_scale=self.cfg.global_dp_scale,
        )

        avg_norm = sum(
            math.sqrt(sum(v.norm().pow(2).item() for v in d.values()))
            for d in raw_deltas
        ) / len(raw_deltas)

        metrics = {
            "round":         round_num,
            "num_nodes":     len(active_nodes),
            "avg_delta_norm": round(avg_norm, 6),
        }
        print("    aggregated update norm (avg): {:.4f}".format(avg_norm))
        return metrics

    # ------------------------------------------------------------------

    def run(
        self,
        model: nn.Module,
        prediction_config: PredictionConfig,
        num_rounds: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run all FL rounds. Returns history list (one dict per round).

        Parameters
        ----------
        model : nn.Module
            The global model to update in-place.
        prediction_config : PredictionConfig
            Task configuration for loss computation during local training.
        num_rounds : int, optional
            Override FLConfig.num_rounds for this call.
        """
        rounds = num_rounds if num_rounds is not None else self.cfg.num_rounds
        history: List[Dict[str, float]] = []
        print("\n=== Federated Learning ({} rounds) ===".format(rounds))
        for r in range(rounds):
            metrics = self.run_one_round(model, prediction_config, round_num=r)
            history.append(metrics)
        print("=== FL complete ===\n")
        return history

    # ------------------------------------------------------------------

    @staticmethod
    def build_nodes_from_dataset(
        full_dataset,             # RealDepressionDataset
        fl_config: FLConfig,
        prediction_config: PredictionConfig,
        collate_fn: Callable,
        device: torch.device,
        batch_size: int = 16,
    ) -> "FederatedLearningLayer":
        """Factory: split full_dataset by source (wu3d / reddit) into LocalNodes.

        Parameters
        ----------
        full_dataset : RealDepressionDataset
            Combined WU3D + Reddit dataset with .records attribute.
        fl_config : FLConfig
        prediction_config : PredictionConfig
        collate_fn : callable
        device : torch.device
        batch_size : int
        """
        records = full_dataset.records
        sources = sorted(set(r.source for r in records))
        nodes: List[LocalNode] = []

        for src in sources:
            indices = [i for i, r in enumerate(records) if r.source == src]
            if not indices:
                continue
            subset = Subset(full_dataset, indices)
            node = LocalNode(
                node_id="{}_node".format(src),
                dataset=subset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                device=device,
            )
            nodes.append(node)
            print("LocalNode '{}': {:,} samples".format(node.node_id, node.n_samples))

        if not nodes:
            raise RuntimeError("No LocalNodes could be created from the dataset.")

        return FederatedLearningLayer(nodes=nodes, config=fl_config)
