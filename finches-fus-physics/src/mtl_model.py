"""
mtl_model.py - Multi-task neural network for H_eff prediction.

Addresses the overfitting problem in coupling constant calibration:
instead of fitting 5 free parameters to 3 data points (memorization),
we train a shared-representation network on multiple tasks simultaneously.

Architecture:
    Input features (per variant):
        - sticker_fraction, n_stickers, omega, mean_linker_length
        - clustering_coefficient, graph_density, mean_degree, n_components
        - percolation_threshold
        - h0_total_persistence, betti_0_auc
        - spacing_entropy, composition_entropy
        → 13 features total

    Shared hidden layers:
        Linear(13, 16) → ReLU → Linear(16, 8) → ReLU

    Task-specific output heads:
        Head 1: H_eff prediction    (regression, 1 output)
        Head 2: c_sat prediction    (regression, log-scale, 1 output)
        Head 3: phase_separates     (binary classification, 1 output)
        Head 4: H_chemistry reconst.(regression, 1 output — physics anchor)

Why MTL helps:
    - Head 3 provides labels for ALL variants (we know which phase separate)
    - Head 4 provides exact supervision on ALL variants (H_chem is computable)
    - The shared layers learn a general feature→energy mapping constrained
      by physics, not just by 3 c_sat measurements
    - Implicit regularization: network can't memorize c_sat without also
      getting phase classification and H_chem reconstruction right

Implemented in pure numpy (no torch dependency). The network is tiny
(~300 parameters) so autograd is unnecessary — we use explicit gradients.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

FEATURE_NAMES = [
    "sticker_fraction", "n_stickers", "omega", "mean_linker_length",
    "clustering_coefficient", "graph_density", "mean_degree", "n_components",
    "percolation_threshold",
    "h0_total_persistence", "betti_0_auc",
    "spacing_entropy", "composition_entropy",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(
    metrics,
    topology,
    homology,
    entropy_metrics,
) -> np.ndarray:
    """
    Extract feature vector from computed metrics.

    Parameters
    ----------
    metrics : VariantMetrics or dict-like with sticker/linker properties
    topology : TopologyMetrics
    homology : HomologyMetrics
    entropy_metrics : EntropyMetrics

    Returns
    -------
    np.ndarray
        Feature vector of length N_FEATURES
    """
    omega_val = metrics.omega if hasattr(metrics, 'omega') else metrics.get('omega', 0.0)
    if not np.isfinite(omega_val):
        omega_val = 1.0

    return np.array([
        metrics.sticker_fraction if hasattr(metrics, 'sticker_fraction') else 0.0,
        metrics.n_stickers if hasattr(metrics, 'n_stickers') else 0,
        omega_val,
        metrics.mean_linker_length if hasattr(metrics, 'mean_linker_length') else 0.0,
        topology.sticker_clustering_coefficient,
        topology.sticker_graph_density,
        topology.sticker_mean_degree,
        topology.sticker_n_components,
        topology.percolation_threshold,
        homology.h0_total_persistence,
        homology.betti_0_auc,
        entropy_metrics.spacing_entropy,
        entropy_metrics.composition_entropy,
    ], dtype=np.float64)


# =============================================================================
# NETWORK LAYERS (numpy implementation)
# =============================================================================

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_grad(s: np.ndarray) -> np.ndarray:
    """Gradient of sigmoid given sigmoid output s."""
    return s * (1.0 - s)


class Linear:
    """Simple linear layer: y = Wx + b, with Adam optimizer state."""

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        # He initialization
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Adam state: first moment (m) and second moment (v) for W and b
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        # Cache for backward
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x.copy()
        return self.W @ x + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients and return grad w.r.t. input."""
        self.dW += np.outer(grad_output, self._input)
        self.db += grad_output
        return self.W.T @ grad_output

    def zero_grad(self):
        self.dW[:] = 0.0
        self.db[:] = 0.0

    def step_adam(self, lr: float, t: int, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8):
        """
        Adam optimizer update (Kingma & Ba, 2015).

        Adaptive learning rates per-parameter using first and second
        moment estimates of the gradient. Key advantages over SGD:
        - Handles sparse gradients (important for MTL with missing labels)
        - Automatic per-parameter learning rate scaling
        - Bias correction for early training steps

        Parameters
        ----------
        lr : float
            Base learning rate (alpha in the paper)
        t : int
            Current timestep (1-indexed, for bias correction)
        beta1 : float
            Exponential decay rate for first moment (default: 0.9)
        beta2 : float
            Exponential decay rate for second moment (default: 0.999)
        eps : float
            Numerical stability constant (default: 1e-8)
        """
        # Update biased first moment estimate
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        # Update biased second raw moment estimate
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        # Bias-corrected estimates
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        mW_hat = self.mW / bc1
        mb_hat = self.mb / bc1
        vW_hat = self.vW / bc2
        vb_hat = self.vb / bc2

        # Parameter update
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# =============================================================================
# MULTI-TASK NETWORK
# =============================================================================

@dataclass
class MTLConfig:
    """Configuration for the multi-task network."""
    hidden_1: int = 16
    hidden_2: int = 8
    lr: float = 0.001
    n_epochs: int = 2000
    # Adam hyperparameters
    beta1: float = 0.9       # first moment decay
    beta2: float = 0.999     # second moment decay
    eps: float = 1e-8        # numerical stability
    # Task loss weights
    lambda_h_eff: float = 1.0
    lambda_csat: float = 1.0
    lambda_phase: float = 0.5
    lambda_h_chem: float = 0.5
    seed: int = 42


class MultiTaskNetwork:
    """
    Multi-task neural network for H_eff prediction.

    Shared backbone:
        Input(13) → Linear(16) → ReLU → Linear(8) → ReLU

    Task heads:
        Head 1: Linear(8, 1) → H_eff
        Head 2: Linear(8, 1) → log(c_sat)
        Head 3: Linear(8, 1) → sigmoid → P(phase_separates)
        Head 4: Linear(8, 1) → H_chemistry
    """

    def __init__(self, config: Optional[MTLConfig] = None):
        if config is None:
            config = MTLConfig()
        self.config = config

        s = config.seed
        # Shared backbone
        self.layer1 = Linear(N_FEATURES, config.hidden_1, seed=s)
        self.layer2 = Linear(config.hidden_1, config.hidden_2, seed=s + 1)

        # Task heads
        self.head_h_eff = Linear(config.hidden_2, 1, seed=s + 2)
        self.head_csat = Linear(config.hidden_2, 1, seed=s + 3)
        self.head_phase = Linear(config.hidden_2, 1, seed=s + 4)
        self.head_h_chem = Linear(config.hidden_2, 1, seed=s + 5)

        # Feature normalization params (set during training)
        self._feat_mean = np.zeros(N_FEATURES)
        self._feat_std = np.ones(N_FEATURES)

        self.training_history: List[Dict[str, float]] = []

    def _all_layers(self):
        return [self.layer1, self.layer2,
                self.head_h_eff, self.head_csat,
                self.head_phase, self.head_h_chem]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._feat_mean) / (self._feat_std + 1e-8)

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass. Returns all task outputs.

        Parameters
        ----------
        x : np.ndarray
            Raw feature vector (N_FEATURES,)

        Returns
        -------
        Dict with keys: "h_eff", "log_csat", "phase_prob", "h_chem",
                        "z1", "a1", "z2", "a2" (intermediate activations)
        """
        x_norm = self._normalize(x)

        z1 = self.layer1.forward(x_norm)
        a1 = _relu(z1)
        z2 = self.layer2.forward(a1)
        a2 = _relu(z2)

        h_eff = self.head_h_eff.forward(a2)[0]
        log_csat = self.head_csat.forward(a2)[0]
        phase_logit = self.head_phase.forward(a2)[0]
        phase_prob = _sigmoid(phase_logit)
        h_chem = self.head_h_chem.forward(a2)[0]

        return {
            "h_eff": h_eff,
            "log_csat": log_csat,
            "phase_prob": phase_prob,
            "phase_logit": phase_logit,
            "h_chem": h_chem,
            "z1": z1, "a1": a1,
            "z2": z2, "a2": a2,
        }

    def predict(self, x: np.ndarray) -> Dict[str, float]:
        """Predict all outputs for a single feature vector."""
        out = self.forward(x)
        return {
            "H_eff": float(out["h_eff"]),
            "csat_relative": float(np.exp(out["log_csat"])),
            "phase_prob": float(out["phase_prob"]),
            "H_chemistry": float(out["h_chem"]),
        }

    def _backward_and_step(
        self,
        features: np.ndarray,
        out: Dict,
        targets: Dict[str, Optional[float]],
    ):
        """
        Compute loss, backpropagate, and update weights for one sample.

        Parameters
        ----------
        features : np.ndarray
            Input features
        out : Dict
            Forward pass outputs
        targets : Dict
            Target values. Keys: "h_eff", "log_csat", "phase", "h_chem"
            None values are skipped (missing labels).
        """
        cfg = self.config
        a2 = out["a2"]
        z2 = out["z2"]
        a1 = out["a1"]
        z1 = out["z1"]

        # Gradient accumulator for shared backbone output (a2)
        grad_a2 = np.zeros_like(a2)
        total_loss = 0.0

        # --- Head 1: H_eff (MSE) ---
        if targets.get("h_eff") is not None:
            err_h = out["h_eff"] - targets["h_eff"]
            total_loss += cfg.lambda_h_eff * err_h ** 2
            grad_out = np.array([2.0 * cfg.lambda_h_eff * err_h])
            grad_a2 += self.head_h_eff.backward(grad_out)

        # --- Head 2: log(c_sat) (MSE on log-scale) ---
        if targets.get("log_csat") is not None:
            err_c = out["log_csat"] - targets["log_csat"]
            total_loss += cfg.lambda_csat * err_c ** 2
            grad_out = np.array([2.0 * cfg.lambda_csat * err_c])
            grad_a2 += self.head_csat.backward(grad_out)

        # --- Head 3: phase_separates (binary cross-entropy) ---
        if targets.get("phase") is not None:
            p = np.clip(out["phase_prob"], 1e-7, 1 - 1e-7)
            y = targets["phase"]
            bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            total_loss += cfg.lambda_phase * bce
            # d(BCE)/d(logit) = p - y
            grad_logit = cfg.lambda_phase * (p - y)
            grad_out = np.array([grad_logit])
            grad_a2 += self.head_phase.backward(grad_out)

        # --- Head 4: H_chemistry reconstruction (MSE) ---
        if targets.get("h_chem") is not None:
            err_hc = out["h_chem"] - targets["h_chem"]
            total_loss += cfg.lambda_h_chem * err_hc ** 2
            grad_out = np.array([2.0 * cfg.lambda_h_chem * err_hc])
            grad_a2 += self.head_h_chem.backward(grad_out)

        # --- Backprop through shared backbone ---
        # Through ReLU at layer 2
        grad_z2 = grad_a2 * _relu_grad(z2)
        grad_a1 = self.layer2.backward(grad_z2)

        # Through ReLU at layer 1
        grad_z1 = grad_a1 * _relu_grad(z1)
        self.layer1.backward(grad_z1)

        return total_loss

    def train(self, training_data: List['MTLSample']) -> List[Dict[str, float]]:
        """
        Train the network on multi-task data.

        Parameters
        ----------
        training_data : List[MTLSample]
            Training samples with features and task-specific labels

        Returns
        -------
        List of per-epoch loss dictionaries
        """
        cfg = self.config

        # Compute feature normalization from training data
        all_features = np.array([s.features for s in training_data])
        self._feat_mean = np.mean(all_features, axis=0)
        self._feat_std = np.std(all_features, axis=0)

        history = []
        rng = np.random.RandomState(cfg.seed)

        for epoch in range(cfg.n_epochs):
            # Zero gradients
            for layer in self._all_layers():
                layer.zero_grad()

            epoch_loss = 0.0
            n_samples = len(training_data)

            # Shuffle training order
            order = rng.permutation(n_samples)

            for idx in order:
                sample = training_data[idx]

                # Forward
                out = self.forward(sample.features)

                # Build targets dict (None = missing label)
                targets = {
                    "h_eff": sample.h_eff_target,
                    "log_csat": np.log(sample.csat_target) if sample.csat_target is not None else None,
                    "phase": sample.phase_target,
                    "h_chem": sample.h_chem_target,
                }

                # Backward
                loss = self._backward_and_step(sample.features, out, targets)
                epoch_loss += loss

            # Average gradients and Adam step
            for layer in self._all_layers():
                layer.dW /= n_samples
                layer.db /= n_samples
                layer.step_adam(
                    lr=cfg.lr, t=epoch + 1,
                    beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps,
                )

            epoch_loss /= n_samples

            if epoch % 200 == 0 or epoch == cfg.n_epochs - 1:
                history.append({"epoch": epoch, "loss": epoch_loss})

        self.training_history = history
        return history


# =============================================================================
# TRAINING DATA
# =============================================================================

@dataclass
class MTLSample:
    """One training sample for the multi-task network."""
    name: str
    features: np.ndarray           # (N_FEATURES,) feature vector

    # Task targets (None = missing / no label for this task)
    h_eff_target: Optional[float] = None        # target H_eff (from calibration)
    csat_target: Optional[float] = None          # experimental c_sat (relative)
    phase_target: Optional[float] = None         # 1.0 = phase separates, 0.0 = doesn't
    h_chem_target: Optional[float] = None        # computed H_chemistry (physics anchor)


def build_training_data(
    variant_names: List[str],
    features: Dict[str, np.ndarray],
    h_chem_values: Dict[str, float],
    csat_experimental: Optional[Dict[str, float]] = None,
    phase_labels: Optional[Dict[str, bool]] = None,
) -> List[MTLSample]:
    """
    Build MTL training samples from computed data.

    Parameters
    ----------
    variant_names : List[str]
        Variant names
    features : Dict[str, np.ndarray]
        Feature vectors per variant
    h_chem_values : Dict[str, float]
        Computed H_chemistry values (available for ALL variants)
    csat_experimental : Optional[Dict[str, float]]
        Experimental c_sat (available for few variants)
    phase_labels : Optional[Dict[str, bool]]
        Phase separation labels (can be assigned for all variants)

    Returns
    -------
    List[MTLSample]
    """
    samples = []

    for name in variant_names:
        csat = csat_experimental.get(name) if csat_experimental else None
        phase = float(phase_labels[name]) if phase_labels and name in phase_labels else None

        samples.append(MTLSample(
            name=name,
            features=features[name],
            h_eff_target=None,  # we don't have ground truth H_eff
            csat_target=csat,
            phase_target=phase,
            h_chem_target=h_chem_values.get(name),
        ))

    return samples


# =============================================================================
# MTL RESULTS
# =============================================================================

@dataclass
class MTLResult:
    """Results from multi-task model training and prediction."""
    predictions: Dict[str, Dict[str, float]]  # name -> {H_eff, csat, phase_prob, H_chem}
    training_history: List[Dict[str, float]]
    n_training_samples: int
    n_csat_labeled: int
    n_phase_labeled: int
    n_hchem_labeled: int

    def to_dict(self) -> Dict:
        return {
            "predictions": self.predictions,
            "n_training_samples": self.n_training_samples,
            "n_csat_labeled": self.n_csat_labeled,
            "n_phase_labeled": self.n_phase_labeled,
            "n_hchem_labeled": self.n_hchem_labeled,
            "final_loss": self.training_history[-1]["loss"] if self.training_history else None,
        }


def train_and_predict(
    training_data: List[MTLSample],
    config: Optional[MTLConfig] = None,
) -> Tuple[MultiTaskNetwork, MTLResult]:
    """
    Train MTL network and generate predictions for all variants.

    Parameters
    ----------
    training_data : List[MTLSample]
        Training data
    config : Optional[MTLConfig]
        Network configuration

    Returns
    -------
    Tuple of (trained network, results)
    """
    net = MultiTaskNetwork(config)
    history = net.train(training_data)

    predictions = {}
    for sample in training_data:
        pred = net.predict(sample.features)
        predictions[sample.name] = pred

    n_csat = sum(1 for s in training_data if s.csat_target is not None)
    n_phase = sum(1 for s in training_data if s.phase_target is not None)
    n_hchem = sum(1 for s in training_data if s.h_chem_target is not None)

    result = MTLResult(
        predictions=predictions,
        training_history=history,
        n_training_samples=len(training_data),
        n_csat_labeled=n_csat,
        n_phase_labeled=n_phase,
        n_hchem_labeled=n_hchem,
    )

    return net, result
