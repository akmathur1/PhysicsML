"""
calibration.py - Fit Hamiltonian coupling constants to experimental data.

Two calibration approaches:

1. Least-squares fit: minimize |log(c_sat_pred/c_sat_exp)|^2
   over the coupling constants, using variants with known c_sat.

2. Rank-order calibration: ensure H_eff rank-ordering matches
   experimental c_sat ordering (weaker constraint, more robust).

Also provides:
- Parameter robustness analysis (sweep each α and check rank stability)
- Leave-one-out cross-validation
- Comparison of H_eff vs H_chem-only prediction quality
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .hamiltonian import (
    HamiltonianParams, HamiltonianDecomposition,
    compute_H_eff, predict_csat, compute_sensitivity,
)
from .topology import TopologyMetrics
from .homology import HomologyMetrics
from .entropy import EntropyMetrics
from .segmentation import StickerMask
from .variants import ExperimentalData


# =============================================================================
# CALIBRATION DATA CONTAINER
# =============================================================================

@dataclass
class CalibrationInput:
    """All data needed for one variant during calibration."""
    name: str
    intermap: np.ndarray
    sticker_mask: StickerMask
    topology: TopologyMetrics
    homology: HomologyMetrics
    entropy: EntropyMetrics
    omega: float
    csat_experimental: float  # relative to WT


@dataclass
class CalibrationResult:
    """Results from coupling constant calibration."""
    optimal_params: HamiltonianParams
    initial_params: HamiltonianParams
    loss_initial: float
    loss_final: float
    n_variants_used: int
    variant_names: List[str]

    # Per-variant predictions with optimal params
    H_eff_predicted: Dict[str, float]
    csat_predicted: Dict[str, float]
    csat_experimental: Dict[str, float]

    # Rank correlation
    rank_correlation_initial: float
    rank_correlation_final: float

    def to_dict(self) -> Dict:
        return {
            "optimal_params": {
                "alpha_clustering": self.optimal_params.alpha_clustering,
                "alpha_connectivity": self.optimal_params.alpha_connectivity,
                "alpha_percolation": self.optimal_params.alpha_percolation,
                "alpha_arrangement": self.optimal_params.alpha_arrangement,
                "alpha_homology": self.optimal_params.alpha_homology,
            },
            "loss_initial": self.loss_initial,
            "loss_final": self.loss_final,
            "n_variants_used": self.n_variants_used,
            "rank_correlation_initial": self.rank_correlation_initial,
            "rank_correlation_final": self.rank_correlation_final,
            "csat_predicted": self.csat_predicted,
            "csat_experimental": self.csat_experimental,
            "H_eff_predicted": self.H_eff_predicted,
        }


# =============================================================================
# CORE CALIBRATION
# =============================================================================

def _params_from_vector(
    x: np.ndarray,
    base_params: HamiltonianParams,
) -> HamiltonianParams:
    """Create HamiltonianParams from optimization vector."""
    return HamiltonianParams(
        w_sticker_sticker=base_params.w_sticker_sticker,
        w_cross=base_params.w_cross,
        w_linker_linker=base_params.w_linker_linker,
        alpha_clustering=x[0],
        alpha_connectivity=x[1],
        alpha_percolation=x[2],
        alpha_arrangement=x[3],
        alpha_homology=x[4],
    )


def _params_to_vector(params: HamiltonianParams) -> np.ndarray:
    """Extract optimization vector from params."""
    return np.array([
        params.alpha_clustering,
        params.alpha_connectivity,
        params.alpha_percolation,
        params.alpha_arrangement,
        params.alpha_homology,
    ])


def _compute_loss(
    x: np.ndarray,
    calibration_data: List[CalibrationInput],
    base_params: HamiltonianParams,
    wt_index: int,
) -> float:
    """
    Loss function for calibration.

    L = Σ_i [log(c_sat_pred_i / c_sat_exp_i)]^2

    Using log-ratio so that fold-changes are weighted equally
    (a 2x error at c_sat=1 is the same as a 2x error at c_sat=100).
    """
    params = _params_from_vector(x, base_params)

    # Compute H_eff for all variants
    H_eff_values = []
    for data in calibration_data:
        decomp = compute_H_eff(
            intermap=data.intermap,
            sticker_mask=data.sticker_mask,
            topology=data.topology,
            homology=data.homology,
            entropy_metrics=data.entropy,
            omega=data.omega,
            params=params,
        )
        H_eff_values.append(decomp.H_eff)

    # Reference H_eff (WT)
    H_ref = H_eff_values[wt_index]

    # Compute c_sat predictions and loss
    loss = 0.0
    for i, data in enumerate(calibration_data):
        csat_pred = predict_csat(H_eff_values[i], reference_csat=1.0, reference_H=H_ref)
        csat_exp = data.csat_experimental

        if csat_exp > 0 and csat_pred > 0:
            log_ratio = np.log(csat_pred / csat_exp)
            loss += log_ratio ** 2

    return loss


def calibrate_coupling_constants(
    calibration_data: List[CalibrationInput],
    initial_params: Optional[HamiltonianParams] = None,
    n_restarts: int = 5,
    seed: int = 42,
) -> CalibrationResult:
    """
    Fit coupling constants to minimize log-ratio loss against experimental c_sat.

    Uses Nelder-Mead simplex optimization with multiple random restarts.

    Parameters
    ----------
    calibration_data : List[CalibrationInput]
        Calibration data (must include WT)
    initial_params : Optional[HamiltonianParams]
        Starting point for optimization
    n_restarts : int
        Number of random restarts
    seed : int
        Random seed

    Returns
    -------
    CalibrationResult
    """
    from scipy.optimize import minimize

    if initial_params is None:
        initial_params = HamiltonianParams()

    # Find WT index
    wt_index = None
    for i, data in enumerate(calibration_data):
        if data.name == "WT":
            wt_index = i
            break
    if wt_index is None:
        raise ValueError("Calibration data must include WT variant")

    x0 = _params_to_vector(initial_params)
    loss_initial = _compute_loss(x0, calibration_data, initial_params, wt_index)

    # Bounds: all alphas must be non-negative (except percolation which can be any sign)
    bounds = [
        (0.0, 1.0),    # alpha_clustering
        (0.0, 1.0),    # alpha_connectivity
        (0.0, 10.0),   # alpha_percolation
        (0.0, 1.0),    # alpha_arrangement
        (0.0, 5.0),    # alpha_homology
    ]

    rng = np.random.RandomState(seed)
    best_result = None
    best_loss = np.inf

    for restart in range(n_restarts):
        if restart == 0:
            x_start = x0.copy()
        else:
            # Random perturbation within bounds
            x_start = np.array([
                rng.uniform(lo, hi) for lo, hi in bounds
            ])

        result = minimize(
            _compute_loss,
            x_start,
            args=(calibration_data, initial_params, wt_index),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
        )

        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    # Extract optimal params
    optimal_params = _params_from_vector(best_result.x, initial_params)
    loss_final = best_result.fun

    # Compute predictions with optimal params
    H_eff_pred = {}
    for data in calibration_data:
        decomp = compute_H_eff(
            intermap=data.intermap,
            sticker_mask=data.sticker_mask,
            topology=data.topology,
            homology=data.homology,
            entropy_metrics=data.entropy,
            omega=data.omega,
            params=optimal_params,
        )
        H_eff_pred[data.name] = decomp.H_eff

    H_ref = H_eff_pred["WT"]
    csat_pred = {name: predict_csat(h, reference_csat=1.0, reference_H=H_ref)
                 for name, h in H_eff_pred.items()}
    csat_exp = {data.name: data.csat_experimental for data in calibration_data}

    # Rank correlations
    rank_initial = _compute_rank_correlation(calibration_data, initial_params, wt_index)
    rank_final = _compute_rank_correlation(calibration_data, optimal_params, wt_index)

    return CalibrationResult(
        optimal_params=optimal_params,
        initial_params=initial_params,
        loss_initial=loss_initial,
        loss_final=loss_final,
        n_variants_used=len(calibration_data),
        variant_names=[d.name for d in calibration_data],
        H_eff_predicted=H_eff_pred,
        csat_predicted=csat_pred,
        csat_experimental=csat_exp,
        rank_correlation_initial=rank_initial,
        rank_correlation_final=rank_final,
    )


def _compute_rank_correlation(
    calibration_data: List[CalibrationInput],
    params: HamiltonianParams,
    wt_index: int,
) -> float:
    """Compute Spearman rank correlation between predicted and experimental c_sat."""
    H_eff_values = []
    csat_exp = []

    for data in calibration_data:
        decomp = compute_H_eff(
            intermap=data.intermap,
            sticker_mask=data.sticker_mask,
            topology=data.topology,
            homology=data.homology,
            entropy_metrics=data.entropy,
            omega=data.omega,
            params=params,
        )
        H_eff_values.append(decomp.H_eff)
        csat_exp.append(data.csat_experimental)

    H_ref = H_eff_values[wt_index]
    csat_pred = [predict_csat(h, reference_csat=1.0, reference_H=H_ref) for h in H_eff_values]

    # Spearman rank correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(csat_pred, csat_exp)
    return float(corr) if np.isfinite(corr) else 0.0


# =============================================================================
# PARAMETER ROBUSTNESS ANALYSIS
# =============================================================================

@dataclass
class RobustnessResult:
    """Results from parameter robustness sweep."""
    param_names: List[str]
    sweep_values: Dict[str, np.ndarray]       # param_name -> tested values
    sweep_H_eff: Dict[str, Dict[str, np.ndarray]]  # param_name -> variant -> H_eff array
    sweep_ranks: Dict[str, np.ndarray]        # param_name -> rank correlation at each value
    rank_stable: Dict[str, bool]              # does rank order hold across sweep?

    def to_dict(self) -> Dict:
        return {
            "param_names": self.param_names,
            "rank_stable": self.rank_stable,
            "sweep_ranks": {k: v.tolist() for k, v in self.sweep_ranks.items()},
        }


def compute_robustness(
    calibration_data: List[CalibrationInput],
    base_params: Optional[HamiltonianParams] = None,
    n_points: int = 20,
    sweep_range: float = 5.0,
) -> RobustnessResult:
    """
    Sweep each coupling constant independently and check rank stability.

    For each alpha_k, vary it from base/sweep_range to base*sweep_range
    while holding others fixed. Check if the rank ordering of H_eff
    across variants is preserved.

    Parameters
    ----------
    calibration_data : List[CalibrationInput]
        Calibration data
    base_params : Optional[HamiltonianParams]
        Base parameter set
    n_points : int
        Number of sweep points per parameter
    sweep_range : float
        Multiplicative range for sweep

    Returns
    -------
    RobustnessResult
    """
    if base_params is None:
        base_params = HamiltonianParams()

    # Find WT index
    wt_index = None
    for i, data in enumerate(calibration_data):
        if data.name == "WT":
            wt_index = i
            break
    if wt_index is None:
        wt_index = 0

    param_names = [
        "alpha_clustering", "alpha_connectivity", "alpha_percolation",
        "alpha_arrangement", "alpha_homology",
    ]
    base_vec = _params_to_vector(base_params)

    sweep_values = {}
    sweep_H_eff = {}
    sweep_ranks = {}
    rank_stable = {}

    for pi, pname in enumerate(param_names):
        base_val = base_vec[pi]
        if base_val < 1e-10:
            vals = np.linspace(0.0, 1.0, n_points)
        else:
            vals = np.linspace(base_val / sweep_range, base_val * sweep_range, n_points)

        sweep_values[pname] = vals
        sweep_H_eff[pname] = {d.name: np.zeros(n_points) for d in calibration_data}
        ranks = np.zeros(n_points)

        for vi, val in enumerate(vals):
            x = base_vec.copy()
            x[pi] = val
            params = _params_from_vector(x, base_params)

            H_effs = []
            csat_exps = []
            for di, data in enumerate(calibration_data):
                decomp = compute_H_eff(
                    intermap=data.intermap,
                    sticker_mask=data.sticker_mask,
                    topology=data.topology,
                    homology=data.homology,
                    entropy_metrics=data.entropy,
                    omega=data.omega,
                    params=params,
                )
                sweep_H_eff[pname][data.name][vi] = decomp.H_eff
                H_effs.append(decomp.H_eff)
                csat_exps.append(data.csat_experimental)

            # Rank correlation at this parameter value
            H_ref = H_effs[wt_index]
            csat_preds = [predict_csat(h, reference_csat=1.0, reference_H=H_ref) for h in H_effs]
            from scipy.stats import spearmanr
            corr, _ = spearmanr(csat_preds, csat_exps)
            ranks[vi] = corr if np.isfinite(corr) else 0.0

        sweep_ranks[pname] = ranks
        # Rank is "stable" if correlation stays > 0.7 across >80% of sweep
        rank_stable[pname] = float(np.mean(ranks > 0.7)) > 0.8

    return RobustnessResult(
        param_names=param_names,
        sweep_values=sweep_values,
        sweep_H_eff=sweep_H_eff,
        sweep_ranks=sweep_ranks,
        rank_stable=rank_stable,
    )


# =============================================================================
# H_CHEM VS H_EFF COMPARISON
# =============================================================================

@dataclass
class ModelComparison:
    """Compare H_chem-only vs H_eff prediction quality."""
    variant_names: List[str]
    csat_experimental: List[float]
    csat_pred_chem_only: List[float]
    csat_pred_H_eff: List[float]
    rank_corr_chem_only: float
    rank_corr_H_eff: float
    rmse_log_chem_only: float
    rmse_log_H_eff: float

    def to_dict(self) -> Dict:
        return {
            "rank_corr_chem_only": self.rank_corr_chem_only,
            "rank_corr_H_eff": self.rank_corr_H_eff,
            "rmse_log_chem_only": self.rmse_log_chem_only,
            "rmse_log_H_eff": self.rmse_log_H_eff,
            "improvement_rank": self.rank_corr_H_eff - self.rank_corr_chem_only,
            "improvement_rmse": self.rmse_log_chem_only - self.rmse_log_H_eff,
        }


def compare_models(
    calibration_data: List[CalibrationInput],
    params: Optional[HamiltonianParams] = None,
) -> ModelComparison:
    """
    Compare chemistry-only vs full H_eff predictions.

    Parameters
    ----------
    calibration_data : List[CalibrationInput]
        Calibration data with experimental c_sat
    params : Optional[HamiltonianParams]
        Parameters for H_eff (uses defaults if None)

    Returns
    -------
    ModelComparison
    """
    from scipy.stats import spearmanr

    if params is None:
        params = HamiltonianParams()

    # Chemistry-only params (zero out topology)
    chem_only_params = HamiltonianParams(
        w_sticker_sticker=params.w_sticker_sticker,
        w_cross=params.w_cross,
        w_linker_linker=params.w_linker_linker,
        alpha_clustering=0.0,
        alpha_connectivity=0.0,
        alpha_percolation=0.0,
        alpha_arrangement=0.0,
        alpha_homology=0.0,
    )

    names = []
    csat_exp = []
    H_chem_vals = []
    H_eff_vals = []

    for data in calibration_data:
        names.append(data.name)
        csat_exp.append(data.csat_experimental)

        # Chemistry-only
        d_chem = compute_H_eff(
            data.intermap, data.sticker_mask, data.topology,
            data.homology, data.entropy, data.omega,
            params=chem_only_params,
        )
        H_chem_vals.append(d_chem.H_eff)

        # Full H_eff
        d_full = compute_H_eff(
            data.intermap, data.sticker_mask, data.topology,
            data.homology, data.entropy, data.omega,
            params=params,
        )
        H_eff_vals.append(d_full.H_eff)

    # Find WT reference
    wt_idx = names.index("WT") if "WT" in names else 0

    csat_chem = [predict_csat(h, 1.0, H_chem_vals[wt_idx]) for h in H_chem_vals]
    csat_full = [predict_csat(h, 1.0, H_eff_vals[wt_idx]) for h in H_eff_vals]

    # Rank correlation
    rc_chem, _ = spearmanr(csat_chem, csat_exp)
    rc_full, _ = spearmanr(csat_full, csat_exp)

    # RMSE of log-ratios
    def rmse_log(pred, exp):
        log_ratios = [np.log(p / e) for p, e in zip(pred, exp) if p > 0 and e > 0]
        return float(np.sqrt(np.mean(np.array(log_ratios) ** 2))) if log_ratios else np.inf

    return ModelComparison(
        variant_names=names,
        csat_experimental=csat_exp,
        csat_pred_chem_only=csat_chem,
        csat_pred_H_eff=csat_full,
        rank_corr_chem_only=float(rc_chem) if np.isfinite(rc_chem) else 0.0,
        rank_corr_H_eff=float(rc_full) if np.isfinite(rc_full) else 0.0,
        rmse_log_chem_only=rmse_log(csat_chem, csat_exp),
        rmse_log_H_eff=rmse_log(csat_full, csat_exp),
    )
