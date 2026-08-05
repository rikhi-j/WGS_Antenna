# =============================================================================
# MODULE: design_evaluator.py
# PURPOSE:
# Evaluates HFSS simulation results and converts them into a single numerical
# score that the Particle Swarm Optimization (PSO) algorithm can use to compare
# antenna designs.
#
# This module defines what the optimizer considers to be a "good" antenna by
# analyzing S11 performance, antenna gain, and resonance location relative to
# the target frequency.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - optimization_controller.py passes the exported S11 and gain data to this
#   module after every HFSS simulation.
# - This module analyzes the simulation results, calculates individual scoring
#   components, combines them into one final score, and returns that score to
#   the optimizer.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Analyze the complete S11 frequency sweep.
# 2. Determine the deepest S11 resonance and its frequency.
# 3. Analyze the gain pattern exported from HFSS.
# 4. Calculate an S11 performance score.
# 5. Calculate a gain score.
# 6. Calculate a resonance proximity score based on how close the resonance is
#    to the target frequency.
# 7. Reduce the S11 reward if the resonance occurs far from the target
#    frequency.
# 8. Combine all weighted score components into one final display score.
# 9. Convert the display score into an optimizer score for PSO.
# -----------------------------------------------------------------------------
# HOW THE SCORING SYSTEM WORKS:
# Three performance metrics contribute to the final antenna score:
#
# S11 Performance:
#     Measures how well the antenna is impedance matched. Deeper (more negative)
#     S11 values receive higher scores because they indicate less reflected
#     power.
#
# Gain:
#     Measures the antenna's maximum radiation gain at the target frequency.
#     Higher gain receives a higher score.
#
# Resonance Location:
#     Rewards antennas whose strongest resonance occurs close to the target
#     operating frequency (2.4 GHz). Even a very deep S11 resonance receives
#     reduced credit if it occurs far from the desired frequency.
#
# The weighted contributions are:
#
#     45% S11 performance
#     45% Gain
#     10% Resonance proximity
#
# -----------------------------------------------------------------------------
# OPTIMIZER SCORE vs DISPLAY SCORE:
# display_score:
#     Human-readable performance score where HIGHER values indicate better
#     antenna performance. This is the score that should be used when comparing
#     antenna designs.
#
# optimizer_score:
#     Internal score used only by the PSO algorithm. Since PSO performs
#     minimization, the optimizer score is simply the negative of the display
#     score, making LOWER values better for the optimizer.
#
# Example:
#
#     Display Score = 8.42   (better antenna)
#     Optimizer Score = -8.42
#
# -----------------------------------------------------------------------------
# KEY DATA STORED:
# This module returns a statistics dictionary containing:
# - S11 analysis results.
# - Gain analysis results.
# - Individual score components.
# - Weighted score contributions.
# - Display score.
# - Optimizer score.
# - Scoring configuration information.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - S11 frequency sweep exported from HFSS.
# - S11 values (dB).
# - Gain values (dB).
# - Theta values for the gain plot.
# - Scoring configuration from ansys_config.py.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Final optimizer score.
# - Human-readable display score.
# - Complete scoring statistics.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - The complete S11 frequency sweep is analyzed rather than only the value at
#   the target frequency.
# - The S11 score is weighted by resonance proximity so deep resonances away
#   from the target frequency receive reduced credit.
# - Gain is evaluated using the maximum value from the configured gain plot.
# - The optimizer always minimizes the optimizer score, while users should
#   compare antenna performance using the display score.
# - All scoring weights and limits are loaded from ansys_config.py, allowing
#   the scoring behavior to be modified without changing this module.
# =============================================================================

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

import ansys_config as cfg


# -----------------------------------------------------------------------------
# Scoring configuration
# -----------------------------------------------------------------------------
TARGET_FREQ_GHZ = float(
    getattr(
        cfg,
        "TARGET_FREQ_GHZ",
        2.4,
    )
)

S11_WEIGHT = float(
    getattr(
        cfg,
        "S11_WEIGHT",
        0.45,
    )
)

GAIN_WEIGHT = float(
    getattr(
        cfg,
        "GAIN_WEIGHT",
        0.45,
    )
)

RESONANCE_WEIGHT = float(
    getattr(
        cfg,
        "RESONANCE_WEIGHT",
        0.10,
    )
)

GAIN_SCORE_SCALE_DB = float(
    getattr(
        cfg,
        "GAIN_SCORE_SCALE_DB",
        4.0,
    )
)

GAIN_SCORE_MIN_DB = float(
    getattr(
        cfg,
        "GAIN_SCORE_MIN_DB",
        -15.0,
    )
)

GAIN_SCORE_MAX_DB = float(
    getattr(
        cfg,
        "GAIN_SCORE_MAX_DB",
        7.0,
    )
)

RESONANCE_SCALE_GHZ = float(
    getattr(
        cfg,
        "RESONANCE_SCALE_GHZ",
        0.5,
    )
)

S11_SOFT_FLOOR_START = float(
    getattr(
        cfg,
        "S11_SOFT_FLOOR_START",
        -10.0,
    )
)

S11_SOFT_FLOOR_SCALE = float(
    getattr(
        cfg,
        "S11_SOFT_FLOOR_SCALE",
        10.0,
    )
)

GAIN_PHI_DEG = float(
    getattr(
        cfg,
        "GAIN_PHI_DEG",
        0.0,
    )
)

SCORING_VERSION = str(
    getattr(
        cfg,
        "SCORING_VERSION",
        "proximity_weighted_s11_gain_v1",
    )
)


def _validate_configuration() -> None:
    if TARGET_FREQ_GHZ <= 0.0:
        raise ValueError(
            "TARGET_FREQ_GHZ must be greater than zero."
        )

    weights = (
        S11_WEIGHT,
        GAIN_WEIGHT,
        RESONANCE_WEIGHT,
    )

    if any(
        weight < 0.0
        for weight in weights
    ):
        raise ValueError(
            "All scoring weights must be non-negative."
        )

    total_weight = sum(weights)

    if not math.isclose(
        total_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "S11_WEIGHT + GAIN_WEIGHT + "
            "RESONANCE_WEIGHT must equal 1.0; "
            f"got {total_weight}."
        )

    if GAIN_SCORE_SCALE_DB <= 0.0:
        raise ValueError(
            "GAIN_SCORE_SCALE_DB must be greater than zero."
        )

    if (
        GAIN_SCORE_MAX_DB
        <= GAIN_SCORE_MIN_DB
    ):
        raise ValueError(
            "GAIN_SCORE_MAX_DB must be greater than "
            "GAIN_SCORE_MIN_DB."
        )

    if RESONANCE_SCALE_GHZ <= 0.0:
        raise ValueError(
            "RESONANCE_SCALE_GHZ must be greater than zero."
        )

    if S11_SOFT_FLOOR_SCALE <= 0.0:
        raise ValueError(
            "S11_SOFT_FLOOR_SCALE must be greater than zero."
        )


_validate_configuration()


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------
def _prepare_xy(
    x_values,
    y_values,
    x_name: str,
    y_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        x_values,
        dtype=float,
    ).reshape(-1)

    y = np.asarray(
        y_values,
        dtype=float,
    ).reshape(-1)

    if x.size == 0:
        raise ValueError(
            f"No {x_name}/{y_name} samples were supplied."
        )

    if x.size != y.size:
        raise ValueError(
            f"{x_name} and {y_name} arrays must have equal length; "
            f"got {x.size} and {y.size}."
        )

    finite_mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[finite_mask]
    y = y[finite_mask]

    if x.size == 0:
        raise ValueError(
            f"No finite {x_name}/{y_name} samples were supplied."
        )

    order = np.argsort(x)

    return (
        x[order],
        y[order],
    )


# -----------------------------------------------------------------------------
# S11 analysis
# -----------------------------------------------------------------------------
def analyze_s11(
    freq_ghz,
    s11_db,
    target_freq_ghz: float = TARGET_FREQ_GHZ,
) -> Dict[str, object]:
    """
    Analyze the full S11 frequency sweep.

    The full sweep remains important because it identifies:

        - the deepest S11 resonance;
        - the frequency of that resonance;
        - the distance between that resonance and 2.4 GHz;
        - S11 interpolated at 2.4 GHz.

    The final scoring does not cap S11 depth. Instead, the raw S11
    depth score is multiplied by a resonance-proximity factor so that
    a very deep resonance receives full credit only when it occurs
    near the target frequency.
    """
    freq, s11 = _prepare_xy(
        freq_ghz,
        s11_db,
        "frequency",
        "S11",
    )

    target = float(
        target_freq_ghz
    )

    if (
        target < float(freq[0])
        or target > float(freq[-1])
    ):
        raise ValueError(
            f"Target frequency {target:.6g} GHz is outside the "
            f"S11 data range {float(freq[0]):.6g} to "
            f"{float(freq[-1]):.6g} GHz."
        )

    min_index = int(
        np.argmin(s11)
    )

    nearest_index = int(
        np.argmin(
            np.abs(freq - target)
        )
    )

    min_s11_db = float(
        s11[min_index]
    )

    resonance_freq_ghz = float(
        freq[min_index]
    )

    resonance_delta_ghz = abs(
        resonance_freq_ghz
        - target
    )

    s11_at_target_db = float(
        np.interp(
            target,
            freq,
            s11,
        )
    )

    return {
        "target_freq_ghz": target,
        "min_s11_db": min_s11_db,
        "resonance_freq_ghz": (
            resonance_freq_ghz
        ),
        "freq_of_min_s11_ghz": (
            resonance_freq_ghz
        ),
        "resonance_delta_ghz": float(
            resonance_delta_ghz
        ),
        "resonance_delta_hz": float(
            resonance_delta_ghz
            * 1e9
        ),
        "s11_at_target_db": (
            s11_at_target_db
        ),
        "nearest_s11_sample_freq_ghz": float(
            freq[nearest_index]
        ),
        "nearest_s11_sample_db": float(
            s11[nearest_index]
        ),
        "s11_frequency_min_ghz": float(
            freq[0]
        ),
        "s11_frequency_max_ghz": float(
            freq[-1]
        ),
        "s11_sample_count": int(
            freq.size
        ),
    }


# -----------------------------------------------------------------------------
# Gain analysis
# -----------------------------------------------------------------------------
def analyze_gain(
    theta_deg,
    gain_db,
) -> Dict[str, object]:
    """
    Analyze Gain Plot1.

    Gain Plot1 represents:

        x-axis: Theta in degrees
        y-axis: dB(GainTotal)
        frequency: 2.4 GHz
        Phi: 0 degrees

    The optimizer uses the maximum gain across the configured Theta cut.
    """
    theta, gain = _prepare_xy(
        theta_deg,
        gain_db,
        "Theta",
        "gain",
    )

    max_index = int(
        np.argmax(gain)
    )

    return {
        "max_gain_db": float(
            gain[max_index]
        ),
        "theta_of_max_gain_deg": float(
            theta[max_index]
        ),
        "gain_frequency_ghz": (
            TARGET_FREQ_GHZ
        ),
        "gain_phi_deg": (
            GAIN_PHI_DEG
        ),
        "theta_min_deg": float(
            theta[0]
        ),
        "theta_max_deg": float(
            theta[-1]
        ),
        "gain_sample_count": int(
            theta.size
        ),
    }


# -----------------------------------------------------------------------------
# Component scores
# -----------------------------------------------------------------------------
def calculate_s11_score(
    min_s11_db: float,
) -> float:
    """
    Score the depth of the best S11 resonance.

    More-negative S11 values receive higher scores.

    S11 depth is not capped. A logarithmic soft floor only prevents
    extremely weak S11 values from producing unbounded negative scores.
    """
    magnitude = abs(
        float(min_s11_db)
    )

    if (
        not math.isfinite(magnitude)
        or magnitude <= 0.0
    ):
        raise ValueError(
            f"Invalid minimum S11 value: {min_s11_db!r}"
        )

    raw_score = (
        10.0
        - 10000.0 / magnitude**3
    )

    if (
        raw_score
        >= S11_SOFT_FLOOR_START
    ):
        return float(
            raw_score
        )

    excess = (
        S11_SOFT_FLOOR_START
        - raw_score
    )

    softened_score = (
        S11_SOFT_FLOOR_START
        - S11_SOFT_FLOOR_SCALE
        * math.log1p(
            excess
            / S11_SOFT_FLOOR_SCALE
        )
    )

    return float(
        softened_score
    )


def calculate_gain_score(
    max_gain_db: float,
) -> float:
    """
    Map maximum gain linearly onto a bounded 0-to-10 score.

    Gain at or below -15 dB receives 0 points.
    Gain at or above +7 dB receives 10 points.

    Examples
    --------
    -15 dB -> 0.00
    -10 dB -> 2.27
     -5 dB -> 4.55
      0 dB -> 6.82
     +1 dB -> 7.27
     +3 dB -> 8.18
     +5 dB -> 9.09
     +7 dB -> 10.00
    """
    gain = float(
        max_gain_db
    )

    if not math.isfinite(gain):
        raise ValueError(
            f"Invalid maximum gain value: "
            f"{max_gain_db!r}"
        )

    normalized = (
        gain
        - GAIN_SCORE_MIN_DB
    ) / (
        GAIN_SCORE_MAX_DB
        - GAIN_SCORE_MIN_DB
    )

    bounded = min(
        1.0,
        max(
            0.0,
            normalized,
        ),
    )

    return float(
        10.0 * bounded
    )

def calculate_resonance_score(
    resonance_freq_ghz: float,
    target_freq_ghz: float = TARGET_FREQ_GHZ,
) -> Tuple[float, float]:
    """
    Reward an S11 resonance that occurs close to the target frequency.

    The score approaches 10 at the target and approaches zero as the
    resonance moves farther away.
    """
    resonance_frequency = float(
        resonance_freq_ghz
    )

    target = float(
        target_freq_ghz
    )

    if not math.isfinite(
        resonance_frequency
    ):
        raise ValueError(
            "Resonance frequency must be finite."
        )

    delta_ghz = abs(
        resonance_frequency
        - target
    )

    resonance_score = (
        10.0
        * math.exp(
            -(
                delta_ghz
                / RESONANCE_SCALE_GHZ
            )
            ** 2
        )
    )

    return (
        float(resonance_score),
        float(delta_ghz),
    )


def calculate_proximity_weighted_s11_score(
    raw_s11_score: float,
    resonance_score: float,
) -> Tuple[float, float]:
    """
    Make the S11-depth reward conditional on resonance proximity.

    proximity_factor is between 0 and 1:

        1.0 near 2.4 GHz
        approaches 0 far from 2.4 GHz

    No S11 cap is used. A very deep target-frequency resonance therefore
    still receives its complete raw S11 score.

    A deep off-target resonance receives a reduced effective S11 score.
    """
    raw_score = float(
        raw_s11_score
    )

    proximity = float(
        resonance_score
    )

    if not math.isfinite(raw_score):
        raise ValueError(
            "Raw S11 score must be finite."
        )

    if not math.isfinite(proximity):
        raise ValueError(
            "Resonance score must be finite."
        )

    proximity_factor = (
        proximity / 10.0
    )

    proximity_factor = min(
        1.0,
        max(
            0.0,
            proximity_factor,
        ),
    )

    effective_s11_score = (
        raw_score
        * proximity_factor
    )

    return (
        float(effective_s11_score),
        float(proximity_factor),
    )


# -----------------------------------------------------------------------------
# Legacy S11-only scorer
# -----------------------------------------------------------------------------
def score_design(
    freq_ghz,
    s11_db,
    target_freq_ghz: float = TARGET_FREQ_GHZ,
):
    """
    Legacy S11-only scoring entry point.

    The deepest S11 value is weighted by resonance proximity.
    """
    s11_stats = analyze_s11(
        freq_ghz,
        s11_db,
        target_freq_ghz=target_freq_ghz,
    )

    raw_s11_score = (
        calculate_s11_score(
            s11_stats["min_s11_db"]
        )
    )

    resonance_score, delta_ghz = (
        calculate_resonance_score(
            s11_stats[
                "resonance_freq_ghz"
            ],
            target_freq_ghz=(
                target_freq_ghz
            ),
        )
    )

    (
        effective_s11_score,
        proximity_factor,
    ) = (
        calculate_proximity_weighted_s11_score(
            raw_s11_score,
            resonance_score,
        )
    )

    display_score = (
        0.9
        * effective_s11_score
        + 0.1
        * resonance_score
    )

    optimizer_score = (
        -display_score
    )

    stats = {
        **s11_stats,
        "raw_s11_score": float(
            raw_s11_score
        ),
        "s11_score": float(
            effective_s11_score
        ),
        "effective_s11_score": float(
            effective_s11_score
        ),
        "resonance_score": float(
            resonance_score
        ),
        "s11_proximity_factor": float(
            proximity_factor
        ),
        "resonance_delta_ghz": float(
            delta_ghz
        ),
        "display_score": float(
            display_score
        ),
        "optimizer_score": float(
            optimizer_score
        ),
        "total_score": float(
            optimizer_score
        ),
        "scoring_version": (
            SCORING_VERSION
        ),
        "scoring_mode": (
            "proximity_weighted_s11_only"
        ),
    }

    return (
        float(optimizer_score),
        stats,
    )


# -----------------------------------------------------------------------------
# Final S11 + gain scorer
# -----------------------------------------------------------------------------
def score_design_with_gain(
    s11_freq_ghz,
    s11_db,
    gain_theta_deg,
    gain_db,
):
    """
    Final antenna scoring.

    The score uses:

        45% proximity-weighted best S11 depth
        45% maximum gain at 2.4 GHz
        10% explicit resonance proximity

    The full 1.2–3.6 GHz sweep remains in use.

    S11 depth is not capped. A very deep resonance at 2.4 GHz receives
    its complete raw S11-depth score.

    A deep resonance far away from 2.4 GHz receives reduced S11 credit.

    Higher display_score means a better antenna.

    PSO minimizes, so optimizer_score is the negative of display_score.
    """
    s11_stats = analyze_s11(
        s11_freq_ghz,
        s11_db,
    )

    gain_stats = analyze_gain(
        gain_theta_deg,
        gain_db,
    )

    raw_s11_score = (
        calculate_s11_score(
            s11_stats["min_s11_db"]
        )
    )

    gain_score = (
        calculate_gain_score(
            gain_stats["max_gain_db"]
        )
    )

    (
        resonance_score,
        resonance_delta_ghz,
    ) = calculate_resonance_score(
        s11_stats[
            "resonance_freq_ghz"
        ]
    )

    (
        effective_s11_score,
        proximity_factor,
    ) = (
        calculate_proximity_weighted_s11_score(
            raw_s11_score,
            resonance_score,
        )
    )

    s11_contribution = (
        S11_WEIGHT
        * effective_s11_score
    )

    gain_contribution = (
        GAIN_WEIGHT
        * gain_score
    )

    resonance_contribution = (
        RESONANCE_WEIGHT
        * resonance_score
    )

    display_score = (
        s11_contribution
        + gain_contribution
        + resonance_contribution
    )

    optimizer_score = (
        -display_score
    )

    stats = {
        **s11_stats,
        **gain_stats,
        "raw_s11_score": float(
            raw_s11_score
        ),
        "s11_score": float(
            effective_s11_score
        ),
        "effective_s11_score": float(
            effective_s11_score
        ),
        "s11_proximity_factor": float(
            proximity_factor
        ),
        "gain_score": float(
            gain_score
        ),
        "resonance_score": float(
            resonance_score
        ),
        "s11_weight": float(
            S11_WEIGHT
        ),
        "gain_weight": float(
            GAIN_WEIGHT
        ),
        "resonance_weight": float(
            RESONANCE_WEIGHT
        ),
        "s11_contribution": float(
            s11_contribution
        ),
        "gain_contribution": float(
            gain_contribution
        ),
        "resonance_contribution": float(
            resonance_contribution
        ),
        "resonance_delta_ghz": float(
            resonance_delta_ghz
        ),
        "resonance_delta_hz": float(
            resonance_delta_ghz
            * 1e9
        ),
        "display_score": float(
            display_score
        ),
        "optimizer_score": float(
            optimizer_score
        ),
        "total_score": float(
            optimizer_score
        ),
        "scoring_version": (
            SCORING_VERSION
        ),
        "scoring_mode": (
            "proximity_weighted_s11_gain_and_resonance"
        ),
        "gain_stats": {
            **gain_stats,
            "gain_score": float(
                gain_score
            ),
            "gain_contribution": float(
                gain_contribution
            ),
        },
    }

    return (
        float(optimizer_score),
        stats,
    )