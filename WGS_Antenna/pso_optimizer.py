# =============================================================================
# MODULE: pso_optimizer.py
# PURPOSE:
# Implements the Particle Swarm Optimization (PSO) algorithm used to optimize
# antenna designs. Each particle represents one complete antenna design as a
# numerical design vector, and lower scores indicate better-performing antennas.
# This module is responsible only for optimizing design vectors—it does not
# generate antenna geometry, run HFSS simulations, or calculate scores.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - optimization_controller.py creates the ParticleSwarmOptimizer object.
# - csv_optimizer.py calls:
#     • ask() to retrieve the current particle design vectors.
#     • set_particle_position() after geometry repair.
#     • tell() after every particle in a generation has been simulated and
#       assigned a score.
#     • best() to retrieve the best antenna design found so far.
# -----------------------------------------------------------------------------
# HOW PARTICLE SWARM OPTIMIZATION (PSO) WORKS:
# Each particle represents one complete antenna design. Every value within a
# particle's design vector corresponds to one editable WGS conductor cell that
# will later become either metal or empty when converted into antenna geometry.
#
# The optimization begins by randomly generating a population of particles
# within the allowed design-variable bounds. These particles act as the swarm's
# initial antenna designs.
#
# Each particle is then:
# 1. Converted into a WGS antenna.
# 2. Repaired if necessary.
# 3. Simulated in HFSS.
# 4. Assigned a score based on its S11, gain, and resonance performance.
#
# Lower optimizer scores represent better antenna designs.
# For interpreting optimization results, the display score should be used.
# Interpreting display score: higher values indicate better antenna performance.
# 
# After every particle has been evaluated, the PSO updates the swarm by allowing
# particles to learn from both their own previous success and the success of
# other particles. New antenna designs are then generated and the process
# repeats until all generations have been completed.
# -----------------------------------------------------------------------------
# PSO COEFFICIENTS:
# Inertia:
#     Determines how much of a particle's previous movement is carried into the
#     next iteration. Higher values encourage particles to continue exploring
#     the design space, while lower values cause particles to slow down and
#     focus more on refining good solutions.
#
# Cognitive Coefficient:
#     Determines how strongly a particle is attracted toward its personal best 
#     antenna design that it has discovered. This allows each particle to
#     learn from its own successful designs.
#
# Social Coefficient:
#     Determines how strongly a particle is attracted toward the global best 
#     design discovered by the entire swarm. This allows particles to learn
#     from each other and gradually converge toward high-performing solutions.
#
# Random numbers are generated during every PSO update so that particles do not
# all move identically. This encourages exploration of different regions of the
# design space while still guiding the swarm toward promising antenna designs.
# -----------------------------------------------------------------------------
# KEY DATA STORED:
# - positions               : Current design vector for every particle.
# - velocities              : Current movement vector for every particle.
# - personal_best_positions : Best design found by each particle.
# - personal_best_scores    : Score of each particle's best design.
# - global_best_position    : Best design found by the entire swarm.
# - global_best_score       : Score of the swarm's best design.
# - iteration               : Number of completed PSO updates.
# - rng                     : Random number generator used by the optimizer.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - Design-variable lower and upper bounds.
# - Number of particles.
# - PSO coefficients (inertia, cognitive, and social).
# - One score for every particle after HFSS evaluation.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Updated particle positions and velocities.
# - Updated personal-best and global-best antenna designs.
# - Current particle positions (ask()).
# - Best antenna design and score found so far (best()).
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - Invalid scores are ignored and never become personal-best or global-best
#   solutions.
# - Geometry repair may modify a particle before HFSS simulation, so
#   set_particle_position() replaces the original PSO vector with the repaired
#   design to ensure the optimizer stores the exact antenna that was evaluated.
# - Particle positions are always clipped so they remain within the configured
#   design-variable bounds.
# - Older optimizer checkpoints remain compatible through __setstate__().
# =============================================================================
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class ParticleSwarmOptimizer:
    """
    Continuous particle-swarm optimizer using minimization.

    Lower scores are better.

    Invalid or failed antenna evaluations may use a large sentinel score,
    such as 1e9. Scores at or above invalid_score_threshold are completely
    excluded from PSO memory:

        - they do not update personal bests;
        - they do not update the global best;
        - particles without a valid personal best receive no cognitive pull;
        - the swarm receives no social pull until a valid global best exists.
    """

    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        n_particles=5,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=None,
        invalid_score_threshold=1e8,
    ):
        self.lower_bounds = np.asarray(
            lower_bounds,
            dtype=float,
        ).reshape(-1)

        self.upper_bounds = np.asarray(
            upper_bounds,
            dtype=float,
        ).reshape(-1)

        self._validate_initial_configuration(
            n_particles=n_particles,
            inertia=inertia,
            cognitive=cognitive,
            social=social,
            invalid_score_threshold=(
                invalid_score_threshold
            ),
        )

        self.n_particles = int(n_particles)
        self.dim = int(
            self.lower_bounds.size
        )

        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)

        self.invalid_score_threshold = float(
            invalid_score_threshold
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.positions = self.rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(
                self.n_particles,
                self.dim,
            ),
        )

        variable_ranges = (
            self.upper_bounds
            - self.lower_bounds
        )

        self.velocities = self.rng.uniform(
            -0.1 * variable_ranges,
            0.1 * variable_ranges,
            size=(
                self.n_particles,
                self.dim,
            ),
        )

        # Positions are initialized here only to preserve the expected
        # array shape. They are not trusted until the corresponding
        # personal_best_score becomes finite.
        self.personal_best_positions = (
            self.positions.copy()
        )

        self.personal_best_scores = np.full(
            self.n_particles,
            np.inf,
            dtype=float,
        )

        # No valid global best exists at initialization.
        self.global_best_position = (
            self.positions[0].copy()
        )

        self.global_best_score = float(
            np.inf
        )

        self.iteration = 0

    def _validate_initial_configuration(
        self,
        n_particles,
        inertia,
        cognitive,
        social,
        invalid_score_threshold,
    ) -> None:
        if self.lower_bounds.size == 0:
            raise ValueError(
                "At least one design variable is required."
            )

        if (
            self.lower_bounds.shape
            != self.upper_bounds.shape
        ):
            raise ValueError(
                "lower_bounds and upper_bounds must have "
                "the same shape. "
                f"Got {self.lower_bounds.shape} and "
                f"{self.upper_bounds.shape}."
            )

        if not np.all(
            np.isfinite(
                self.lower_bounds
            )
        ):
            raise ValueError(
                "lower_bounds contains non-finite values."
            )

        if not np.all(
            np.isfinite(
                self.upper_bounds
            )
        ):
            raise ValueError(
                "upper_bounds contains non-finite values."
            )

        invalid_bounds = (
            self.lower_bounds
            >= self.upper_bounds
        )

        if np.any(invalid_bounds):
            invalid_indices = np.flatnonzero(
                invalid_bounds
            ).tolist()

            raise ValueError(
                "Every lower bound must be strictly less "
                "than its upper bound. Invalid indices: "
                f"{invalid_indices}"
            )

        if (
            isinstance(
                n_particles,
                (bool, np.bool_),
            )
            or not isinstance(
                n_particles,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "n_particles must be an integer."
            )

        if int(n_particles) <= 0:
            raise ValueError(
                "n_particles must be greater than zero."
            )

        for name, value in (
            ("inertia", inertia),
            ("cognitive", cognitive),
            ("social", social),
        ):
            try:
                numeric_value = float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise TypeError(
                    f"{name} must be numeric."
                ) from exc

            if not np.isfinite(
                numeric_value
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

            if numeric_value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        try:
            threshold = float(
                invalid_score_threshold
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "invalid_score_threshold must be numeric."
            ) from exc

        if not np.isfinite(
            threshold
        ):
            raise ValueError(
                "invalid_score_threshold must be finite."
            )

        if threshold <= 0.0:
            raise ValueError(
                "invalid_score_threshold must be "
                "greater than zero."
            )

    def _validate_internal_state(
        self,
    ) -> None:
        expected_shape = (
            self.n_particles,
            self.dim,
        )

        arrays_with_expected_shape = {
            "positions": self.positions,
            "velocities": self.velocities,
            "personal_best_positions": (
                self.personal_best_positions
            ),
        }

        for name, array in (
            arrays_with_expected_shape.items()
        ):
            if (
                np.asarray(array).shape
                != expected_shape
            ):
                raise RuntimeError(
                    "Optimizer state is corrupted: "
                    f"{name} has shape "
                    f"{np.asarray(array).shape}, "
                    f"expected {expected_shape}."
                )

        if (
            np.asarray(
                self.personal_best_scores
            ).shape
            != (self.n_particles,)
        ):
            raise RuntimeError(
                "Optimizer state is corrupted: "
                "personal_best_scores has the wrong shape."
            )

        if (
            np.asarray(
                self.global_best_position
            ).shape
            != (self.dim,)
        ):
            raise RuntimeError(
                "Optimizer state is corrupted: "
                "global_best_position has the wrong shape."
            )

        if not np.all(
            np.isfinite(
                self.positions
            )
        ):
            raise RuntimeError(
                "Optimizer state contains non-finite "
                "particle positions."
            )

        if not np.all(
            np.isfinite(
                self.velocities
            )
        ):
            raise RuntimeError(
                "Optimizer state contains non-finite "
                "particle velocities."
            )

    def __setstate__(
        self,
        state,
    ) -> None:
        """
        Restore optimizer checkpoints created by older code versions.

        Older optimizer_state.pkl files do not contain
        invalid_score_threshold. This migration supplies the new default
        while preserving the saved particle positions, velocities, scores,
        RNG state, generation progress, and iteration count.
        """
        self.__dict__.update(
            state
        )

        if not hasattr(
            self,
            "invalid_score_threshold",
        ):
            self.invalid_score_threshold = 1e8

        if not hasattr(
            self,
            "iteration",
        ):
            self.iteration = 0

        # Defensive conversion for older checkpoints.
        self.positions = np.asarray(
            self.positions,
            dtype=float,
        )

        self.velocities = np.asarray(
            self.velocities,
            dtype=float,
        )

        self.personal_best_positions = np.asarray(
            self.personal_best_positions,
            dtype=float,
        )

        self.personal_best_scores = np.asarray(
            self.personal_best_scores,
            dtype=float,
        )

        self.global_best_position = np.asarray(
            self.global_best_position,
            dtype=float,
        )

        self.global_best_score = float(
            self.global_best_score
        )

        # Remove any invalid sentinel values that an older optimizer may
        # already have stored as personal bests.
        invalid_personal_best_mask = (
            ~np.isfinite(
                self.personal_best_scores
            )
            | (
                self.personal_best_scores
                >= self.invalid_score_threshold
            )
        )

        self.personal_best_scores[
            invalid_personal_best_mask
        ] = np.inf

        # If the old checkpoint stored 1e9 as the global best, clear it.
        if (
            not np.isfinite(
                self.global_best_score
            )
            or self.global_best_score
            >= self.invalid_score_threshold
        ):
            valid_personal_indices = np.flatnonzero(
                np.isfinite(
                    self.personal_best_scores
                )
            )

            if valid_personal_indices.size > 0:
                best_offset = int(
                    np.argmin(
                        self.personal_best_scores[
                            valid_personal_indices
                        ]
                    )
                )

                best_index = int(
                    valid_personal_indices[
                        best_offset
                    ]
                )

                self.global_best_score = float(
                    self.personal_best_scores[
                        best_index
                    ]
                )

                self.global_best_position = (
                    self.personal_best_positions[
                        best_index
                    ].copy()
                )

            else:
                self.global_best_score = np.inf
                self.global_best_position = (
                    self.positions[0].copy()
                )

        self._validate_internal_state()

    def ask(self) -> np.ndarray:
        """
        Return a copy of the current particle positions.

        Calling ask() does not modify the optimizer.
        """
        self._validate_internal_state()

        return self.positions.copy()

    def set_particle_position(
        self,
        particle_index: int,
        position,
    ) -> None:
        """
        Replace one particle's current position.

        This is used after geometry repair so PSO stores the same valid
        design that is actually sent to HFSS.
        """
        index = int(
            particle_index
        )

        if (
            index < 0
            or index >= self.n_particles
        ):
            raise IndexError(
                f"Particle index {index} is outside the "
                f"valid range 0 to {self.n_particles - 1}."
            )

        position_array = np.asarray(
            position,
            dtype=float,
        ).reshape(-1)

        if (
            position_array.size
            != self.dim
        ):
            raise ValueError(
                "Replacement particle position has the "
                f"wrong length. Expected {self.dim}, "
                f"got {position_array.size}."
            )

        if not np.all(
            np.isfinite(
                position_array
            )
        ):
            raise ValueError(
                "Replacement particle position contains "
                "non-finite values."
            )

        position_array = np.clip(
            position_array,
            self.lower_bounds,
            self.upper_bounds,
        )

        self.positions[index] = (
            position_array
        )

    def tell(
        self,
        scores: Sequence[float],
    ) -> None:
        """
        Submit one score for every current particle and advance the swarm.

        Scores at or above invalid_score_threshold are treated as failed
        or invalid evaluations and are excluded from all PSO best-memory
        updates.
        """
        self._validate_internal_state()

        scores_array = np.asarray(
            scores,
            dtype=float,
        )

        if scores_array.ndim != 1:
            raise ValueError(
                "scores must be a one-dimensional sequence. "
                f"Got shape {scores_array.shape}."
            )

        if (
            scores_array.size
            != self.n_particles
        ):
            raise ValueError(
                "tell() requires exactly one score per particle. "
                f"Expected {self.n_particles}, got "
                f"{scores_array.size}."
            )

        if not np.all(
            np.isfinite(
                scores_array
            )
        ):
            invalid_indices = np.flatnonzero(
                ~np.isfinite(
                    scores_array
                )
            ).tolist()

            raise ValueError(
                "scores contains non-finite values at "
                f"indices {invalid_indices}."
            )

        # A score such as 1e9 may remain in generation logging, but it
        # is never allowed to enter personal-best or global-best memory.
        valid_score_mask = (
            scores_array
            < self.invalid_score_threshold
        )

        improved_personal_best = (
            valid_score_mask
            & (
                scores_array
                < self.personal_best_scores
            )
        )

        if np.any(
            improved_personal_best
        ):
            self.personal_best_scores[
                improved_personal_best
            ] = scores_array[
                improved_personal_best
            ]

            self.personal_best_positions[
                improved_personal_best
            ] = self.positions[
                improved_personal_best
            ]

        valid_indices = np.flatnonzero(
            valid_score_mask
        )

        if valid_indices.size > 0:
            valid_scores = scores_array[
                valid_indices
            ]

            local_best_offset = int(
                np.argmin(
                    valid_scores
                )
            )

            generation_best_index = int(
                valid_indices[
                    local_best_offset
                ]
            )

            generation_best_score = float(
                scores_array[
                    generation_best_index
                ]
            )

            if (
                generation_best_score
                < self.global_best_score
            ):
                self.global_best_score = (
                    generation_best_score
                )

                self.global_best_position = (
                    self.positions[
                        generation_best_index
                    ].copy()
                )

        r1 = self.rng.random(
            size=self.positions.shape
        )

        r2 = self.rng.random(
            size=self.positions.shape
        )

        # Only particles that have previously produced a valid design
        # are allowed to use their personal-best position.
        has_valid_personal_best = np.isfinite(
            self.personal_best_scores
        )

        cognitive_velocity = np.zeros_like(
            self.velocities
        )

        if np.any(
            has_valid_personal_best
        ):
            cognitive_velocity[
                has_valid_personal_best
            ] = (
                self.cognitive
                * r1[
                    has_valid_personal_best
                ]
                * (
                    self.personal_best_positions[
                        has_valid_personal_best
                    ]
                    - self.positions[
                        has_valid_personal_best
                    ]
                )
            )

        # Until at least one valid antenna has been evaluated, there is
        # no legitimate global best and therefore no social attraction.
        if np.isfinite(
            self.global_best_score
        ):
            social_velocity = (
                self.social
                * r2
                * (
                    self.global_best_position
                    - self.positions
                )
            )
        else:
            social_velocity = np.zeros_like(
                self.velocities
            )

        self.velocities = (
            self.inertia
            * self.velocities
            + cognitive_velocity
            + social_velocity
        )

        self.positions = (
            self.positions
            + self.velocities
        )

        self.positions = np.clip(
            self.positions,
            self.lower_bounds,
            self.upper_bounds,
        )

        self.iteration += 1

        self._validate_internal_state()

    def best(
        self,
    ) -> Tuple[np.ndarray, float]:
        """
        Return the best valid position and score seen so far.

        If no valid antenna has been evaluated yet, the returned score
        remains infinity.
        """
        self._validate_internal_state()

        return (
            self.global_best_position.copy(),
            float(
                self.global_best_score
            ),
        )