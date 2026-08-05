# =============================================================================
# MODULE: optimization_controller.py
# PURPOSE:
# Connects the numerical Particle Swarm Optimization (PSO) system to the WGS
# antenna geometry, repair, validation, HFSS export, and scoring systems.
# This module acts as the main translation layer between the optimizer and the
# physical antenna simulation.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# csv_optimizer.py calls this module to:
# - Create the PSO optimizer.
# - Convert PSO vectors into WGS antenna objects.
# - Repair and validate proposed antenna geometries.
# - Convert repaired antennas back into PSO vectors.
# - Generate the ANSYS candidate geometry script.
# - Read and score S11 and gain CSV results.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Define the WGS antenna dimensions, grid resolution, and repair settings.
# 2. Create a base WGS_Antenna object.
# 3. Determine how many valid WGS cells are controlled by the PSO.
# 4. Convert a continuous PSO vector into a binary conductor grid:
#       value > 0.5  = metal
#       value <= 0.5 = empty
# 5. Run the complete WGS repair process.
# 6. Strictly validate the repaired antenna.
# 7. Convert the repaired antenna back into a binary PSO vector.
# 8. Verify that the repaired vector recreates the same valid geometry.
# 9. Generate current_candidate.py for ANSYS/HFSS.
# 10. Read the S11 and gain CSV files after HFSS finishes.
# 11. Pass the simulation data to design_evaluator.py for scoring.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - Continuous PSO design vectors containing values between 0.0 and 1.0.
# - S11 CSV files exported by HFSS.
# - Gain CSV files exported by HFSS.
# - Optional output filenames, ANSYS object prefixes, and repair settings.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - WGS_Antenna objects representing candidate geometries.
# - Repaired binary PSO vectors containing only 0.0 and 1.0.
# - Generated ANSYS/HFSS candidate-script paths.
# - Validation and repair summaries.
# - Final numerical design scores and detailed scoring statistics.
# - Configured ParticleSwarmOptimizer objects.
# -----------------------------------------------------------------------------
# MAIN DEPENDENCIES:
# WGS_pixels.py:
#     Defines the WGS_Antenna geometry object.
# wgs_full_repair_handler.py:
#     Repairs illegal, isolated, or disconnected antenna geometry.
# validator_updated_only.py:
#     Performs strict validation and creates readable validation summaries.
# wgs_to_ansys_geometry.py:
#     Converts a valid WGS antenna into an ANSYS cutout script.
# ansys_s11_bridge.py:
#     Reads the S11 and gain CSV files exported by HFSS.
# design_evaluator.py:
#     Converts S11, gain, and resonance performance into an optimizer score.
# pso_optimizer.py:
#     Provides the ParticleSwarmOptimizer class.
# -----------------------------------------------------------------------------
# IMPORTANT CONFIGURATION:
# DS:
#     Approximate WGS grid-cell size in meters.
# A_SIZE, B_SIZE, C_SIZE:
#     Physical dimensions used to construct the unfolded WGS antenna.
# RESOLUTION:
#     Number of grid cells along each WGS dimension.
# PAD_RING_T:
#     Padding-ring thickness of fixed border surrounding antenna.
# THRESHOLD:
#     PSO values above 0.5 become metal; values at or below 0.5 become empty.
# REPAIR_MAX_ITERATIONS:
#     Maximum number of complete repair cycles.
# REPAIR_MIN_ISLAND_AREA:
#     Conductor components smaller than this may be removed during repair.
# REPAIR_MAX_BRIDGES_PER_ITERATION:
#     Limits how many component-connecting bridges may be created per cycle.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - InvalidCandidateError is raised when a candidate remains invalid after all
#   permitted repair attempts. Invalid geometry is never exported to HFSS.
# - The repaired vector must be written back into the PSO because HFSS evaluates
#   the repaired geometry, not necessarily the original PSO proposal.
# - The vector-to-WGS and WGS-to-vector conversions use only locations where
#   map_mask is True. Non-WGS positions are not included in the PSO vector.
# - A round-trip validation confirms that converting the repaired antenna into
#   a vector and back does not change or invalidate the geometry.
# - score_hfss_csv() and update_optimizer_from_csv() are legacy S11-only helpers.
#   The production workflow uses score_hfss_csv_with_gain().
# - export_candidate_batch() is a batch-export helper and is not used by the
#   main csv_optimizer.py execution path.
# =============================================================================

from __future__ import annotations

import numpy as np

from WGS_pixels import WGS_Antenna
from ansys_s11_bridge import read_gain_csv, read_s11_csv
from design_evaluator import score_design, score_design_with_gain
from pso_optimizer import ParticleSwarmOptimizer
from validator_updated_only import (
    summarize_validation_report,
    validate_wgs_particle,
)
from wgs_full_repair_handler import (
    repair_wgs_particle_full,
)
from wgs_to_ansys_geometry import write_wgs_cutout_script


# -----------------------------------------------------------------------------
# WGS geometry configuration
# -----------------------------------------------------------------------------
DS = 10e-3

A_SIZE = 88.9e-3
B_SIZE = 44.45e-3
C_SIZE = 250e-3

RESOLUTION = np.array(
    [
        int(A_SIZE / DS),
        int(B_SIZE / DS),
        int(C_SIZE / DS),
    ]
)

PAD_RING_T = np.array(
    [
        0.5e-2,
        0.5e-2,
    ]
)

THRESHOLD = 0.5

# Full WGS-aware repair settings.
REPAIR_MAX_ITERATIONS = 20
REPAIR_MIN_ISLAND_AREA = 4
REPAIR_MAX_BRIDGES_PER_ITERATION = 500


class InvalidCandidateError(ValueError):
    """
    Raised when a WGS candidate remains invalid after the complete,
    bounded repair pipeline.

    The invalid geometry is not exported and therefore cannot reach HFSS.
    """

    def __init__(
        self,
        summary,
        report,
        repair_summary=None,
    ):
        self.summary = summary
        self.report = report
        self.repair_summary = (
            repair_summary
        )

        message = (
            "Generated WGS design is still invalid "
            "after the complete repair pipeline.\n"
            f"Validation summary: {summary}"
        )

        if repair_summary is not None:
            message += (
                "\nRepair summary: "
                f"{repair_summary}"
            )

        super().__init__(message)


# -----------------------------------------------------------------------------
# Particle construction and encoding
# -----------------------------------------------------------------------------
def make_base_particle():
    return WGS_Antenna(
        size=np.array(
            [
                A_SIZE,
                B_SIZE,
                C_SIZE,
            ]
        ),
        resolution=RESOLUTION,
        pad_ring_t=PAD_RING_T,
        randomized=False,
        alpha=0.5,
    )


def get_design_bounds():
    particle = make_base_particle()

    lower_bounds = []
    upper_bounds = []

    for row_index in range(
        len(particle.map_mask)
    ):
        for column_index in range(
            len(particle.map_mask[row_index])
        ):
            if particle.map_mask[
                row_index
            ][column_index]:
                lower_bounds.append(0.0)
                upper_bounds.append(1.0)

    return lower_bounds, upper_bounds


def get_design_dimension():
    """
    Return the number of mutable WGS cells represented by one PSO vector.
    """
    lower_bounds, _ = get_design_bounds()

    return len(lower_bounds)


def vector_to_wgs_particle(vector):
    """
    Convert a continuous PSO vector into a binary WGS particle.

    Values greater than 0.5 become metal.
    Values at or below 0.5 become empty.
    """
    particle = make_base_particle()

    vector_array = np.asarray(
        vector,
        dtype=float,
    ).reshape(-1)

    expected_length = get_design_dimension()

    if (
        vector_array.size
        != expected_length
    ):
        raise ValueError(
            "Design-vector length does not match the number "
            "of optimizable WGS cells. "
            f"Expected {expected_length}, "
            f"got {vector_array.size}."
        )

    if not np.all(
        np.isfinite(
            vector_array
        )
    ):
        raise ValueError(
            "Design vector contains non-finite values."
        )

    vector_index = 0

    for row_index in range(
        len(particle.map_mask)
    ):
        for column_index in range(
            len(particle.map_mask[row_index])
        ):
            if particle.map_mask[
                row_index
            ][column_index]:
                particle.conductor_grid[
                    row_index
                ][column_index] = bool(
                    vector_array[
                        vector_index
                    ]
                    > THRESHOLD
                )

                vector_index += 1

    return particle


def wgs_particle_to_vector(particle):
    """
    Convert a repaired WGS particle back into the PSO design-vector format.

    This is essential because HFSS must evaluate the same geometry that PSO
    stores as the particle's current position.

    Metal becomes 1.0.
    Empty becomes 0.0.
    """
    if not hasattr(
        particle,
        "map_mask",
    ):
        raise TypeError(
            "particle does not contain map_mask."
        )

    if not hasattr(
        particle,
        "conductor_grid",
    ):
        raise TypeError(
            "particle does not contain conductor_grid."
        )

    repaired_vector = []

    for row_index in range(
        len(particle.map_mask)
    ):
        for column_index in range(
            len(particle.map_mask[row_index])
        ):
            if particle.map_mask[
                row_index
            ][column_index]:
                repaired_vector.append(
                    1.0
                    if bool(
                        particle.conductor_grid[
                            row_index
                        ][column_index]
                    )
                    else 0.0
                )

    repaired_array = np.asarray(
        repaired_vector,
        dtype=float,
    )

    expected_length = get_design_dimension()

    if (
        repaired_array.size
        != expected_length
    ):
        raise RuntimeError(
            "Repaired WGS vector has the wrong size. "
            f"Expected {expected_length}, "
            f"got {repaired_array.size}."
        )

    return repaired_array


# -----------------------------------------------------------------------------
# HFSS result reading and scoring
# -----------------------------------------------------------------------------
def score_hfss_csv(s11_csv_path):
    """
    Legacy S11-only scoring.

    The production optimization uses score_hfss_csv_with_gain().
    """
    s11_result = read_s11_csv(
        s11_csv_path
    )

    score, stats = score_design(
        s11_result.frequency,
        s11_result.s11,
    )

    stats["s11_source"] = {
        "csv_path": (
            s11_result.csv_path
        ),
        "frequency_column": (
            s11_result.frequency_column
        ),
        "s11_column": (
            s11_result.s11_column
        ),
    }

    return float(score), stats


def read_hfss_results(
    s11_csv_path,
    gain_csv_path,
):
    """
    Load both HFSS result files without scoring them.

    S11 data:
        x-axis = frequency in GHz
        y-axis = S11 in dB

    Gain data:
        x-axis = Theta in degrees
        y-axis = dB(GainTotal)
        fixed at 2.4 GHz and Phi = 0 degrees
    """
    s11_result = read_s11_csv(
        s11_csv_path
    )

    gain_result = read_gain_csv(
        gain_csv_path
    )

    s11_source_stats = {
        "csv_path": (
            s11_result.csv_path
        ),
        "frequency_column": (
            s11_result.frequency_column
        ),
        "s11_column": (
            s11_result.s11_column
        ),
        "min_s11_db": (
            s11_result.min_s11_db
        ),
        "freq_of_min_s11_ghz": (
            s11_result.min_s11_frequency
        ),
        "sample_count": int(
            s11_result.frequency.size
        ),
    }

    gain_source_stats = {
        "csv_path": (
            gain_result.csv_path
        ),
        "theta_column": (
            gain_result.theta_column
        ),
        "gain_column": (
            gain_result.gain_column
        ),
        "max_gain_db": (
            gain_result.max_gain_db
        ),
        "theta_of_max_gain_deg": (
            gain_result.theta_of_max_gain
        ),
        "gain_frequency_ghz": (
            gain_result.frequency_ghz
        ),
        "gain_phi_deg": (
            gain_result.phi_deg
        ),
        "sample_count": int(
            gain_result.theta.size
        ),
    }

    return (
        s11_result,
        gain_result,
        s11_source_stats,
        gain_source_stats,
    )


def score_hfss_csv_with_gain(
    s11_csv_path,
    gain_csv_path,
):
    """
    Read and score one completed HFSS particle.

    The final score uses:

        45% proximity-weighted best S11 depth
        45% peak gain at 2.4 GHz
        10% explicit resonance proximity
    """
    (
        s11_result,
        gain_result,
        s11_source_stats,
        gain_source_stats,
    ) = read_hfss_results(
        s11_csv_path,
        gain_csv_path,
    )

    score, stats = score_design_with_gain(
        s11_result.frequency,
        s11_result.s11,
        gain_result.theta,
        gain_result.gain,
    )

    stats["s11_source"] = (
        s11_source_stats
    )

    evaluator_gain_stats = dict(
        stats.get(
            "gain_stats",
            {},
        )
    )

    evaluator_gain_stats.update(
        gain_source_stats
    )

    stats["gain_stats"] = (
        evaluator_gain_stats
    )

    return float(score), stats


# -----------------------------------------------------------------------------
# Candidate repair and validation
# -----------------------------------------------------------------------------
def repair_and_validate_design_vector(
    vector,
    verbose=False,
):
    """
    Convert, fully repair, validate, and re-encode one PSO design.

    Repair sequence inside repair_wgs_particle_full():

        1. Repair illegal diagonal/checkerboard contacts.
        2. Remove small conductor islands.
        3. Connect disconnected conductor components.
        4. Repair new diagonal contacts caused by bridges.
        5. Validate.
        6. Repeat for a bounded number of iterations.

    Returns
    -------
    repaired_particle
        The valid WGS particle that may be exported to HFSS.

    repaired_vector
        The repaired binary design vector that must be written back into PSO.

    repair_summary
        Information about changes made by the repair process.

    validation_summary
        Final strict validation summary.

    Raises
    ------
    InvalidCandidateError
        If the candidate remains invalid after all bounded repair attempts.
    """
    original_particle = (
        vector_to_wgs_particle(
            vector
        )
    )

    repair_result = (
        repair_wgs_particle_full(
            original_particle,
            in_place=False,
            include_fixed=False,
            min_island_area=(
                REPAIR_MIN_ISLAND_AREA
            ),
            repair_illegal_diagonals=True,
            remove_small_islands=True,
            connect_components=True,
            max_iterations=(
                REPAIR_MAX_ITERATIONS
            ),
            max_bridges_per_iteration=(
                REPAIR_MAX_BRIDGES_PER_ITERATION
            ),
            validate_after=True,
            verbose=verbose,
        )
    )

    repaired_particle = (
        repair_result.particle
    )

    # Perform a fresh strict validation instead of relying only on the
    # validation object stored by the repair handler.
    final_report = (
        validate_wgs_particle(
            repaired_particle
        )
    )

    final_summary = (
        summarize_validation_report(
            final_report
        )
    )

    repair_summary = (
        repair_result.summary()
    )

    if not bool(
        final_report.get(
            "is_valid",
            False,
        )
    ):
        raise InvalidCandidateError(
            summary=final_summary,
            report=final_report,
            repair_summary=(
                repair_summary
            ),
        )

    repaired_vector = (
        wgs_particle_to_vector(
            repaired_particle
        )
    )

    # Verify the round trip. The vector returned to PSO must reconstruct
    # the same valid binary geometry.
    round_trip_particle = (
        vector_to_wgs_particle(
            repaired_vector
        )
    )

    round_trip_report = (
        validate_wgs_particle(
            round_trip_particle
        )
    )

    if not bool(
        round_trip_report.get(
            "is_valid",
            False,
        )
    ):
        round_trip_summary = (
            summarize_validation_report(
                round_trip_report
            )
        )

        raise InvalidCandidateError(
            summary=round_trip_summary,
            report=round_trip_report,
            repair_summary={
                **repair_summary,
                "round_trip_failure": True,
            },
        )

    return (
        repaired_particle,
        repaired_vector,
        repair_summary,
        final_summary,
    )


# -----------------------------------------------------------------------------
# HFSS script export
# -----------------------------------------------------------------------------
def export_design_vector_to_hfss(
    vector,
    filename="generated_candidate.py",
    object_prefix="WGS_OPT",
    return_repair_details=False,
    repair_verbose=False,
):
    """
    Repair and validate a design before writing its HFSS candidate script.

    Invalid geometry never reaches write_wgs_cutout_script().

    When return_repair_details=True, the function also returns the repaired
    vector so the caller can overwrite the PSO particle position.
    """
    (
        repaired_particle,
        repaired_vector,
        repair_summary,
        validation_summary,
    ) = repair_and_validate_design_vector(
        vector,
        verbose=repair_verbose,
    )

    # This function is reached only after strict validation succeeds.
    script_path = write_wgs_cutout_script(
        repaired_particle,
        script_path=filename,
        object_prefix=object_prefix,
        clear_existing=True,
        close_ansys=False,
    )

    if return_repair_details:
        return {
            "script_path": (
                script_path
            ),
            "repaired_vector": (
                repaired_vector
            ),
            "repair_summary": (
                repair_summary
            ),
            "validation_summary": (
                validation_summary
            ),
        }

    # Preserve compatibility with older scripts that expect only a path.
    return script_path


# -----------------------------------------------------------------------------
# PSO creation and legacy helpers
# -----------------------------------------------------------------------------
def create_optimizer(
    n_particles=8,
    seed=1,
):
    lower_bounds, upper_bounds = (
        get_design_bounds()
    )

    return ParticleSwarmOptimizer(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_particles=n_particles,
        seed=seed,
        invalid_score_threshold=1e8,
    )


def export_candidate_batch(
    optimizer,
    folder_prefix="candidate",
):
    """
    Export a batch of valid, repaired candidates.

    Repaired positions are written back into the optimizer before export.
    """
    designs = optimizer.ask()
    paths = []

    for particle_index, design in enumerate(
        designs
    ):
        export_result = (
            export_design_vector_to_hfss(
                design,
                filename=(
                    f"{folder_prefix}_"
                    f"{particle_index}.py"
                ),
                object_prefix=(
                    f"WGS_OPT_"
                    f"{particle_index}"
                ),
                return_repair_details=True,
            )
        )

        optimizer.set_particle_position(
            particle_index,
            export_result[
                "repaired_vector"
            ],
        )

        paths.append(
            export_result[
                "script_path"
            ]
        )

    return paths


def update_optimizer_from_csv(
    optimizer,
    csv_paths,
):
    """
    Legacy S11-only batch update helper.

    The production optimizer uses both S11 and gain and does not call
    this function.
    """
    scores = []

    for csv_path in csv_paths:
        score, stats = score_hfss_csv(
            csv_path
        )

        scores.append(score)

        print("CSV:", csv_path)
        print("Score:", score)
        print("Stats:", stats)
        print()

    optimizer.tell(scores)

    best_design, best_score = (
        optimizer.best()
    )

    return best_design, best_score