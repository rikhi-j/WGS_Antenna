
# =============================================================================
# MODULE: csv_optimizer.py
#
# PURPOSE:
# Main execution module. This file manages the complete
# PSO optimization process by coordinating geometry generation,
# repair, HFSS simulation, scoring, checkpointing, and result archiving.
#
# HOW IT IS USED:
# Run this file directly to start a new optimization or resume an existing one:
#
#     python csv_optimizer.py
#
# OVERALL MODULE PROCESS:
# 1. Load an existing optimizer checkpoint or create a new PSO.
# 2. Request the current particle design from the PSO.
# 3. Repair and validate the proposed antenna geometry.
# 4. Generate the current HFSS geometry script.
# 5. Reset the working HFSS project from the protected master project.
# 6. Run HFSS and export S11 and gain results.
# 7. Calculate the particle's objective score.
# 8. Archive results and save a checkpoint.
# 9. Repeat until all particles and generations have completed.
#
# MAIN INPUTS:
# - Project configuration (ansys_config.py)
# - PSO design vectors
# - Master HFSS project
# - S11 and gain CSV files
# - Optional optimizer checkpoint
#
# MAIN OUTPUTS:
# - optimizer_state.pkl (checkpoint)
# - particle_history.jsonl
# - Archived particle results
# - Updated PSO state
#
# DEPENDENCIES:
# - ansys_config.py
# - optimization_controller.py
# - ansys_s11_bridge.py
#
# IMPORTANT NOTES:
# - Invalid geometries are repaired or replaced before HFSS is run.
# - Every particle starts from a clean copy of the master HFSS project.
# - The repaired geometry is written back to the PSO because it is the design
#   that HFSS actually evaluates.
# - Checkpoints allow the optimization to resume after interruption.
# =============================================================================

from __future__ import annotations

import json
import os
import pickle
import shutil
from datetime import datetime
from typing import Any, Dict, List
import time
import numpy as np

import ansys_config as cfg

from ansys_s11_bridge import (
    HFSSTimeoutError,
    run_ansys_export,
)

from optimization_controller import (
    InvalidCandidateError,
    create_optimizer,
    export_design_vector_to_hfss,
    score_hfss_csv_with_gain,
)


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
N_PARTICLES = 10

# Generations 0, 1, 2, and 3.
MAX_GENERATIONS = 4

PAUSE_AFTER_EACH_GENERATION = False

INVALID_OPTIMIZER_SCORE = 1e9

# An invalid proposal is discarded and replaced in the same PSO slot.
# HFSS runs only after a valid repaired geometry is found.
MAX_VALIDITY_ATTEMPTS_PER_PARTICLE = 100

EXPECTED_WORKING_FILENAME = "Project_Working.aedt"

# -----------------------------------------------------------------------------
# Project-relative runtime paths
# -----------------------------------------------------------------------------
S11_CSV_PATH = cfg.EXPORT_CSV
GAIN_CSV_PATH = cfg.EXPORT_GAIN_CSV

STATE_FILE = os.path.join(
    cfg.PROJECT_FOLDER,
    "optimizer_state.pkl",
)

PARTICLE_HISTORY_FILE = os.path.join(
    cfg.EXPORT_DIR,
    "particle_history.jsonl",
)

PARTICLE_RESULTS_DIR = os.path.join(
    cfg.EXPORT_DIR,
    "particle_results",
)


# -----------------------------------------------------------------------------
# Scoring configuration signature
# -----------------------------------------------------------------------------
def get_scoring_configuration() -> Dict[str, Any]:
    """
    Return every setting that changes the meaning of an optimizer score.

    The saved optimizer state is rejected if any of these settings change.
    """
    return {
        "scoring_version": str(cfg.SCORING_VERSION),
        "target_freq_ghz": float(cfg.TARGET_FREQ_GHZ),
        "gain_phi_deg": float(cfg.GAIN_PHI_DEG),
        "s11_weight": float(cfg.S11_WEIGHT),
        "gain_weight": float(cfg.GAIN_WEIGHT),
        "resonance_weight": float(cfg.RESONANCE_WEIGHT),
        "gain_score_min_db": float(
            cfg.GAIN_SCORE_MIN_DB
        ),
        "gain_score_max_db": float(
            cfg.GAIN_SCORE_MAX_DB
        ),
        "resonance_scale_ghz": float(
            cfg.RESONANCE_SCALE_GHZ
        ),
        "s11_soft_floor_start": float(
            cfg.S11_SOFT_FLOOR_START
        ),
        "s11_soft_floor_scale": float(
            cfg.S11_SOFT_FLOOR_SCALE
        ),
    }


# -----------------------------------------------------------------------------
# Path safety
# -----------------------------------------------------------------------------
def _normalized(path: str) -> str:
    return os.path.normcase(
        os.path.abspath(path)
    )


def safe_delete_working_project() -> None:
    master = _normalized(
        cfg.MASTER_PROJECT_PATH
    )

    working = _normalized(
        cfg.WORKING_PROJECT_PATH
    )

    if master == working:
        raise RuntimeError(
            "SAFETY STOP: master and working "
            "project paths are identical."
        )

    if (
        os.path.basename(working).lower()
        != EXPECTED_WORKING_FILENAME.lower()
    ):
        raise RuntimeError(
            "SAFETY STOP: refusing to delete "
            f"unexpected file: {working}"
        )

    working_dir = _normalized(
        os.path.dirname(
            cfg.WORKING_PROJECT_PATH
        )
    )

    if os.path.exists(
        cfg.WORKING_PROJECT_PATH
    ):
        print(
            "Deleting old working project:",
            cfg.WORKING_PROJECT_PATH,
        )

        os.remove(
            cfg.WORKING_PROJECT_PATH
        )

    possible_results_folder = os.path.join(
        os.path.dirname(
            cfg.WORKING_PROJECT_PATH
        ),
        "Project_Working.aedtresults",
    )

    if os.path.isdir(
        possible_results_folder
    ):
        results_folder_norm = _normalized(
            possible_results_folder
        )

        if not results_folder_norm.startswith(
            working_dir
        ):
            raise RuntimeError(
                "SAFETY STOP: results folder "
                "is outside the working folder."
            )

        print(
            "Deleting old working results folder:",
            possible_results_folder,
        )

        shutil.rmtree(
            possible_results_folder
        )


def safe_reset_working_project() -> None:
    master = _normalized(
        cfg.MASTER_PROJECT_PATH
    )

    working = _normalized(
        cfg.WORKING_PROJECT_PATH
    )

    if master == working:
        raise RuntimeError(
            "SAFETY STOP: master and working "
            "project paths are identical."
        )

    if not os.path.exists(
        cfg.MASTER_PROJECT_PATH
    ):
        raise FileNotFoundError(
            "Master project not found: "
            f"{cfg.MASTER_PROJECT_PATH}"
        )

    if (
        os.path.basename(working).lower()
        != EXPECTED_WORKING_FILENAME.lower()
    ):
        raise RuntimeError(
            "SAFETY STOP: working project must "
            f"be named {EXPECTED_WORKING_FILENAME}"
        )

    os.makedirs(
        os.path.dirname(
            cfg.WORKING_PROJECT_PATH
        ),
        exist_ok=True,
    )

    safe_delete_working_project()

    shutil.copy2(
        cfg.MASTER_PROJECT_PATH,
        cfg.WORKING_PROJECT_PATH,
    )

    print("Copied clean master project:")
    print("  FROM:", cfg.MASTER_PROJECT_PATH)
    print("  TO:  ", cfg.WORKING_PROJECT_PATH)


# -----------------------------------------------------------------------------
# JSON conversion
# -----------------------------------------------------------------------------
def vector_to_list(vector) -> List[float]:
    if hasattr(vector, "tolist"):
        return vector.tolist()

    return list(vector)


def make_json_safe(value):
    if isinstance(
        value,
        (
            np.integer,
            np.floating,
            np.bool_,
        ),
    ):
        return value.item()

    if isinstance(value, np.ndarray):
        return [
            make_json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


# -----------------------------------------------------------------------------
# Optimizer state
# -----------------------------------------------------------------------------
def save_state(
    optimizer,
    particle_index: int,
    scores_so_far,
    generation: int,
) -> None:
    os.makedirs(
        cfg.PROJECT_FOLDER,
        exist_ok=True,
    )

    temporary_state_file = (
        STATE_FILE + ".tmp"
    )

    state = {
        "optimizer": optimizer,
        "particle_index": int(
            particle_index
        ),
        "scores_so_far": list(
            scores_so_far
        ),
        "generation": int(
            generation
        ),
        "n_particles": int(
            N_PARTICLES
        ),
        "max_generations": int(
            MAX_GENERATIONS
        ),
        "scoring_configuration": (
            get_scoring_configuration()
        ),
    }

    with open(
        temporary_state_file,
        "wb",
    ) as file:
        pickle.dump(
            state,
            file,
        )

    os.replace(
        temporary_state_file,
        STATE_FILE,
    )


def load_state():
    if not os.path.exists(
        STATE_FILE
    ):
        return (
            create_optimizer(
                n_particles=N_PARTICLES,
                seed=1,
            ),
            0,
            [],
            0,
        )

    with open(
        STATE_FILE,
        "rb",
    ) as file:
        data = pickle.load(file)

    saved_n_particles = data.get(
        "n_particles"
    )

    if (
        saved_n_particles is not None
        and saved_n_particles
        != N_PARTICLES
    ):
        raise RuntimeError(
            "Existing optimizer_state.pkl "
            f"was created with "
            f"{saved_n_particles} particles, "
            f"but this script expects "
            f"{N_PARTICLES}."
        )

    saved_max_generations = data.get(
        "max_generations"
    )

    if (
        saved_max_generations is not None
        and saved_max_generations
        != MAX_GENERATIONS
    ):
        raise RuntimeError(
            "Existing optimizer_state.pkl "
            f"was created for "
            f"{saved_max_generations} generations, "
            f"but this script expects "
            f"{MAX_GENERATIONS}."
        )

    saved_scoring_configuration = data.get(
        "scoring_configuration"
    )

    current_scoring_configuration = (
        get_scoring_configuration()
    )

    if (
        saved_scoring_configuration
        != current_scoring_configuration
    ):
        raise RuntimeError(
            "Existing optimizer_state.pkl uses "
            "a different scoring configuration.\n"
            "Delete or rename optimizer_state.pkl "
            "before beginning the final run.\n"
            f"Saved:   "
            f"{saved_scoring_configuration}\n"
            f"Current: "
            f"{current_scoring_configuration}"
        )

    return (
        data["optimizer"],
        int(
            data.get(
                "particle_index",
                0,
            )
        ),
        list(
            data.get(
                "scores_so_far",
                [],
            )
        ),
        int(
            data.get(
                "generation",
                0,
            )
        ),
    )


# -----------------------------------------------------------------------------
# Permanent history
# -----------------------------------------------------------------------------
def append_particle_history(
    generation: int,
    particle_index: int,
    design_vector,
    score: float,
    stats,
    status: str,
    result_directory: str,
) -> None:
    os.makedirs(
        cfg.EXPORT_DIR,
        exist_ok=True,
    )

    record = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "generation": int(
            generation
        ),
        "particle_index": int(
            particle_index
        ),
        "status": str(status),
        "design_vector": vector_to_list(
            design_vector
        ),
        "optimizer_score": float(
            score
        ),
        "display_score": make_json_safe(
            stats.get(
                "display_score"
            )
        ),
        "scoring_configuration": (
            get_scoring_configuration()
        ),
        "result_directory": (
            result_directory
        ),
        "stats": make_json_safe(
            stats
        ),
        # Kept for compatibility with the
        # earlier history format.
        "s11_stats": make_json_safe(
            stats
        ),
        "gain_stats": make_json_safe(
            stats.get(
                "gain_stats"
            )
        ),
    }

    with open(
        PARTICLE_HISTORY_FILE,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(record)
            + "\n"
        )

    print(
        "Saved permanent particle history:"
    )

    print(
        "  ",
        PARTICLE_HISTORY_FILE,
    )


# -----------------------------------------------------------------------------
# Per-particle archives
# -----------------------------------------------------------------------------
def get_particle_result_directory(
    generation: int,
    particle_index: int,
) -> str:
    return os.path.join(
        PARTICLE_RESULTS_DIR,
        f"generation_{generation:03d}",
        f"particle_{particle_index:03d}",
    )


def write_summary_json(
    result_directory: str,
    summary,
) -> str:
    os.makedirs(
        result_directory,
        exist_ok=True,
    )

    summary_path = os.path.join(
        result_directory,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(summary),
            file,
            indent=2,
        )

    return summary_path


def archive_valid_particle(
    generation: int,
    particle_index: int,
    design_vector,
    stats,
) -> str:
    result_directory = (
        get_particle_result_directory(
            generation,
            particle_index,
        )
    )

    os.makedirs(
        result_directory,
        exist_ok=True,
    )

    archived_s11_path = os.path.join(
        result_directory,
        "s11.csv",
    )

    archived_gain_path = os.path.join(
        result_directory,
        "gain.csv",
    )

    archived_candidate_path = os.path.join(
        result_directory,
        "candidate.py",
    )

    if not os.path.exists(
        S11_CSV_PATH
    ):
        raise FileNotFoundError(
            "Cannot archive missing S11 CSV: "
            f"{S11_CSV_PATH}"
        )

    if not os.path.exists(
        GAIN_CSV_PATH
    ):
        raise FileNotFoundError(
            "Cannot archive missing gain CSV: "
            f"{GAIN_CSV_PATH}"
        )

    shutil.copy2(
        S11_CSV_PATH,
        archived_s11_path,
    )

    shutil.copy2(
        GAIN_CSV_PATH,
        archived_gain_path,
    )

    if os.path.exists(
        cfg.CURRENT_CANDIDATE_SCRIPT
    ):
        shutil.copy2(
            cfg.CURRENT_CANDIDATE_SCRIPT,
            archived_candidate_path,
        )

    summary = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "generation": int(
            generation
        ),
        "particle_index": int(
            particle_index
        ),
        "status": "valid",
        "hfss_was_run": True,
        "design_vector": vector_to_list(
            design_vector
        ),
        "scoring_configuration": (
            get_scoring_configuration()
        ),
        "archived_s11_csv": (
            archived_s11_path
        ),
        "archived_gain_csv": (
            archived_gain_path
        ),
        "archived_candidate_script": (
            archived_candidate_path
            if os.path.exists(
                archived_candidate_path
            )
            else None
        ),
        "stats": make_json_safe(
            stats
        ),
    }

    write_summary_json(
        result_directory,
        summary,
    )

    print(
        "Archived valid particle results:"
    )

    print(
        "  ",
        result_directory,
    )

    return result_directory


def archive_invalid_particle(
    generation: int,
    particle_index: int,
    design_vector,
    invalid_stats,
) -> str:
    result_directory = (
        get_particle_result_directory(
            generation,
            particle_index,
        )
    )

    os.makedirs(
        result_directory,
        exist_ok=True,
    )

    summary = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "generation": int(
            generation
        ),
        "particle_index": int(
            particle_index
        ),
        "status": "invalid_geometry",
        "hfss_was_run": False,
        "design_vector": vector_to_list(
            design_vector
        ),
        "scoring_configuration": (
            get_scoring_configuration()
        ),
        "stats": make_json_safe(
            invalid_stats
        ),
    }

    write_summary_json(
        result_directory,
        summary,
    )

    print(
        "Archived invalid particle result:"
    )

    print(
        "  ",
        result_directory,
    )

    return result_directory


# -----------------------------------------------------------------------------
# Temporary file handling
# -----------------------------------------------------------------------------
def delete_temp_csvs() -> None:
    if os.path.exists(
        S11_CSV_PATH
    ):
        os.remove(
            S11_CSV_PATH
        )

        print(
            "Deleted used temporary "
            "S11 CSV:",
            S11_CSV_PATH,
        )

    if os.path.exists(
        GAIN_CSV_PATH
    ):
        os.remove(
            GAIN_CSV_PATH
        )

        print(
            "Deleted used temporary "
            "gain CSV:",
            GAIN_CSV_PATH,
        )


def delete_stale_csvs_before_run() -> None:
    if os.path.exists(
        S11_CSV_PATH
    ):
        print(
            "Deleting stale S11 CSV "
            "before ANSYS run:",
            S11_CSV_PATH,
        )

        os.remove(
            S11_CSV_PATH
        )

    if os.path.exists(
        GAIN_CSV_PATH
    ):
        print(
            "Deleting stale gain CSV "
            "before ANSYS run:",
            GAIN_CSV_PATH,
        )

        os.remove(
            GAIN_CSV_PATH
        )


# -----------------------------------------------------------------------------
# Progress and scoring
# -----------------------------------------------------------------------------
def print_progress(
    generation: int,
    particle_index: int,
) -> None:
    overall_index = (
        generation * N_PARTICLES
        + particle_index
        + 1
    )

    total_runs = (
        MAX_GENERATIONS
        * N_PARTICLES
    )

    print()
    print("=" * 60)

    print(
        f"Generation: "
        f"{generation} / "
        f"{MAX_GENERATIONS - 1}"
    )

    print(
        f"Particle:   "
        f"{particle_index + 1} / "
        f"{N_PARTICLES}"
    )

    print(
        f"Overall:    "
        f"{overall_index} / "
        f"{total_runs}"
    )

    print("=" * 60)


def score_completed_particle(
    design_vector,
    generation: int,
    particle_index: int,
    repair_summary=None,
    validation_summary=None,
):
    """
    Score and archive one completed HFSS simulation.

    design_vector must be the repaired vector written back into PSO,
    not the original potentially invalid proposal.
    """
    current_design = np.asarray(
        design_vector,
        dtype=float,
    ).reshape(-1)

    score, stats = (
        score_hfss_csv_with_gain(
            S11_CSV_PATH,
            GAIN_CSV_PATH,
        )
    )

    stats["geometry_status"] = (
        "valid_after_full_repair"
    )

    stats["hfss_was_run"] = True

    stats["repair_summary"] = (
        make_json_safe(
            repair_summary
        )
    )

    stats["validation_summary"] = (
        make_json_safe(
            validation_summary
        )
    )

    print(
        "Read HFSS CSV files:"
    )

    print(
        "Generation:",
        generation,
    )

    print(
        "Particle index:",
        particle_index,
    )

    print(
        "Optimizer score:",
        score,
    )

    print(
        "Display score:",
        stats.get(
            "display_score"
        ),
    )

    print(
        "Repair summary:",
        repair_summary,
    )

    print(
        "Final validation:",
        validation_summary,
    )

    print(
        "Stats:",
        stats,
    )

    # Archive before deleting temporary files.
    result_directory = (
        archive_valid_particle(
            generation=generation,
            particle_index=particle_index,
            design_vector=current_design,
            stats=stats,
        )
    )

    append_particle_history(
        generation=generation,
        particle_index=particle_index,
        design_vector=current_design,
        score=score,
        stats=stats,
        status="valid_after_full_repair",
        result_directory=result_directory,
    )

    delete_temp_csvs()

    return float(score)


def run_particle_slot_until_success(
    optimizer,
    particle_index: int,
    scores_so_far,
    generation: int,
) -> float:
    """
    Produce and simulate one successful candidate for the current slot.

    Rejected candidates do not advance particle_index and do not receive
    a PSO score.

    A candidate is rejected when:

        1. It remains geometrically invalid after repair.
        2. HFSS exceeds the configured timeout.

    The same slot is resampled until one valid candidate completes HFSS.
    """
    designs = optimizer.ask()

    if particle_index >= len(
        designs
    ):
        raise RuntimeError(
            f"Particle index {particle_index} is outside "
            f"the designs array of size {len(designs)}."
        )

    attempt = 0

    while (
        attempt
        < MAX_VALIDITY_ATTEMPTS_PER_PARTICLE
    ):
        attempt += 1

        delete_stale_csvs_before_run()

        if attempt == 1:
            candidate_design = np.asarray(
                designs[
                    particle_index
                ],
                dtype=float,
            ).reshape(-1)
        else:
            candidate_design = (
                optimizer.rng.uniform(
                    optimizer.lower_bounds,
                    optimizer.upper_bounds,
                )
            )

            optimizer.set_particle_position(
                particle_index,
                candidate_design,
            )

        print()
        print(
            "Candidate attempt:",
            attempt,
            "/",
            MAX_VALIDITY_ATTEMPTS_PER_PARTICLE,
        )

        print(
            "Generation:",
            generation,
        )

        print(
            "Particle slot:",
            particle_index,
        )

        # -------------------------------------------------------------
        # Geometry repair and validation
        # -------------------------------------------------------------
        try:
            export_result = (
                export_design_vector_to_hfss(
                    candidate_design,
                    filename=(
                        cfg.CURRENT_CANDIDATE_SCRIPT
                    ),
                    object_prefix=(
                        "WGS_CURRENT"
                    ),
                    return_repair_details=True,
                    repair_verbose=False,
                )
            )

        except InvalidCandidateError as exc:
            print(
                "Candidate remained invalid "
                "after repair."
            )

            print(
                "Candidate discarded before HFSS."
            )

            print(
                "Validation summary:",
                exc.summary,
            )

            continue

        repaired_design = np.asarray(
            export_result[
                "repaired_vector"
            ],
            dtype=float,
        ).reshape(-1)

        repair_summary = (
            export_result[
                "repair_summary"
            ]
        )

        validation_summary = (
            export_result[
                "validation_summary"
            ]
        )

        # PSO must store the exact geometry evaluated by HFSS.
        optimizer.set_particle_position(
            particle_index,
            repaired_design,
        )

        print(
            "Valid geometry found."
        )

        print(
            "Repair summary:",
            repair_summary,
        )

        print(
            "Validation summary:",
            validation_summary,
        )

        # -------------------------------------------------------------
        # Reset project and run HFSS
        # -------------------------------------------------------------
        safe_reset_working_project()

        save_state(
            optimizer,
            particle_index,
            scores_so_far,
            generation,
        )

        print(
            "Running ANSYS/HFSS for "
            "repaired valid particle..."
        )

        try:
            run_ansys_export(
                timeout=(
                    cfg.HFSS_TIMEOUT_SECONDS
                )
            )

        except HFSSTimeoutError as exc:
            print()
            print(
                "HFSS TIMEOUT:"
            )

            print(
                f"Candidate exceeded "
                f"{cfg.HFSS_TIMEOUT_SECONDS} seconds."
            )

            print(
                "Timed-out candidate discarded."
            )

            print(
                "The same particle slot will receive "
                "a replacement geometry."
            )

            print(
                exc
            )

            delete_temp_csvs()

            # Give Windows time to release AEDT files after taskkill.
            time.sleep(
                5
            )

            lock_path = (
                cfg.WORKING_PROJECT_PATH
                + ".lock"
            )

            if os.path.exists(
                lock_path
            ):
                print(
                    "Deleting stale working-project lock:",
                    lock_path,
                )

                os.remove(
                    lock_path
                )

            # Return to the top of the loop and resample this slot.
            continue

        print(
            "ANSYS/HFSS run complete."
        )

        if not os.path.exists(
            S11_CSV_PATH
        ):
            raise FileNotFoundError(
                "ANSYS finished, but the S11 CSV "
                "was not created: "
                f"{S11_CSV_PATH}"
            )

        if not os.path.exists(
            GAIN_CSV_PATH
        ):
            raise FileNotFoundError(
                "ANSYS finished, but the gain CSV "
                "was not created: "
                f"{GAIN_CSV_PATH}"
            )

        particle_score = (
            score_completed_particle(
                design_vector=(
                    repaired_design
                ),
                generation=generation,
                particle_index=(
                    particle_index
                ),
                repair_summary=(
                    repair_summary
                ),
                validation_summary=(
                    validation_summary
                ),
            )
        )

        return float(
            particle_score
        )

    raise RuntimeError(
        "Could not complete a valid HFSS candidate "
        f"for generation {generation}, particle slot "
        f"{particle_index}, after "
        f"{MAX_VALIDITY_ATTEMPTS_PER_PARTICLE} attempts."
    )

# -----------------------------------------------------------------------------
# Main optimization loop
# -----------------------------------------------------------------------------
def main() -> None:
    os.makedirs(
        cfg.EXPORT_DIR,
        exist_ok=True,
    )

    (
        optimizer,
        particle_index,
        scores_so_far,
        generation,
    ) = load_state()

    if generation >= MAX_GENERATIONS:
        print(
            "Optimization complete."
        )

        print(
            "Maximum generations reached:",
            MAX_GENERATIONS,
        )

        return

    while generation < MAX_GENERATIONS:
        designs = optimizer.ask()

        if particle_index >= len(
            designs
        ):
            raise RuntimeError(
                f"Particle index "
                f"{particle_index} is outside "
                f"the designs array of size "
                f"{len(designs)}."
            )

        print_progress(
            generation,
            particle_index,
        )

        delete_stale_csvs_before_run()

        repaired_design = None
        repair_summary = None
        validation_summary = None
        export_result = None

        validity_attempt = 0

        # -------------------------------------------------------------
        # Find one valid geometry for this exact PSO particle slot.
        #
        # Attempt 1 uses the particle proposed by PSO.
        #
        # If that proposal cannot be repaired, it is discarded and the
        # same particle slot is resampled. The particle index does not
        # advance, no score is appended, and HFSS is not launched.
        # -------------------------------------------------------------
        while export_result is None:
            validity_attempt += 1

            if validity_attempt == 1:
                candidate_design = np.asarray(
                    designs[particle_index],
                    dtype=float,
                ).reshape(-1)

            else:
                candidate_design = (
                    optimizer.rng.uniform(
                        optimizer.lower_bounds,
                        optimizer.upper_bounds,
                    )
                )

                # Replace only the current invalid slot.
                optimizer.set_particle_position(
                    particle_index,
                    candidate_design,
                )

                designs[particle_index] = (
                    candidate_design
                )

            print()
            print(
                "Validity attempt:",
                validity_attempt,
                "/",
                MAX_VALIDITY_ATTEMPTS_PER_PARTICLE,
            )

            print(
                "Generation:",
                generation,
            )

            print(
                "Particle slot:",
                particle_index,
            )

            try:
                export_result = (
                    export_design_vector_to_hfss(
                        candidate_design,
                        filename=(
                            cfg.CURRENT_CANDIDATE_SCRIPT
                        ),
                        object_prefix=(
                            "WGS_CURRENT"
                        ),
                        return_repair_details=True,
                        repair_verbose=False,
                    )
                )

            except InvalidCandidateError as exc:
                print(
                    "Candidate remained invalid "
                    "after repair."
                )

                print(
                    "Candidate discarded."
                )

                print(
                    "HFSS was not launched."
                )

                print(
                    "Validation summary:",
                    exc.summary,
                )

                repair_failure_summary = (
                    getattr(
                        exc,
                        "repair_summary",
                        None,
                    )
                )

                if (
                    repair_failure_summary
                    is not None
                ):
                    print(
                        "Repair summary:",
                        repair_failure_summary,
                    )

                if (
                    validity_attempt
                    >=
                    MAX_VALIDITY_ATTEMPTS_PER_PARTICLE
                ):
                    raise RuntimeError(
                        "Could not produce a valid "
                        "candidate for generation "
                        f"{generation}, particle slot "
                        f"{particle_index}, after "
                        f"{MAX_VALIDITY_ATTEMPTS_PER_PARTICLE} "
                        "attempts. HFSS was never "
                        "launched for this slot."
                    ) from exc

                # Try a new random candidate in the same slot.
                continue

        # -------------------------------------------------------------
        # A valid repaired candidate now exists.
        # -------------------------------------------------------------
        repaired_design = np.asarray(
            export_result[
                "repaired_vector"
            ],
            dtype=float,
        ).reshape(-1)

        repair_summary = (
            export_result[
                "repair_summary"
            ]
        )

        validation_summary = (
            export_result[
                "validation_summary"
            ]
        )

        # HFSS must evaluate the exact geometry stored by PSO.
        optimizer.set_particle_position(
            particle_index,
            repaired_design,
        )

        designs[particle_index] = (
            repaired_design
        )

        print()
        print(
            "Valid candidate found after",
            validity_attempt,
            "attempt(s).",
        )

        print(
            "Repair summary:",
            repair_summary,
        )

        print(
            "Validation summary:",
            validation_summary,
        )

        # -------------------------------------------------------------
        # Run HFSS only for the valid repaired candidate.
        # -------------------------------------------------------------
        safe_reset_working_project()

        # Save the exact valid geometry before starting HFSS.
        # If the computer restarts during the solve, this same particle
        # slot and repaired position remain in the checkpoint.
        save_state(
            optimizer,
            particle_index,
            scores_so_far,
            generation,
        )

        print(
            "Running ANSYS/HFSS "
            "for repaired valid particle..."
        )

        try:
            run_ansys_export(
                timeout=(
                    cfg.HFSS_TIMEOUT_SECONDS
                )
            )

        except HFSSTimeoutError as exc:
            print()
            print(
                "HFSS TIMEOUT:"
            )
            print(
                f"Candidate exceeded "
                f"{cfg.HFSS_TIMEOUT_SECONDS} seconds."
            )
            print(
                "Timed-out candidate discarded."
            )
            print(
                "Generating a replacement for "
                f"particle slot {particle_index}."
            )
            print(
                exc
            )

            # Remove partial output from the timed-out solve.
            delete_temp_csvs()

            # Allow Windows a moment to release AEDT files.
            time.sleep(
                5
            )

            lock_path = (
                cfg.WORKING_PROJECT_PATH
                + ".lock"
            )

            if os.path.exists(
                lock_path
            ):
                print(
                    "Deleting stale working-project lock:",
                    lock_path,
                )

                try:
                    os.remove(
                        lock_path
                    )
                except OSError as lock_error:
                    print(
                        "Warning: could not remove stale lock:",
                        lock_error,
                    )

            # Create a different proposal for this same particle slot.
            replacement_design = (
                optimizer.rng.uniform(
                    optimizer.lower_bounds,
                    optimizer.upper_bounds,
                )
            )

            optimizer.set_particle_position(
                particle_index,
                replacement_design,
            )

            # Do not append a score and do not increment particle_index.
            # Restart the outer loop for the same slot.
            continue

        print(
            "ANSYS/HFSS run complete."
        )

        if not os.path.exists(
            S11_CSV_PATH
        ):
            raise FileNotFoundError(
                "ANSYS finished, but the "
                "S11 CSV was not created: "
                f"{S11_CSV_PATH}"
            )

        if not os.path.exists(
            GAIN_CSV_PATH
        ):
            raise FileNotFoundError(
                "ANSYS finished, but the "
                "gain CSV was not created: "
                f"{GAIN_CSV_PATH}"
            )

        particle_score = (
            score_completed_particle(
                design_vector=(
                    repaired_design
                ),
                generation=generation,
                particle_index=(
                    particle_index
                ),
                repair_summary=(
                    repair_summary
                ),
                validation_summary=(
                    validation_summary
                ),
            )
        )

        if not np.isfinite(
            particle_score
        ):
            raise RuntimeError(
                "A particle produced a "
                "non-finite optimizer score: "
                f"{particle_score}"
            )

        # Only valid HFSS scores enter the generation.
        scores_so_far.append(
            float(particle_score)
        )

        particle_index += 1

        # Checkpoint immediately after every successful HFSS particle.
        save_state(
            optimizer,
            particle_index,
            scores_so_far,
            generation,
        )

        if particle_index >= N_PARTICLES:
            print()

            print(
                "All valid particles scored "
                "for generation",
                generation,
            )

            print(
                "Optimizer scores:",
                scores_so_far,
            )

            if (
                len(scores_so_far)
                != N_PARTICLES
            ):
                raise RuntimeError(
                    "Cannot update PSO: expected "
                    f"{N_PARTICLES} valid scores, "
                    f"got {len(scores_so_far)}."
                )

            score_array = np.asarray(
                scores_so_far,
                dtype=float,
            )

            if not np.all(
                np.isfinite(
                    score_array
                )
            ):
                raise RuntimeError(
                    "Cannot update PSO: one or "
                    "more scores are not finite."
                )

            if np.any(
                score_array
                >=
                optimizer.invalid_score_threshold
            ):
                raise RuntimeError(
                    "Cannot update PSO: the "
                    "generation unexpectedly contains "
                    "an invalid sentinel score."
                )

            print(
                "Valid HFSS scores:",
                len(scores_so_far),
            )

            print(
                "Invalid sentinel scores:",
                0,
            )

            optimizer.tell(
                scores_so_far
            )

            best_design, best_score = (
                optimizer.best()
            )

            if np.isfinite(
                best_score
            ):
                print(
                    "Best valid optimizer "
                    "score so far:",
                    best_score,
                )
            else:
                raise RuntimeError(
                    "PSO completed a generation "
                    "of valid scores but did not "
                    "establish a global best."
                )

            particle_index = 0
            scores_so_far = []
            generation += 1

            save_state(
                optimizer,
                particle_index,
                scores_so_far,
                generation,
            )

            if generation >= MAX_GENERATIONS:
                safe_delete_working_project()

                print()
                print(
                    "Optimization complete."
                )

                print(
                    "Finished",
                    MAX_GENERATIONS,
                    "generations with",
                    N_PARTICLES,
                    "valid HFSS particles each.",
                )

                print(
                    "Total valid HFSS simulations:",
                    (
                        MAX_GENERATIONS
                        * N_PARTICLES
                    ),
                )

                print(
                    "Particle history saved at:"
                )

                print(
                    "  ",
                    PARTICLE_HISTORY_FILE,
                )

                print(
                    "Particle result folders "
                    "saved at:"
                )

                print(
                    "  ",
                    PARTICLE_RESULTS_DIR,
                )

                return

            if PAUSE_AFTER_EACH_GENERATION:
                input(
                    "\nGeneration complete. "
                    "Press Enter to continue..."
                )

    print(
        "Optimization complete."
    )


if __name__ == "__main__":
    main()