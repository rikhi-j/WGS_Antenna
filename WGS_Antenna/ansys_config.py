# =============================================================================
# MODULE: ansys_config.py
# PURPOSE:
# Stores the centralized configuration settings used by the ANSYS/HFSS
# automation, antenna optimization, result export, scoring, and reliability
# systems.
#
# This module does not perform calculations or run HFSS by itself. Other modules
# import these values so paths, report names, scoring settings, and run options
# are defined in one place instead of being repeated throughout the project.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - csv_optimizer.py uses the master/working project paths, output paths,
#   timeout, and candidate-script location.
# - ansys_s11_bridge.py uses the ANSYS executable, project/design/report names,
#   CSV paths, and run options.
# - wgs_to_ansys_geometry.py uses project names and generated-script locations.
# - design_evaluator.py uses the target frequency, scoring weights, score ranges,
#   and resonance settings.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Define the ANSYS Electronics Desktop executable.
# 2. Define the protected master HFSS project and disposable working copy.
# 3. Define the project, design, and report names used inside HFSS.
# 4. Define where generated scripts and result files are stored.
# 5. Define the antenna target frequency and scoring configuration.
# 6. Define whether HFSS analyzes, saves, closes, or runs non-graphically.
# 7. Define the maximum allowed runtime for one HFSS simulation.
# -----------------------------------------------------------------------------
# MASTER AND WORKING PROJECTS:
# MASTER_PROJECT_PATH:
#     Protected original HFSS project. The optimizer should never modify or run
#     directly on this file.
#
# WORKING_PROJECT_PATH:
#     Disposable copy created from the master project for each candidate.
#
# PROJECT_PATH:
#     Points to WORKING_PROJECT_PATH so automation always opens the disposable
#     project instead of the protected master.
# -----------------------------------------------------------------------------
# HFSS REPORT REQUIREMENTS:
# REPORT_NAME:
#     Existing HFSS report used to export the S11 frequency sweep.
#
# GAIN_REPORT_NAME:
#     Existing HFSS report used to export Theta versus dB(GainTotal) at the
#     target frequency and configured Phi angle.
#
# These reports must already exist inside the HFSS project with matching names.
# -----------------------------------------------------------------------------
# SCORING CONFIGURATION:
# TARGET_FREQ_GHZ:
#     Desired antenna operating frequency.
#
# S11_WEIGHT:
#     Fraction of the display score contributed by proximity-weighted S11.
#
# GAIN_WEIGHT:
#     Fraction of the display score contributed by maximum gain.
#
# RESONANCE_WEIGHT:
#     Fraction of the display score contributed by resonance proximity.
#
# GAIN_SCORE_MIN_DB and GAIN_SCORE_MAX_DB:
#     Define the linear gain range mapped onto a score from 0 to 10.
#
# RESONANCE_SCALE_GHZ:
#     Controls how quickly the resonance score decreases as the strongest S11
#     resonance moves away from the target frequency.
#
# S11_SOFT_FLOOR_START and S11_SOFT_FLOOR_SCALE:
#     Compress extremely poor S11 scores so they do not become excessively
#     negative.
#
# SCORING_VERSION:
#     Identifies the exact scoring configuration saved with checkpoints and
#     particle histories.
# -----------------------------------------------------------------------------
# SCORE INTERPRETATION:
# display_score:
#     Human-readable score where higher values indicate better antenna
#     performance. This is the score that should be used when comparing designs.
#
# optimizer_score:
#     Internal score used by PSO. It is the negative of the display score because
#     the optimizer performs minimization, so lower values are better internally.
# -----------------------------------------------------------------------------
# ANSYS RUN OPTIONS:
# ANALYZE_BEFORE_EXPORT:
#     When True, HFSS runs AnalyzeAll() before exporting reports.
#
# SAVE_PROJECT_AFTER_RUN:
#     Controls whether the disposable working project is saved after simulation.
#
# CLOSE_ANSYS_AFTER_RUN:
#     Controls whether ANSYS closes after exporting results.
#
# RUN_NON_GRAPHICAL:
#     Controls whether ANSYS runs without the graphical interface.
# -----------------------------------------------------------------------------
# MAIN OUTPUT LOCATIONS:
# EXPORT_CSV:
#     Temporary S11 CSV exported by HFSS.
#
# EXPORT_GAIN_CSV:
#     Temporary gain CSV exported by HFSS.
#
# GENERATED_HFSS_SCRIPT:
#     Generated IronPython script that opens the project, applies the current
#     candidate, runs HFSS, and exports the reports.
#
# CURRENT_CANDIDATE_SCRIPT:
#     Generated geometry script containing the cutout operations for the current
#     antenna design.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - Machine-specific paths must be updated if the project is moved to another
#   computer or user account.
# - The three scoring weights must add to exactly 1.0.
# - PROJECT_NAME, DESIGN_NAME, REPORT_NAME, and GAIN_REPORT_NAME must exactly
#   match the names inside the HFSS project.
# - Temporary CSV files are archived by csv_optimizer.py before being deleted.
# - HFSS_TIMEOUT_SECONDS is currently 4800 seconds, or 80 minutes, per run.
# =============================================================================

from __future__ import annotations

import os


# -----------------------------------------------------------------------------
# 1) ANSYS Electronics Desktop executable
# -----------------------------------------------------------------------------
ANSYS_EXE = r"C:\Program Files\ANSYS Inc\v252\AnsysEM\ansysedt.exe"


# -----------------------------------------------------------------------------
# 2) HFSS project/design/report names
# -----------------------------------------------------------------------------
# NEVER run optimization directly on the master project.
MASTER_PROJECT_PATH = (
    r"C:\Users\rikhi\Desktop\HFSS_MASTER_DO_NOT_TOUCH"
    r"\Project1_WORKING_MASTER.aedt"
)

# Disposable project copy used during optimization.
WORKING_PROJECT_PATH = (
    r"C:\Users\rikhi\Desktop\HFSS_WORKING_TEMP"
    r"\Project_Working.aedt"
)

# The rest of the code must open only this working copy.
PROJECT_PATH = WORKING_PROJECT_PATH

# Project and design names inside AEDT.
PROJECT_NAME = "Project_Working"
DESIGN_NAME = "HFSSDesign1"

# These reports must already exist in HFSS.
REPORT_NAME = "Terminal S Parameter Plot1"
GAIN_REPORT_NAME = "Gain Plot1"


# -----------------------------------------------------------------------------
# 3) Project-relative output locations
# -----------------------------------------------------------------------------
PROJECT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

EXPORT_DIR = os.path.join(
    PROJECT_FOLDER,
    "ansys_output",
)

# Temporary CSV exports produced by HFSS.
# These are archived into per-particle folders before deletion.
EXPORT_CSV = (
    r"C:\Users\rikhi\Desktop\s11_export.csv"
)

EXPORT_GAIN_CSV = (
    r"C:\Users\rikhi\Desktop\gain_export.csv"
)

GENERATED_HFSS_SCRIPT = os.path.join(
    EXPORT_DIR,
    "generated_export_s11.py",
)

CURRENT_CANDIDATE_SCRIPT = os.path.join(
    PROJECT_FOLDER,
    "current_candidate.py",
)


# -----------------------------------------------------------------------------
# 4) Antenna operating point and scoring configuration
# -----------------------------------------------------------------------------
TARGET_FREQ_GHZ = 2.4

# Gain Plot1 configuration:
#
# x-axis: Theta
# y-axis: dB(GainTotal)
# frequency: 2.4 GHz
# Phi: 0 degrees
GAIN_PHI_DEG = 0.0

# Final objective weighting:
#
# 45% proximity-weighted deepest S11 resonance
# 45% maximum gain at 2.4 GHz
# 10% explicit resonance proximity to 2.4 GHz
S11_WEIGHT = 0.45
GAIN_WEIGHT = 0.45
RESONANCE_WEIGHT = 0.10

# Gain score:
#
# 10 * tanh(max_gain_db / GAIN_SCORE_SCALE_DB)
GAIN_SCORE_SCALE_DB = 4.0
# Linear gain-score mapping:
#
# -15 dB or lower -> 0 points
# +7 dB or higher -> 10 points
GAIN_SCORE_MIN_DB = -15.0
GAIN_SCORE_MAX_DB = 7.0

# Resonance proximity score:
#
# 10 * exp(-(delta_f_ghz / RESONANCE_SCALE_GHZ)^2)
#
# With 0.5 GHz:
#
# delta = 0.0 GHz  -> score 10.00
# delta = 0.1 GHz  -> score about 9.61
# delta = 0.25 GHz -> score about 7.79
# delta = 0.5 GHz  -> score about 3.68
# delta = 1.0 GHz  -> score about 0.18
RESONANCE_SCALE_GHZ = 0.5

# The original cubic S11 score remains uncapped in its useful range.
# Extremely weak S11 values are compressed below this soft-floor point
# to prevent enormous negative scores.
S11_SOFT_FLOOR_START = -10.0
S11_SOFT_FLOOR_SCALE = 10.0

# Identifies the exact scoring equation saved with particle histories.
SCORING_VERSION = (
    "proximity_weighted_s11_linear_gain_v2"
)

# -----------------------------------------------------------------------------
# 5) ANSYS run options
# -----------------------------------------------------------------------------
# True:
#     run oDesign.AnalyzeAll() before exporting.
#
# False:
#     export reports from an already-solved project.
ANALYZE_BEFORE_EXPORT = True

# Do not save the disposable working project after simulation/export.
SAVE_PROJECT_AFTER_RUN = False

# Close ANSYS after export so the working project can be safely replaced.
CLOSE_ANSYS_AFTER_RUN = True

# Keep graphical mode enabled until automation has been fully validated.
RUN_NON_GRAPHICAL = False


# -----------------------------------------------------------------------------
# 6) Reliability settings
# -----------------------------------------------------------------------------
# Maximum allowed duration for one ANSYS/HFSS process = 80 min.
HFSS_TIMEOUT_SECONDS = 4800