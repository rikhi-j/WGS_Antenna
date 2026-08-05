# =============================================================================
# MODULE: ansys_s11_bridge.py
# PURPOSE:
# Provides the communication bridge between the Python optimization code and
# ANSYS Electronics Desktop (HFSS). This module automatically generates and
# executes IronPython scripts inside HFSS, exports simulation results, and reads
# the resulting S11 and gain CSV files back into Python.
#
# This module is responsible for automating the HFSS simulation process. It
# does not evaluate antenna performance or calculate optimization scores.
# Instead, it simply retrieves the simulation data that will later be scored by
# design_evaluator.py.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - csv_optimizer.py calls run_ansys_export() after a repaired antenna has been
#   converted into current_candidate.py.
# - This module generates the HFSS automation script.
# - HFSS executes current_candidate.py to modify the antenna geometry.
# - HFSS analyzes the modified antenna.
# - HFSS exports S11 and gain reports as CSV files.
# - This module reads those CSV files and returns the simulation results to the
#   optimization process.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Generate an IronPython automation script for HFSS.
# 2. Open the working HFSS project.
# 3. Execute current_candidate.py to modify the antenna geometry.
# 4. Run the HFSS electromagnetic simulation.
# 5. Export the S11 report to a CSV file.
# 6. Export the gain report to a CSV file.
# 7. Save and optionally close HFSS.
# 8. Read the exported CSV files.
# 9. Return the simulation results to the optimizer.
# -----------------------------------------------------------------------------
# WHAT THIS MODULE DOES:
# - Generates HFSS automation scripts.
# - Launches ANSYS Electronics Desktop.
# - Monitors HFSS execution.
# - Detects and handles simulation timeouts.
# - Parses exported S11 CSV files.
# - Parses exported gain CSV files.
# - Returns structured simulation data for later scoring.
# -----------------------------------------------------------------------------
# KEY DATA STORED:
# S11Result:
#     Stores the frequency sweep, S11 values, CSV information, and useful
#     properties such as minimum S11 and resonant frequency.
#
# GainResult:
#     Stores gain values, theta angles, CSV information, and useful properties
#     such as maximum gain and the angle where maximum gain occurs.
#
# HFSSTimeoutError:
#     Custom exception raised when an HFSS simulation exceeds the configured
#     timeout period.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - current_candidate.py geometry script.
# - HFSS project information from ansys_config.py.
# - Optional HFSS variable updates.
# - Generated S11 and gain report definitions.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Generated HFSS IronPython automation script.
# - Exported S11 CSV file.
# - Exported gain CSV file.
# - Parsed S11Result object.
# - Parsed GainResult object.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - This module only communicates with HFSS. It does not calculate antenna
#   performance scores.
# - The generated automation script executes current_candidate.py before running
#   the HFSS simulation so the latest optimized antenna geometry is applied.
# - CSV column names are automatically detected instead of assuming fixed column
#   positions, making the parser more tolerant of different HFSS report formats.
# - Frequency values are automatically converted to GHz and theta values are
#   converted to degrees when necessary.
# - If HFSS exceeds the configured timeout, the entire ANSYS process tree is
#   terminated to prevent the working project from remaining locked.
# =============================================================================

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import ansys_config as cfg

class HFSSTimeoutError(RuntimeError):
    """
    Raised only when an ANSYS/HFSS process exceeds the configured timeout.

    The optimizer may safely treat this as a rejected candidate and
    generate a replacement in the same particle slot.
    """


_NUMBER_RE = re.compile(
    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
)


@dataclass
class S11Result:
    frequency: np.ndarray
    s11: np.ndarray
    csv_path: str
    frequency_column: str
    s11_column: str

    @property
    def min_s11_db(self) -> float:
        return float(np.min(self.s11))

    @property
    def min_s11_index(self) -> int:
        return int(np.argmin(self.s11))

    @property
    def min_s11_frequency(self) -> float:
        return float(
            self.frequency[self.min_s11_index]
        )


@dataclass
class GainResult:
    theta: np.ndarray
    gain: np.ndarray
    csv_path: str
    theta_column: str
    gain_column: str
    frequency_ghz: float
    phi_deg: float

    @property
    def max_gain_db(self) -> float:
        return float(np.max(self.gain))

    @property
    def max_gain_index(self) -> int:
        return int(np.argmax(self.gain))

    @property
    def theta_of_max_gain(self) -> float:
        return float(
            self.theta[self.max_gain_index]
        )


def _require_config_is_edited() -> None:
    if (
        "CHANGE_ME" in cfg.PROJECT_PATH
        or "CHANGE_ME" in cfg.PROJECT_NAME
    ):
        raise ValueError(
            "Please edit ansys_config.py first."
        )


def _ironpython_literal(
    value: Any,
) -> str:
    if isinstance(value, str):
        return repr(value)

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return repr(float(value))

    raise TypeError(
        f"Unsupported HFSS variable value: {value!r}"
    )


def _variable_update_block(
    variable_name: str,
    value: Any,
) -> str:
    return """
oDesign.ChangeProperty(
    [
        "NAME:AllTabs",
        [
            "NAME:LocalVariableTab",
            ["NAME:PropServers", "LocalVariables"],
            [
                "NAME:ChangedProps",
                [
                    "NAME:%s",
                    "Value:=", %s
                ]
            ]
        ]
    ])
""".strip() % (
        str(variable_name),
        _ironpython_literal(value),
    )


def build_export_script(
    variables: Optional[
        Mapping[str, Any]
    ] = None,
) -> str:
    variables = variables or {}

    variable_blocks = "\n\n".join(
        _variable_update_block(
            name,
            value,
        )
        for name, value in variables.items()
    )

    candidate_script = getattr(
        cfg,
        "CURRENT_CANDIDATE_SCRIPT",
        None,
    )

    if not candidate_script:
        raise ValueError(
            "ansys_config.py must define "
            "CURRENT_CANDIDATE_SCRIPT."
        )

    analyze_line = (
        "oDesign.AnalyzeAll()"
        if cfg.ANALYZE_BEFORE_EXPORT
        else "# Analyze disabled"
    )

    save_line = (
        "oProject.Save()"
        if cfg.SAVE_PROJECT_AFTER_RUN
        else "# Save disabled"
    )

    close_line = (
        "oDesktop.QuitApplication()"
        if cfg.CLOSE_ANSYS_AFTER_RUN
        else "# Close disabled"
    )

    return """# Auto-generated by ansys_s11_bridge.py
# Runs inside Ansys Electronics Desktop / HFSS IronPython.

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()

oDesktop.OpenProject(%r)
oProject = oDesktop.SetActiveProject(%r)
oDesign = oProject.SetActiveDesign(%r)

%s

# Run the generated WGS candidate geometry script.
execfile(%r)

# Re-select the project and design after candidate generation.
oProject = oDesktop.SetActiveProject(%r)
oDesign = oProject.SetActiveDesign(%r)

%s

oModule = oDesign.GetModule("ReportSetup")

# Export the S11 frequency sweep.
oModule.ExportToFile(%r, %r, False)

# Export Gain Plot1:
# Theta versus dB(GainTotal) at 2.4 GHz and Phi = 0 degrees.
oModule.ExportToFile(%r, %r, False)

%s
%s
""" % (
        cfg.PROJECT_PATH,
        cfg.PROJECT_NAME,
        cfg.DESIGN_NAME,
        (
            variable_blocks
            if variable_blocks
            else "# No variable updates requested"
        ),
        candidate_script,
        cfg.PROJECT_NAME,
        cfg.DESIGN_NAME,
        analyze_line,
        cfg.REPORT_NAME,
        cfg.EXPORT_CSV,
        cfg.GAIN_REPORT_NAME,
        cfg.EXPORT_GAIN_CSV,
        save_line,
        close_line,
    )


def write_export_script(
    variables: Optional[
        Mapping[str, Any]
    ] = None,
) -> str:
    os.makedirs(
        os.path.dirname(
            cfg.GENERATED_HFSS_SCRIPT
        ),
        exist_ok=True,
    )

    script_text = build_export_script(
        variables=variables
    )

    with open(
        cfg.GENERATED_HFSS_SCRIPT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(script_text)

    return cfg.GENERATED_HFSS_SCRIPT


def run_ansys_script(
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """
    Launch ANSYS and run the generated automation script.

    If the timeout is reached, terminate the full Windows process tree
    so a solver child process cannot keep the working project locked.
    """
    _require_config_is_edited()

    command = [
        cfg.ANSYS_EXE
    ]

    if cfg.RUN_NON_GRAPHICAL:
        command.append(
            "-ng"
        )

    command.extend(
        [
            "-RunScript",
            cfg.GENERATED_HFSS_SCRIPT,
        ]
    )

    creation_flags = 0

    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
        )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )

    try:
        stdout, stderr = (
            process.communicate(
                timeout=timeout
            )
        )

    except subprocess.TimeoutExpired as exc:
        if os.name == "nt":
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            process.kill()

        try:
            stdout, stderr = (
                process.communicate(
                    timeout=30
                )
            )
        except subprocess.TimeoutExpired:
            process.kill()

            stdout, stderr = (
                process.communicate()
            )

        raise HFSSTimeoutError(
            "ANSYS/HFSS exceeded the configured "
            f"timeout of {timeout} seconds.\n"
            "The timed-out ANSYS process tree was "
            "terminated.\n"
            f"Generated script: "
            f"{cfg.GENERATED_HFSS_SCRIPT}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        ) from exc

    if process.returncode != 0:
        raise RuntimeError(
            "ANSYS/HFSS exited with an error.\n"
            f"Return code: {process.returncode}\n"
            f"Generated script: "
            f"{cfg.GENERATED_HFSS_SCRIPT}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )

def _parse_number(
    value: str,
) -> Optional[float]:
    match = _NUMBER_RE.search(
        str(value)
    )

    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def _read_csv_rows(
    csv_path: str,
) -> Tuple[
    List[str],
    List[List[str]],
]:
    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = [
            row
            for row in csv.reader(file)
            if row
        ]

    if len(rows) < 2:
        raise ValueError(
            "CSV must contain a header and at "
            f"least one data row: {csv_path}"
        )

    header = rows[0]
    data_rows = rows[1:]

    common_data_width = max(
        set(
            len(row)
            for row in data_rows
        ),
        key=[
            len(row)
            for row in data_rows
        ].count,
    )

    if (
        len(header) > common_data_width
        and common_data_width == 2
    ):
        header = [
            header[0],
            ",".join(header[1:]),
        ]

        data_rows = [
            row[:2]
            for row in data_rows
            if len(row) >= 2
        ]

    return header, data_rows


def _compact(
    value: str,
) -> str:
    return (
        str(value)
        .lower()
        .replace(" ", "")
    )


def _find_required_index(
    header: Sequence[str],
    tokens: Sequence[str],
    label: str,
) -> int:
    for index, name in enumerate(header):
        compact = _compact(name)

        if any(
            _compact(token) in compact
            for token in tokens
        ):
            return index

    raise ValueError(
        f"Could not find a recognizable "
        f"{label} column. "
        f"Header: {list(header)!r}"
    )


def _find_frequency_index(
    header: Sequence[str],
) -> int:
    return _find_required_index(
        header,
        ("freq", "frequency"),
        "frequency",
    )


def _find_theta_index(
    header: Sequence[str],
) -> int:
    return _find_required_index(
        header,
        ("theta",),
        "Theta",
    )


def _find_s11_index(
    header: Sequence[str],
    excluded_index: int,
) -> int:
    recognized_tokens = (
        "s(1,1)",
        "st(1,1)",
        "s11",
        "s_11",
        "db(s",
        "db(st",
    )

    for index, name in enumerate(header):
        if index == excluded_index:
            continue

        compact = _compact(name)

        if any(
            token in compact
            for token in recognized_tokens
        ):
            return index

    raise ValueError(
        "Could not find a recognizable "
        "S11 column. "
        f"Header: {list(header)!r}"
    )


def _find_gain_index(
    header: Sequence[str],
    excluded_index: int,
) -> int:
    for index, name in enumerate(header):
        if index == excluded_index:
            continue

        compact = _compact(name)

        if (
            "gain" in compact
            and "db(" in compact
        ):
            return index

    raise ValueError(
        "Could not find a recognizable "
        "dB gain column. Expected a "
        "dB(GainTotal)-style column. "
        f"Header: {list(header)!r}"
    )


def _frequency_scale_to_ghz(
    column_name: str,
) -> float:
    compact = _compact(column_name)

    if (
        "[ghz]" in compact
        or "(ghz)" in compact
    ):
        return 1.0

    if (
        "[mhz]" in compact
        or "(mhz)" in compact
    ):
        return 1e-3

    if (
        "[khz]" in compact
        or "(khz)" in compact
    ):
        return 1e-6

    if (
        "[hz]" in compact
        or "(hz)" in compact
    ):
        return 1e-9

    raise ValueError(
        "Frequency units are not explicit "
        f"in column {column_name!r}. "
        "Expected Hz, kHz, MHz, or GHz."
    )


def _theta_scale_to_degrees(
    column_name: str,
) -> float:
    compact = _compact(column_name)

    if (
        "deg" in compact
        or "degree" in compact
    ):
        return 1.0

    if "rad" in compact:
        return 180.0 / math.pi

    raise ValueError(
        "Theta units are not explicit in "
        f"column {column_name!r}. "
        "Expected degrees or radians."
    )


def read_s11_csv(
    csv_path: Optional[str] = None,
) -> S11Result:
    csv_path = (
        csv_path
        or cfg.EXPORT_CSV
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"S11 CSV was not found: "
            f"{csv_path}"
        )

    header, rows = _read_csv_rows(
        csv_path
    )

    freq_index = _find_frequency_index(
        header
    )

    s11_index = _find_s11_index(
        header,
        freq_index,
    )

    freq_scale = (
        _frequency_scale_to_ghz(
            str(header[freq_index])
        )
    )

    frequency_values: List[float] = []
    s11_values: List[float] = []

    for row in rows:
        if len(row) <= max(
            freq_index,
            s11_index,
        ):
            continue

        freq = _parse_number(
            row[freq_index]
        )

        s11 = _parse_number(
            row[s11_index]
        )

        if (
            freq is None
            or s11 is None
        ):
            continue

        frequency_values.append(
            freq * freq_scale
        )

        s11_values.append(s11)

    if not frequency_values:
        raise ValueError(
            "No numeric frequency/S11 "
            f"data could be parsed from "
            f"{csv_path}"
        )

    frequency = np.asarray(
        frequency_values,
        dtype=float,
    )

    s11 = np.asarray(
        s11_values,
        dtype=float,
    )

    if (
        not np.all(np.isfinite(frequency))
        or not np.all(np.isfinite(s11))
    ):
        raise ValueError(
            "S11 CSV contains non-finite "
            f"values: {csv_path}"
        )

    return S11Result(
        frequency=frequency,
        s11=s11,
        csv_path=csv_path,
        frequency_column=str(
            header[freq_index]
        ),
        s11_column=str(
            header[s11_index]
        ),
    )


def read_gain_csv(
    csv_path: Optional[str] = None,
) -> GainResult:
    csv_path = (
        csv_path
        or cfg.EXPORT_GAIN_CSV
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Gain CSV was not found: "
            f"{csv_path}"
        )

    header, rows = _read_csv_rows(
        csv_path
    )

    theta_index = _find_theta_index(
        header
    )

    gain_index = _find_gain_index(
        header,
        theta_index,
    )

    theta_scale = (
        _theta_scale_to_degrees(
            str(header[theta_index])
        )
    )

    theta_values: List[float] = []
    gain_values: List[float] = []

    for row in rows:
        if len(row) <= max(
            theta_index,
            gain_index,
        ):
            continue

        theta = _parse_number(
            row[theta_index]
        )

        gain = _parse_number(
            row[gain_index]
        )

        if (
            theta is None
            or gain is None
        ):
            continue

        theta_values.append(
            theta * theta_scale
        )

        gain_values.append(gain)

    if not theta_values:
        raise ValueError(
            "No numeric Theta/gain data "
            f"could be parsed from "
            f"{csv_path}"
        )

    theta = np.asarray(
        theta_values,
        dtype=float,
    )

    gain = np.asarray(
        gain_values,
        dtype=float,
    )

    if (
        not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(gain))
    ):
        raise ValueError(
            "Gain CSV contains non-finite "
            f"values: {csv_path}"
        )

    return GainResult(
        theta=theta,
        gain=gain,
        csv_path=csv_path,
        theta_column=str(
            header[theta_index]
        ),
        gain_column=str(
            header[gain_index]
        ),
        frequency_ghz=float(
            cfg.TARGET_FREQ_GHZ
        ),
        phi_deg=float(
            cfg.GAIN_PHI_DEG
        ),
    )


def run_ansys_export(
    variables: Optional[
        Mapping[str, Any]
    ] = None,
    timeout: Optional[float] = None,
) -> S11Result:
    write_export_script(
        variables=variables
    )

    run_ansys_script(
        timeout=timeout
    )

    return read_s11_csv(
        cfg.EXPORT_CSV
    )


def objective_from_s11(
    result: S11Result,
) -> float:
    return -result.min_s11_db


if __name__ == "__main__":
    script_path = write_export_script()

    print(
        "Generated HFSS export script:",
        script_path,
    )

    print(
        "Configured S11 CSV output:",
        cfg.EXPORT_CSV,
    )

    print(
        "Configured gain CSV output:",
        cfg.EXPORT_GAIN_CSV,
    )