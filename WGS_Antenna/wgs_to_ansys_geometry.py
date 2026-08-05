# =============================================================================
# MODULE: wgs_to_ansys_geometry.py
# PURPOSE:
# Converts a repaired WGS_Antenna object into an ANSYS Electronics Desktop
# (HFSS) IronPython script that modifies the antenna geometry inside the HFSS
# project.
#
# The current optimization workflow uses a cutout-based approach rather than
# creating a completely new antenna for every simulation. The master HFSS
# project already contains fully conductive antenna sheets. This module
# determines which regions should remain metal and which regions should be
# removed, then generates the ANSYS commands needed to perform those cutout
# operations.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - optimization_controller.py calls write_wgs_cutout_script().
# - The generated script becomes current_candidate.py.
# - ansys_s11_bridge.py later executes current_candidate.py inside HFSS before
#   each simulation.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Read the repaired WGS_Antenna conductor grid.
# 2. Determine which conductor cells should remain metal and which should become
#    cutouts.
# 3. Create a small rectangular box for every cutout cell.
# 4. Merge neighboring cutout boxes on each antenna face into larger polygons.
# 5. Convert every merged polygon into its correct 3D coordinates.
# 6. Generate IronPython commands that create temporary cutout sheets.
# 7. Subtract the temporary sheets from the existing HFSS conductor layers.
# 8. Save the generated IronPython script for execution by HFSS.
# -----------------------------------------------------------------------------
# HOW THE CUTOUT PROCESS WORKS:
# The master HFSS project begins with fully conductive antenna surfaces.
#
# Instead of rebuilding the antenna geometry for every particle, this module
# removes conductor from locations where the WGS conductor grid contains False
# (empty) cells.
#
# Example:
#
#     True   = Keep conductor.
#     False  = Remove conductor.
#
# Every False cell is first represented as a small rectangular box on its
# corresponding antenna face. Touching boxes on the same face are then merged
# together using Shapely's polygon-union operation (unary_union), producing one
# or more larger cutout polygons instead of hundreds of individual rectangles.
#
# Each merged polygon is converted into a temporary HFSS sheet located on the
# correct 3D face of the antenna. Finally, those temporary sheets are subtracted
# from the existing conductor layer, removing all of the required conductor in
# one operation.
#
# This greatly reduces the number of HFSS objects and subtraction operations,
# making the optimization significantly faster than rebuilding the complete
# antenna geometry for every simulation.
# -----------------------------------------------------------------------------
# KEY DATA STORED:
# CUT_TARGET_SHEETS:
#     Maps each WGS antenna face to the corresponding conductor sheet already
#     present in the HFSS project.
#
# Generated polygons:
#     Merged conductor or cutout regions represented as Shapely polygons before
#     conversion into HFSS geometry.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - Repaired WGS_Antenna objects.
# - WGS conductor grid.
# - Existing HFSS conductor-sheet names.
# - ANSYS project configuration from ansys_config.py.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Generated IronPython geometry scripts.
# - current_candidate.py used during HFSS simulation.
# - Temporary cutout-sheet definitions.
# - HFSS subtraction commands.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - The production workflow uses build_wgs_cutout_script().
# - build_wgs_geometry_script() is an older geometry-generation method retained
#   for debugging and development.
# - Only the four main conductor sheets (Top, Bottom, Left, and Right) are
#   modified during cutout generation.
# - Neighboring cutout cells are merged into larger polygons to minimize the
#   number of HFSS objects and subtraction operations.
# - Every merged polygon is converted from the unfolded WGS representation back
#   into its correct 3D coordinates before being written to the ANSYS script.
# =============================================================================

from __future__ import annotations

from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import re
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import ansys_config as cfg

try:
    from validator_updated_only import validate_wgs_particle, summarize_validation_report
except Exception:  # pragma: no cover - lets this helper still write flat polygons
    validate_wgs_particle = None
    summarize_validation_report = None


Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]

CUT_TARGET_SHEETS = {
    "TF": "Top_Layer",
    "BF": "Bottom_Layer",
    "LS": "yneg_side",
    "RS": "ypos_side",
}

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def _require_config_is_edited() -> None:
    """Catch the most common mistake before launching ANSYS."""
    if "CHANGE_ME" in cfg.PROJECT_PATH or "CHANGE_ME" in cfg.PROJECT_NAME:
        raise ValueError(
            "Please edit ansys_config.py first. Set PROJECT_PATH and PROJECT_NAME "
            "to your actual HFSS .aedt project."
        )


def _safe_hfss_name(text: str) -> str:
    """Make a string safe as an HFSS object name."""
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(text))
    if not text:
        return "obj"
    if text[0].isdigit():
        text = "obj_" + text
    return text


def _hfss_length(value_m: float, unit: str = "mm") -> str:
    """
    Convert a Python length in meters to an HFSS length string.

    Your WGS code stores sizes in meters, so unit='mm' turns 0.0005 into '0.5mm'.
    """
    unit = unit.lower()
    scale_by_unit = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "mm": 1.0e3,
        "millimeter": 1.0e3,
        "millimeters": 1.0e3,
        "um": 1.0e6,
        "micron": 1.0e6,
        "microns": 1.0e6,
    }

    if unit not in scale_by_unit:
        raise ValueError(f"Unsupported unit {unit!r}. Use 'm', 'mm', or 'um'.")

    suffix = "m" if unit in ("m", "meter", "meters") else "mm" if unit.startswith("milli") or unit == "mm" else "um"
    scaled = float(value_m) * scale_by_unit[unit]

    if abs(scaled) < 1e-15:
        scaled = 0.0

    return f"{scaled:.12g}{suffix}"


def _hfss_point_block(point: Point3D, unit: str) -> str:
    x, y, z = point
    return (
        '["NAME:PLPoint", '
        '"X:=", %r, '
        '"Y:=", %r, '
        '"Z:=", %r]'
    ) % (_hfss_length(x, unit), _hfss_length(y, unit), _hfss_length(z, unit))


def _hfss_segment_block(start_index: int) -> str:
    return (
        '["NAME:PLSegment", '
        '"SegmentType:=", "Line", '
        '"StartIndex:=", %d, '
        '"NoOfPoints:=", 2]'
    ) % start_index


def _build_polyline_sheet_command(
    name: str,
    points: Sequence[Point3D],
    *,
    unit: str = "mm",
    material: str = "copper",
    color: str = "(255 128 0)",
    transparency: float = 0.0,
) -> str:
    """Return AEDT/IronPython code that creates one closed covered polyline sheet."""
    if len(points) < 3:
        raise ValueError("A sheet polygon needs at least 3 points.")

    clean_name = _safe_hfss_name(name)
    point_blocks = ",\n            ".join(_hfss_point_block(p, unit) for p in points)
    segment_blocks = ",\n            ".join(_hfss_segment_block(i) for i in range(len(points)))

    # AEDT expects MaterialValue to include quotation marks inside the string.
    material_value = '\\"%s\\"' % material

    return '''oEditor.CreatePolyline(
    [
        "NAME:PolylineParameters",
        "IsPolylineCovered:=", True,
        "IsPolylineClosed:=", True,
        [
            "NAME:PolylinePoints",
            %s
        ],
        [
            "NAME:PolylineSegments",
            %s
        ],
        [
            "NAME:PolylineXSection",
            "XSectionType:=", "None",
            "XSectionOrient:=", "Auto",
            "XSectionWidth:=", "0mm",
            "XSectionTopWidth:=", "0mm",
            "XSectionHeight:=", "0mm",
            "XSectionNumSegments:=", "0",
            "XSectionBendType:=", "Corner"
        ]
    ],
    [
        "NAME:Attributes",
        "Name:=", %r,
        "Flags:=", "",
        "Color:=", %r,
        "Transparency:=", %s,
        "PartCoordinateSystem:=", "Global",
        "UDMId:=", "",
        "MaterialValue:=", "%s",
        "SolveInside:=", False,
        "ShellElement:=", False,
        "ShellElementThickness:=", "0mm",
        "IsMaterialEditable:=", True,
        "UseMaterialAppearance:=", False
    ])''' % (
        point_blocks,
        segment_blocks,
        clean_name,
        color,
        repr(float(transparency)),
        material_value,
    )


def _script_header() -> str:
    return '''# Auto-generated by wgs_to_ansys_geometry.py
# This file runs inside ANSYS Electronics Desktop / HFSS IronPython.

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()

# Project is already open in HFSS.
oProject = oDesktop.SetActiveProject(%r)
oDesign = oProject.SetActiveDesign(%r)
oEditor = oDesign.SetActiveEditor("3D Modeler")
''' % (
        cfg.PROJECT_NAME,
        cfg.DESIGN_NAME,
    )

def _clear_existing_block(object_prefix: str) -> str:
    pattern = _safe_hfss_name(object_prefix) + "*"
    return '''
# Delete old generated WGS objects with the same prefix.
try:
    old_objects = list(oEditor.GetMatchedObjectName(%r))
    if old_objects:
        oEditor.Delete(["NAME:Selections", "Selections:=", ",".join(old_objects)])
except Exception as exc:
    print("Could not delete old generated objects:", exc)
''' % pattern


def _script_footer(save_project: bool = True, close_ansys: bool = False) -> str:
    lines = [""]
    lines.append("oProject.Save()" if save_project else "# Save disabled")
    lines.append("oDesktop.QuitApplication()" if close_ansys else "# Close disabled")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# WGS_Antenna -> real 3D HFSS sheet rectangles
# -----------------------------------------------------------------------------


def _looks_like_wgs_particle(particle: Any) -> bool:
    return all(hasattr(particle, name) for name in ("map_grid", "conductor_grid", "a", "b", "c", "A", "B", "C"))


def wgs_cell_corners_3d(particle: Any, i: int, j: int) -> List[Point3D]:
    """
    Return the four 3D corners for one WGS cell in meters.

    This mirrors the geometry used by WGS_Antenna.__make_segment(), but it does
    not call the private method, so it can be used from this add-on file.
    """
    if not _looks_like_wgs_particle(particle):
        raise TypeError("wgs_cell_corners_3d() expects a WGS_Antenna-like object.")

    a = float(particle.a)
    b = float(particle.b)
    c = float(particle.c)
    A = int(particle.A)
    B = int(particle.B)
    C = int(particle.C)

    da = a / A
    db = b / B
    dc = c / C

    side = particle.map_grid[i][j]

    if side == "NC":
        x0 = c / 2.0
        return [
            (x0, a / 2.0 - da * j,     -b / 2.0 + db * i),
            (x0, a / 2.0 - da * (j+1), -b / 2.0 + db * i),
            (x0, a / 2.0 - da * (j+1), -b / 2.0 + db * (i+1)),
            (x0, a / 2.0 - da * j,     -b / 2.0 + db * (i+1)),
        ]

    if side == "FC":
        x0 = -c / 2.0
        ii = i - (B + C)
        return [
            (x0, a / 2.0 - da * j,     b / 2.0 - db * ii),
            (x0, a / 2.0 - da * (j+1), b / 2.0 - db * ii),
            (x0, a / 2.0 - da * (j+1), b / 2.0 - db * (ii+1)),
            (x0, a / 2.0 - da * j,     b / 2.0 - db * (ii+1)),
        ]

    if side == "TF":
        z0 = b / 2.0
        ii = i - B
        return [
            (c / 2.0 - dc * ii,     a / 2.0 - da * j,     z0),
            (c / 2.0 - dc * (ii+1), a / 2.0 - da * j,     z0),
            (c / 2.0 - dc * (ii+1), a / 2.0 - da * (j+1), z0),
            (c / 2.0 - dc * ii,     a / 2.0 - da * (j+1), z0),
        ]

    if side == "BF":
        z0 = -b / 2.0
        ii = i - B
        jj = j - (A + B)
        return [
            (c / 2.0 - dc * ii,     -a / 2.0 + da * jj,     z0),
            (c / 2.0 - dc * (ii+1), -a / 2.0 + da * jj,     z0),
            (c / 2.0 - dc * (ii+1), -a / 2.0 + da * (jj+1), z0),
            (c / 2.0 - dc * ii,     -a / 2.0 + da * (jj+1), z0),
        ]

    if side == "LS":
        y0 = -a / 2.0
        ii = i - B
        jj = j - A
        return [
            (c / 2.0 - dc * ii,     y0, b / 2.0 - db * jj),
            (c / 2.0 - dc * (ii+1), y0, b / 2.0 - db * jj),
            (c / 2.0 - dc * (ii+1), y0, b / 2.0 - db * (jj+1)),
            (c / 2.0 - dc * ii,     y0, b / 2.0 - db * (jj+1)),
        ]

    if side == "RS":
        y0 = a / 2.0
        ii = i - B
        jj = j - (2*A + B)
        return [
            (c / 2.0 - dc * ii,     y0, -b / 2.0 + db * jj),
            (c / 2.0 - dc * (ii+1), y0, -b / 2.0 + db * jj),
            (c / 2.0 - dc * (ii+1), y0, -b / 2.0 + db * (jj+1)),
            (c / 2.0 - dc * ii,     y0, -b / 2.0 + db * (jj+1)),
        ]

    raise ValueError(f"Cell {(i, j)} has invalid side {side!r}; expected NC/FC/TF/BF/LS/RS.")


def iter_wgs_metal_cells(particle: Any) -> Iterable[Tuple[int, int, str, List[Point3D]]]:
    """Yield (map_i, map_j, side_name, four_3d_corners) for each metal WGS cell."""
    if not _looks_like_wgs_particle(particle):
        raise TypeError("iter_wgs_metal_cells() expects a WGS_Antenna-like object.")

    for i, side_row in enumerate(particle.map_grid):
        conductor_j = 0

        for j, side in enumerate(side_row):
            if side == "X":
                continue

            is_metal = bool(particle.conductor_grid[i][conductor_j])
            conductor_j += 1

            if is_metal:
                yield i, j, str(side), wgs_cell_corners_3d(particle, i, j)

def iter_wgs_cutout_cells(particle: Any):
    """
    Yield cells that should be cut out.

    True  = keep metal
    False = cut hole
    Only cuts the four main sheets:
        TF, BF, LS, RS
    """

    for i, side_row in enumerate(particle.map_grid):
        conductor_j = 0

        for j, side in enumerate(side_row):
            if side == "X":
                continue

            keep_metal = bool(particle.conductor_grid[i][conductor_j])
            conductor_j += 1

            if side not in CUT_TARGET_SHEETS:
                continue

            if not keep_metal:
                yield i, j, str(side), wgs_cell_corners_3d(particle, i, j)

        
def _local_cell_index(particle, i, j, side):
    A = int(particle.A)
    B = int(particle.B)
    C = int(particle.C)

    if side == "NC":
        return i, j

    if side == "FC":
        return i - (B + C), j

    if side == "TF":
        return i - B, j

    if side == "BF":
        return i - B, j - (A + B)

    if side == "LS":
        return i - B, j - A

    if side == "RS":
        return i - B, j - (2 * A + B)

    raise ValueError(f"Unknown side: {side}")


def _face_vertex_to_3d(particle, side, x, y):
    a = float(particle.a)
    b = float(particle.b)
    c = float(particle.c)

    A = int(particle.A)
    B = int(particle.B)
    C = int(particle.C)

    da = a / A
    db = b / B
    dc = c / C

    if side == "NC":
        return (c / 2.0, a / 2.0 - da * x, -b / 2.0 + db * y)

    if side == "FC":
        return (-c / 2.0, a / 2.0 - da * x, b / 2.0 - db * y)

    if side == "TF":
        return (c / 2.0 - dc * y, a / 2.0 - da * x, b / 2.0)

    if side == "BF":
        return (c / 2.0 - dc * y, -a / 2.0 + da * x, -b / 2.0)

    if side == "LS":
        return (c / 2.0 - dc * y, -a / 2.0, b / 2.0 - db * x)

    if side == "RS":
        return (c / 2.0 - dc * y, a / 2.0, -b / 2.0 + db * x)

    raise ValueError(f"Unknown side: {side}")


def iter_wgs_merged_face_polygons(particle):
    """
    Yield merged metal polygons per box face.

    Instead of:
        one metal cell -> one HFSS sheet

    this does:
        touching metal cells on same face -> one merged HFSS polygon
    """

    face_boxes = {
        "NC": [],
        "FC": [],
        "TF": [],
        "BF": [],
        "LS": [],
        "RS": [],
    }

    for i, j, side, corners in iter_wgs_metal_cells(particle):
        row, col = _local_cell_index(particle, i, j, side)

        # shapely box uses x=column, y=row
        face_boxes[side].append(
            box(col, row, col + 1, row + 1)
        )

    for side, boxes in face_boxes.items():
        if not boxes:
            continue

        merged = unary_union(boxes)

        if isinstance(merged, Polygon):
            polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            polygons = list(merged.geoms)
        else:
            continue

        for poly in polygons:
            exterior = list(poly.exterior.coords)

            # remove repeated closing point
            if exterior[0] == exterior[-1]:
                exterior = exterior[:-1]

            points_3d = [
                _face_vertex_to_3d(particle, side, x, y)
                for x, y in exterior
            ]

            yield side, points_3d

def iter_wgs_merged_cutout_polygons(particle):
    """
    Yield merged cutout polygons per box face.

    True  = keep metal
    False = cut hole

    Touching False cells on the same face become one merged polygon.
    """

    face_boxes = {
        "TF": [],
        "BF": [],
        "LS": [],
        "RS": [],
    }

    for i, j, side, corners in iter_wgs_cutout_cells(particle):
        row, col = _local_cell_index(particle, i, j, side)

        # shapely box uses x=column, y=row
        face_boxes[side].append(
            box(col, row, col + 1, row + 1)
        )

    for side, boxes in face_boxes.items():
        if not boxes:
            continue

        merged = unary_union(boxes)

        if isinstance(merged, Polygon):
            polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            polygons = list(merged.geoms)
        else:
            continue

        for poly in polygons:
            exterior = list(poly.exterior.coords)

            # remove repeated closing point
            if exterior[0] == exterior[-1]:
                exterior = exterior[:-1]

            points_3d = [
                _face_vertex_to_3d(particle, side, x, y)
                for x, y in exterior
            ]

            yield side, points_3d

def build_wgs_geometry_script(
    particle: Any,
    *,
    object_prefix: str = "WGS",
    unit: str = "mm",
    material: str = "copper",
    clear_existing: bool = True,
    validate_geometry: bool = True,
    save_project: Optional[bool] = None,
    close_ansys: bool = False,
) -> str:
    """
    Build an AEDT/IronPython script that creates WGS metal cells in HFSS.

    Parameters
    ----------
    particle:
        WGS_Antenna-like object.
    object_prefix:
        Prefix used for generated HFSS object names. Old objects with this prefix
        are deleted when clear_existing=True.
    unit:
        HFSS display unit for coordinates. Your particle is still assumed meters.
    material:
        HFSS material name, usually 'copper' or 'pec'.
    validate_geometry:
        If True, calls validator_updated_only.validate_wgs_particle() first and
        refuses to export invalid geometry.
    close_ansys:
        Default False so you can inspect the generated geometry in the GUI.
    """
    if not _looks_like_wgs_particle(particle):
        raise TypeError("build_wgs_geometry_script() expects a WGS_Antenna-like object.")

    if save_project is None:
        save_project = bool(getattr(cfg, "SAVE_PROJECT_AFTER_RUN", True))

    if validate_geometry:
        if validate_wgs_particle is None or summarize_validation_report is None:
            raise RuntimeError("validator_updated_only.py could not be imported, so validation is unavailable.")

        report = validate_wgs_particle(particle)
        summary = summarize_validation_report(report)

        if not report.get("is_valid", False):
            raise ValueError(
                "WGS particle is invalid; not exporting to HFSS. "
                f"Errors={summary.get('errors')}. "
                "Repair the particle first, or call write_wgs_geometry_script(..., validate_geometry=False) for debugging only."
            )

    polygons = list(iter_wgs_merged_face_polygons(particle))
    if not polygons:
        raise ValueError("The WGS particle has no metal polygons to export.")

    prefix = _safe_hfss_name(object_prefix)
    commands: List[str] = []

    for k, (side, points) in enumerate(polygons):
        name = f"{prefix}_{k:05d}_{side}_merged"
        commands.append(
            _build_polyline_sheet_command(
                name,
                points,
                unit=unit,
                material=material,
            )
        )

    script = _script_header()

    if clear_existing:
        script += _clear_existing_block(prefix)

    script += "\n# Create %d merged WGS face polygons.\n" % len(polygons)
    script += "\n\n".join(commands)
    script += _script_footer(save_project=save_project, close_ansys=close_ansys)

    return script

def build_wgs_cutout_script(
    particle: Any,
    *,
    object_prefix: str = "WGS_CUT",
    unit: str = "mm",
    material: str = "copper",
    clear_existing: bool = True,
    save_project: Optional[bool] = None,
    close_ansys: bool = False,
) -> str:
    """
    Build HFSS script that cuts holes out of existing sheet faces.

    True cells are kept.
    False cells are removed from existing HFSS sheets.
    """

    if save_project is None:
        save_project = bool(getattr(cfg, "SAVE_PROJECT_AFTER_RUN", True))

    prefix = _safe_hfss_name(object_prefix)

    cut_polygons = list(iter_wgs_merged_cutout_polygons(particle))

    if not cut_polygons:
        raise ValueError("No cutout polygons found.")
    
    script = _script_header()

    if clear_existing:
        script += _clear_existing_block(prefix)

    commands = []

    cutters_by_target = {}

    for k, (side, points) in enumerate(cut_polygons):
        target_sheet = CUT_TARGET_SHEETS[side]
        cutter_name = f"{prefix}_{k:05d}_{side}_merged_cutout"

        commands.append(
            _build_polyline_sheet_command(
                cutter_name,
                points,
                unit=unit,
                material=material,
                color="(255 0 0)",
                transparency=0.6,
            )
        )

        cutters_by_target.setdefault(target_sheet, []).append(_safe_hfss_name(cutter_name))
    script += "\n# Create temporary cutout sheets.\n"
    script += "\n\n".join(commands)

    script += "\n\n# Subtract cutouts from existing antenna sheets.\n"

    for target_sheet, cutters in cutters_by_target.items():
        script += f'''
oEditor.Subtract(
    [
        "NAME:Selections",
        "Blank Parts:=", "{target_sheet}",
        "Tool Parts:=", "{",".join(cutters)}"
    ],
    [
        "NAME:SubtractParameters",
        "KeepOriginals:=", False
    ])
'''

    script += _script_footer(save_project=save_project, close_ansys=close_ansys)

    return script


def write_wgs_geometry_script(
    particle: Any,
    script_path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Write the WGS -> HFSS geometry script and return the path."""
    if script_path is None:
        script_path = os.path.join(cfg.EXPORT_DIR, "generated_wgs_geometry.py")

    os.makedirs(os.path.dirname(os.path.abspath(script_path)), exist_ok=True)
    script_text = build_wgs_geometry_script(particle, **kwargs)

    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script_text)

    return script_path

def write_wgs_cutout_script(
    particle: Any,
    script_path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    if script_path is None:
        script_path = os.path.join(cfg.EXPORT_DIR, "generated_wgs_cutouts.py")

    os.makedirs(os.path.dirname(os.path.abspath(script_path)), exist_ok=True)

    script_text = build_wgs_cutout_script(particle, **kwargs)

    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script_text)

    return script_path

def run_ansys_geometry_script(script_path: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Launch ANSYS Electronics Desktop and run a generated geometry script."""
    _require_config_is_edited()

    command = [cfg.ANSYS_EXE]
    if bool(getattr(cfg, "RUN_NON_GRAPHICAL", False)):
        command.append("-ng")
    command.extend(["-RunScript", script_path])

    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def export_wgs_particle_to_ansys(
    particle: Any,
    *,
    run: bool = True,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> str:
    """
    Write the geometry script, optionally run ANSYS, and return the script path.

    Set run=False when you only want to inspect the generated IronPython file.
    """
    script_path = write_wgs_geometry_script(particle, **kwargs)

    if run:
        run_ansys_geometry_script(script_path, timeout=timeout)

    return script_path


# -----------------------------------------------------------------------------
# Flat polygon list -> XY-plane HFSS sheet polygons
# -----------------------------------------------------------------------------


def _polygon_vertices(poly: Any) -> List[Point2D]:
    """Accept either {'vertices': [...]} dictionaries or raw [(x, y), ...] lists."""
    if isinstance(poly, Mapping):
        vertices = poly.get("vertices", [])
    else:
        vertices = poly

    clean: List[Point2D] = []
    for point in vertices:
        if len(point) < 2:
            raise ValueError(f"Bad polygon vertex {point!r}; expected at least x,y.")
        clean.append((float(point[0]), float(point[1])))

    if len(clean) >= 2 and clean[0] == clean[-1]:
        clean = clean[:-1]

    if len(clean) < 3:
        raise ValueError("Polygon must contain at least 3 unique vertices.")

    return clean


def build_flat_polygon_script(
    polygons: Sequence[Any],
    *,
    z: float = 0.0,
    object_prefix: str = "WGS_flat_poly",
    unit: str = "mm",
    material: str = "copper",
    clear_existing: bool = True,
    save_project: Optional[bool] = None,
    close_ansys: bool = False,
) -> str:
    """
    Build a script that exports flat 2D polygon loops to HFSS in the XY plane.

    This is useful for debugging polygon outlines from grid_or_wgs_to_polygons().
    It does NOT fold the unfolded WGS polygon back onto the 3D antenna box. For
    real antenna geometry, use build_wgs_geometry_script() instead.
    """
    if save_project is None:
        save_project = bool(getattr(cfg, "SAVE_PROJECT_AFTER_RUN", True))

    prefix = _safe_hfss_name(object_prefix)
    script = _script_header()

    if clear_existing:
        script += _clear_existing_block(prefix)

    commands: List[str] = []
    for k, poly in enumerate(polygons):
        vertices2d = _polygon_vertices(poly)
        points3d: List[Point3D] = [(x, y, float(z)) for x, y in vertices2d]

        poly_type = "poly"
        if isinstance(poly, Mapping):
            poly_type = str(poly.get("type", "poly"))

        name = f"{prefix}_{k:04d}_{poly_type}"
        commands.append(
            _build_polyline_sheet_command(
                name,
                points3d,
                unit=unit,
                material=material,
            )
        )

    script += "\n# Create %d flat polygon sheet loops.\n" % len(commands)
    script += "\n\n".join(commands)
    script += _script_footer(save_project=save_project, close_ansys=close_ansys)

    return script


def write_flat_polygon_script(
    polygons: Sequence[Any],
    script_path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Write a flat polygon -> HFSS script and return the path."""
    if script_path is None:
        script_path = os.path.join(cfg.EXPORT_DIR, "generated_flat_polygons.py")

    os.makedirs(os.path.dirname(os.path.abspath(script_path)), exist_ok=True)
    script_text = build_flat_polygon_script(polygons, **kwargs)

    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script_text)

    return script_path
