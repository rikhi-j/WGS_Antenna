# =============================================================================
# MODULE: validator_updated_only.py
# PURPOSE:
# Validates WGS antenna designs to determine whether they satisfy all geometry
# requirements before being exported to HFSS. This module defines what is
# considered a valid antenna by checking conductor connectivity, isolated cells,
# illegal diagonal connections, and overall conductor topology.
#
# It also provides helper functions for converting WGS antennas into
# rectangular grids and polygon outlines used by other parts of the project.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - optimization_controller.py validates every repaired antenna before it is
#   exported to HFSS.
# - wgs_illegal_handler.py uses this module to convert WGS antennas into
#   rectangular grids and verify illegal-diagonal repairs.
# - wgs_full_repair_handler.py repeatedly validates the antenna after each
#   repair iteration until a valid design is produced.
# - wgs_to_ansys_geometry.py uses the polygon-generation functions to create
#   merged conductor outlines for ANSYS geometry generation.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Convert a WGS antenna into an unfolded rectangular grid when required.
# 2. Identify every conductor cell within the antenna.
# 3. Determine each cell's physical neighbors using the WGS neighbor map.
# 4. Detect isolated conductor cells.
# 5. Detect illegal diagonal (checkerboard) conductor contacts.
# 6. Find all physically connected conductor components.
# 7. Determine whether the antenna contains multiple disconnected components.
# 8. Generate a validation report describing every detected issue.
# 9. Optionally convert the validated conductor layout into merged polygon
#    outlines for geometry export.
# -----------------------------------------------------------------------------
# WHAT MAKES AN ANTENNA VALID?
# A valid antenna must satisfy all of the following conditions:
#
# - Every conductor cell must be connected to at least one neighboring
#   conductor cell.
#
# - Conductor cells may only connect through shared edges. Diagonal
#   corner-to-corner conductor connections are not allowed.
#
# - The conductor should form one physically connected component unless
#   multiple components are intentionally permitted.
#
# - Only valid WGS cells are considered during validation. Placeholder ("X")
#   cells in the unfolded grid are ignored.
# -----------------------------------------------------------------------------
# KEY DATA STORED:
# Validation reports contain:
# - Overall validity.
# - Validation errors.
# - Isolated conductor cells.
# - Illegal diagonal checkerboard locations.
# - Number of conductor components.
# - Lists of disconnected conductor components.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - Rectangular binary conductor grids.
# - WGS_Antenna objects.
# - Optional validation settings controlling connectivity requirements.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Validation reports.
# - Connected conductor components.
# - Lists of isolated conductor cells.
# - Lists of illegal diagonal contacts.
# - Polygon outlines generated from conductor geometry.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - Physical conductor connectivity is determined using the WGS neighbor map,
#   not simple row/column neighbors, because opposite edges of the unfolded net
#   may actually be adjacent on the folded 3D antenna.
# - Illegal diagonal (checkerboard) detection is intentionally performed on the
#   unfolded rectangular grid because diagonal corner-touch rules are strictly a
#   2D layout constraint.
# - This module only detects geometry problems. It does not repair them. Repair
#   operations are performed by wgs_illegal_handler.py and
#   wgs_full_repair_handler.py.
# - The validation report generated here is used throughout the project to
#   determine whether an antenna is safe to export and simulate in HFSS.
# =============================================================================

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, Any

Grid = List[List[int]]
Cell = Tuple[int, int]
Point = Tuple[float, float]
IntPoint = Tuple[int, int]
Edge = Tuple[IntPoint, IntPoint]


# ---------------------------------------------------------------------------
# Rectangular 2D-grid helpers
# ---------------------------------------------------------------------------

def check_grid_shape(grid: Sequence[Sequence[Union[int, bool]]]) -> None:
    if not grid or not grid[0]:
        raise ValueError("Grid cannot be empty.")

    width = len(grid[0])

    for row in grid:
        if len(row) != width:
            raise ValueError(
                "All rows must have the same length. "
                "For WGS_Antenna objects, call validate_wgs_particle() or "
                "wgs_to_rect_grid() first."
            )

        for value in row:
            if value not in (0, 1, False, True):
                raise ValueError("Grid values must only be 0 or 1 / False or True.")


def in_bounds(grid: Sequence[Sequence[Union[int, bool]]], r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def is_metal(grid: Sequence[Sequence[Union[int, bool]]], r: int, c: int) -> bool:
    return in_bounds(grid, r, c) and bool(grid[r][c])


def four_neighbors(r: int, c: int) -> List[Cell]:
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]


# ---------------------------------------------------------------------------
# Rectangular 2D-grid validation
# ---------------------------------------------------------------------------

def find_isolated_metal_cells(grid: Grid) -> List[Cell]:
    """Find metal cells that have no edge-connected metal neighbor."""
    check_grid_shape(grid)

    isolated: List[Cell] = []

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if bool(grid[r][c]):
                has_edge_neighbor = any(is_metal(grid, nr, nc) for nr, nc in four_neighbors(r, c))
                if not has_edge_neighbor:
                    isolated.append((r, c))

    return isolated


def find_illegal_diagonal_touches(grid: Grid) -> List[Dict[str, object]]:
    """
    Find illegal 2x2 cases like 1/0 diagonally touching 0/1.
    This is a flat rectangular-grid check.
    """
    check_grid_shape(grid)

    bad_blocks: List[Dict[str, object]] = []
    rows = len(grid)
    cols = len(grid[0])

    for r in range(rows - 1):
        for c in range(cols - 1):
            top_left = int(bool(grid[r][c]))
            top_right = int(bool(grid[r][c + 1]))
            bottom_left = int(bool(grid[r + 1][c]))
            bottom_right = int(bool(grid[r + 1][c + 1]))

            if top_left == 1 and bottom_right == 1 and top_right == 0 and bottom_left == 0:
                bad_blocks.append({
                    "block_top_left": (r, c),
                    "metal_cells": [(r, c), (r + 1, c + 1)],
                })

            if top_right == 1 and bottom_left == 1 and top_left == 0 and bottom_right == 0:
                bad_blocks.append({
                    "block_top_left": (r, c),
                    "metal_cells": [(r, c + 1), (r + 1, c)],
                })

    return bad_blocks


def find_metal_components(grid: Grid) -> List[Set[Cell]]:
    """Find edge-connected metal groups in a rectangular grid."""
    check_grid_shape(grid)

    visited: Set[Cell] = set()
    components: List[Set[Cell]] = []

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if bool(grid[r][c]) and (r, c) not in visited:
                component: Set[Cell] = set()
                queue: deque[Cell] = deque([(r, c)])
                visited.add((r, c))

                while queue:
                    current_r, current_c = queue.popleft()
                    component.add((current_r, current_c))

                    for nr, nc in four_neighbors(current_r, current_c):
                        if is_metal(grid, nr, nc) and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

                components.append(component)

    return components


def validate_waffle_grid(grid: Grid, require_single_component: bool = True) -> Dict[str, object]:
    """Validate an ordinary rectangular binary grid."""
    isolated_cells = find_isolated_metal_cells(grid)
    diagonal_touches = find_illegal_diagonal_touches(grid)
    components = find_metal_components(grid)

    disconnected_components: List[Set[Cell]] = []

    if require_single_component and len(components) > 1:
        main_component = max(components, key=len)
        disconnected_components = [comp for comp in components if comp is not main_component]

    errors: List[str] = []

    if isolated_cells:
        errors.append("Isolated metal cells found.")

    if diagonal_touches:
        errors.append("Illegal diagonal corner touches found.")

    if disconnected_components:
        errors.append("Disconnected metal components found.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "isolated_cells": isolated_cells,
        "diagonal_touches": diagonal_touches,
        "num_metal_components": len(components),
        "disconnected_components": disconnected_components,
    }


# ---------------------------------------------------------------------------
# Rectangular 2D polygon generation
# ---------------------------------------------------------------------------

def add_boundary_edges_for_cell(grid: Grid, r: int, c: int, edges: List[Edge]) -> None:
    """
    Add only the outside edges of a metal cell.

    Coordinate system:
    - row 0 is the top row of the grid
    - polygon coordinates use origin at bottom-left
    """
    rows = len(grid)

    x0 = c
    x1 = c + 1
    y0 = rows - r - 1
    y1 = rows - r

    if not is_metal(grid, r + 1, c):
        edges.append(((x0, y0), (x1, y0)))

    if not is_metal(grid, r, c + 1):
        edges.append(((x1, y0), (x1, y1)))

    if not is_metal(grid, r - 1, c):
        edges.append(((x1, y1), (x0, y1)))

    if not is_metal(grid, r, c - 1):
        edges.append(((x0, y1), (x0, y0)))


def trace_boundary_loops(edges: List[Edge]) -> List[List[IntPoint]]:
    """Convert boundary edges into closed polygon loops."""
    outgoing: Dict[IntPoint, List[IntPoint]] = defaultdict(list)

    for start, end in edges:
        outgoing[start].append(end)

    unused_edges = set(edges)
    loops: List[List[IntPoint]] = []

    while unused_edges:
        start, end = unused_edges.pop()
        loop = [start, end]
        current = end

        while current != start:
            candidates: List[Edge] = []

            for next_point in outgoing[current]:
                edge = (current, next_point)

                if edge in unused_edges:
                    candidates.append(edge)

            if not candidates:
                raise RuntimeError(
                    f"Boundary tracing failed at {current}. "
                    "Check for illegal diagonal touches or invalid geometry."
                )

            next_edge = candidates[0]
            unused_edges.remove(next_edge)

            current = next_edge[1]
            loop.append(current)

        loops.append(loop)

    return loops


def simplify_polygon(points: List[IntPoint]) -> List[IntPoint]:
    """Remove unnecessary middle points along straight lines."""
    if points and points[0] == points[-1]:
        points = points[:-1]

    changed = True

    while changed and len(points) > 2:
        changed = False
        new_points: List[IntPoint] = []

        for i in range(len(points)):
            prev_pt = points[i - 1]
            curr_pt = points[i]
            next_pt = points[(i + 1) % len(points)]

            same_x = prev_pt[0] == curr_pt[0] == next_pt[0]
            same_y = prev_pt[1] == curr_pt[1] == next_pt[1]

            if same_x or same_y:
                changed = True
            else:
                new_points.append(curr_pt)

        points = new_points

    return points


def polygon_area(points: Sequence[Union[IntPoint, Point]]) -> float:
    """Signed polygon area."""
    area = 0.0

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1

    return area / 2.0


def grid_to_polygons(grid: Grid, cell_size: float = 1.0) -> List[Dict[str, object]]:
    """Convert a rectangular binary grid into merged 2D polygon outlines."""
    check_grid_shape(grid)

    edges: List[Edge] = []

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if bool(grid[r][c]):
                add_boundary_edges_for_cell(grid, r, c, edges)

    if not edges:
        return []

    raw_loops = trace_boundary_loops(edges)

    polygons: List[Dict[str, object]] = []

    for loop in raw_loops:
        simplified = simplify_polygon(loop)
        area = polygon_area(simplified)

        scaled_vertices = [(x * cell_size, y * cell_size) for x, y in simplified]

        polygons.append({
            "type": "outer" if area > 0 else "hole",
            "area": abs(area) * (cell_size ** 2),
            "vertices": scaled_vertices,
        })

    polygons.sort(key=lambda p: p["area"], reverse=True)
    return polygons


# ---------------------------------------------------------------------------
# WGS_Antenna helpers
# ---------------------------------------------------------------------------

def _looks_like_wgs_particle(particle: Any) -> bool:
    return all(hasattr(particle, name) for name in ("map_grid", "conductor_grid", "neighbor_map"))


def wgs_to_rect_grid(particle: Any) -> Grid:
    """
    Convert a WGS_Antenna's ragged conductor_grid to a rectangular 0/1 net.

    Invalid cells marked as "X" in particle.map_grid become 0. This is useful
    for plotting/exporting an unfolded net, but physical connectivity across
    folded seams should use validate_wgs_particle().
    """
    if not _looks_like_wgs_particle(particle):
        raise TypeError("wgs_to_rect_grid() expects a WGS_Antenna-like object.")

    rect: Grid = []

    for i, side_row in enumerate(particle.map_grid):
        out_row: List[int] = []
        conductor_j = 0

        for side in side_row:
            if side == "X":
                out_row.append(0)
            else:
                out_row.append(int(bool(particle.conductor_grid[i][conductor_j])))
                conductor_j += 1

        rect.append(out_row)

    return rect


def wgs_metal_positions(particle: Any) -> Tuple[Set[Cell], Set[Cell]]:
    """
    Return (metal_cells, valid_cells) using full map_grid coordinates.

    These coordinates match particle.neighbor_map, not the compressed/ragged
    conductor_grid column numbers.
    """
    if not _looks_like_wgs_particle(particle):
        raise TypeError("wgs_metal_positions() expects a WGS_Antenna-like object.")

    metal_cells: Set[Cell] = set()
    valid_cells: Set[Cell] = set()

    for i, side_row in enumerate(particle.map_grid):
        conductor_j = 0

        for j, side in enumerate(side_row):
            if side == "X":
                continue

            valid_cells.add((i, j))

            if bool(particle.conductor_grid[i][conductor_j]):
                metal_cells.add((i, j))

            conductor_j += 1

    return metal_cells, valid_cells


def wgs_neighbors(particle: Any, cell: Cell, valid_cells: Optional[Set[Cell]] = None) -> List[Cell]:
    """Return physical WGS neighbors using particle.neighbor_map."""
    if not _looks_like_wgs_particle(particle):
        raise TypeError("wgs_neighbors() expects a WGS_Antenna-like object.")

    i, j = cell
    neighbor_dict = particle.neighbor_map[i][j]
    neighbors: List[Cell] = []

    for raw_neighbor in neighbor_dict.values():
        if raw_neighbor is None:
            continue

        ni = int(raw_neighbor[0])
        nj = int(raw_neighbor[1])
        neighbor = (ni, nj)

        if valid_cells is None or neighbor in valid_cells:
            neighbors.append(neighbor)

    return neighbors


def find_wgs_isolated_metal_cells(particle: Any) -> List[Cell]:
    """Find metal cells without any physically edge-connected metal neighbor."""
    metal_cells, valid_cells = wgs_metal_positions(particle)

    isolated: List[Cell] = []

    for cell in sorted(metal_cells):
        if not any(neighbor in metal_cells for neighbor in wgs_neighbors(particle, cell, valid_cells)):
            isolated.append(cell)

    return isolated


def find_wgs_metal_components(particle: Any) -> List[Set[Cell]]:
    """Find physically connected metal components using WGS seam-aware neighbors."""
    metal_cells, valid_cells = wgs_metal_positions(particle)

    visited: Set[Cell] = set()
    components: List[Set[Cell]] = []

    for start in sorted(metal_cells):
        if start in visited:
            continue

        component: Set[Cell] = set()
        queue: deque[Cell] = deque([start])
        visited.add(start)

        while queue:
            current = queue.popleft()
            component.add(current)

            for neighbor in wgs_neighbors(particle, current, valid_cells):
                if neighbor in metal_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def validate_wgs_particle(
    particle: Any,
    require_single_component: bool = True,
    check_flat_diagonals: bool = True,
) -> Dict[str, object]:
    """
    Validate a WGS_Antenna object.

    Connectivity and isolation use the WGS seam-aware neighbor_map. Diagonal
    checks are performed on the unfolded rectangular net because diagonal
    corner-touch rules are fundamentally 2D layout checks.
    """
    isolated_cells = find_wgs_isolated_metal_cells(particle)
    components = find_wgs_metal_components(particle)

    diagonal_touches: List[Dict[str, object]] = []
    if check_flat_diagonals:
        diagonal_touches = find_illegal_diagonal_touches(wgs_to_rect_grid(particle))

    disconnected_components: List[Set[Cell]] = []

    if require_single_component and len(components) > 1:
        main_component = max(components, key=len)
        disconnected_components = [comp for comp in components if comp is not main_component]

    errors: List[str] = []

    if isolated_cells:
        errors.append("Isolated metal cells found.")

    if diagonal_touches:
        errors.append("Illegal diagonal corner touches found in unfolded WGS net.")

    if disconnected_components:
        errors.append("Disconnected metal components found.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "isolated_cells": isolated_cells,
        "diagonal_touches": diagonal_touches,
        "num_metal_components": len(components),
        "disconnected_components": disconnected_components,
    }


def grid_or_wgs_to_polygons(grid_or_particle: Union[Grid, Any], cell_size: float = 1.0) -> List[Dict[str, object]]:
    """
    Polygon export helper.

    - Rectangular grid input: exports that grid directly.
    - WGS_Antenna input: exports the unfolded rectangular net produced by
      wgs_to_rect_grid().
    """
    if _looks_like_wgs_particle(grid_or_particle):
        return grid_to_polygons(wgs_to_rect_grid(grid_or_particle), cell_size=cell_size)

    return grid_to_polygons(grid_or_particle, cell_size=cell_size)


def summarize_validation_report(report: Dict[str, object], max_items: int = 10) -> Dict[str, object]:
    """Return a compact, printable version of a validation report."""
    disconnected_components = report.get("disconnected_components", []) or []

    return {
        "is_valid": report.get("is_valid"),
        "errors": report.get("errors", []),
        "num_isolated_cells": len(report.get("isolated_cells", []) or []),
        "sample_isolated_cells": list(report.get("isolated_cells", []) or [])[:max_items],
        "num_diagonal_touches": len(report.get("diagonal_touches", []) or []),
        "sample_diagonal_touches": list(report.get("diagonal_touches", []) or [])[:max_items],
        "num_metal_components": report.get("num_metal_components", 0),
        "num_disconnected_components": len(disconnected_components),
        "sample_disconnected_component_sizes": [len(comp) for comp in disconnected_components[:max_items]],
    }
