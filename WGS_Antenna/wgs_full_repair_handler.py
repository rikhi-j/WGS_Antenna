# =============================================================================
# MODULE: wgs_full_repair_handler.py
# PURPOSE:
# Performs the complete manufacturability repair process for WGS antenna
# designs. This module coordinates all major repair operations required to
# transform a raw PSO-generated antenna into a valid antenna that can safely be
# exported to HFSS.
#
# Unlike wgs_illegal_handler.py, which only repairs illegal diagonal
# checkerboard connections, this module performs the complete repair pipeline,
# including illegal-pattern repair, island removal, and reconnection of
# disconnected conductor regions.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - optimization_controller.py calls repair_wgs_particle_full() before every
#   candidate is exported to HFSS.
# - The repaired WGS_Antenna object is returned to the optimizer and later
#   converted into ANSYS geometry.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Validate that all required repair modules have been imported.
# 2. Save the original WGS conductor grid.
# 3. Repair illegal diagonal (checkerboard) conductor patterns.
# 4. Remove small isolated conductor islands below a minimum size.
# 5. Detect disconnected conductor components.
# 6. Find the shortest valid bridge path between disconnected components.
# 7. Add conductor cells along the bridge path to reconnect the antenna.
# 8. Repair illegal checkerboards that may have been introduced by bridging.
# 9. Validate the repaired antenna.
# 10. Repeat until the antenna becomes valid or the maximum repair iterations
#     have been reached.
# -----------------------------------------------------------------------------
# MAJOR REPAIR OPERATIONS:
# Illegal Checkerboard Repair:
#     Removes conductor cells that touch only diagonally by calling
#     wgs_illegal_handler.py.
#
# Small Island Removal:
#     Removes isolated conductor regions whose area is smaller than the
#     configured minimum island size.
#
# Component Bridging:
#     Connects disconnected conductor regions by finding the shortest editable
#     path between them and converting the required empty cells into conductor.
#
# Final Validation:
#     Confirms the repaired antenna satisfies all geometry rules before it is
#     exported to HFSS.
# -----------------------------------------------------------------------------
# KEY DATA STORED:
# FullRepairResult:
#     Stores the repaired particle, original and repaired conductor grids,
#     modified cells, added bridge cells, removed cells, bridge paths,
#     validation reports, repair statistics, and warning messages.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - WGS_Antenna objects.
# - Editable-cell mask (map_mask).
# - Valid WGS-cell mask (map_grid).
# - Minimum conductor-island area.
# - Maximum repair iterations.
# - Maximum bridges allowed per iteration.
# - Optional validation settings.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Fully repaired WGS_Antenna objects.
# - Complete repair summaries.
# - Lists of changed, added, and removed cells.
# - Bridge paths created during repair.
# - Validation reports before and after repair.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - This module coordinates the entire geometry repair pipeline.
# - It performs repair operations iteratively because one repair may create a
#   new issue that must be corrected during the next repair cycle.
# - Bridge paths are created using the WGS neighbor map rather than simple
#   rectangular-grid neighbors so connectivity matches the true folded antenna
#   geometry.
# - Fixed conductor regions are preserved unless include_fixed=True is
#   explicitly requested.
# - If no valid bridge can be created without modifying fixed cells, the module
#   reports a warning rather than forcing an invalid repair.
# =============================================================================

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from validator_updated_only import (
        validate_wgs_particle,
        summarize_validation_report,
        wgs_metal_positions,
        wgs_neighbors,
        find_wgs_metal_components,
    )
except Exception as exc:  # pragma: no cover
    validate_wgs_particle = None
    summarize_validation_report = None
    wgs_metal_positions = None
    wgs_neighbors = None
    find_wgs_metal_components = None
    _VALIDATOR_IMPORT_ERROR = exc
else:
    _VALIDATOR_IMPORT_ERROR = None

try:
    from wgs_illegal_handler import (
        repair_wgs_particle_illegal_positions,
        get_wgs_rect_grid,
        apply_rect_grid_to_wgs_particle,
        wgs_valid_rect_mask,
        wgs_mutable_rect_mask,
        count_checkerboards,
    )
except Exception as exc:  # pragma: no cover
    repair_wgs_particle_illegal_positions = None
    get_wgs_rect_grid = None
    apply_rect_grid_to_wgs_particle = None
    wgs_valid_rect_mask = None
    wgs_mutable_rect_mask = None
    count_checkerboards = None
    _ILLEGAL_HANDLER_IMPORT_ERROR = exc
else:
    _ILLEGAL_HANDLER_IMPORT_ERROR = None


Cell = Tuple[int, int]
Path = List[Cell]


@dataclass
class FullRepairResult:
    """Result returned by repair_wgs_particle_full()."""

    particle: Any
    original_grid: np.ndarray
    repaired_grid: np.ndarray
    changed_cells: List[Cell] = field(default_factory=list)
    removed_cells: List[Cell] = field(default_factory=list)
    added_cells: List[Cell] = field(default_factory=list)
    illegal_changed_cells: List[Cell] = field(default_factory=list)
    bridge_paths: List[Path] = field(default_factory=list)
    iterations: int = 0
    before_validation: Optional[Dict[str, object]] = None
    after_validation: Optional[Dict[str, object]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def changed_count(self) -> int:
        return len(set(self.changed_cells))

    @property
    def added_count(self) -> int:
        return len(set(self.added_cells))

    @property
    def removed_count(self) -> int:
        return len(set(self.removed_cells))

    @property
    def bridge_count(self) -> int:
        return len(self.bridge_paths)

    @property
    def is_valid_after(self) -> Optional[bool]:
        if self.after_validation is None:
            return None
        return bool(self.after_validation.get("is_valid", False))

    def summary(self) -> Dict[str, object]:
        """Return a compact printable summary."""
        before = None
        after = None

        if self.before_validation is not None:
            before = (
                summarize_validation_report(self.before_validation)
                if summarize_validation_report is not None
                else self.before_validation
            )

        if self.after_validation is not None:
            after = (
                summarize_validation_report(self.after_validation)
                if summarize_validation_report is not None
                else self.after_validation
            )

        return {
            "is_valid_after": self.is_valid_after,
            "iterations": self.iterations,
            "changed_count": self.changed_count,
            "illegal_changed_count": len(set(self.illegal_changed_cells)),
            "removed_count": self.removed_count,
            "added_count": self.added_count,
            "bridge_count": self.bridge_count,
            "bridge_path_lengths": [len(path) for path in self.bridge_paths],
            "warnings": self.warnings,
            "before_validation": before,
            "after_validation": after,
        }


# -----------------------------------------------------------------------------
# Import checks / masks
# -----------------------------------------------------------------------------


def _require_imports() -> None:
    missing = []

    if _VALIDATOR_IMPORT_ERROR is not None:
        missing.append(f"validator_updated_only.py import failed: {_VALIDATOR_IMPORT_ERROR}")

    if _ILLEGAL_HANDLER_IMPORT_ERROR is not None:
        missing.append(f"wgs_illegal_handler.py import failed: {_ILLEGAL_HANDLER_IMPORT_ERROR}")

    if missing:
        raise ImportError("\n".join(missing))


def _looks_like_wgs_particle(particle: Any) -> bool:
    return all(hasattr(particle, name) for name in ("map_grid", "conductor_grid", "neighbor_map"))


def _mask_to_cell_set(mask: np.ndarray) -> Set[Cell]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _editable_cells(particle: Any, include_fixed: bool) -> Set[Cell]:
    mask = wgs_mutable_rect_mask(particle, include_fixed=include_fixed)
    return _mask_to_cell_set(mask)


def _valid_cells(particle: Any) -> Set[Cell]:
    mask = wgs_valid_rect_mask(particle)
    return _mask_to_cell_set(mask)


def _set_rect_cells(
    particle: Any,
    cells: Iterable[Cell],
    value: int,
    *,
    include_fixed: bool,
) -> List[Cell]:
    """Set rectangular WGS cells to 0 or 1 and write into conductor_grid."""
    rect = get_wgs_rect_grid(particle)
    editable = _editable_cells(particle, include_fixed=include_fixed)
    valid = _valid_cells(particle)

    actually_changed: List[Cell] = []

    for cell in cells:
        r, c = int(cell[0]), int(cell[1])
        if (r, c) not in valid:
            continue
        if (r, c) not in editable:
            continue
        if int(rect[r, c]) != int(value):
            rect[r, c] = int(value)
            actually_changed.append((r, c))

    apply_rect_grid_to_wgs_particle(
        particle,
        rect,
        only_mutable=not include_fixed,
    )

    return actually_changed


# -----------------------------------------------------------------------------
# WGS-aware component helpers
# -----------------------------------------------------------------------------


def _metal_components(particle: Any) -> List[Set[Cell]]:
    comps = find_wgs_metal_components(particle)
    # Stable order: largest first, then top-left-ish for reproducible output.
    return sorted(
        [set(comp) for comp in comps],
        key=lambda comp: (-len(comp), min(comp) if comp else (10**9, 10**9)),
    )


def remove_small_wgs_conductor_islands(
    particle: Any,
    *,
    min_area: int = 4,
    include_fixed: bool = False,
) -> Tuple[List[Cell], List[str]]:
    """
    Remove WGS metal components smaller than min_area.

    Returns
    -------
    removed_cells, warnings
    """
    _require_imports()

    if min_area <= 1:
        return [], []

    components = _metal_components(particle)
    editable = _editable_cells(particle, include_fixed=include_fixed)

    cells_to_remove: List[Cell] = []
    warnings: List[str] = []

    for comp in components:
        if len(comp) >= min_area:
            continue

        removable = sorted(cell for cell in comp if cell in editable)
        blocked = sorted(cell for cell in comp if cell not in editable)

        if removable:
            cells_to_remove.extend(removable)

        if blocked:
            warnings.append(
                f"Small component of size {len(comp)} contains {len(blocked)} fixed cell(s); "
                "those fixed cells were not removed. Use include_fixed=True to allow it."
            )

    removed = _set_rect_cells(
        particle,
        cells_to_remove,
        0,
        include_fixed=include_fixed,
    )

    return removed, warnings


def _shortest_editable_wgs_path(
    particle: Any,
    source_component: Set[Cell],
    target_component: Set[Cell],
    *,
    include_fixed: bool,
) -> Optional[Path]:
    """
    Find the shortest WGS-neighbor path that can connect source_component to
    target_component by turning editable zero cells into metal.

    The graph is seam-aware because it uses particle.neighbor_map through
    validator_updated_only.wgs_neighbors().
    """
    metal_cells, valid_cells = wgs_metal_positions(particle)
    editable = _editable_cells(particle, include_fixed=include_fixed)

    # You can walk through existing metal and through editable cells that can be
    # converted into metal. Fixed zero cells are not passable unless
    # include_fixed=True made them editable.
    passable = set(metal_cells) | set(editable)

    source_component = set(source_component)
    target_component = set(target_component)

    queue: deque[Cell] = deque()
    parent: Dict[Cell, Optional[Cell]] = {}

    for cell in sorted(source_component):
        if cell in valid_cells:
            queue.append(cell)
            parent[cell] = None

    while queue:
        current = queue.popleft()

        if current in target_component:
            path: Path = []
            node: Optional[Cell] = current
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        for neighbor in wgs_neighbors(particle, current, valid_cells):
            if neighbor in parent:
                continue
            if neighbor not in passable:
                continue
            parent[neighbor] = current
            queue.append(neighbor)

    return None


def connect_wgs_components(
    particle: Any,
    *,
    include_fixed: bool = False,
    max_bridges: int = 500,
) -> Tuple[List[Cell], List[Path], List[str]]:
    """
    Connect disconnected WGS conductor components using seam-aware editable paths.

    Returns
    -------
    added_cells, bridge_paths, warnings
    """
    _require_imports()

    added_cells: List[Cell] = []
    bridge_paths: List[Path] = []
    warnings: List[str] = []

    for _ in range(max_bridges):
        components = _metal_components(particle)

        if len(components) <= 1:
            return added_cells, bridge_paths, warnings

        main = components[0]
        candidates: List[Tuple[int, int, Set[Cell], Path]] = []

        for comp in components[1:]:
            path = _shortest_editable_wgs_path(
                particle,
                source_component=comp,
                target_component=main,
                include_fixed=include_fixed,
            )
            if path is not None:
                # Prefer shortest path; tie-break by connecting larger components first.
                candidates.append((len(path), -len(comp), comp, path))

        if not candidates:
            warnings.append(
                f"Could not connect {len(components) - 1} disconnected component(s) "
                "without crossing fixed/invalid cells. Try include_fixed=True or inspect the layout."
            )
            return added_cells, bridge_paths, warnings

        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, _, best_path = candidates[0]

        # Only zero cells along the path need to be added. Existing metal cells
        # at the endpoints stay unchanged.
        rect = get_wgs_rect_grid(particle)
        path_zero_cells = [cell for cell in best_path if int(rect[cell[0], cell[1]]) == 0]

        added = _set_rect_cells(
            particle,
            path_zero_cells,
            1,
            include_fixed=include_fixed,
        )

        if not added:
            warnings.append(
                "A bridge path was found, but no editable zero cells could be changed. "
                "Stopping to avoid an infinite loop."
            )
            return added_cells, bridge_paths, warnings

        added_cells.extend(added)
        bridge_paths.append(best_path)

    warnings.append(f"Stopped after max_bridges={max_bridges}; components may remain disconnected.")
    return added_cells, bridge_paths, warnings


# -----------------------------------------------------------------------------
# High-level full repair
# -----------------------------------------------------------------------------


def repair_wgs_particle_full(
    particle: Any,
    *,
    in_place: bool = False,
    include_fixed: bool = False,
    min_island_area: int = 4,
    repair_illegal_diagonals: bool = True,
    remove_small_islands: bool = True,
    connect_components: bool = True,
    max_iterations: int = 20,
    max_bridges_per_iteration: int = 500,
    validate_after: bool = True,
    verbose: bool = False,
) -> FullRepairResult:
    """
    Repair a WGS_Antenna particle for manufacturability-style validity checks.

    The repair order per iteration is:
        1. repair illegal diagonal/checkerboard contacts
        2. remove conductor components smaller than min_island_area
        3. connect remaining disconnected components with WGS-neighbor bridges
        4. repair illegal diagonals again, because bridges can introduce them

    By default, only mutable cells from particle.map_mask may be changed.
    """
    _require_imports()

    if not _looks_like_wgs_particle(particle):
        raise TypeError("repair_wgs_particle_full() expects a WGS_Antenna-like object.")

    target = particle if in_place else copy.deepcopy(particle)
    original_grid = get_wgs_rect_grid(target)

    before_validation = validate_wgs_particle(target) if validate_wgs_particle is not None else None

    all_changed: List[Cell] = []
    all_removed: List[Cell] = []
    all_added: List[Cell] = []
    all_illegal_changed: List[Cell] = []
    all_bridge_paths: List[Path] = []
    warnings: List[str] = []

    iterations_done = 0
    seen_states: Set[bytes] = set()

    for iteration in range(1, max_iterations + 1):
        start_grid = get_wgs_rect_grid(target)
        state_key = start_grid.tobytes()

        if state_key in seen_states:
            warnings.append("Full repair stopped because the WGS grid repeated a previous state.")
            break
        seen_states.add(state_key)

        iteration_changed: Set[Cell] = set()

        if repair_illegal_diagonals:
            illegal_result = repair_wgs_particle_illegal_positions(
                target,
                in_place=True,
                include_fixed=include_fixed,
                validate_after=False,
                verbose=False,
            )
            all_illegal_changed.extend(illegal_result.changed_cells)
            iteration_changed.update(illegal_result.changed_cells)
            warnings.extend(illegal_result.warnings)

        if remove_small_islands:
            removed, remove_warnings = remove_small_wgs_conductor_islands(
                target,
                min_area=min_island_area,
                include_fixed=include_fixed,
            )
            all_removed.extend(removed)
            iteration_changed.update(removed)
            warnings.extend(remove_warnings)

        if connect_components:
            added, bridge_paths, connect_warnings = connect_wgs_components(
                target,
                include_fixed=include_fixed,
                max_bridges=max_bridges_per_iteration,
            )
            all_added.extend(added)
            all_bridge_paths.extend(bridge_paths)
            iteration_changed.update(added)
            warnings.extend(connect_warnings)

        # Bridges may create flat diagonal checkerboards. Clean them once more.
        if repair_illegal_diagonals:
            illegal_result = repair_wgs_particle_illegal_positions(
                target,
                in_place=True,
                include_fixed=include_fixed,
                validate_after=False,
                verbose=False,
            )
            all_illegal_changed.extend(illegal_result.changed_cells)
            iteration_changed.update(illegal_result.changed_cells)
            warnings.extend(illegal_result.warnings)

        iterations_done = iteration
        all_changed.extend(sorted(iteration_changed))

        after_iteration_validation = validate_wgs_particle(target)
        after_summary = summarize_validation_report(after_iteration_validation)

        if verbose:
            checkerboards = count_checkerboards(get_wgs_rect_grid(target))
            print(
                f"Iteration {iteration:3d} | "
                f"changed={len(iteration_changed):4d} | "
                f"checkerboards={checkerboards:4d} | "
                f"components={after_summary['num_metal_components']:4d} | "
                f"isolated={after_summary['num_isolated_cells']:4d} | "
                f"valid={after_summary['is_valid']}"
            )

        if after_iteration_validation.get("is_valid", False):
            break

        end_grid = get_wgs_rect_grid(target)
        if np.array_equal(start_grid, end_grid):
            warnings.append(
                "Full repair stopped because an iteration made no changes. "
                "Remaining problems likely require editing fixed cells or changing repair settings."
            )
            break

    else:
        warnings.append(f"Stopped after max_iterations={max_iterations}; validation problems may remain.")

    after_validation = validate_wgs_particle(target) if validate_after else None
    repaired_grid = get_wgs_rect_grid(target)

    # De-duplicate warnings while preserving order.
    deduped_warnings: List[str] = []
    seen_warning: Set[str] = set()
    for warning in warnings:
        if warning not in seen_warning:
            seen_warning.add(warning)
            deduped_warnings.append(warning)

    return FullRepairResult(
        particle=target,
        original_grid=original_grid,
        repaired_grid=repaired_grid,
        changed_cells=all_changed,
        removed_cells=all_removed,
        added_cells=all_added,
        illegal_changed_cells=all_illegal_changed,
        bridge_paths=all_bridge_paths,
        iterations=iterations_done,
        before_validation=before_validation,
        after_validation=after_validation,
        warnings=deduped_warnings,
    )


def repair_wgs_particle_components(
    particle: Any,
    *,
    in_place: bool = False,
    include_fixed: bool = False,
    min_island_area: int = 4,
    verbose: bool = False,
) -> Any:
    """
    Convenience wrapper that returns only the repaired particle.
    """
    return repair_wgs_particle_full(
        particle,
        in_place=in_place,
        include_fixed=include_fixed,
        min_island_area=min_island_area,
        verbose=verbose,
    ).particle


# -----------------------------------------------------------------------------
# Small smoke demo
# -----------------------------------------------------------------------------


def demo() -> None:
    """Run a small smoke test using WGS_pixels.WGS_Antenna."""
    from WGS_pixels import WGS_Antenna

    ds = 0.5e-3
    particle = WGS_Antenna(
        size=np.array([20 * ds, 10 * ds, 100 * ds]),
        resolution=np.array([20, 10, 100]),
        pad_ring_t=np.array([0.5e-2, 0.5e-2]),
        randomized=True,
        alpha=0.5,
    )

    result = repair_wgs_particle_full(
        particle,
        in_place=False,
        include_fixed=False,
        min_island_area=4,
        verbose=True,
    )
    print(result.summary())


if __name__ == "__main__":
    demo()
