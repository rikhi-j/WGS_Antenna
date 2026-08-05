# =============================================================================
# MODULE: wgs_illegal_handler.py
# PURPOSE:
# Repairs illegal diagonal (checkerboard) conductor patterns in WGS antenna
# designs. These illegal patterns occur when two conductor cells touch only at
# a corner instead of sharing a full edge, creating ambiguous or physically
# invalid conductor connections.
#
# This module performs the first stage of the antenna repair process by removing
# illegal local conductor patterns before the antenna undergoes larger repairs,
# such as reconnecting separated conductor regions.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - optimization_controller.py indirectly calls this module through
#   wgs_full_repair_handler.py.
# - A PSO-generated WGS_Antenna is converted into a rectangular binary grid.
# - Illegal checkerboard patterns are detected and repaired.
# - The repaired grid is written back into the WGS_Antenna object.
# - The repaired antenna is optionally revalidated before continuing through the
#   remainder of the repair pipeline.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Convert the WGS antenna into a rectangular binary grid.
# 2. Determine which cells are valid WGS cells and which cells are allowed to
#    be modified.
# 3. Search the grid for illegal 2x2 diagonal (checkerboard) conductor patterns.
# 4. For each illegal block, evaluate every editable one-cell flip.
# 5. Score each possible repair and choose the best local solution.
# 6. Apply the repair to the rectangular grid.
# 7. Write the repaired grid back into the WGS_Antenna object.
# 8. Optionally validate the repaired antenna before returning it.
# -----------------------------------------------------------------------------
# WHAT IS AN ILLEGAL CHECKERBOARD?
# Two conductor cells are only allowed to connect through a shared edge.
# Connections that touch only at a single corner are considered illegal.
#
# Illegal examples:
#
#     1 0        0 1
#     0 1    or  1 0
#
# where:
#     1 = conductor
#     0 = empty
#
# These patterns are repaired before any larger connectivity repairs are
# performed.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - WGS_Antenna objects.
# - Rectangular binary conductor grids.
# - Editable-cell masks (map_mask).
# - Valid WGS-cell masks (map_grid).
# - Maximum repair iterations.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Repaired WGS_Antenna objects.
# - Repaired rectangular conductor grids.
# - Repair summaries.
# - Lists of modified cells.
# - Validation reports.
# -----------------------------------------------------------------------------
# KEY DATA STRUCTURES:
# IllegalRepairResult:
#     Stores the repaired particle, repaired grid, modified cells, repair
#     statistics, remaining unrepaired checkerboards, warnings, and optional
#     validation results.
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - Only illegal diagonal/checkerboard conductor patterns are repaired here.
# - This module does NOT reconnect disconnected conductor regions or remove
#   conductor islands. Those operations are performed later by
#   wgs_full_repair_handler.py.
# - Fixed conductor cells remain unchanged unless include_fixed=True is
#   explicitly requested.
# - Only real WGS cells may be modified; placeholder ("X") cells are ignored.
# - Every repair is scored so the algorithm chooses the locally best cell to
#   flip rather than always flipping the same location.
# =============================================================================

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from validator_updated_only import (
        wgs_to_rect_grid,
        validate_wgs_particle,
        summarize_validation_report,
    )
except Exception:  # pragma: no cover - keeps the file importable during partial setups
    wgs_to_rect_grid = None
    validate_wgs_particle = None
    summarize_validation_report = None


Cell = Tuple[int, int]
Block = Tuple[int, int]
GridArray = np.ndarray


FOUR_CONNECTED_STRUCTURE = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=int,
)


@dataclass
class IllegalRepairResult:
    """Result returned by the high-level repair functions."""

    particle: Optional[Any]
    original_grid: GridArray
    repaired_grid: GridArray
    changed_cells: List[Cell] = field(default_factory=list)
    unrepaired_blocks: List[Block] = field(default_factory=list)
    before_illegal_count: int = 0
    after_illegal_count: int = 0
    iterations: int = 0
    validation_report: Optional[Dict[str, object]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def changed_count(self) -> int:
        return len(self.changed_cells)

    @property
    def fully_repaired(self) -> bool:
        return self.after_illegal_count == 0 and not self.unrepaired_blocks

    def summary(self) -> Dict[str, object]:
        """Return a compact printable summary."""
        data: Dict[str, object] = {
            "fully_repaired": self.fully_repaired,
            "before_illegal_count": self.before_illegal_count,
            "after_illegal_count": self.after_illegal_count,
            "changed_count": self.changed_count,
            "iterations": self.iterations,
            "unrepaired_blocks": self.unrepaired_blocks,
            "warnings": self.warnings,
        }

        if self.validation_report is not None and summarize_validation_report is not None:
            data["validation"] = summarize_validation_report(self.validation_report)
        elif self.validation_report is not None:
            data["validation"] = self.validation_report

        return data


# -----------------------------------------------------------------------------
# Rectangular-grid helpers
# -----------------------------------------------------------------------------


def as_binary_numpy_grid(grid: Any) -> GridArray:
    """Convert a rectangular grid-like object to a 2D int NumPy array of 0/1."""
    arr = np.asarray(grid, dtype=int)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grid. Got shape {arr.shape!r}.")

    bad = ~np.isin(arr, [0, 1])
    if np.any(bad):
        bad_positions = np.argwhere(bad)
        first_bad = tuple(int(x) for x in bad_positions[0])
        raise ValueError(
            "Grid values must be 0/1 or bool. "
            f"Found {arr[first_bad]!r} at {first_bad}."
        )

    return arr.astype(int, copy=True)


def four_neighbors(cell: Cell, shape: Tuple[int, int]) -> Iterable[Cell]:
    r, c = cell
    rows, cols = shape

    if r > 0:
        yield (r - 1, c)
    if r + 1 < rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < cols:
        yield (r, c + 1)


def illegal_blocks(grid: Any) -> List[Block]:
    """
    Find illegal 2x2 diagonal/checkerboard conductor contacts.

    Illegal patterns are:
        1 0      0 1
        0 1  or  1 0

    Returns the top-left cell of every illegal 2x2 block.
    """
    arr = as_binary_numpy_grid(grid)
    rows, cols = arr.shape
    blocks: List[Block] = []

    for r in range(rows - 1):
        for c in range(cols - 1):
            block = arr[r : r + 2, c : c + 2]

            if (
                block[0, 0] == 1
                and block[1, 1] == 1
                and block[0, 1] == 0
                and block[1, 0] == 0
            ):
                blocks.append((r, c))
            elif (
                block[0, 1] == 1
                and block[1, 0] == 1
                and block[0, 0] == 0
                and block[1, 1] == 0
            ):
                blocks.append((r, c))

    return blocks


def count_checkerboards(grid: Any) -> int:
    return len(illegal_blocks(grid))


def connected_components(grid: Any) -> List[Set[Cell]]:
    """Return 4-connected metal components in a rectangular 0/1 grid."""
    arr = as_binary_numpy_grid(grid)
    rows, cols = arr.shape
    visited: Set[Cell] = set()
    components: List[Set[Cell]] = []

    for r in range(rows):
        for c in range(cols):
            start = (r, c)
            if arr[r, c] == 0 or start in visited:
                continue

            comp: Set[Cell] = set()
            queue: deque[Cell] = deque([start])
            visited.add(start)

            while queue:
                current = queue.popleft()
                comp.add(current)

                for nbr in four_neighbors(current, arr.shape):
                    nr, nc = nbr
                    if arr[nr, nc] == 1 and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

            components.append(comp)

    return components


def count_components(grid: Any) -> int:
    return len(connected_components(grid))


def isolated_metal_cells(grid: Any) -> List[Cell]:
    """Find metal cells with no 4-connected metal neighbor."""
    arr = as_binary_numpy_grid(grid)
    isolated: List[Cell] = []

    for r, c in np.argwhere(arr == 1):
        cell = (int(r), int(c))
        if not any(arr[nr, nc] == 1 for nr, nc in four_neighbors(cell, arr.shape)):
            isolated.append(cell)

    return isolated


def count_isolated_cells(grid: Any) -> int:
    return len(isolated_metal_cells(grid))


# -----------------------------------------------------------------------------
# WGS mapping helpers
# -----------------------------------------------------------------------------


def looks_like_wgs_particle(particle: Any) -> bool:
    return all(hasattr(particle, name) for name in ("map_grid", "conductor_grid", "neighbor_map"))


def wgs_valid_rect_mask(particle: Any) -> GridArray:
    """
    Rectangular mask: True where particle.map_grid is a real WGS cell, False at X.
    """
    if not looks_like_wgs_particle(particle):
        raise TypeError("Expected a WGS_Antenna-like object.")

    rows = len(particle.map_grid)
    cols = max(len(row) for row in particle.map_grid)
    mask = np.zeros((rows, cols), dtype=bool)

    for r, row in enumerate(particle.map_grid):
        for c, side in enumerate(row):
            if side != "X":
                mask[r, c] = True

    return mask


def wgs_mutable_rect_mask(particle: Any, include_fixed: bool = False) -> GridArray:
    """
    Rectangular mask of cells that the repair function is allowed to change.

    include_fixed=False, default:
        only cells whose compressed WGS map_mask entry is True are editable.

    include_fixed=True:
        all real/non-X cells are editable. Use carefully.
    """
    valid_mask = wgs_valid_rect_mask(particle)

    if include_fixed:
        return valid_mask.copy()

    mask = np.zeros_like(valid_mask, dtype=bool)

    for r, row in enumerate(particle.map_grid):
        conductor_j = 0

        for c, side in enumerate(row):
            if side == "X":
                continue

            editable = False
            if r < len(particle.map_mask) and conductor_j < len(particle.map_mask[r]):
                editable = bool(particle.map_mask[r][conductor_j])

            mask[r, c] = editable
            conductor_j += 1

    return mask


def get_wgs_rect_grid(particle: Any) -> GridArray:
    """Convert WGS_Antenna to the rectangular 0/1 grid used by your validator."""
    if wgs_to_rect_grid is None:
        raise ImportError(
            "Could not import wgs_to_rect_grid from validator_updated_only.py. "
            "Place wgs_illegal_handler.py next to validator_updated_only.py."
        )

    return as_binary_numpy_grid(wgs_to_rect_grid(particle))


def apply_rect_grid_to_wgs_particle(
    particle: Any,
    rect_grid: Any,
    *,
    only_mutable: bool = True,
) -> List[Cell]:
    """
    Write a repaired rectangular grid back into particle.conductor_grid.

    Returns a list of rectangular-grid cells that changed inside conductor_grid.
    Invalid X cells are ignored. If only_mutable=True, fixed WGS cells are preserved.
    """
    if not looks_like_wgs_particle(particle):
        raise TypeError("Expected a WGS_Antenna-like object.")

    arr = as_binary_numpy_grid(rect_grid)
    expected_shape = (len(particle.map_grid), max(len(row) for row in particle.map_grid))

    if arr.shape != expected_shape:
        raise ValueError(f"rect_grid shape {arr.shape} does not match WGS map shape {expected_shape}.")

    changed: List[Cell] = []

    for r, row in enumerate(particle.map_grid):
        conductor_j = 0

        for c, side in enumerate(row):
            if side == "X":
                continue

            if r >= len(particle.conductor_grid) or conductor_j >= len(particle.conductor_grid[r]):
                raise IndexError(
                    "particle.conductor_grid does not match particle.map_grid. "
                    f"Problem at rectangular cell {(r, c)} / conductor index {conductor_j}."
                )

            editable = True
            if only_mutable:
                editable = (
                    r < len(particle.map_mask)
                    and conductor_j < len(particle.map_mask[r])
                    and bool(particle.map_mask[r][conductor_j])
                )

            if editable:
                old_value = bool(particle.conductor_grid[r][conductor_j])
                new_value = bool(arr[r, c])

                if old_value != new_value:
                    particle.conductor_grid[r][conductor_j] = new_value
                    changed.append((r, c))

            conductor_j += 1

    return changed


# -----------------------------------------------------------------------------
# Repair scoring
# -----------------------------------------------------------------------------


def _block_cells(block: Block) -> List[Cell]:
    r, c = block
    return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]


def _is_illegal_block_at(arr: GridArray, block: Block) -> bool:
    r, c = block
    rows, cols = arr.shape

    if r < 0 or c < 0 or r + 1 >= rows or c + 1 >= cols:
        return False

    b = arr[r : r + 2, c : c + 2]
    return bool(
        (b[0, 0] == 1 and b[1, 1] == 1 and b[0, 1] == 0 and b[1, 0] == 0)
        or (b[0, 1] == 1 and b[1, 0] == 1 and b[0, 0] == 0 and b[1, 1] == 0)
    )


def _nearby_block_topleft_cells(cell: Cell, shape: Tuple[int, int]) -> List[Block]:
    """
    Top-left coordinates of every 2x2 block that contains cell.
    A cell can belong to at most four 2x2 blocks.
    """
    r, c = cell
    rows, cols = shape
    blocks: List[Block] = []

    for br in (r - 1, r):
        for bc in (c - 1, c):
            if 0 <= br < rows - 1 and 0 <= bc < cols - 1:
                blocks.append((br, bc))

    return blocks


def _local_checkerboard_count(arr: GridArray, cells: Sequence[Cell]) -> int:
    """Count illegal 2x2 blocks only near the given cells."""
    blocks: Set[Block] = set()
    for cell in cells:
        blocks.update(_nearby_block_topleft_cells(cell, arr.shape))
    return sum(1 for block in blocks if _is_illegal_block_at(arr, block))


def _metal_neighbor_count(arr: GridArray, cell: Cell) -> int:
    return sum(int(arr[nr, nc] == 1) for nr, nc in four_neighbors(cell, arr.shape))


def _candidate_local_score(before: GridArray, candidate: GridArray, flipped_cell: Cell) -> Tuple[int, int, int, int]:
    """
    Fast local score. Lower is better.

    Priority:
        1. fewer illegal blocks near the changed cell
        2. prefer adding metal over removing metal
        3. prefer the changed cell to have more 4-neighbor metal contacts
        4. prefer more total metal in the 2x2 neighborhood
    """
    r, c = flipped_cell
    new_value = int(candidate[r, c])

    neighborhood_rows = range(max(0, r - 1), min(candidate.shape[0], r + 2))
    neighborhood_cols = range(max(0, c - 1), min(candidate.shape[1], c + 2))
    local_metal = int(sum(candidate[rr, cc] for rr in neighborhood_rows for cc in neighborhood_cols))

    return (
        _local_checkerboard_count(candidate, [flipped_cell]),
        0 if new_value == 1 else 1,      # prefer adding a bridge
        -_metal_neighbor_count(candidate, flipped_cell),
        -local_metal,
    )


def _choose_repair_flip_for_block(
    grid: GridArray,
    block: Block,
    editable: GridArray,
) -> Optional[Tuple[Cell, GridArray]]:
    """Choose the best editable one-cell flip for one illegal 2x2 block."""
    if not _is_illegal_block_at(grid, block):
        return None

    block_cells = _block_cells(block)
    before_local = _local_checkerboard_count(grid, block_cells)

    candidates: List[Tuple[Tuple[int, int, int, int], Cell, GridArray]] = []

    for cell in block_cells:
        r, c = cell
        if not editable[r, c]:
            continue

        candidate = grid.copy()
        candidate[r, c] = 1 - candidate[r, c]
        after_local = _local_checkerboard_count(candidate, block_cells)

        # Only accept flips that improve the local checkerboard situation.
        if after_local < before_local:
            score = _candidate_local_score(grid, candidate, cell)
            candidates.append((score, cell, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, best_cell, best_grid = candidates[0]
    return best_cell, best_grid


def repair_rect_grid_illegal_positions(
    grid: Any,
    *,
    mutable_mask: Optional[Any] = None,
    valid_mask: Optional[Any] = None,
    max_iterations: int = 200,
    prefer_add_metal: bool = True,
    verbose: bool = False,
) -> IllegalRepairResult:
    """
    Repair illegal diagonal/checkerboard positions in a rectangular 0/1 grid.

    This is the fast version intended for your WGS grid sizes. It scans the grid
    for illegal 2x2 blocks, then flips one editable cell in each currently-illegal
    block using local scoring.

    mutable_mask:
        Boolean grid marking cells that may be changed. If omitted, all cells are
        editable.

    valid_mask:
        Boolean grid marking cells that represent real geometry. If provided,
        invalid cells are never changed.

    max_iterations:
        Number of full-grid repair passes, not number of cell flips.
    """
    original = as_binary_numpy_grid(grid)
    repaired = original.copy()

    if mutable_mask is None:
        editable = np.ones_like(repaired, dtype=bool)
    else:
        editable = np.asarray(mutable_mask, dtype=bool)
        if editable.shape != repaired.shape:
            raise ValueError(f"mutable_mask shape {editable.shape} does not match grid shape {repaired.shape}.")

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != repaired.shape:
            raise ValueError(f"valid_mask shape {valid.shape} does not match grid shape {repaired.shape}.")
        editable = editable & valid

    before_count = count_checkerboards(repaired)
    changed_cells: List[Cell] = []
    unrepaired_blocks: List[Block] = []
    warnings: List[str] = []
    seen_states: Set[bytes] = set()

    iteration = 0

    while iteration < max_iterations:
        blocks = illegal_blocks(repaired)
        if not blocks:
            break

        state_key = repaired.tobytes()
        if state_key in seen_states:
            warnings.append("Repair stopped because the grid entered a repeated state.")
            unrepaired_blocks = blocks
            break
        seen_states.add(state_key)

        changed_this_pass = 0
        blocked_this_pass: List[Block] = []

        for block in blocks:
            # This block may already have been fixed by a previous flip in this pass.
            if not _is_illegal_block_at(repaired, block):
                continue

            chosen = _choose_repair_flip_for_block(repaired, block, editable)
            if chosen is None:
                blocked_this_pass.append(block)
                continue

            flipped_cell, repaired = chosen
            changed_cells.append(flipped_cell)
            changed_this_pass += 1

        iteration += 1

        if verbose:
            print(
                f"Pass {iteration:4d} | "
                f"flips={changed_this_pass:4d} | "
                f"checkerboards={count_checkerboards(repaired):4d} | "
                f"blocked={len(blocked_this_pass):4d}"
            )

        if changed_this_pass == 0:
            unrepaired_blocks = illegal_blocks(repaired)
            warnings.append(
                "No editable local repair was available for the remaining illegal blocks. "
                "Try include_fixed=True, or inspect fixed feed/ring cells around the reported blocks."
            )
            break

    if iteration >= max_iterations and count_checkerboards(repaired) > 0:
        warnings.append(f"Stopped after max_iterations={max_iterations} repair passes.")
        unrepaired_blocks = illegal_blocks(repaired)

    return IllegalRepairResult(
        particle=None,
        original_grid=original,
        repaired_grid=repaired,
        changed_cells=changed_cells,
        unrepaired_blocks=unrepaired_blocks,
        before_illegal_count=before_count,
        after_illegal_count=count_checkerboards(repaired),
        iterations=iteration,
        validation_report=None,
        warnings=warnings,
    )


# -----------------------------------------------------------------------------
# WGS-facing API
# -----------------------------------------------------------------------------


def repair_wgs_particle_illegal_positions(
    particle: Any,
    *,
    in_place: bool = False,
    include_fixed: bool = False,
    validate_after: bool = True,
    max_iterations: int = 10_000,
    verbose: bool = False,
) -> IllegalRepairResult:
    """
    Repair illegal diagonal/checkerboard positions in a WGS_Antenna particle.

    Parameters
    ----------
    particle:
        A WGS_Antenna-like object.

    in_place:
        False by default. If False, the particle is deep-copied and the repaired
        copy is returned. If True, particle.conductor_grid is edited directly.

    include_fixed:
        False by default. If False, only cells where particle.map_mask is True
        can be flipped. If True, all real/non-X WGS cells may be flipped.

    validate_after:
        If True and validator_updated_only.py is available, run
        validate_wgs_particle() after writing the repair back into the particle.

    Returns
    -------
    IllegalRepairResult
        The result object contains the repaired particle, original/repaired
        rectangular grids, changed cells, warnings, and optional validation data.
    """
    if not looks_like_wgs_particle(particle):
        raise TypeError("repair_wgs_particle_illegal_positions() expects a WGS_Antenna-like object.")

    target = particle if in_place else copy.deepcopy(particle)

    original_grid = get_wgs_rect_grid(target)
    valid_mask = wgs_valid_rect_mask(target)
    mutable_mask = wgs_mutable_rect_mask(target, include_fixed=include_fixed)

    rect_result = repair_rect_grid_illegal_positions(
        original_grid,
        mutable_mask=mutable_mask,
        valid_mask=valid_mask,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    written_changed_cells = apply_rect_grid_to_wgs_particle(
        target,
        rect_result.repaired_grid,
        only_mutable=not include_fixed,
    )

    validation_report: Optional[Dict[str, object]] = None
    if validate_after and validate_wgs_particle is not None:
        validation_report = validate_wgs_particle(target)

    result = IllegalRepairResult(
        particle=target,
        original_grid=rect_result.original_grid,
        repaired_grid=get_wgs_rect_grid(target),
        changed_cells=written_changed_cells,
        unrepaired_blocks=rect_result.unrepaired_blocks,
        before_illegal_count=rect_result.before_illegal_count,
        after_illegal_count=count_checkerboards(get_wgs_rect_grid(target)),
        iterations=rect_result.iterations,
        validation_report=validation_report,
        warnings=rect_result.warnings,
    )

    return result


def repair_wgs_particle(
    particle: Any,
    *,
    in_place: bool = False,
    include_fixed: bool = False,
    verbose: bool = False,
) -> Any:
    """
    Convenience wrapper that returns only the repaired particle.

    Use repair_wgs_particle_illegal_positions() if you also want the repair
    report/summary.
    """
    return repair_wgs_particle_illegal_positions(
        particle,
        in_place=in_place,
        include_fixed=include_fixed,
        verbose=verbose,
    ).particle


# -----------------------------------------------------------------------------
# Optional demo / smoke test
# -----------------------------------------------------------------------------


def demo_rect_grid() -> None:
    """Small standalone demo that does not require WGS_Antenna."""
    grid = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )

    result = repair_rect_grid_illegal_positions(grid, verbose=True)
    print("Original grid:\n", result.original_grid)
    print("Repaired grid:\n", result.repaired_grid)
    print(result.summary())


if __name__ == "__main__":
    demo_rect_grid()
