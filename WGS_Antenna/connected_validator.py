import os

import matplotlib.pyplot as plt

# Prevent matplotlib from opening a plot window when main.py is imported.
# os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import main

from WGS_pixels import WGS_Antenna, child, mutate

from validator_updated_only import (
    validate_wgs_particle,
    summarize_validation_report,
    grid_or_wgs_to_polygons,
    wgs_to_rect_grid,
)


def validate_particle(name, particle, cell_size):
    report = validate_wgs_particle(particle)
    summary = summarize_validation_report(report)
    polygons = grid_or_wgs_to_polygons(particle, cell_size=cell_size)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Valid: {summary['is_valid']}")
    print(f"Errors: {summary['errors']}")
    print(f"Metal components: {summary['num_metal_components']}")
    print(f"Isolated cells: {summary['num_isolated_cells']}")
    print(f"Diagonal touches: {summary['num_diagonal_touches']}")
    print(f"Disconnected components: {summary['num_disconnected_components']}")
    print(f"Polygons generated: {len(polygons)}")

    return report, polygons

def plot_polygons(name, polygons):
    plt.figure()

    for polygon in polygons:
        vertices = polygon["vertices"]

        if not vertices:
            continue

        closed_vertices = vertices + [vertices[0]]

        xs = [point[0] for point in closed_vertices]
        ys = [point[1] for point in closed_vertices]

        plt.plot(xs, ys)

    plt.axis("equal")
    plt.title(f"{name} unfolded 2D polygon outline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def get_particles_from_main():
    """
    Try to grab particles that main.py already created.
    This works only if particle0, particle1, particle2, particle3
    exist as module-level variables in main.py.
    """
    particles = {}

    for name in ["particle0", "particle1", "particle2", "particle3"]:
        particle = getattr(main, name, None)

        if isinstance(particle, WGS_Antenna):
            particles[name] = particle

    return particles


def make_particles_here():
    """
    Fallback option:
    If main.py does not expose particle0, particle1, particle2, particle3,
    create the same kind of particles here without changing main.py.
    """
    ds = 0.5E-3

    a = 20 * ds
    b = 10 * ds
    c = 100 * ds

    A = int(a / ds)
    B = int(b / ds)
    C = int(c / ds)

    particle0 = WGS_Antenna(
        size=np.array([a, b, c]),
        resolution=np.array([A, B, C]),
        pad_ring_t=np.array([0.5E-2, 0.5E-2]),
        randomized=True,
        alpha=0.5,
    )

    particle1 = WGS_Antenna(
        size=np.array([a, b, c]),
        resolution=np.array([A, B, C]),
        pad_ring_t=np.array([0.5E-2, 0.5E-2]),
        randomized=True,
        alpha=0.5,
    )

    particle2, particle3 = child(particle0, particle1)

    particle2 = mutate(particle2)
    particle3 = mutate(particle3)

    particles = {
        "particle0": particle0,
        "particle1": particle1,
        "particle2": particle2,
        "particle3": particle3,
    }

    return ds, particles

def plot_unfolded_grid(name, particle):
    grid = wgs_to_rect_grid(particle)

    plt.figure()
    plt.imshow(grid, origin="upper", interpolation="none")
    plt.title(f"{name} unfolded 0/1 conductor grid")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.show()

def run_validation():
    # Use main.ds if it exists. Otherwise use your original ds value.
    cell_size = getattr(main, "ds", 0.5E-3)

    particles = get_particles_from_main()

    if not particles:
        print("main.py did not expose particle variables.")
        print("Creating test particles inside connected_validator.py instead.")
        cell_size, particles = make_particles_here()

    results = {}

    for name, particle in particles.items():
        report, polygons = validate_particle(name, particle, cell_size)

        if report["is_valid"]:
            print(f"{name} is valid. Plotting polygon.")
            plot_polygons(name, polygons)
        else:
            print(f"{name} is invalid. Plotting grid only for debugging.")
            plot_unfolded_grid(name, particle)

        results[name] = {
            "report": report,
            "polygons": polygons,
        }
    return results


if __name__ == "__main__":
    run_validation()