import numpy as np

from WGS_pixels import WGS_Antenna, child, mutate

from validator_updated_only import (
    validate_wgs_particle,
    summarize_validation_report,
    grid_or_wgs_to_polygons,
)


def validate_and_print(name, particle, cell_size):
    """
    Validate one WGS_Antenna particle and print a readable summary.
    Also converts the particle into polygon outlines.
    """
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


def main():
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

    validate_and_print("Parent 0", particle0, ds)
    validate_and_print("Parent 1", particle1, ds)

    particle0.display(mask=False)

    particle2, particle3 = child(particle0, particle1)

    particle2 = mutate(particle2)
    particle3 = mutate(particle3)

    report2, polygons2 = validate_and_print("Child 2", particle2, ds)
    report3, polygons3 = validate_and_print("Child 3", particle3, ds)

    if report2["is_valid"]:
        particle2.display(mask=False)
    else:
        print("\nChild 2 is invalid, so it was not displayed.")

    if report3["is_valid"]:
        particle3.display(mask=False)
    else:
        print("Child 3 is invalid, so it was not displayed.")


if __name__ == "__main__":
    main()