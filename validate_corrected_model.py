"""
STEP 36: CORRECTED NUMERICAL CONVERGENCE VALIDATION

Corrects the wound-mask reconstruction problem
identified in Steps 34 and 35.

Experiments:
- Time refinement: dt = 0.1, 0.05, 0.025 seconds
- Grid refinement: dx = 0.2, 0.1, 0.05 mm
- Normal and diabetic wound perfusion
- Duration = 600 seconds

The physical wound radius stays fixed at 1.5 mm.
The baseline grid reproduces the original wound mask.

All results are stored in a new directory.
"""

from pathlib import Path
import csv

import numpy as np

import final_gradient_simulation_corrected as f


OUT = Path("corrected_model_convergence_results_v2")
OUT.mkdir(parents=True, exist_ok=True)

DURATION = 600.0

LX = f.Lx
LY = f.Ly

ORIGINAL_SPACING = f.dx
WOUND_RADIUS_M = f.WOUND_RADIUS_CELLS * f.dx

ALPHA = f.alpha_O2

SURROUNDING_PO2 = f.SURROUNDING_INITIAL_PO2
CENTRE_PO2 = f.WOUND_CENTRE_INITIAL_PO2
EDGE_PO2 = f.WOUND_EDGE_INITIAL_PO2


def make_grid(spacing):
    nx = int(round(LX / spacing)) + 1
    ny = int(round(LY / spacing)) + 1

    dx = LX / (nx - 1)
    dy = LY / (ny - 1)

    if not np.isclose(dx, spacing, rtol=1e-10, atol=1e-15):
        raise ValueError("Unexpected x spacing.")

    if not np.isclose(dy, spacing, rtol=1e-10, atol=1e-15):
        raise ValueError("Unexpected y spacing.")

    ii, jj = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        indexing="ij",
    )

    cx = nx // 2
    cy = ny // 2

    # Integer grid offsets avoid floating-point
    # coordinate cancellation at the wound boundary.
    offset_x = ii - cx
    offset_y = jj - cy

    # Physical squared distance expressed in grid units.
    # At the baseline resolution this is exactly the
    # same integer comparison as the original model.
    if np.isclose(
        spacing,
        ORIGINAL_SPACING,
        rtol=1e-10,
        atol=1e-15,
    ):
        radius_cells = f.WOUND_RADIUS_CELLS

        distance_squared_cells = (
            offset_x**2 + offset_y**2
        )

        mask = (
            distance_squared_cells < radius_cells**2
        )

        radial_fraction = np.clip(
            np.sqrt(distance_squared_cells)
            / radius_cells,
            0.0,
            1.0,
        )

    else:
        # Keep the physical radius fixed on refined
        # and coarsened grids.
        distance_squared = (
            (offset_x * dx)**2
            + (offset_y * dy)**2
        )

        # Rounded-grid geometry may differ slightly
        # across resolutions: assess this in results.
        mask = distance_squared < WOUND_RADIUS_M**2

        radial_fraction = np.clip(
            np.sqrt(distance_squared)
            / WOUND_RADIUS_M,
            0.0,
            1.0,
        )

    return dx, dy, mask, radial_fraction


def initial_concentration(mask, radial_fraction):
    wound_po2 = (
        CENTRE_PO2
        + (EDGE_PO2 - CENTRE_PO2)
        * radial_fraction
    )

    po2 = np.full(
        mask.shape,
        SURROUNDING_PO2,
        dtype=float,
    )

    po2[mask] = wound_po2[mask]

    return po2 * ALPHA


def perfusion_field(mask, condition):
    beta = np.full(
        mask.shape,
        f.beta_normal,
        dtype=float,
    )

    if condition == "normal":
        wound_beta = (
            f.beta_normal
            * f.WOUND_PERFUSION_FRACTION
        )

    elif condition == "diabetic":
        wound_beta = (
            f.beta_diabetic
            * f.WOUND_PERFUSION_FRACTION
        )

    else:
        raise ValueError("Unknown condition.")

    beta[mask] = wound_beta

    return beta


def laplacian_zero_flux(C, dx, dy):
    padded = np.pad(C, 1, mode="edge")

    second_x = (
        padded[2:, 1:-1]
        - 2.0 * C
        + padded[:-2, 1:-1]
    ) / dx**2

    second_y = (
        padded[1:-1, 2:]
        - 2.0 * C
        + padded[1:-1, :-2]
    ) / dy**2

    return second_x + second_y


def verify_baseline_geometry():
    dx, dy, mask, radial_fraction = make_grid(
        ORIGINAL_SPACING
    )

    generated_C = initial_concentration(
        mask, radial_fraction
    )

    reference_C = f.initial_concentration()

    different_cells = np.count_nonzero(
        mask != f.WOUND_MASK
    )

    max_initial_difference = np.max(
        np.abs(generated_C - reference_C)
    ) / ALPHA

    print("=" * 75)
    print("BASELINE REPRODUCTION CHECK")
    print("=" * 75)

    print("Reference wound cells:", f.WOUND_MASK.sum())
    print("Generated wound cells:", mask.sum())
    print("Different mask cells:", different_cells)

    print(
        "Maximum initial oxygen difference:",
        f"{max_initial_difference:.12e} mmHg"
    )

    if different_cells != 0:
        raise RuntimeError(
            "Baseline wound mask still differs."
        )

    if max_initial_difference > 1e-8:
        raise RuntimeError(
            "Baseline initial oxygen does not match."
        )

    print("Baseline geometry and initial state: PASS")


def simulate(condition, spacing, dt):
    dx, dy, mask, radial_fraction = make_grid(
        spacing
    )

    diffusion_number = (
        f.D * dt
        * (1.0 / dx**2 + 1.0 / dy**2)
    )

    if diffusion_number > 0.5:
        raise ValueError(
            "Explicit diffusion stability limit exceeded."
        )

    C = initial_concentration(
        mask, radial_fraction
    )

    beta = perfusion_field(mask, condition)

    steps = int(round(DURATION / dt))

    if not np.isclose(
        steps * dt,
        DURATION,
        rtol=1e-12,
        atol=1e-10,
    ):
        raise ValueError("Invalid number of time steps.")

    centre = (
        C.shape[0] // 2,
        C.shape[1] // 2,
    )

    initial_centre = float(C[centre] / ALPHA)

    initial_mean = float(
        C[mask].mean() / ALPHA
    )

    for step in range(steps):
        diffusion = (
            f.D
            * laplacian_zero_flux(C, dx, dy)
        )

        consumption = (
            f.rho_cell * f.Vmax
            * C / (f.Km + C)
        )

        perfusion = (
            beta * (f.SOURCE_CONCENTRATION - C)
        )

        C = C + dt * (
            diffusion - consumption + perfusion
        )

        if not np.all(np.isfinite(C)):
            raise RuntimeError(
                f"Non-finite concentration at step {step}"
            )

        if C.min() < -1e-12:
            raise RuntimeError(
                f"Negative concentration at step {step}"
            )

    po2 = C / ALPHA

    return {
        "condition": condition,
        "spacing_mm": spacing * 1000.0,
        "dt_s": dt,
        "grid_size": f"{C.shape[0]}x{C.shape[1]}",
        "wound_cells": int(mask.sum()),
        "diffusion_number": diffusion_number,
        "initial_centre_mmHg": initial_centre,
        "initial_wound_mean_mmHg": initial_mean,
        "final_centre_mmHg": float(po2[centre]),
        "final_wound_mean_mmHg": float(
            po2[mask].mean()
        ),
        "final_wound_min_mmHg": float(
            po2[mask].min()
        ),
    }


verify_baseline_geometry()

CONFIGURATIONS = [
    ("Baseline", 0.0001, 0.1),
    ("Time 0.05", 0.0001, 0.05),
    ("Time 0.025", 0.0001, 0.025),
    ("Grid 0.2 mm", 0.0002, 0.1),
    ("Grid 0.05 mm", 0.00005, 0.1),
]

results = []

print()
print("=" * 75)
print("STEP 36 - CORRECTED NUMERICAL CONVERGENCE")
print("=" * 75)

for condition in ["normal", "diabetic"]:
    print(f"\n{condition.upper()}")
    print("-" * 75)

    for label, spacing, dt in CONFIGURATIONS:
        print(f"Running {label} ...", flush=True)

        result = simulate(
            condition,
            spacing,
            dt,
        )

        result["test_name"] = label
        results.append(result)

        print(
            f"  Grid={result['grid_size']}, "
            f"Wound cells={result['wound_cells']}, "
            f"Centre={result['final_centre_mmHg']:.6f}, "
            f"Mean={result['final_wound_mean_mmHg']:.6f}"
        )


def get_result(condition, label):
    return next(
        row for row in results
        if row["condition"] == condition
        and row["test_name"] == label
    )


print()
print("=" * 75)
print("TIME-STEP CONVERGENCE")
print("=" * 75)

for condition in ["normal", "diabetic"]:
    reference = get_result(
        condition,
        "Time 0.025",
    )

    print(f"\n{condition.upper()}")

    for label in [
        "Baseline",
        "Time 0.05",
        "Time 0.025",
    ]:
        result = get_result(condition, label)

        centre_error = abs(
            result["final_centre_mmHg"]
            - reference["final_centre_mmHg"]
        )

        mean_error = abs(
            result["final_wound_mean_mmHg"]
            - reference["final_wound_mean_mmHg"]
        )

        print(
            f"{label:<15} "
            f"Centre error={centre_error:.8f} mmHg | "
            f"Mean error={mean_error:.8f} mmHg"
        )


print()
print("=" * 75)
print("SPATIAL-GRID CONVERGENCE")
print("=" * 75)

for condition in ["normal", "diabetic"]:
    reference = get_result(
        condition,
        "Grid 0.05 mm",
    )

    print(f"\n{condition.upper()}")

    for label in [
        "Grid 0.2 mm",
        "Baseline",
        "Grid 0.05 mm",
    ]:
        result = get_result(condition, label)

        centre_error = abs(
            result["final_centre_mmHg"]
            - reference["final_centre_mmHg"]
        )

        mean_error = abs(
            result["final_wound_mean_mmHg"]
            - reference["final_wound_mean_mmHg"]
        )

        print(
            f"{label:<15} "
            f"Centre error={centre_error:.8f} mmHg | "
            f"Mean error={mean_error:.8f} mmHg"
        )


print()
print("=" * 75)
print("CROSS-CHECK AGAINST ORIGINAL BASELINE")
print("=" * 75)

previous = {
    "normal": {
        "centre": 52.3717,
        "mean": 53.4020,
    },
    "diabetic": {
        "centre": 49.9386,
        "mean": 51.7327,
    },
}

for condition in ["normal", "diabetic"]:
    result = get_result(condition, "Baseline")

    centre_diff = (
        result["final_centre_mmHg"]
        - previous[condition]["centre"]
    )

    mean_diff = (
        result["final_wound_mean_mmHg"]
        - previous[condition]["mean"]
    )

    print(
        f"{condition.capitalize()}: "
        f"Centre difference={centre_diff:+.6f} mmHg, "
        f"Mean difference={mean_diff:+.6f} mmHg"
    )

    if (
        abs(centre_diff) > 0.001
        or abs(mean_diff) > 0.001
    ):
        print(
            "WARNING: Baseline still differs. "
            "Investigate before accepting validation."
        )
    else:
        print("Baseline reproduction: PASS")


columns = [
    "condition",
    "test_name",
    "spacing_mm",
    "dt_s",
    "grid_size",
    "wound_cells",
    "diffusion_number",
    "initial_centre_mmHg",
    "initial_wound_mean_mmHg",
    "final_centre_mmHg",
    "final_wound_mean_mmHg",
    "final_wound_min_mmHg",
]

csv_path = OUT / "convergence_results.csv"

with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=columns,
    )
    writer.writeheader()
    writer.writerows(results)

print()
print("Saved:", csv_path.resolve())
print("Main simulation and previous results unchanged.")
print()
print(
    "Interpretation reminder: grid refinement changes "
    "the discrete representation of the circular wound. "
    "Convergence does not establish physiological validity."
)
