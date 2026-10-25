
"""
Numerical resolution validation for the final spatial-gradient model.

Configurations:
1. Baseline: dx = 0.1 mm, dt = 0.1 s
2. Finer time: dx = 0.1 mm, dt = 0.05 s
3. Finer grid: dx = 0.05 mm, dt = 0.1 s

All simulations run for 600 seconds.

The wound mask uses integer grid coordinates, matching
final_gradient_simulation.py at the baseline resolution.

This is a numerical resolution-sensitivity assessment,
not clinical validation or a formal convergence-order study.
"""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from parameters import (
    D,
    Vmax,
    Km,
    rho_cell,
    alpha_O2,
    beta_normal,
    beta_diabetic,
    Lx,
    Ly,
)

from model import (
    full_oxygen_model,
    create_wound_mask,
    create_perfusion_field,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("final_gradient_validation")

DURATION = 600.0

SOURCE_PO2 = 60.0
SURROUNDING_INITIAL_PO2 = 50.0

WOUND_CENTRE_INITIAL_PO2 = 10.0
WOUND_EDGE_INITIAL_PO2 = 40.0

WOUND_RADIUS_M = 0.0015
WOUND_PERFUSION_FRACTION = 0.5

HYPOXIA_THRESHOLDS = (20.0, 40.0, 48.41)

CONDITIONS = {
    "normal": beta_normal,
    "diabetic": beta_diabetic,
}

CONFIGURATIONS = [
    {
        "name": "Baseline",
        "dx": 1e-4,
        "dy": 1e-4,
        "dt": 0.1,
    },
    {
        "name": "Finer time",
        "dx": 1e-4,
        "dy": 1e-4,
        "dt": 0.05,
    },
    {
        "name": "Finer grid",
        "dx": 5e-5,
        "dy": 5e-5,
        "dt": 0.1,
    },
]


# ============================================================
# 2. INITIAL CONDITIONS
# ============================================================

def prepare_grid(dx, dy):
    nx = int(round(Lx / dx)) + 1
    ny = int(round(Ly / dy)) + 1

    # Use the same integer-coordinate wound-mask
    # convention as the final simulation.
    radius_cells_x = WOUND_RADIUS_M / dx
    radius_cells_y = WOUND_RADIUS_M / dy

    if not np.isclose(radius_cells_x, radius_cells_y):
        raise ValueError(
            "This validation requires equal spatial spacing."
        )

    radius_cells = int(round(radius_cells_x))

    wound_mask = create_wound_mask(
        nx,
        ny,
        radius=radius_cells,
    )

    xx, yy = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        indexing="ij",
    )

    centre_x = nx // 2
    centre_y = ny // 2

    radial_distance = np.sqrt(
        (xx - centre_x) ** 2
        + (yy - centre_y) ** 2
    )

    radial_fraction = np.clip(
        radial_distance / radius_cells,
        0.0,
        1.0,
    )

    wound_po2 = (
        WOUND_CENTRE_INITIAL_PO2
        + (
            WOUND_EDGE_INITIAL_PO2
            - WOUND_CENTRE_INITIAL_PO2
        ) * radial_fraction
    )

    po2 = np.full(
        (nx, ny),
        SURROUNDING_INITIAL_PO2,
        dtype=float,
    )

    po2[wound_mask] = wound_po2[wound_mask]

    return po2 * alpha_O2, wound_mask


# ============================================================
# 3. SIMULATION
# ============================================================

def run_simulation(beta_surrounding, dx, dy, dt):
    diffusion_number = D * dt * (
        1.0 / dx**2 + 1.0 / dy**2
    )

    if diffusion_number > 0.5:
        raise ValueError(
            "Explicit diffusion stability condition exceeded."
        )

    steps = int(round(DURATION / dt))

    if not np.isclose(steps * dt, DURATION):
        raise ValueError(
            "Duration must be divisible by time step."
        )

    C, wound_mask = prepare_grid(dx, dy)

    nx, ny = C.shape
    surrounding_mask = ~wound_mask

    beta_field = create_perfusion_field(
        wound_mask,
        beta_surrounding=beta_surrounding,
        wound_perfusion_fraction=WOUND_PERFUSION_FRACTION,
    )

    source_concentration = SOURCE_PO2 * alpha_O2

    clearance_times = {
        threshold: None
        for threshold in HYPOXIA_THRESHOLDS
    }

    start = perf_counter()

    for step in range(steps + 1):
        time = step * dt

        wound_po2 = C[wound_mask] / alpha_O2

        for threshold in HYPOXIA_THRESHOLDS:
            if (
                clearance_times[threshold] is None
                and np.all(wound_po2 >= threshold)
            ):
                clearance_times[threshold] = round(
                    time, 6
                )

        if step == steps:
            break

        derivative = full_oxygen_model(
            C,
            D,
            Vmax,
            Km,
            rho_cell,
            beta_field,
            source_concentration,
            dx,
            dy,
            boundary_type="zero_flux",
        )

        C = C + dt * derivative

        if (
            not np.all(np.isfinite(C))
            or np.any(C < 0)
        ):
            raise ValueError(
                f"Invalid concentration at step {step + 1}."
            )

    elapsed = perf_counter() - start
    final_po2 = C / alpha_O2

    return {
        "grid": f"{nx} x {ny}",
        "wound_cells": int(np.sum(wound_mask)),
        "dx_mm": dx * 1000.0,
        "dt_s": dt,
        "diffusion_number": diffusion_number,
        "final_centre_po2_mmHg": float(
            final_po2[nx // 2, ny // 2]
        ),
        "final_mean_wound_po2_mmHg": float(
            np.mean(final_po2[wound_mask])
        ),
        "final_mean_surrounding_po2_mmHg": float(
            np.mean(final_po2[surrounding_mask])
        ),
        "clearance_below_20_s": clearance_times[20.0],
        "clearance_below_40_s": clearance_times[40.0],
        "clearance_below_48_41_s": clearance_times[48.41],
        "runtime_s": elapsed,
    }


# ============================================================
# 4. CSV EXPORT
# ============================================================

def save_csv(path, records):
    if not records:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 5. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL SPATIAL-GRADIENT NUMERICAL VALIDATION")
    print("=" * 70)
    print(f"Duration: {DURATION} seconds")
    print("Boundary condition: zero flux")
    print("Integration method: explicit Euler")

    results = []

    for configuration in CONFIGURATIONS:
        print(
            "\nConfiguration:",
            configuration["name"],
        )

        for condition, beta_value in CONDITIONS.items():
            result = run_simulation(
                beta_value,
                configuration["dx"],
                configuration["dy"],
                configuration["dt"],
            )

            row = {
                "configuration": configuration["name"],
                "condition": condition,
                **result,
            }

            results.append(row)

            print(
                f"  {condition.capitalize()}: "
                f"grid={result['grid']}, "
                f"wound cells={result['wound_cells']}, "
                f"dt={result['dt_s']} s"
            )

            print(
                "    Final wound mean:",
                f"{result['final_mean_wound_po2_mmHg']:.4f}",
                "mmHg",
            )

            print(
                "    Final surrounding mean:",
                f"{result['final_mean_surrounding_po2_mmHg']:.4f}",
                "mmHg",
            )

            print(
                "    Clearance below 40 mmHg:",
                result["clearance_below_40_s"],
                "s",
            )

            print(
                "    Runtime:",
                f"{result['runtime_s']:.2f}",
                "s",
            )

    save_csv(
        OUTPUT_DIR / "resolution_comparison.csv",
        results,
    )

    print("\n" + "=" * 70)
    print("DEVIATIONS FROM BASELINE")
    print("=" * 70)

    for condition in CONDITIONS:
        baseline = next(
            row
            for row in results
            if (
                row["configuration"] == "Baseline"
                and row["condition"] == condition
            )
        )

        for row in results:
            if (
                row["condition"] != condition
                or row["configuration"] == "Baseline"
            ):
                continue

            wound_difference = abs(
                row["final_mean_wound_po2_mmHg"]
                - baseline["final_mean_wound_po2_mmHg"]
            )

            clearance_difference = abs(
                row["clearance_below_40_s"]
                - baseline["clearance_below_40_s"]
            )

            print(
                f"{condition.capitalize()} — "
                f"{row['configuration']}: "
                f"wound difference={wound_difference:.6f} mmHg, "
                f"clearance difference={clearance_difference:.3f} s"
            )

    print("\nResults saved to:", OUTPUT_DIR)
    print(
        "These comparisons assess numerical resolution "
        "sensitivity, not clinical accuracy."
    )


if __name__ == "__main__":
    main()
