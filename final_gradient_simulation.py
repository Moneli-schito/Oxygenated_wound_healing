
"""
Final exploratory spatial-gradient oxygen simulation.

Compares normal and diabetic oxygen transport in a 2D wound bed
using identical initial oxygen fields and different effective
perfusion coefficients.

This model predicts oxygen transport, not clinical wound healing.
All perfusion and initial-condition scenarios are exploratory.

Existing oxygen_recovery_refinement.py and its results are preserved.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    dx,
    dy,
)

from model import (
    full_oxygen_model,
    create_wound_mask,
    create_perfusion_field,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("final_gradient_results")

DT = 0.1
DURATION = 600.0
SAVE_INTERVAL = 1.0

SOURCE_PO2 = 60.0
SURROUNDING_INITIAL_PO2 = 50.0

WOUND_CENTRE_INITIAL_PO2 = 10.0
WOUND_EDGE_INITIAL_PO2 = 40.0

WOUND_RADIUS_CELLS = 15
WOUND_PERFUSION_FRACTION = 0.5

PRIMARY_HYPOXIA_THRESHOLD = 40.0
HYPOXIA_THRESHOLDS = (20.0, 40.0, 48.41)

SNAPSHOT_TIMES = (0, 120, 300, 600)

CONDITIONS = {
    "normal": beta_normal,
    "diabetic": beta_diabetic,
}

NX = int(round(Lx / dx)) + 1
NY = int(round(Ly / dy)) + 1

X_MM = np.linspace(0.0, Lx * 1000.0, NX)
Y_MM = np.linspace(0.0, Ly * 1000.0, NY)

WOUND_MASK = create_wound_mask(
    NX, NY, radius=WOUND_RADIUS_CELLS
)
SURROUNDING_MASK = ~WOUND_MASK

SOURCE_CONCENTRATION = SOURCE_PO2 * alpha_O2

STEPS = int(round(DURATION / DT))
SAVE_EVERY = int(round(SAVE_INTERVAL / DT))

if not np.isclose(STEPS * DT, DURATION):
    raise ValueError("Duration must be divisible by DT.")

if not np.isclose(SAVE_EVERY * DT, SAVE_INTERVAL):
    raise ValueError("Save interval must be divisible by DT.")

if any(
    not np.isclose(round(t / DT) * DT, t)
    for t in SNAPSHOT_TIMES
):
    raise ValueError("Snapshot times must align with DT.")

if D * DT * (1.0 / dx**2 + 1.0 / dy**2) > 0.5:
    raise ValueError("Explicit diffusion stability limit exceeded.")

if not (0.0 <= WOUND_PERFUSION_FRACTION <= 1.0):
    raise ValueError("Invalid wound perfusion fraction.")

if alpha_O2 <= 0:
    raise ValueError("Oxygen conversion factor must be positive.")


# ============================================================
# 2. INITIAL SPATIAL GRADIENT
# ============================================================

def initial_concentration():
    """
    Set the same initial oxygen field for both conditions.

    Outside wound: 50 mmHg.
    Wound centre: 10 mmHg.
    Wound edge: approaches 40 mmHg.

    The gradient is linear with radial distance, and is a
    hypothetical initial condition rather than measured data.
    """

    xx, yy = np.meshgrid(
        np.arange(NX),
        np.arange(NY),
        indexing="ij",
    )

    centre_x = NX // 2
    centre_y = NY // 2

    radial_distance = np.sqrt(
        (xx - centre_x) ** 2
        + (yy - centre_y) ** 2
    )

    radial_fraction = np.clip(
        radial_distance / WOUND_RADIUS_CELLS,
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
        (NX, NY),
        SURROUNDING_INITIAL_PO2,
        dtype=float,
    )

    po2[WOUND_MASK] = wound_po2[WOUND_MASK]

    return po2 * alpha_O2


# ============================================================
# 3. METRICS
# ============================================================

def threshold_column(threshold):
    return f"hypoxic_area_below_{threshold:g}_percent"


def calculate_metrics(C):
    po2 = C / alpha_O2

    metrics = {
        "centre_po2_mmHg": float(
            po2[NX // 2, NY // 2]
        ),
        "mean_wound_po2_mmHg": float(
            np.mean(po2[WOUND_MASK])
        ),
        "mean_surrounding_po2_mmHg": float(
            np.mean(po2[SURROUNDING_MASK])
        ),
        "min_wound_po2_mmHg": float(
            np.min(po2[WOUND_MASK])
        ),
    }

    for threshold in HYPOXIA_THRESHOLDS:
        metrics[threshold_column(threshold)] = float(
            100.0
            * np.mean(po2[WOUND_MASK] < threshold)
        )

    return metrics


# ============================================================
# 4. SIMULATION
# ============================================================

def run_simulation(beta_surrounding):
    C = initial_concentration()

    beta_field = create_perfusion_field(
        WOUND_MASK,
        beta_surrounding=beta_surrounding,
        wound_perfusion_fraction=WOUND_PERFUSION_FRACTION,
    )

    records = []
    snapshots = {}
    clearance_times = {
        threshold: None
        for threshold in HYPOXIA_THRESHOLDS
    }

    snapshot_steps = {
        int(round(t / DT)): t
        for t in SNAPSHOT_TIMES
    }

    for step in range(STEPS + 1):
        time = step * DT

        # Check threshold clearance at every simulation step.
        wound_po2 = C[WOUND_MASK] / alpha_O2

        for threshold in HYPOXIA_THRESHOLDS:
            if (
                clearance_times[threshold] is None
                and np.all(wound_po2 >= threshold)
            ):
                clearance_times[threshold] = round(
                    time, 6
                )

        if step % SAVE_EVERY == 0:
            records.append({
                "time_s": round(time, 6),
                **calculate_metrics(C),
            })

        if step in snapshot_steps:
            snapshots[snapshot_steps[step]] = C.copy()

        if step == STEPS:
            break

        derivative = full_oxygen_model(
            C,
            D,
            Vmax,
            Km,
            rho_cell,
            beta_field,
            SOURCE_CONCENTRATION,
            dx,
            dy,
            boundary_type="zero_flux",
        )

        C = C + DT * derivative

        if (
            not np.all(np.isfinite(C))
            or np.any(C < 0)
        ):
            raise ValueError(
                "Invalid oxygen concentration "
                f"after step {step + 1}."
            )

    return {
        "records": records,
        "snapshots": snapshots,
        "final": C.copy(),
        "clearance_times": clearance_times,
    }


# ============================================================
# 5. CSV EXPORT
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


def save_parameters():
    parameters = [
        ("D_m2_per_s", D),
        ("Vmax_mol_per_cell_per_s", Vmax),
        ("Km_mol_per_m3", Km),
        ("rho_cell_cells_per_m3", rho_cell),
        ("alpha_O2_mol_per_m3_per_mmHg", alpha_O2),
        ("normal_beta_per_s", beta_normal),
        ("diabetic_beta_per_s", beta_diabetic),
        ("wound_perfusion_fraction", WOUND_PERFUSION_FRACTION),
        ("source_po2_mmHg", SOURCE_PO2),
        ("surrounding_initial_po2_mmHg", SURROUNDING_INITIAL_PO2),
        ("wound_centre_initial_po2_mmHg", WOUND_CENTRE_INITIAL_PO2),
        ("wound_edge_nominal_po2_mmHg", WOUND_EDGE_INITIAL_PO2),
        ("primary_hypoxia_threshold_mmHg", PRIMARY_HYPOXIA_THRESHOLD),
        ("domain_length_x_m", Lx),
        ("domain_length_y_m", Ly),
        ("dx_m", dx),
        ("dy_m", dy),
        ("grid_nx", NX),
        ("grid_ny", NY),
        ("wound_radius_cells", WOUND_RADIUS_CELLS),
        ("time_step_s", DT),
        ("duration_s", DURATION),
        ("save_interval_s", SAVE_INTERVAL),
        ("boundary_condition", "zero_flux"),
        ("time_integrator", "explicit_Euler"),
    ]

    save_csv(
        OUTPUT_DIR / "simulation_parameters.csv",
        [
            {"parameter": key, "value": value}
            for key, value in parameters
        ],
    )


# ============================================================
# 6. TIME-SERIES FIGURES
# ============================================================

def plot_time_series(results):
    plots = [
        (
            "mean_wound_po2_mmHg",
            "Mean wound oxygen (mmHg)",
            "mean_wound_oxygen.png",
        ),
        (
            "mean_surrounding_po2_mmHg",
            "Mean surrounding oxygen (mmHg)",
            "mean_surrounding_oxygen.png",
        ),
        (
            "centre_po2_mmHg",
            "Wound-centre oxygen (mmHg)",
            "centre_oxygen.png",
        ),
    ]

    for metric, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(9, 5))

        for condition in CONDITIONS:
            records = results[condition]["records"]

            ax.plot(
                [r["time_s"] for r in records],
                [r[metric] for r in records],
                linewidth=2,
                label=condition.capitalize(),
            )

        if metric != "mean_surrounding_po2_mmHg":
            ax.axhline(
                PRIMARY_HYPOXIA_THRESHOLD,
                color="gray",
                linestyle=":",
                label="Exploratory 40 mmHg threshold",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=250,
        )
        plt.close(fig)


def plot_hypoxic_area(results):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, threshold in zip(
        axes, HYPOXIA_THRESHOLDS
    ):
        metric = threshold_column(threshold)

        for condition in CONDITIONS:
            records = results[condition]["records"]

            ax.plot(
                [r["time_s"] for r in records],
                [r[metric] for r in records],
                linewidth=2,
                label=condition.capitalize(),
            )

        ax.set_title(f"Below {threshold:g} mmHg")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Hypoxic wound area (%)")
    axes[0].legend()

    fig.suptitle(
        "Hypoxic-area sensitivity to oxygen threshold"
    )

    fig.savefig(
        OUTPUT_DIR / "hypoxic_area_thresholds.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 7. SPATIAL MAPS
# ============================================================

def plot_spatial_maps(results):
    fig, axes = plt.subplots(
        2,
        len(SNAPSHOT_TIMES),
        figsize=(16, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    extent = [
        X_MM[0], X_MM[-1],
        Y_MM[0], Y_MM[-1],
    ]

    image = None

    for row, condition in enumerate(CONDITIONS):
        for col, time in enumerate(SNAPSHOT_TIMES):
            ax = axes[row, col]

            field = (
                results[condition]["snapshots"][time]
                / alpha_O2
            )

            image = ax.imshow(
                field.T,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                vmax=SOURCE_PO2,
                aspect="equal",
                interpolation="nearest",
            )

            ax.contour(
                X_MM,
                Y_MM,
                WOUND_MASK.T.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.8,
            )

            ax.set_title(
                f"{condition.capitalize()} — {time} s"
            )

            if row == 1:
                ax.set_xlabel("x (mm)")

            if col == 0:
                ax.set_ylabel("y (mm)")

    fig.colorbar(
        image,
        ax=axes,
        label="Oxygen partial pressure (mmHg)",
        shrink=0.8,
    )

    fig.suptitle(
        "Spatial oxygen distribution: normal vs diabetic"
    )

    fig.savefig(
        OUTPUT_DIR / "spatial_oxygen_maps.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 8. SPATIAL PROFILES
# ============================================================

def plot_centreline_profiles(results):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    centre_y = NY // 2

    initial = (
        initial_concentration()[:, centre_y]
        / alpha_O2
    )

    ax.plot(
        X_MM,
        initial,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Shared initial condition",
    )

    for condition in CONDITIONS:
        final = (
            results[condition]["final"][:, centre_y]
            / alpha_O2
        )

        ax.plot(
            X_MM,
            final,
            linewidth=2,
            label=f"{condition.capitalize()} at 600 s",
        )

    centre_x_mm = X_MM[NX // 2]
    wound_radius_mm = WOUND_RADIUS_CELLS * dx * 1000.0

    ax.axvline(
        centre_x_mm - wound_radius_mm,
        color="gray",
        linestyle=":",
    )
    ax.axvline(
        centre_x_mm + wound_radius_mm,
        color="gray",
        linestyle=":",
    )

    ax.axhline(
        PRIMARY_HYPOXIA_THRESHOLD,
        color="gray",
        linestyle="-.",
        label="40 mmHg threshold",
    )

    ax.set_xlabel("Position across wound (mm)")
    ax.set_ylabel("Oxygen partial pressure (mmHg)")
    ax.set_title("Oxygen profile through wound centre")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "wound_centreline_profiles.png",
        dpi=250,
    )
    plt.close(fig)


# ============================================================
# 9. OXYGEN-CHANGE MAPS
# ============================================================

def plot_oxygen_change(results):
    changes = {}

    for condition in CONDITIONS:
        initial = results[condition]["snapshots"][0]
        final = results[condition]["snapshots"][600]

        changes[condition] = (
            final - initial
        ) / alpha_O2

    maximum = max(
        float(np.max(np.abs(field)))
        for field in changes.values()
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    extent = [
        X_MM[0], X_MM[-1],
        Y_MM[0], Y_MM[-1],
    ]

    for ax, condition in zip(axes, CONDITIONS):
        field = changes[condition]

        image = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-maximum,
            vmax=maximum,
            aspect="equal",
        )

        ax.contour(
            X_MM,
            Y_MM,
            WOUND_MASK.T.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=0.8,
        )

        ax.set_title(condition.capitalize())
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.colorbar(
        image,
        ax=axes,
        label="Change in pO2 over 600 s (mmHg)",
        shrink=0.85,
    )

    fig.suptitle("Change in oxygenation from initial state")

    fig.savefig(
        OUTPUT_DIR / "oxygen_change_maps.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 10. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FINAL EXPLORATORY SPATIAL-GRADIENT SIMULATION")
    print("=" * 65)

    print(f"Grid: {NX} x {NY}")
    print(f"Time step: {DT} s")
    print(f"Duration: {DURATION} s")
    print(f"Source oxygen: {SOURCE_PO2} mmHg")
    print(
        "Initial wound mean oxygen:",
        f"{np.mean(initial_concentration()[WOUND_MASK] / alpha_O2):.2f}",
        "mmHg",
    )

    save_parameters()

    results = []
    simulation_results = {}

    for condition, beta_value in CONDITIONS.items():
        print(f"\nRunning {condition} simulation...")

        result = run_simulation(beta_value)
        simulation_results[condition] = result

        save_csv(
            OUTPUT_DIR / f"{condition}_timeseries.csv",
            result["records"],
        )

        final_metrics = calculate_metrics(result["final"])

        summary = {
            "condition": condition,
            "beta_surrounding_s_inv": beta_value,
            "wound_perfusion_fraction": WOUND_PERFUSION_FRACTION,
            **final_metrics,
        }

        for threshold in HYPOXIA_THRESHOLDS:
            summary[
                f"clearance_below_{threshold:g}_time_s"
            ] = result["clearance_times"][threshold]

        results.append(summary)

        print(
            "Final mean wound oxygen:",
            f"{final_metrics['mean_wound_po2_mmHg']:.4f}",
            "mmHg",
        )

        print(
            "Final mean surrounding oxygen:",
            f"{final_metrics['mean_surrounding_po2_mmHg']:.4f}",
            "mmHg",
        )

        print(
            "40 mmHg threshold clearance time:",
            result["clearance_times"][
                PRIMARY_HYPOXIA_THRESHOLD
            ],
            "s",
        )

    save_csv(
        OUTPUT_DIR / "final_comparison_summary.csv",
        results,
    )

    plot_time_series(simulation_results)
    plot_hypoxic_area(simulation_results)
    plot_spatial_maps(simulation_results)
    plot_centreline_profiles(simulation_results)
    plot_oxygen_change(simulation_results)

    print("\nSimulation complete.")
    print("Results saved in:", OUTPUT_DIR)
    print(
        "Interpretation: exploratory oxygen transport, "
        "not clinical wound-healing prediction."
    )


if __name__ == "__main__":
    main()
