
"""
Exploratory oxygen-recovery refinement.

Compares normal and diabetic oxygen transport using
a common initial condition and a 60 mmHg oxygen source.

This is a mechanistic oxygenation model, not a
clinically validated prediction of wound healing.

Original simulations and their outputs remain unchanged.
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
    C_hyp,
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

OUTPUT_DIR = Path("oxygen_recovery_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DT = 0.1
DURATION = 600.0
SAVE_INTERVAL = 1.0

SOURCE_PO2 = 60.0
INITIAL_SURROUNDING_PO2 = 50.0
INITIAL_WOUND_CONCENTRATION = 0.05

WOUND_RADIUS_CELLS = 15
WOUND_PERFUSION_FRACTION = 0.5

NX = int(round(Lx / dx)) + 1
NY = int(round(Ly / dy)) + 1

X = np.linspace(0, Lx * 1000, NX)
Y = np.linspace(0, Ly * 1000, NY)

WOUND_MASK = create_wound_mask(
    NX, NY, radius=WOUND_RADIUS_CELLS
)

SURROUNDING_MASK = ~WOUND_MASK

SOURCE_CONCENTRATION = SOURCE_PO2 * alpha_O2
INITIAL_SURROUNDING_CONCENTRATION = (
    INITIAL_SURROUNDING_PO2 * alpha_O2
)

STEPS = int(round(DURATION / DT))
SAVE_EVERY = int(round(SAVE_INTERVAL / DT))

SNAPSHOT_TIMES = [0, 60, 120, 180, 240, 360, 480, 600]

CONDITIONS = {
    "normal": beta_normal,
    "diabetic": beta_diabetic,
}

diffusion_number = D * DT * (
    1.0 / dx**2 + 1.0 / dy**2
)

if diffusion_number > 0.5:
    raise ValueError("Diffusion stability condition violated.")

if not np.isclose(STEPS * DT, DURATION):
    raise ValueError("DURATION must be a multiple of DT.")

if not np.isclose(SAVE_EVERY * DT, SAVE_INTERVAL):
    raise ValueError("SAVE_INTERVAL must be a multiple of DT.")


# ============================================================
# 2. INITIAL CONDITIONS
# ============================================================

def initial_concentration():
    concentration = np.full(
        (NX, NY),
        INITIAL_SURROUNDING_CONCENTRATION,
        dtype=float,
    )

    concentration[WOUND_MASK] = (
        INITIAL_WOUND_CONCENTRATION
    )

    return concentration


# ============================================================
# 3. METRICS
# ============================================================

def calculate_metrics(C):
    po2 = C / alpha_O2

    return {
        "centre_po2_mmHg": float(
            po2[NX // 2, NY // 2]
        ),
        "mean_wound_po2_mmHg": float(
            np.mean(po2[WOUND_MASK])
        ),
        "mean_surrounding_po2_mmHg": float(
            np.mean(po2[SURROUNDING_MASK])
        ),
        "hypoxic_area_percent": float(
            100 * np.mean(C[WOUND_MASK] < C_hyp)
        ),
    }


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

    snapshot_steps = {
        int(round(t / DT)): t
        for t in SNAPSHOT_TIMES
    }

    for step in range(STEPS + 1):
        time = step * DT

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
                f"Invalid oxygen concentration at {time:.2f} s"
            )

    return {
        "records": records,
        "snapshots": snapshots,
        "final": C.copy(),
    }


# ============================================================
# 5. SAVE CSV
# ============================================================

def save_csv(path, records):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 6. TIME-SERIES PLOTS
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
        (
            "hypoxic_area_percent",
            "Hypoxic wound area (%)",
            "hypoxic_area.png",
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

        if metric in (
            "mean_wound_po2_mmHg",
            "centre_po2_mmHg",
        ):
            ax.axhline(
                C_hyp / alpha_O2,
                color="gray",
                linestyle=":",
                label="Provisional hypoxia threshold",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            "Exploratory oxygen recovery: " + ylabel
        )
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)


# ============================================================
# 7. SPATIAL OXYGEN HEATMAPS
# ============================================================

def plot_spatial_maps(results):
    times = SNAPSHOT_TIMES

    # Shared scale across both conditions and all times.
    all_fields = [
        results[condition]["snapshots"][time] / alpha_O2
        for condition in CONDITIONS
        for time in times
    ]

    vmin = min(float(field.min()) for field in all_fields)
    vmax = max(float(field.max()) for field in all_fields)

    fig, axes = plt.subplots(
        2,
        len(times),
        figsize=(24, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    extent = [X[0], X[-1], Y[0], Y[-1]]
    image = None

    for row, condition in enumerate(CONDITIONS):
        for column, time in enumerate(times):
            ax = axes[row, column]

            field = (
                results[condition]["snapshots"][time]
                / alpha_O2
            )

            image = ax.imshow(
                field.T,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )

            ax.contour(
                X,
                Y,
                WOUND_MASK.T.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.8,
            )

            ax.set_title(
                f"{condition.capitalize()}\n{time} s"
            )

            if row == 1:
                ax.set_xlabel("x (mm)")

            if column == 0:
                ax.set_ylabel("y (mm)")

    fig.colorbar(
        image,
        ax=axes,
        label="Oxygen partial pressure (mmHg)",
        shrink=0.75,
    )

    fig.suptitle(
        "Exploratory oxygen recovery: spatial pO2",
        fontsize=15,
    )

    fig.savefig(
        OUTPUT_DIR / "spatial_oxygen_maps.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. OXYGEN-CHANGE MAPS
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
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    extent = [X[0], X[-1], Y[0], Y[-1]]

    for ax, (condition, field) in zip(
        axes, changes.items()
    ):
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
            X,
            Y,
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
        label="Change in pO2 from 0 to 600 s (mmHg)",
        shrink=0.85,
    )

    fig.suptitle(
        "Spatial oxygenation change",
        fontsize=14,
    )

    fig.savefig(
        OUTPUT_DIR / "oxygen_change_maps.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. SUMMARY
# ============================================================

def main():
    print("=" * 65)
    print("EXPLORATORY OXYGEN RECOVERY REFINEMENT")
    print("=" * 65)

    print(f"Grid: {NX} x {NY}")
    print(f"Duration: {DURATION} s")
    print(f"Time step: {DT} s")
    print(f"Source oxygen: {SOURCE_PO2} mmHg")
    print(
        "Initial surrounding oxygen:",
        INITIAL_SURROUNDING_PO2,
        "mmHg",
    )
    print(
        "Initial wound oxygen:",
        INITIAL_WOUND_CONCENTRATION / alpha_O2,
        "mmHg",
    )

    results = {}
    summary = []

    for condition, beta_value in CONDITIONS.items():
        print(f"\nRunning {condition} simulation...")

        result = run_simulation(beta_value)
        results[condition] = result

        save_csv(
            OUTPUT_DIR / f"{condition}_timeseries.csv",
            result["records"],
        )

        final = result["records"][-1]

        summary.append({
            "condition": condition,
            "source_po2_mmHg": SOURCE_PO2,
            "initial_surrounding_po2_mmHg": (
                INITIAL_SURROUNDING_PO2
            ),
            "beta_surrounding_s_inv": beta_value,
            "wound_perfusion_fraction": (
                WOUND_PERFUSION_FRACTION
            ),
            **final,
        })

        print(
            "Final mean wound pO2:",
            f"{final['mean_wound_po2_mmHg']:.2f} mmHg",
        )
        print(
            "Final mean surrounding pO2:",
            f"{final['mean_surrounding_po2_mmHg']:.2f} mmHg",
        )
        print(
            "Final hypoxic area:",
            f"{final['hypoxic_area_percent']:.2f}%",
        )

    save_csv(
        OUTPUT_DIR / "recovery_summary.csv",
        summary,
    )

    plot_time_series(results)
    plot_spatial_maps(results)
    plot_oxygen_change(results)

    print("\nSimulation complete.")
    print("Results saved in:", OUTPUT_DIR)
    print(
        "Reminder: Results are exploratory and "
        "not clinically validated."
    )


if __name__ == "__main__":
    main()
