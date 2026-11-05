
"""
Final simulation figures for oxygen transport in wound healing.

Compares normal and diabetic perfusion conditions under:
1. Original baseline oxygen supply.
2. Exploratory reduced oxygen supply.

All outputs are saved separately from previous analyses.

IMPORTANT:
These simulations are mechanistic and exploratory.
They are not clinically validated wound-healing predictions.
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
    Cb,
    C0,
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

OUTPUT_DIR = Path("final_simulation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DT = 0.1
DURATION = 600.0
SAVE_INTERVAL = 1.0

WOUND_RADIUS_CELLS = 15
WOUND_INITIAL_CONCENTRATION = 0.05
WOUND_PERFUSION_FRACTION = 0.5

DIABETIC_PERFUSION_RATIO = beta_diabetic / beta_normal

NX = int(round(Lx / dx)) + 1
NY = int(round(Ly / dy)) + 1

X = np.linspace(0.0, Lx * 1000.0, NX)
Y = np.linspace(0.0, Ly * 1000.0, NY)

CENTRE_X = NX // 2
CENTRE_Y = NY // 2

WOUND_MASK = create_wound_mask(
    NX,
    NY,
    radius=WOUND_RADIUS_CELLS,
)

SURROUNDING_MASK = ~WOUND_MASK

STEPS = int(round(DURATION / DT))
SAVE_EVERY_STEPS = int(round(SAVE_INTERVAL / DT))

if SAVE_EVERY_STEPS < 1:
    raise ValueError("SAVE_INTERVAL must be >= DT.")

if not np.isclose(
    SAVE_EVERY_STEPS * DT,
    SAVE_INTERVAL,
):
    raise ValueError(
        "SAVE_INTERVAL must be a multiple of DT."
    )

DIFFUSION_NUMBER = D * DT * (
    1.0 / dx**2 + 1.0 / dy**2
)

if DIFFUSION_NUMBER > 0.5:
    raise ValueError(
        "Explicit diffusion stability limit exceeded."
    )


SCENARIOS = {
    "original_baseline": {
        "title": "Original baseline",
        "source_concentration": Cb,
        "perfusion_multiplier": 1.0,
    },
    "reduced_supply": {
        "title": "Exploratory reduced oxygen supply",
        "source_concentration": 60.0 * alpha_O2,
        "perfusion_multiplier": 0.25,
    },
}


# ============================================================
# 2. INITIAL CONDITIONS
# ============================================================

def initial_concentration():
    concentration = np.full(
        (NX, NY),
        C0,
        dtype=float,
    )

    concentration[WOUND_MASK] = (
        WOUND_INITIAL_CONCENTRATION
    )

    return concentration


# ============================================================
# 3. SUMMARY METRICS
# ============================================================

def calculate_metrics(concentration):
    po2 = concentration / alpha_O2

    centre_po2 = float(
        po2[CENTRE_X, CENTRE_Y]
    )

    mean_wound_po2 = float(
        np.mean(po2[WOUND_MASK])
    )

    mean_surrounding_po2 = float(
        np.mean(po2[SURROUNDING_MASK])
    )

    hypoxic_fraction = float(
        np.mean(
            concentration[WOUND_MASK] < C_hyp
        )
    )

    wound_surrounding_difference = (
        mean_surrounding_po2 - mean_wound_po2
    )

    return {
        "centre_po2_mmHg": centre_po2,
        "mean_wound_po2_mmHg": mean_wound_po2,
        "mean_surrounding_po2_mmHg": (
            mean_surrounding_po2
        ),
        "wound_surrounding_difference_mmHg": (
            wound_surrounding_difference
        ),
        "hypoxic_fraction": hypoxic_fraction,
    }


# ============================================================
# 4. RUN SIMULATION
# ============================================================

def run_simulation(
    source_concentration,
    surrounding_beta,
):
    beta_field = create_perfusion_field(
        WOUND_MASK,
        beta_surrounding=surrounding_beta,
        wound_perfusion_fraction=(
            WOUND_PERFUSION_FRACTION
        ),
    )

    concentration = initial_concentration()

    records = []
    snapshots = {}

    snapshot_times = [
        0.0,
        60.0,
        180.0,
        300.0,
        600.0,
    ]

    snapshot_steps = {
        int(round(time / DT)): time
        for time in snapshot_times
    }

    for step in range(STEPS + 1):
        time = step * DT

        if step % SAVE_EVERY_STEPS == 0:
            metrics = calculate_metrics(
                concentration
            )

            records.append({
                "time_s": time,
                **metrics,
            })

        if step in snapshot_steps:
            snapshot_time = snapshot_steps[step]

            snapshots[snapshot_time] = (
                concentration.copy()
            )

        if step == STEPS:
            break

        derivative = full_oxygen_model(
            concentration,
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

        concentration = (
            concentration + DT * derivative
        )

        if (
            not np.all(np.isfinite(concentration))
            or np.any(concentration < 0)
        ):
            raise ValueError(
                f"Invalid concentration at t={time:.2f} s"
            )

    return {
        "records": records,
        "snapshots": snapshots,
        "final_concentration": concentration.copy(),
    }


# ============================================================
# 5. SAVE CSV FILES
# ============================================================

def save_time_series_csv(path, records):
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


def save_summary_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 6. SPATIAL OXYGEN MAPS
# ============================================================

def plot_spatial_maps(
    scenario_name,
    scenario_title,
    results,
):
    normal_snapshots = results["normal"]["snapshots"]
    diabetic_snapshots = results["diabetic"]["snapshots"]

    times = [0.0, 60.0, 180.0, 600.0]

    all_maps = []

    for condition in ["normal", "diabetic"]:
        for time in times:
            field = (
                results[condition]["snapshots"][time]
                / alpha_O2
            )
            all_maps.append(field)

    vmin = min(
        float(np.min(field))
        for field in all_maps
    )

    vmax = max(
        float(np.max(field))
        for field in all_maps
    )

    fig, axes = plt.subplots(
        2,
        len(times),
        figsize=(16, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    extent = [
        X[0],
        X[-1],
        Y[0],
        Y[-1],
    ]

    image = None

    for row_index, condition in enumerate(
        ["normal", "diabetic"]
    ):
        for column_index, time in enumerate(times):
            ax = axes[row_index, column_index]

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
                interpolation="nearest",
                aspect="equal",
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
                f"{condition.capitalize()}, "
                f"t = {time:.0f} s"
            )

            if row_index == 1:
                ax.set_xlabel("x (mm)")

            if column_index == 0:
                ax.set_ylabel("y (mm)")

    fig.colorbar(
        image,
        ax=axes,
        label="Oxygen partial pressure (mmHg)",
        shrink=0.75,
    )

    fig.suptitle(
        f"{scenario_title}: Spatial oxygen distribution",
        fontsize=14,
    )

    fig.savefig(
        OUTPUT_DIR / f"{scenario_name}_spatial_maps.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 7. TIME SERIES FIGURES
# ============================================================

def plot_time_series(
    scenario_name,
    scenario_title,
    results,
):
    metrics = [
        (
            "centre_po2_mmHg",
            "Wound-centre pO2 (mmHg)",
            "centre_po2",
        ),
        (
            "mean_wound_po2_mmHg",
            "Mean wound pO2 (mmHg)",
            "mean_wound_po2",
        ),
        (
            "hypoxic_fraction",
            "Hypoxic wound area (%)",
            "hypoxic_area",
        ),
        (
            "wound_surrounding_difference_mmHg",
            "Surrounding minus wound pO2 (mmHg)",
            "wound_surrounding_difference",
        ),
    ]

    for metric_key, ylabel, filename_suffix in metrics:
        fig, ax = plt.subplots(
            figsize=(9, 5),
        )

        for condition, label in [
            ("normal", "Normal"),
            ("diabetic", "Diabetic"),
        ]:
            records = results[condition]["records"]

            times = [
                record["time_s"]
                for record in records
            ]

            values = [
                record[metric_key]
                for record in records
            ]

            if metric_key == "hypoxic_fraction":
                values = [
                    value * 100.0
                    for value in values
                ]

            ax.plot(
                times,
                values,
                label=label,
                linewidth=2,
            )

        if metric_key == "centre_po2_mmHg":
            ax.axhline(
                C_hyp / alpha_O2,
                color="gray",
                linestyle=":",
                label="Provisional hypoxia threshold",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{scenario_title}: {ylabel}"
        )

        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"{scenario_name}_{filename_suffix}.png",
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

def create_summary(scenario_name, results):
    rows = []

    for condition in ["normal", "diabetic"]:
        records = results[condition]["records"]

        final = records[-1]

        threshold = C_hyp / alpha_O2

        exit_time = None

        if records[0]["centre_po2_mmHg"] >= threshold:
            exit_time = 0.0
        else:
            for previous, current in zip(
                records[:-1],
                records[1:],
            ):
                previous_po2 = previous[
                    "centre_po2_mmHg"
                ]

                current_po2 = current[
                    "centre_po2_mmHg"
                ]

                if (
                    previous_po2 < threshold
                    and current_po2 >= threshold
                ):
                    change = (
                        current_po2 - previous_po2
                    )

                    fraction = (
                        (threshold - previous_po2)
                        / change
                    )

                    exit_time = (
                        previous["time_s"]
                        + fraction
                        * (
                            current["time_s"]
                            - previous["time_s"]
                        )
                    )
                    break

        rows.append({
            "scenario": scenario_name,
            "condition": condition,
            "final_centre_po2_mmHg": (
                final["centre_po2_mmHg"]
            ),
            "final_mean_wound_po2_mmHg": (
                final["mean_wound_po2_mmHg"]
            ),
            "final_mean_surrounding_po2_mmHg": (
                final["mean_surrounding_po2_mmHg"]
            ),
            "final_wound_surrounding_difference_mmHg": (
                final[
                    "wound_surrounding_difference_mmHg"
                ]
            ),
            "final_hypoxic_area_percent": (
                100.0 * final["hypoxic_fraction"]
            ),
            "centre_threshold_crossing_time_s": (
                exit_time
            ),
        })

    return rows


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("FINAL WOUND OXYGEN SIMULATIONS")
    print("=" * 65)

    print(f"Grid: {NX} x {NY}")
    print(f"Time step: {DT} s")
    print(f"Duration: {DURATION} s")
    print(
        "Provisional hypoxia threshold:",
        f"{C_hyp / alpha_O2:.2f} mmHg",
    )

    all_summary_rows = []

    for scenario_name, config in SCENARIOS.items():
        print("\n" + "-" * 65)
        print("Scenario:", config["title"])
        print("-" * 65)

        normal_beta = (
            beta_normal
            * config["perfusion_multiplier"]
        )

        diabetic_beta = (
            normal_beta
            * DIABETIC_PERFUSION_RATIO
        )

        source = config["source_concentration"]

        results = {}

        for condition, beta in [
            ("normal", normal_beta),
            ("diabetic", diabetic_beta),
        ]:
            print(
                f"Running {condition} condition..."
            )

            result = run_simulation(
                source,
                beta,
            )

            results[condition] = result

            csv_path = (
                OUTPUT_DIR
                / f"{scenario_name}_{condition}_timeseries.csv"
            )

            save_time_series_csv(
                csv_path,
                result["records"],
            )

        plot_spatial_maps(
            scenario_name,
            config["title"],
            results,
        )

        plot_time_series(
            scenario_name,
            config["title"],
            results,
        )

        summary_rows = create_summary(
            scenario_name,
            results,
        )

        all_summary_rows.extend(
            summary_rows
        )

        for row in summary_rows:
            crossing = row[
                "centre_threshold_crossing_time_s"
            ]

            crossing_text = (
                f"{crossing:.2f} s"
                if crossing is not None
                else "Not reached"
            )

            print(
                f"\n{row['condition'].capitalize()}:"
            )

            print(
                "  Final mean wound pO2:",
                f"{row['final_mean_wound_po2_mmHg']:.2f}",
                "mmHg",
            )

            print(
                "  Final hypoxic area:",
                f"{row['final_hypoxic_area_percent']:.2f}%",
            )

            print(
                "  Centre threshold crossing:",
                crossing_text,
            )

    summary_path = (
        OUTPUT_DIR / "final_summary.csv"
    )

    save_summary_csv(
        summary_path,
        all_summary_rows,
    )

    print("\n" + "=" * 65)
    print("FINAL SIMULATIONS COMPLETE")
    print("=" * 65)

    print("Results directory:", OUTPUT_DIR)
    print("Summary CSV:", summary_path)

    print(
        "\nInterpretation reminder:"
        "\nThese are exploratory oxygen transport"
        "\nsimulations, not validated predictions"
        "\nof clinical wound healing."
    )


if __name__ == "__main__":
    main()
