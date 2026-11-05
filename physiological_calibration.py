
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D, Vmax, Km, rho_cell,
    Cb, C0, C_hyp, alpha_O2,
    beta_normal, beta_diabetic,
    Lx, Ly, dx, dy
)

from model import (
    full_oxygen_model,
    create_wound_mask,
    create_perfusion_field
)


# ============================================================
# PHYSIOLOGICAL CALIBRATION - EXPLORATORY SCENARIOS
# ============================================================
#
# This script does not modify the original model parameters.
#
# The scenarios below are illustrative and are NOT validated
# patient-specific or tissue-specific parameter estimates.
#
# The reference oxygen range is used for comparison only.
# ============================================================

OUTPUT_DIR = Path("physiological_calibration_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DT = 0.1
DURATION = 600.0
STEPS = int(round(DURATION / DT))

WOUND_PERFUSION_FRACTION = 0.5
DIABETIC_PERFUSION_RATIO = beta_diabetic / beta_normal

# Illustrative source pO2 scenarios in mmHg.
# Converted to dissolved concentration using the existing
# provisional oxygen solubility coefficient.
SOURCE_PO2_VALUES = [60.0, 80.0, 100.0]

# Illustrative effective perfusion multipliers.
PERFUSION_MULTIPLIERS = [0.25, 0.5, 1.0]

# Literature-informed comparison band for normal skin.
# Not a direct wound-bed calibration target.
REFERENCE_LOW = 35.0
REFERENCE_HIGH = 50.0


# ------------------------------------------------------------
# 1. Grid and wound geometry
# ------------------------------------------------------------

NX = int(round(Lx / dx)) + 1
NY = int(round(Ly / dy)) + 1

CENTRE_X = NX // 2
CENTRE_Y = NY // 2

WOUND_MASK = create_wound_mask(
    NX, NY, radius=15
)

SURROUNDING_MASK = ~WOUND_MASK

DIFFUSION_NUMBER = (
    D * DT * (1.0 / dx**2 + 1.0 / dy**2)
)

if DIFFUSION_NUMBER > 0.5:
    raise ValueError(
        "Time step violates the explicit diffusion "
        "stability condition."
    )


# ------------------------------------------------------------
# 2. Simulation function
# ------------------------------------------------------------

def simulate_condition(
    source_concentration,
    surrounding_beta
):
    beta_field = create_perfusion_field(
        WOUND_MASK,
        beta_surrounding=surrounding_beta,
        wound_perfusion_fraction=(
            WOUND_PERFUSION_FRACTION
        )
    )

    concentration = np.full(
        (NX, NY), C0, dtype=float
    )

    concentration[WOUND_MASK] = 0.05

    initial_centre = concentration[
        CENTRE_X, CENTRE_Y
    ]

    centre_crossing_time = (
        0.0 if initial_centre >= C_hyp else None
    )

    previous_centre = initial_centre

    for step in range(STEPS):
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
            boundary_type="zero_flux"
        )

        concentration = (
            concentration + DT * derivative
        )

        if (
            not np.all(np.isfinite(concentration))
            or np.any(concentration < 0)
        ):
            raise ValueError(
                "Invalid oxygen concentration "
                "during simulation."
            )

        current_centre = concentration[
            CENTRE_X, CENTRE_Y
        ]

        if (
            centre_crossing_time is None
            and previous_centre < C_hyp
            and current_centre >= C_hyp
        ):
            change = (
                current_centre - previous_centre
            )

            fraction = (
                (C_hyp - previous_centre) / change
                if change > 0 else 1.0
            )

            centre_crossing_time = (
                step + fraction
            ) * DT

        previous_centre = current_centre

    return {
        "centre_po2": float(
            concentration[CENTRE_X, CENTRE_Y]
            / alpha_O2
        ),
        "mean_wound_po2": float(
            np.mean(concentration[WOUND_MASK])
            / alpha_O2
        ),
        "mean_surrounding_po2": float(
            np.mean(concentration[SURROUNDING_MASK])
            / alpha_O2
        ),
        "final_hypoxic_fraction": float(
            np.mean(
                concentration[WOUND_MASK] < C_hyp
            )
        ),
        "centre_crossing_time": (
            float(centre_crossing_time)
            if centre_crossing_time is not None
            else None
        )
    }


# ------------------------------------------------------------
# 3. Run exploratory parameter scenarios
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("EXPLORATORY PHYSIOLOGICAL CALIBRATION")
    print("=" * 65)

    print("Grid:", NX, "x", NY)
    print("Duration:", DURATION, "seconds")
    print("Time step:", DT, "seconds")
    print(
        "Diabetic perfusion ratio:",
        round(DIABETIC_PERFUSION_RATIO, 4)
    )
    print(
        "Reference normal skin band:",
        REFERENCE_LOW, "-", REFERENCE_HIGH,
        "mmHg"
    )

    rows = []

    total = (
        len(SOURCE_PO2_VALUES)
        * len(PERFUSION_MULTIPLIERS)
    )

    scenario_number = 0

    for source_po2 in SOURCE_PO2_VALUES:
        source_concentration = (
            alpha_O2 * source_po2
        )

        for multiplier in PERFUSION_MULTIPLIERS:
            scenario_number += 1

            normal_beta = (
                beta_normal * multiplier
            )

            diabetic_beta = (
                normal_beta
                * DIABETIC_PERFUSION_RATIO
            )

            print(
                f"\nScenario {scenario_number}/{total}: "
                f"source={source_po2:.1f} mmHg, "
                f"beta multiplier={multiplier:.2f}"
            )

            normal = simulate_condition(
                source_concentration,
                normal_beta
            )

            diabetic = simulate_condition(
                source_concentration,
                diabetic_beta
            )

            row = {
                "source_po2_mmHg": source_po2,
                "source_concentration_mol_m3": (
                    source_concentration
                ),
                "beta_multiplier": multiplier,
                "normal_beta_s_inverse": normal_beta,
                "diabetic_beta_s_inverse": (
                    diabetic_beta
                ),
                "normal_centre_po2_mmHg": (
                    normal["centre_po2"]
                ),
                "diabetic_centre_po2_mmHg": (
                    diabetic["centre_po2"]
                ),
                "normal_mean_wound_po2_mmHg": (
                    normal["mean_wound_po2"]
                ),
                "diabetic_mean_wound_po2_mmHg": (
                    diabetic["mean_wound_po2"]
                ),
                "normal_mean_surrounding_po2_mmHg": (
                    normal["mean_surrounding_po2"]
                ),
                "diabetic_mean_surrounding_po2_mmHg": (
                    diabetic["mean_surrounding_po2"]
                ),
                "normal_exit_time_s": (
                    normal["centre_crossing_time"]
                ),
                "diabetic_exit_time_s": (
                    diabetic["centre_crossing_time"]
                ),
                "normal_final_hypoxic_fraction": (
                    normal["final_hypoxic_fraction"]
                ),
                "diabetic_final_hypoxic_fraction": (
                    diabetic["final_hypoxic_fraction"]
                )
            }

            rows.append(row)

            print(
                "  Normal mean wound pO2:",
                f'{normal["mean_wound_po2"]:.2f} mmHg'
            )
            print(
                "  Diabetic mean wound pO2:",
                f'{diabetic["mean_wound_po2"]:.2f} mmHg'
            )

    # --------------------------------------------------------
    # 4. Save numerical results
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR / "calibration_scenarios.csv"
    )

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # 5. Plot comparison
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for multiplier in PERFUSION_MULTIPLIERS:
        selected = [
            row for row in rows
            if row["beta_multiplier"] == multiplier
        ]

        x = [
            row["source_po2_mmHg"]
            for row in selected
        ]

        normal_y = [
            row["normal_mean_wound_po2_mmHg"]
            for row in selected
        ]

        diabetic_y = [
            row["diabetic_mean_wound_po2_mmHg"]
            for row in selected
        ]

        ax.plot(
            x,
            normal_y,
            marker="o",
            label=f"Normal, beta x{multiplier}"
        )

        ax.plot(
            x,
            diabetic_y,
            marker="s",
            linestyle="--",
            label=f"Diabetic, beta x{multiplier}"
        )

    ax.axhspan(
        REFERENCE_LOW,
        REFERENCE_HIGH,
        color="gray",
        alpha=0.15,
        label="Normal skin reference band"
    )

    ax.set_xlabel(
        "Assumed source oxygen pO2 (mmHg)"
    )

    ax.set_ylabel(
        "Final mean wound pO2 (mmHg)"
    )

    ax.set_title(
        "Exploratory Oxygen Source and "
        "Perfusion Calibration"
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    figure_path = (
        OUTPUT_DIR / "calibration_comparison.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n" + "=" * 65)
    print("CALIBRATION EXPLORATION COMPLETE")
    print("=" * 65)

    print("Scenarios completed:", len(rows))
    print("CSV saved:", csv_path)
    print("Figure saved:", figure_path)

    print("\nImportant limitations:")
    print("- Scenarios are exploratory, not validated.")
    print("- Oxygen solubility is provisional.")
    print("- Effective perfusion is not calibrated.")
    print("- Normal skin reference values are not")
    print("  direct wound-bed calibration targets.")
    print("- Crossing a threshold does not prove healing.")


if __name__ == "__main__":
    main()
