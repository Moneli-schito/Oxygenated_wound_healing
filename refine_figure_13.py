"""
FIGURE 13 - REFINED NUMERICAL CONVERGENCE

Creates a four-panel numerical error comparison:
1. Normal - time-step refinement
2. Normal - spatial-grid refinement
3. Diabetic - time-step refinement
4. Diabetic - spatial-grid refinement

Uses saved Step 36 results.
Does NOT rerun the PDE simulation.
Does NOT modify original simulation files.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# INPUT AND OUTPUT
# ============================================================

INPUT = Path(
    "corrected_model_convergence_results_v2/"
    "convergence_results.csv"
)

OUTPUT = Path(
    "Final_Report_Figures/Validation/"
    "Figure_13_Refined_Numerical_Convergence.png"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Could not find convergence data: {INPUT}"
    )


# ============================================================
# READ VERIFIED DATA
# ============================================================

with INPUT.open(newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

required_columns = {
    "condition",
    "test_name",
    "final_centre_mmHg",
}

if not rows:
    raise ValueError("Convergence CSV is empty.")

missing = required_columns - set(rows[0].keys())

if missing:
    raise ValueError(
        f"Missing CSV columns: {sorted(missing)}\n"
        f"Available: {list(rows[0].keys())}"
    )


def get_oxygen(condition, test_name):
    matches = [
        row for row in rows
        if row["condition"].strip().lower()
        == condition.lower()
        and row["test_name"].strip().lower()
        == test_name.lower()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one result for "
            f"{condition}, {test_name}; "
            f"found {len(matches)}."
        )

    result = float(matches[0]["final_centre_mmHg"])

    if not np.isfinite(result):
        raise ValueError(
            f"Invalid result: {condition}, {test_name}"
        )

    return result


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

conditions = ["normal", "diabetic"]

# The lists are ordered from finest to coarsest.
time_tests = [
    ("Time 0.025", 0.025),
    ("Time 0.05", 0.05),
    ("Baseline", 0.1),
]

grid_tests = [
    ("Grid 0.05 mm", 0.05),
    ("Baseline", 0.1),
    ("Grid 0.2 mm", 0.2),
]


def calculate_errors(condition, tests):
    values = np.array(
        [get_oxygen(condition, label) for label, _ in tests],
        dtype=float
    )

    resolutions = np.array(
        [resolution for _, resolution in tests],
        dtype=float
    )

    # The finest tested resolution is the numerical reference.
    # This is NOT an exact analytical solution.
    reference = values[0]

    errors = np.abs(values - reference)

    return resolutions, values, errors, reference


# ============================================================
# FOUR-PANEL PLOT
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
    constrained_layout=True
)

colors = {
    "normal": "tab:blue",
    "diabetic": "tab:orange",
}

for row_index, condition in enumerate(conditions):

    for column_index, (tests, resolution_label) in enumerate([
        (time_tests, "Time step (s)"),
        (grid_tests, "Grid spacing (mm)"),
    ]):

        x, oxygen, errors, reference = calculate_errors(
            condition,
            tests
        )

        ax = axes[row_index, column_index]

        ax.plot(
            x,
            errors,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=colors[condition],
        )

        ax.axhline(
            0,
            color="grey",
            linestyle=":",
            linewidth=1
        )

        for xx, error in zip(x, errors):
            if np.isclose(error, 0.0, atol=1e-12):
                label = "Reference"
            else:
                label = f"{error:.6f}"

            ax.annotate(
                label,
                (xx, error),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=9
            )

        metric = (
            "Time-step refinement"
            if column_index == 0
            else "Spatial-grid refinement"
        )

        ax.set_title(
            f"{condition.capitalize()} wound — {metric}",
            fontsize=12
        )

        ax.set_xlabel(resolution_label)
        ax.set_ylabel(
            "Absolute wound-centre difference (mmHg)"
        )

        ax.set_xticks(x)

        maximum_error = float(np.max(errors))

        ax.set_ylim(
            bottom=-0.08 * maximum_error,
            top=1.28 * maximum_error
        )

        ax.grid(alpha=0.3)

        print()
        print(f"{condition.upper()} - {metric}")
        print("-" * 60)

        print(
            f"Finest-resolution reference: "
            f"{reference:.6f} mmHg"
        )

        for resolution, value, error in zip(x, oxygen, errors):
            print(
                f"Resolution={resolution:.3f} | "
                f"Centre={value:.6f} mmHg | "
                f"Difference={error:.8f} mmHg"
            )


fig.suptitle(
    "Numerical refinement: wound-centre oxygen differences at 600 s",
    fontsize=15,
    fontweight="bold"
)

fig.supxlabel(
    "Differences are relative to the finest tested resolution "
    "within each refinement experiment.",
    fontsize=10
)

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print()
print("=" * 65)
print("REFINED FIGURE 13 CREATED")
print("=" * 65)
print("Saved to:", OUTPUT.resolve())
print("Existing simulation data unchanged.")
