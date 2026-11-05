from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("Final_Report_Figures/Validation")
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# FIGURE 12: BOUNDARY COMPARISON
# --------------------------------------------------

source = Path(
    "corrected_boundary_comparison_results/boundary_comparison.csv"
)

with source.open(newline="") as handle:
    data = list(csv.DictReader(handle))


def series(name):
    rows = sorted(
        [r for r in data if r["scenario"] == name],
        key=lambda r: float(r["time_s"])
    )
    if not rows:
        raise ValueError(f"Missing scenario: {name}")
    return (
        np.array([float(r["time_s"]) for r in rows]),
        np.array([float(r["centre_po2_mmHg"]) for r in rows])
    )


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for row, condition in enumerate(["normal", "diabetic"]):

    t_zero, y_zero = series(f"{condition}_zero_flux")
    t_fixed, y_fixed = series(f"{condition}_fixed")

    if not np.array_equal(t_zero, t_fixed):
        raise ValueError("Time points do not match")

    difference = y_fixed - y_zero

    ax = axes[row, 0]
    ax.plot(
        t_zero, y_zero,
        label="Zero flux",
        linewidth=2.5,
    )
    ax.plot(
        t_fixed, y_fixed,
        label="Fixed oxygen",
        linestyle="--",
        linewidth=1.8,
    )
    ax.set_title(f"{condition.capitalize()}: oxygen trajectories")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wound-centre pO2 (mmHg)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[row, 1]
    ax.plot(t_zero, difference, linewidth=2)
    ax.axhline(0, linestyle=":", color="gray")
    ax.set_title(f"{condition.capitalize()}: fixed minus zero flux")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Difference (mmHg)")
    ax.ticklabel_format(
        style="sci", axis="y", scilimits=(0, 0)
    )
    ax.grid(alpha=0.3)

    print(
        f"{condition.capitalize()} maximum difference: "
        f"{np.max(np.abs(difference)):.10e} mmHg"
    )

fig.suptitle(
    "Sensitivity to outer boundary conditions",
    fontsize=15
)
fig.tight_layout()

fig.savefig(
    OUT / "Figure_12_Boundary_Comparison_Four_Panels.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# --------------------------------------------------
# FIGURE 13: CONVERGENCE
# --------------------------------------------------

source = Path(
    "corrected_model_convergence_results_v2/convergence_results.csv"
)

with source.open(newline="") as handle:
    convergence = list(csv.DictReader(handle))


def get_result(condition, label):
    rows = [
        r for r in convergence
        if r["condition"] == condition
        and r["test_name"] == label
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {condition}, {label}"
        )
    return rows[0]


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for condition in ["normal", "diabetic"]:

    time_labels = ["Baseline", "Time 0.05", "Time 0.025"]
    time_steps = [0.1, 0.05, 0.025]

    time_values = [
        float(get_result(condition, label)["final_centre_mmHg"])
        for label in time_labels
    ]

    axes[0].plot(
        time_steps, time_values,
        marker="o",
        label=condition.capitalize()
    )

    grid_labels = ["Grid 0.2 mm", "Baseline", "Grid 0.05 mm"]
    spacings = [0.2, 0.1, 0.05]

    grid_values = [
        float(get_result(condition, label)["final_centre_mmHg"])
        for label in grid_labels
    ]

    axes[1].plot(
        spacings, grid_values,
        marker="o",
        label=condition.capitalize()
    )

axes[0].set_title("Time-step refinement")
axes[0].set_xlabel("Time step (s)")
axes[0].set_ylabel("Wound-centre pO2 at 600 s (mmHg)")

axes[1].set_title("Spatial-grid refinement")
axes[1].set_xlabel("Grid spacing (mm)")
axes[1].set_ylabel("Wound-centre pO2 at 600 s (mmHg)")

for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle("Numerical convergence assessment", fontsize=15)
fig.tight_layout()

fig.savefig(
    OUT / "Figure_13_Numerical_Convergence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print()
print("FINAL PLOTTING COMPLETE")
print("Created Figure 12 and Figure 13.")
print("All existing simulations and outputs are unchanged.")
