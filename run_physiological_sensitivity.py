"""
STEP 38 - PHYSIOLOGICAL PARAMETER SENSITIVITY

Investigates uncertainty in:
1. Effective oxygen source pressure
2. Wound perfusion strength
3. Diabetic perfusion reduction

Uses the verified original wound mask and initial
condition with the same PDE, grid and time step.

The parameter ranges are hypothetical sensitivity
scenarios, not established clinical reference ranges.

No original simulation files are modified.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import final_gradient_simulation_corrected as f


OUT = Path("physiological_sensitivity_results")
OUT.mkdir(parents=True, exist_ok=True)

DT = f.DT
DURATION = f.DURATION
STEPS = int(round(DURATION / DT))

MASK = f.WOUND_MASK
CENTRE = (f.NX // 2, f.NY // 2)

ALPHA = f.alpha_O2

BASE_SOURCE_PO2 = (
    f.SOURCE_CONCENTRATION / ALPHA
)

BASE_REDUCTION = (
    1.0 - f.beta_diabetic / f.beta_normal
)

WOUND_FRACTION = f.WOUND_PERFUSION_FRACTION

SCENARIOS = []


def add_scenario(
    group,
    label,
    condition,
    source_po2=BASE_SOURCE_PO2,
    wound_multiplier=1.0,
    diabetic_reduction=BASE_REDUCTION,
):
    SCENARIOS.append({
        "group": group,
        "label": label,
        "condition": condition,
        "source_po2": source_po2,
        "wound_multiplier": wound_multiplier,
        "diabetic_reduction": diabetic_reduction,
    })


# Experiment 1: Effective oxygen source.
for source in [50.0, 60.0, 70.0]:
    for condition in ["normal", "diabetic"]:
        add_scenario(
            "source",
            f"{source:.0f} mmHg",
            condition,
            source_po2=source,
        )


# Experiment 2: Wound-only perfusion multiplier.
for multiplier in [0.5, 1.0, 1.5]:
    for condition in ["normal", "diabetic"]:
        add_scenario(
            "perfusion_strength",
            f"{multiplier:.1f}x baseline",
            condition,
            wound_multiplier=multiplier,
        )


# Experiment 3: Diabetic reduction relative to
# the normal perfusion coefficient.
for reduction in [0.0, 0.15, BASE_REDUCTION, 0.45]:
    add_scenario(
        "diabetic_reduction",
        f"{100 * reduction:.2f}% reduction",
        "diabetic",
        diabetic_reduction=reduction,
    )


N = len(SCENARIOS)

INITIAL = f.initial_concentration()

C = np.repeat(
    INITIAL[None, :, :],
    N,
    axis=0,
)

BETA = np.zeros_like(C)
SOURCES = np.zeros((N, 1, 1))


for k, scenario in enumerate(SCENARIOS):
    source = scenario["source_po2"] * ALPHA

    # Surrounding tissue uses the same normal
    # perfusion coefficient in all cases.
    beta = np.full(
        MASK.shape,
        f.beta_normal,
        dtype=float,
    )

    if scenario["condition"] == "normal":
        base_wound_beta = f.beta_normal
    else:
        base_wound_beta = (
            f.beta_normal
            * (1.0 - scenario["diabetic_reduction"])
        )

    wound_beta = (
        base_wound_beta
        * WOUND_FRACTION
        * scenario["wound_multiplier"]
    )

    beta[MASK] = wound_beta

    BETA[k] = beta
    SOURCES[k, 0, 0] = source


diffusion_number = (
    f.D * DT
    * (1.0 / f.dx**2 + 1.0 / f.dy**2)
)

if diffusion_number > 0.5:
    raise RuntimeError(
        "Explicit diffusion stability limit exceeded."
    )


def laplacian_zero_flux(fields):
    padded = np.pad(
        fields,
        ((0, 0), (1, 1), (1, 1)),
        mode="edge",
    )

    centre = padded[:, 1:-1, 1:-1]

    d2x = (
        padded[:, 2:, 1:-1]
        - 2.0 * centre
        + padded[:, :-2, 1:-1]
    ) / f.dx**2

    d2y = (
        padded[:, 1:-1, 2:]
        - 2.0 * centre
        + padded[:, 1:-1, :-2]
    ) / f.dy**2

    return d2x + d2y


SAVE_TIMES = [0, 30, 60, 120, 300, 600]

SAVE_STEPS = {
    int(round(t / DT)): t
    for t in SAVE_TIMES
}

records = []


def record_results(time):
    po2 = C / ALPHA

    for k, scenario in enumerate(SCENARIOS):
        wound = po2[k][MASK]

        records.append({
            "group": scenario["group"],
            "label": scenario["label"],
            "condition": scenario["condition"],
            "source_po2_mmHg": scenario["source_po2"],
            "wound_multiplier": scenario["wound_multiplier"],
            "diabetic_reduction_percent": (
                100 * scenario["diabetic_reduction"]
            ),
            "time_s": time,
            "centre_po2_mmHg": float(
                po2[k][CENTRE]
            ),
            "mean_wound_po2_mmHg": float(
                wound.mean()
            ),
            "min_wound_po2_mmHg": float(
                wound.min()
            ),
            "area_below_40_percent": float(
                100 * np.mean(wound < 40.0)
            ),
        })


print("=" * 75)
print("STEP 38 - PHYSIOLOGICAL SENSITIVITY ANALYSIS")
print("=" * 75)
print("Scenarios:", N)
print("Time step:", DT, "s")
print("Duration:", DURATION, "s")
print("Diffusion number:", diffusion_number)
print("Baseline source:", BASE_SOURCE_PO2, "mmHg")
print("Baseline reduction:", 100 * BASE_REDUCTION, "%")
print()
print("Running simulations...", flush=True)


record_results(0)

for step in range(1, STEPS + 1):
    diffusion = (
        f.D * laplacian_zero_flux(C)
    )

    consumption = (
        f.rho_cell * f.Vmax
        * C / (f.Km + C)
    )

    perfusion = (
        BETA * (SOURCES - C)
    )

    C = C + DT * (
        diffusion - consumption + perfusion
    )

    if not np.all(np.isfinite(C)):
        raise RuntimeError(
            f"Non-finite concentration at step {step}"
        )

    if np.min(C) < -1e-12:
        raise RuntimeError(
            f"Negative oxygen at step {step}"
        )

    if step in SAVE_STEPS:
        record_results(SAVE_STEPS[step])


csv_path = OUT / "sensitivity_results.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(records[0].keys()),
    )

    writer.writeheader()
    writer.writerows(records)


def get_final(group, label, condition):
    return next(
        row for row in records
        if row["group"] == group
        and row["label"] == label
        and row["condition"] == condition
        and row["time_s"] == 600
    )


print()
print("=" * 75)
print("EXPERIMENT 1 - EFFECTIVE OXYGEN SOURCE")
print("=" * 75)

for source in [50.0, 60.0, 70.0]:
    normal = get_final(
        "source", f"{source:.0f} mmHg", "normal"
    )

    diabetic = get_final(
        "source", f"{source:.0f} mmHg", "diabetic"
    )

    print(
        f"Source={source:.0f} mmHg | "
        f"Normal mean={normal['mean_wound_po2_mmHg']:.4f} | "
        f"Diabetic mean={diabetic['mean_wound_po2_mmHg']:.4f} | "
        f"Difference="
        f"{normal['mean_wound_po2_mmHg'] - diabetic['mean_wound_po2_mmHg']:.4f}"
    )


print()
print("=" * 75)
print("EXPERIMENT 2 - WOUND PERFUSION STRENGTH")
print("=" * 75)

for multiplier in [0.5, 1.0, 1.5]:
    label = f"{multiplier:.1f}x baseline"

    normal = get_final(
        "perfusion_strength", label, "normal"
    )

    diabetic = get_final(
        "perfusion_strength", label, "diabetic"
    )

    print(
        f"Multiplier={multiplier:.1f}x | "
        f"Normal mean={normal['mean_wound_po2_mmHg']:.4f} | "
        f"Diabetic mean={diabetic['mean_wound_po2_mmHg']:.4f} | "
        f"Difference="
        f"{normal['mean_wound_po2_mmHg'] - diabetic['mean_wound_po2_mmHg']:.4f}"
    )


print()
print("=" * 75)
print("EXPERIMENT 3 - DIABETIC PERFUSION REDUCTION")
print("=" * 75)

for reduction in [
    0.0, 0.15, BASE_REDUCTION, 0.45
]:
    label = f"{100 * reduction:.2f}% reduction"

    result = get_final(
        "diabetic_reduction",
        label,
        "diabetic",
    )

    print(
        f"Reduction={100 * reduction:.2f}% | "
        f"Centre={result['centre_po2_mmHg']:.4f} | "
        f"Mean={result['mean_wound_po2_mmHg']:.4f}"
    )


print()
print("=" * 75)
print("BASELINE REPRODUCTION CHECK")
print("=" * 75)

for condition, expected in [
    ("normal", 53.401956),
    ("diabetic", 51.732702),
]:
    result = get_final(
        "source", "60 mmHg", condition
    )

    actual = result["mean_wound_po2_mmHg"]
    difference = actual - expected

    print(
        f"{condition.capitalize()}: "
        f"actual={actual:.6f}, "
        f"previous={expected:.6f}, "
        f"difference={difference:+.8f} mmHg"
    )

    if abs(difference) > 0.001:
        print("WARNING: Baseline mismatch.")


# Plot mean wound oxygen for source scenarios.
fig, ax = plt.subplots(figsize=(9, 5))

for condition in ["normal", "diabetic"]:
    xs = [50, 60, 70]
    ys = [
        get_final(
            "source",
            f"{source} mmHg",
            condition
        )["mean_wound_po2_mmHg"]
        for source in xs
    ]

    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        label=condition.capitalize(),
    )

ax.set_xlabel("Effective oxygen source (mmHg)")
ax.set_ylabel("Mean wound pO2 at 600 s (mmHg)")
ax.set_title("Sensitivity to effective oxygen source")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "source_sensitivity.png",
    dpi=200,
)
plt.close(fig)


# Plot mean wound oxygen for perfusion multipliers.
fig, ax = plt.subplots(figsize=(9, 5))

for condition in ["normal", "diabetic"]:
    xs = [0.5, 1.0, 1.5]
    ys = [
        get_final(
            "perfusion_strength",
            f"{multiplier:.1f}x baseline",
            condition
        )["mean_wound_po2_mmHg"]
        for multiplier in xs
    ]

    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        label=condition.capitalize(),
    )

ax.set_xlabel("Wound perfusion multiplier")
ax.set_ylabel("Mean wound pO2 at 600 s (mmHg)")
ax.set_title("Sensitivity to wound perfusion")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "perfusion_sensitivity.png",
    dpi=200,
)
plt.close(fig)


# Plot diabetic perfusion reduction.
fig, ax = plt.subplots(figsize=(9, 5))

reductions = [
    0.0, 0.15, BASE_REDUCTION, 0.45
]

values = [
    get_final(
        "diabetic_reduction",
        f"{100 * reduction:.2f}% reduction",
        "diabetic",
    )["mean_wound_po2_mmHg"]
    for reduction in reductions
]

ax.plot(
    [100 * value for value in reductions],
    values,
    marker="o",
    linewidth=2,
)

ax.set_xlabel("Perfusion reduction (%)")
ax.set_ylabel("Mean wound pO2 at 600 s (mmHg)")
ax.set_title("Sensitivity to diabetic perfusion impairment")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUT / "diabetic_reduction_sensitivity.png",
    dpi=200,
)
plt.close(fig)


print()
print("Saved CSV:", csv_path.resolve())
print("Saved three sensitivity figures.")
print("Original simulation files unchanged.")
print()
print(
    "All parameter ranges are hypothetical. "
    "These results are not clinical predictions."
)
