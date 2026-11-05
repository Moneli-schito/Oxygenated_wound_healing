"""
STEP 28: TIME-DEPENDENT INITIAL PROFILE COMPARISON

Compare original and mean-matched continuous initial
oxygen profiles under normal and diabetic perfusion.

All cases use:
- The same governing oxygen transport PDE
- The same diffusion and consumption parameters
- The same healthy surrounding tissue perfusion
- The same effective oxygen source
- The same zero-flux outer boundary
- The same 600-second duration

This models oxygen transport, NOT wound closure.

Existing simulation files and results are preserved.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import final_gradient_simulation_corrected as f


OUTPUT_DIR = Path(
    "transient_initial_profile_comparison_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = f.DT
DURATION = f.DURATION
SAVE_INTERVAL = 1.0

STEPS = int(round(DURATION / DT))
SAVE_STEPS = int(round(SAVE_INTERVAL / DT))

MASK = f.WOUND_MASK
CENTRE = (f.NX // 2, f.NY // 2)

# Load the two initial conditions from Step 27.
SOURCE_DIR = Path("matched_initial_profile_results")

original = np.load(
    SOURCE_DIR / "original_initial_po2.npy"
)

continuous = np.load(
    SOURCE_DIR / "continuous_initial_po2.npy"
)

if original.shape != (f.NX, f.NY):
    raise ValueError("Original profile has incorrect shape.")

if continuous.shape != (f.NX, f.NY):
    raise ValueError("Continuous profile has incorrect shape.")

# Check that the initial conditions are comparable.
assert np.isclose(
    original[MASK].mean(),
    continuous[MASK].mean(),
    atol=1e-8,
)

assert np.isclose(
    original[CENTRE],
    continuous[CENTRE],
    atol=1e-8,
)

# Set healthy surrounding perfusion to the same value
# in all simulations.
normal_wound_beta = (
    f.beta_normal * f.WOUND_PERFUSION_FRACTION
)

diabetic_wound_beta = (
    f.beta_diabetic * f.WOUND_PERFUSION_FRACTION
)


def create_beta_field(wound_beta):
    beta = np.full(
        (f.NX, f.NY),
        f.beta_normal,
        dtype=float,
    )

    beta[MASK] = wound_beta
    return beta


normal_beta_field = create_beta_field(normal_wound_beta)
diabetic_beta_field = create_beta_field(
    diabetic_wound_beta
)

SCENARIOS = [
    ("normal_original", original, normal_beta_field),
    ("normal_continuous", continuous, normal_beta_field),
    ("diabetic_original", original, diabetic_beta_field),
    ("diabetic_continuous", continuous, diabetic_beta_field),
]

NAMES = [item[0] for item in SCENARIOS]

# Simulate all four conditions together.
C = np.stack([
    item[1] * f.alpha_O2
    for item in SCENARIOS
])

BETA = np.stack([
    item[2]
    for item in SCENARIOS
])

# Explicit Euler diffusion stability condition.
diffusion_number = (
    f.D * DT * (1 / f.dx**2 + 1 / f.dy**2)
)

if diffusion_number > 0.5:
    raise ValueError(
        "Time step exceeds explicit diffusion stability limit."
    )

def laplacian_zero_flux(fields):
    """
    Finite-difference Laplacian with reflected
    zero-normal-gradient boundaries using edge padding.

    fields shape: (scenario, x, y)
    """

    padded = np.pad(
        fields,
        ((0, 0), (1, 1), (1, 1)),
        mode="edge",
    )

    centre = padded[:, 1:-1, 1:-1]

    d2x = (
        padded[:, 2:, 1:-1]
        - 2 * centre
        + padded[:, :-2, 1:-1]
    ) / f.dx**2

    d2y = (
        padded[:, 1:-1, 2:]
        - 2 * centre
        + padded[:, 1:-1, :-2]
    ) / f.dy**2

    return d2x + d2y


def calculate_metrics(fields, time):
    po2 = fields / f.alpha_O2

    rows = []

    for index, name in enumerate(NAMES):
        wound = po2[index][MASK]

        rows.append({
            "scenario": name,
            "time_s": time,
            "centre_po2_mmHg": float(
                po2[index][CENTRE]
            ),
            "mean_wound_po2_mmHg": float(
                wound.mean()
            ),
            "minimum_wound_po2_mmHg": float(
                wound.min()
            ),
            "area_below_40_percent": float(
                100 * np.mean(wound < 40.0)
            ),
        })

    return rows


print("=" * 70)
print("STEP 28 - TIME-DEPENDENT PROFILE COMPARISON")
print("=" * 70)

print("Time step:", DT, "seconds")
print("Duration:", DURATION, "seconds")
print("Diffusion stability number:", diffusion_number)
print("Normal wound beta:", normal_wound_beta)
print("Diabetic wound beta:", diabetic_wound_beta)

print()
print("Running four controlled simulations...")

records = []

# Record initial state.
records.extend(calculate_metrics(C, 0.0))

for step in range(1, STEPS + 1):

    diffusion = f.D * laplacian_zero_flux(C)

    consumption = (
        f.rho_cell
        * f.Vmax
        * C
        / (f.Km + C)
    )

    perfusion = (
        BETA * (f.SOURCE_CONCENTRATION - C)
    )

    dCdt = diffusion - consumption + perfusion

    C = C + DT * dCdt

    if not np.all(np.isfinite(C)):
        raise RuntimeError(
            f"Non-finite oxygen concentration at step {step}"
        )

    if np.min(C) < -1e-12:
        raise RuntimeError(
            f"Negative oxygen concentration at step {step}"
        )

    if step % SAVE_STEPS == 0:
        time = step * DT
        records.extend(calculate_metrics(C, time))


# Save time-series data.
csv_path = OUTPUT_DIR / "transient_comparison.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(records[0].keys()),
    )
    writer.writeheader()
    writer.writerows(records)


# Group data by scenario.
grouped = {
    name: [r for r in records if r["scenario"] == name]
    for name in NAMES
}


def get_final(name, key):
    return grouped[name][-1][key]


# Print final results.
print()
print("FINAL RESULTS AT 600 SECONDS")
print("-" * 70)

for name in NAMES:
    centre = get_final(name, "centre_po2_mmHg")
    mean = get_final(name, "mean_wound_po2_mmHg")
    area = get_final(name, "area_below_40_percent")

    print(
        f"{name:<22} "
        f"Centre={centre:8.4f}  "
        f"Mean={mean:8.4f}  "
        f"Below40={area:6.2f}%"
    )


# Calculate how much the initial profile changes outcomes.
print()
print("EFFECT OF CHANGING ONLY THE INITIAL PROFILE")
print("-" * 70)

for condition in ["normal", "diabetic"]:

    old = f"{condition}_original"
    new = f"{condition}_continuous"

    centre_diff = (
        get_final(new, "centre_po2_mmHg")
        - get_final(old, "centre_po2_mmHg")
    )

    mean_diff = (
        get_final(new, "mean_wound_po2_mmHg")
        - get_final(old, "mean_wound_po2_mmHg")
    )

    print(
        f"{condition.capitalize():<10} "
        f"Centre difference: {centre_diff:+.4f} mmHg, "
        f"Mean difference: {mean_diff:+.4f} mmHg"
    )


# Time curves.
STYLES = {
    "normal_original": ("tab:blue", "-"),
    "normal_continuous": ("tab:blue", "--"),
    "diabetic_original": ("tab:orange", "-"),
    "diabetic_continuous": ("tab:orange", "--"),
}


def plot_metric(key, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in NAMES:
        rows = grouped[name]
        times = [r["time_s"] for r in rows]
        values = [r[key] for r in rows]

        colour, linestyle = STYLES[name]

        ax.plot(
            times,
            values,
            label=name.replace("_", " ").title(),
            color=colour,
            linestyle=linestyle,
            linewidth=2,
        )

    ax.set_xlabel("Simulation time (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
    )
    plt.close(fig)


plot_metric(
    "centre_po2_mmHg",
    "Wound centre oxygen (mmHg)",
    "centre_oxygen_comparison.png",
    "Wound centre oxygen versus time",
)

plot_metric(
    "mean_wound_po2_mmHg",
    "Mean wound oxygen (mmHg)",
    "mean_oxygen_comparison.png",
    "Mean wound oxygen versus time",
)

plot_metric(
    "area_below_40_percent",
    "Wound area below 40 mmHg (%)",
    "threshold_area_comparison.png",
    "Exploratory oxygen threshold comparison",
)


# Save final oxygen fields.
final_po2 = C / f.alpha_O2

for index, name in enumerate(NAMES):
    np.save(
        OUTPUT_DIR / f"{name}_final_po2.npy",
        final_po2[index],
    )


print()
print("Saved results to:", OUTPUT_DIR.resolve())

# Compare original-profile results with the previously
# reported corrected simulation as a diagnostic.
print()
print("CROSS-CHECK AGAINST PREVIOUS BASELINE")

previous_means = {
    "normal_original": 53.4020,
    "diabetic_original": 51.7327,
}

for name, previous in previous_means.items():

    current = get_final(
        name,
        "mean_wound_po2_mmHg",
    )

    difference = current - previous

    print(
        f"{name}: "
        f"new={current:.4f}, "
        f"previous={previous:.4f}, "
        f"difference={difference:+.4f} mmHg"
    )

    if abs(difference) > 0.25:
        print(
            "WARNING: This independent solver differs "
            "from the previously reported baseline. "
            "Check the zero-flux boundary discretisation "
            "before interpreting the comparison."
        )

print()
print("Existing simulation files were not modified.")
