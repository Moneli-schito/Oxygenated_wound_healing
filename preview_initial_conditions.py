
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import final_gradient_simulation_corrected as f


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

EQUILIBRIUM_DIR = Path("spatial_equilibrium_results")
OUTPUT_DIR = Path("initial_conditions_preview")

# Preserve the original normal centre value as a
# provisional calibration target for this experiment.
NORMAL_CENTRE_TARGET_PO2 = 10.0


# --------------------------------------------------
# 2. Load model-derived equilibrium fields
# --------------------------------------------------

normal_equilibrium = np.load(
    EQUILIBRIUM_DIR / "normal_equilibrium_po2.npy"
)

diabetic_equilibrium = np.load(
    EQUILIBRIUM_DIR / "diabetic_equilibrium_po2.npy"
)

expected_shape = (f.NX, f.NY)

if normal_equilibrium.shape != expected_shape:
    raise ValueError("Normal equilibrium has incorrect shape.")

if diabetic_equilibrium.shape != expected_shape:
    raise ValueError("Diabetic equilibrium has incorrect shape.")


# --------------------------------------------------
# 3. Create a common wound-deficit profile
# --------------------------------------------------

x, y = np.meshgrid(
    np.arange(f.NX),
    np.arange(f.NY),
    indexing="ij",
)

cx = f.NX // 2
cy = f.NY // 2

radius = f.WOUND_RADIUS_CELLS

distance = np.sqrt(
    (x - cx) ** 2 + (y - cy) ** 2
)

# Greatest deficit at wound centre.
# Zero imposed deficit at and outside wound edge.
deficit_profile = np.clip(
    1.0 - distance / radius,
    0.0,
    1.0,
)

# The same fractional deficit applies to both
# scenarios. This is a hypothetical injury model.
normal_eq_centre = normal_equilibrium[cx, cy]

deficit_fraction = (
    1.0
    - NORMAL_CENTRE_TARGET_PO2 / normal_eq_centre
)

if not 0 <= deficit_fraction <= 1:
    raise ValueError("Invalid deficit fraction.")


# --------------------------------------------------
# 4. Construct scenario-specific initial fields
# --------------------------------------------------

def construct_initial_field(equilibrium):
    return equilibrium * (
        1.0 - deficit_fraction * deficit_profile
    )


normal_initial = construct_initial_field(
    normal_equilibrium
)

diabetic_initial = construct_initial_field(
    diabetic_equilibrium
)


# --------------------------------------------------
# 5. Validate and print initial conditions
# --------------------------------------------------

for name, field in [
    ("Normal", normal_initial),
    ("Diabetic", diabetic_initial),
]:
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} field contains non-finite values.")

    if np.any(field < 0):
        raise ValueError(f"{name} field contains negative oxygen.")

    print(f"\n{name.upper()} INITIAL CONDITIONS")

    print(
        "Wound centre:",
        f"{field[cx, cy]:.4f} mmHg"
    )

    print(
        "Mean wound:",
        f"{np.mean(field[f.WOUND_MASK]):.4f} mmHg"
    )

    print(
        "Mean surrounding:",
        f"{np.mean(field[f.SURROUNDING_MASK]):.4f} mmHg"
    )

print(
    "\nCommon fractional deficit at wound centre:",
    f"{100 * deficit_fraction:.2f}%"
)

print(
    "Difference between starting wound-centre values:",
    f"{normal_initial[cx, cy] - diabetic_initial[cx, cy]:.4f} mmHg"
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(
    OUTPUT_DIR / "normal_initial_po2.npy",
    normal_initial
)

np.save(
    OUTPUT_DIR / "diabetic_initial_po2.npy",
    diabetic_initial
)

summary = []

for name, field in [
    ("Normal", normal_initial),
    ("Diabetic", diabetic_initial),
]:
    summary.append({
        "condition": name,
        "centre_po2_mmHg": float(field[cx, cy]),
        "mean_wound_po2_mmHg": float(
            np.mean(field[f.WOUND_MASK])
        ),
        "mean_surrounding_po2_mmHg": float(
            np.mean(field[f.SURROUNDING_MASK])
        ),
        "common_centre_deficit_fraction": deficit_fraction,
    })

with (
    OUTPUT_DIR / "initial_conditions_summary.csv"
).open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)


# --------------------------------------------------
# 7. Plot initial spatial oxygen distributions
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

fields = [normal_initial, diabetic_initial]

vmin = min(float(np.min(a)) for a in fields)
vmax = max(float(np.max(a)) for a in fields)

for ax, field, title in zip(
    axes,
    fields,
    ["Normal initial condition", "Diabetic initial condition"],
):
    image = ax.imshow(
        field.T,
        origin="lower",
        extent=[0, f.Lx * 1000, 0, f.Ly * 1000],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    ax.contour(
        f.X_MM,
        f.Y_MM,
        f.WOUND_MASK.T.astype(float),
        levels=[0.5],
        colors="white",
    )

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

fig.colorbar(
    image,
    ax=axes,
    label="Initial oxygen partial pressure (mmHg)",
)

fig.savefig(
    OUTPUT_DIR / "initial_oxygen_maps.png",
    dpi=250,
)

plt.close(fig)


# --------------------------------------------------
# 8. Plot initial centreline comparison
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    f.X_MM,
    normal_initial[:, cy],
    label="Normal initial condition",
    linewidth=2,
)

ax.plot(
    f.X_MM,
    diabetic_initial[:, cy],
    label="Diabetic initial condition",
    linewidth=2,
)

ax.set_xlabel("Position across wound (mm)")
ax.set_ylabel("Initial oxygen partial pressure (mmHg)")
ax.set_title("Scenario-specific initial oxygen profiles")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "initial_centreline_profiles.png",
    dpi=250,
)

plt.close(fig)

print("\nPreview files saved to:", OUTPUT_DIR)
