"""
Step 27: Mean-matched initial oxygen profile comparison.

Compares:
1. Original discontinuous wound oxygen profile.
2. Continuous profile with the same centre oxygen
   and mean wound oxygen.

No original simulation files are modified.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import final_gradient_simulation_corrected as f


OUTPUT_DIR = Path("matched_initial_profile_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

original = f.initial_concentration() / f.alpha_O2

mask = f.WOUND_MASK
surrounding = float(f.SURROUNDING_INITIAL_PO2)
centre = float(f.WOUND_CENTRE_INITIAL_PO2)

x_mm = np.arange(f.NX) * f.dx * 1000.0
y_mm = np.arange(f.NY) * f.dy * 1000.0

X, Y = np.meshgrid(x_mm, y_mm, indexing="ij")

cx = x_mm[f.NX // 2]
cy = y_mm[f.NY // 2]

radius_mm = f.WOUND_RADIUS_CELLS * f.dx * 1000.0

r = np.sqrt((X - cx)**2 + (Y - cy)**2)
s = np.clip(r / radius_mm, 0.0, 1.0)

target_mean = float(original[mask].mean())


def create_continuous_profile(power):
    # Smooth radial transition.
    z = s ** power
    smoothstep = 3.0 * z**2 - 2.0 * z**3

    profile = centre + (
        surrounding - centre
    ) * smoothstep

    profile[~mask] = surrounding

    return profile


def mean_error(power):
    profile = create_continuous_profile(power)
    return profile[mask].mean() - target_mean


# Find a profile shape matching original mean oxygen.
power = brentq(mean_error, 0.1, 100.0)

continuous = create_continuous_profile(power)


def boundary_jumps(profile):
    horizontal = mask[:-1, :] != mask[1:, :]
    vertical = mask[:, :-1] != mask[:, 1:]

    return np.concatenate([
        np.abs(np.diff(profile, axis=0))[horizontal],
        np.abs(np.diff(profile, axis=1))[vertical],
    ])


def report(name, profile):
    jumps = boundary_jumps(profile)

    print()
    print(name)
    print("-" * 50)
    print(f"Centre oxygen: {profile[f.NX//2, f.NY//2]:.4f} mmHg")
    print(f"Mean wound oxygen: {profile[mask].mean():.4f} mmHg")
    print(f"Mean boundary jump: {jumps.mean():.4f} mmHg")
    print(f"Maximum boundary jump: {jumps.max():.4f} mmHg")


print("=" * 60)
print("STEP 27 - MEAN-MATCHED PROFILE COMPARISON")
print("=" * 60)

print(f"Healthy surrounding oxygen: {surrounding:.4f} mmHg")
print(f"Matched profile shape exponent: {power:.6f}")

report("ORIGINAL PROFILE", original)
report("CONTINUOUS MATCHED PROFILE", continuous)

print()
print(
    "Mean oxygen difference:",
    abs(original[mask].mean() - continuous[mask].mean()),
    "mmHg"
)

# Save concentration arrays for subsequent transient simulations.
np.save(OUTPUT_DIR / "original_initial_po2.npy", original)
np.save(OUTPUT_DIR / "continuous_initial_po2.npy", continuous)

# Plot centreline comparison.
mid = f.NY // 2

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    x_mm,
    original[:, mid],
    label="Original profile",
    linewidth=2,
)

ax.plot(
    x_mm,
    continuous[:, mid],
    label="Continuous, matched mean",
    linewidth=2,
    linestyle="--",
)

for boundary in [cx - radius_mm, cx + radius_mm]:
    ax.axvline(
        boundary,
        color="grey",
        linestyle=":",
    )

ax.set_title("Initial oxygen profiles — matched wound mean")
ax.set_xlabel("Distance (mm)")
ax.set_ylabel("Oxygen pressure (mmHg)")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "matched_centreline.png",
    dpi=200,
)
plt.close(fig)

# Plot maps on the same colour scale.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, data, title in zip(
    axes,
    [original, continuous],
    ["Original profile", "Continuous, matched mean"],
):
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
        cmap="viridis",
        vmin=0,
        vmax=60,
    )
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")

fig.colorbar(
    im,
    ax=axes,
    label="Oxygen pressure (mmHg)",
    shrink=0.8,
)

fig.savefig(
    OUTPUT_DIR / "matched_oxygen_maps.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

print()
print("Results saved to:", OUTPUT_DIR.resolve())
print("Original files unchanged.")
