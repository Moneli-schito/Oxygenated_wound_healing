"""
Compare two exploratory wound oxygen initial conditions.

1. Original: centre 10 mmHg, wound edge approaches 40 mmHg,
   surrounding tissue 56.17 mmHg, with a boundary jump.

2. Continuous: centre 10 mmHg, smoothly approaches the
   surrounding oxygen level at the wound boundary.

This experiment compares initial profiles only.
It does not simulate wound healing or validate either
profile physiologically.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import final_gradient_simulation_corrected as f


OUTPUT_DIR = Path("initial_profile_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geometry in physical distance units (mm).
x_mm = np.arange(f.NX) * f.dx * 1000.0
y_mm = np.arange(f.NY) * f.dy * 1000.0

X, Y = np.meshgrid(x_mm, y_mm, indexing="ij")

cx = x_mm[f.NX // 2]
cy = y_mm[f.NY // 2]

radius_mm = f.WOUND_RADIUS_CELLS * f.dx * 1000.0

r = np.sqrt((X - cx)**2 + (Y - cy)**2)
s = np.clip(r / radius_mm, 0.0, 1.0)

surrounding = float(f.SURROUNDING_INITIAL_PO2)
centre = float(f.WOUND_CENTRE_INITIAL_PO2)

# Original profile from the active simulation.
original = (
    f.initial_concentration() / f.alpha_O2
)

# Continuous profile: cubic smoothstep.
# Same centre oxygen, smoothly joins the surrounding level.
smoothstep = 3.0 * s**2 - 2.0 * s**3

continuous = (
    centre + (surrounding - centre) * smoothstep
)

# Outside the wound, healthy oxygen remains unchanged.
continuous[~f.WOUND_MASK] = surrounding


def boundary_jumps(po2):
    mask = f.WOUND_MASK

    horizontal = mask[:-1, :] != mask[1:, :]
    vertical = mask[:, :-1] != mask[:, 1:]

    return np.concatenate([
        np.abs(np.diff(po2, axis=0))[horizontal],
        np.abs(np.diff(po2, axis=1))[vertical],
    ])


def report(name, po2):
    jumps = boundary_jumps(po2)

    print(f"\n{name}")
    print("-" * 50)
    print(f"Centre oxygen: {po2[f.NX//2, f.NY//2]:.4f} mmHg")
    print(f"Mean wound oxygen: {po2[f.WOUND_MASK].mean():.4f} mmHg")
    print(f"Mean boundary jump: {jumps.mean():.4f} mmHg")
    print(f"Maximum boundary jump: {jumps.max():.4f} mmHg")
    print(f"Minimum oxygen: {po2.min():.4f} mmHg")
    print(f"Maximum oxygen: {po2.max():.4f} mmHg")


print("=" * 60)
print("INITIAL OXYGEN PROFILE COMPARISON")
print("=" * 60)
print(f"Healthy surrounding oxygen: {surrounding:.4f} mmHg")
print(f"Wound radius: {radius_mm:.3f} mm")

report("ORIGINAL PROFILE", original)
report("CONTINUOUS PROFILE", continuous)

# Save initial oxygen maps.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, data, title in zip(
    axes,
    [original, continuous],
    ["Original initial profile", "Continuous initial profile"],
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
    OUTPUT_DIR / "initial_oxygen_maps.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# Save centreline comparison.
mid = f.NY // 2

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    x_mm,
    original[:, mid],
    label="Original profile",
    linewidth=2,
)

ax.plot(
    x_mm,
    continuous[:, mid],
    label="Continuous profile",
    linewidth=2,
    linestyle="--",
)

ax.axvline(
    cx - radius_mm,
    color="grey",
    linestyle=":",
    label="Wound boundaries",
)
ax.axvline(
    cx + radius_mm,
    color="grey",
    linestyle=":",
)

ax.set_xlabel("Distance across tissue (mm)")
ax.set_ylabel("Oxygen pressure (mmHg)")
ax.set_title("Initial wound oxygen: centreline comparison")
ax.grid(alpha=0.3)
ax.legend()

fig.savefig(
    OUTPUT_DIR / "initial_centreline_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

print("\nSaved figures to:", OUTPUT_DIR.resolve())
print("Existing simulation files were not modified.")
