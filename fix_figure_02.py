"""
FINAL FIGURE 2
Initial oxygen and change from 0 to 600 seconds.

Recalculates oxygen fields from the corrected PDE.
Uses identical initial conditions and healthy surrounding
perfusion in normal and diabetic scenarios.

No hard-coded oxygen outputs.
No existing simulation files are modified.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import final_gradient_simulation_corrected as f


OUTPUT = Path(
    "Final_Report_Figures/Main_Results/"
    "Figure_02_Verified_Oxygen_Changes.png"
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

DT = f.DT
STEPS = int(round(f.DURATION / DT))

MASK = f.WOUND_MASK
ALPHA = f.alpha_O2
CENTRE = (f.NX // 2, f.NY // 2)

INITIAL_C = f.initial_concentration()
INITIAL_PO2 = INITIAL_C / ALPHA


def laplacian(C):
    p = np.pad(C, 1, mode="edge")

    return (
        (p[2:, 1:-1] - 2*C + p[:-2, 1:-1])
        / f.dx**2
        +
        (p[1:-1, 2:] - 2*C + p[1:-1, :-2])
        / f.dy**2
    )


def simulate(wound_beta):
    C = INITIAL_C.copy()

    # Healthy surrounding tissue is identical
    # in both scenarios.
    beta = np.full(
        C.shape,
        f.beta_normal,
        dtype=float
    )

    beta[MASK] = wound_beta

    for step in range(STEPS):
        diffusion = f.D * laplacian(C)

        consumption = (
            f.rho_cell * f.Vmax
            * C / (f.Km + C)
        )

        perfusion = (
            beta * (f.SOURCE_CONCENTRATION - C)
        )

        C = C + DT * (
            diffusion - consumption + perfusion
        )

        if not np.all(np.isfinite(C)):
            raise RuntimeError("Non-finite oxygen result")

    return C / ALPHA


print("Reproducing corrected simulations...")

normal = simulate(
    f.beta_normal * f.WOUND_PERFUSION_FRACTION
)

diabetic = simulate(
    f.beta_diabetic * f.WOUND_PERFUSION_FRACTION
)

normal_change = normal - INITIAL_PO2
diabetic_change = diabetic - INITIAL_PO2

normal_centre = float(normal[CENTRE])
diabetic_centre = float(diabetic[CENTRE])

normal_gain = float(normal_change[CENTRE])
diabetic_gain = float(diabetic_change[CENTRE])

normal_mean = float(normal[MASK].mean())
diabetic_mean = float(diabetic[MASK].mean())

# Reproduce the previously verified baseline.
if not np.isclose(normal_centre, 52.3717, atol=0.02):
    raise RuntimeError(
        "Normal baseline mismatch: "
        f"{normal_centre:.6f}"
    )

if not np.isclose(diabetic_centre, 49.9386, atol=0.02):
    raise RuntimeError(
        "Diabetic baseline mismatch: "
        f"{diabetic_centre:.6f}"
    )

print()
print("VERIFIED RESULTS")
print("-" * 50)
print(f"Initial centre: {INITIAL_PO2[CENTRE]:.4f}")
print(f"Normal final centre: {normal_centre:.4f}")
print(f"Diabetic final centre: {diabetic_centre:.4f}")
print(f"Normal centre increase: {normal_gain:+.4f}")
print(f"Diabetic centre increase: {diabetic_gain:+.4f}")
print(f"Normal mean at 600 s: {normal_mean:.4f}")
print(f"Diabetic mean at 600 s: {diabetic_mean:.4f}")


# -------------------------------------------------
# FIGURE: INITIAL + TWO MATCHED CHANGE MAPS
# -------------------------------------------------

xmax = (f.NX - 1) * f.dx * 1000
ymax = (f.NY - 1) * f.dy * 1000

extent = [0, xmax, 0, ymax]

xs = np.linspace(0, xmax, f.NX)
ys = np.linspace(0, ymax, f.NY)

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 5.5),
    constrained_layout=True
)

fig.suptitle(
    "Simulated oxygen changes between 0 and 600 s",
    fontsize=16
)


def draw_wound_boundary(ax):
    ax.contour(
        xs,
        ys,
        MASK.T.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1
    )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")


# Panel A: Initial condition
im0 = axes[0].imshow(
    INITIAL_PO2.T,
    origin="lower",
    extent=extent,
    cmap="viridis",
    vmin=0,
    vmax=60,
    interpolation="nearest"
)

draw_wound_boundary(axes[0])
axes[0].set_title("A. Shared initial oxygen (0 s)")

axes[0].text(
    0.04, 0.05,
    f"Centre: {INITIAL_PO2[CENTRE]:.2f} mmHg",
    transform=axes[0].transAxes,
    bbox=dict(
        facecolor="white",
        alpha=0.9,
        edgecolor="grey"
    ),
    fontsize=10
)

fig.colorbar(
    im0,
    ax=axes[0],
    label="Initial pO2 (mmHg)",
    fraction=0.047,
    pad=0.04
)


# Panels B and C: Same quantity, same scale.
# The scale is centred on zero to distinguish
# oxygen increases from decreases.
cases = [
    (
        axes[1],
        normal_change,
        "B. Normal: oxygen change",
        normal_gain
    ),
    (
        axes[2],
        diabetic_change,
        "C. Diabetic: oxygen change",
        diabetic_gain
    )
]

for ax, change, title, gain in cases:
    im = ax.imshow(
        change.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-45,
        vmax=45,
        interpolation="nearest"
    )

    draw_wound_boundary(ax)
    ax.set_title(title)

    ax.text(
        0.04, 0.05,
        f"Centre increase: {gain:+.2f} mmHg",
        transform=ax.transAxes,
        bbox=dict(
            facecolor="white",
            alpha=0.9,
            edgecolor="grey"
        ),
        fontsize=10
    )

fig.colorbar(
    im,
    ax=[axes[1], axes[2]],
    label="Change in pO2: 600 s minus 0 s (mmHg)",
    fraction=0.047,
    pad=0.04
)

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print()
print("FIGURE CREATED SUCCESSFULLY")
print("Saved:", OUTPUT.resolve())
print("Original simulation files unchanged.")
