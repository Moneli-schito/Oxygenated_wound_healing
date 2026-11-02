"""
STEP 31 - OUTER BOUNDARY CONDITION COMPARISON

Compare:
1. Zero-flux outer boundary
2. Fixed healthy-oxygen outer boundary

Under:
- Normal wound perfusion
- Diabetic wound perfusion

Identical initial oxygen profile and PDE parameters.

This experiment tests sensitivity to the outer
boundary condition; it does not establish which
boundary is biologically correct.

Existing files and results are preserved.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import final_gradient_simulation_corrected as f


OUT = Path("corrected_boundary_comparison_results")
OUT.mkdir(parents=True, exist_ok=True)

DT = f.DT
DURATION = f.DURATION
N_STEPS = int(round(DURATION / DT))

MASK = f.WOUND_MASK
CENTRE = (f.NX // 2, f.NY // 2)
ALPHA = f.alpha_O2

INITIAL = f.initial_concentration()
FIXED_VALUE = f.SURROUNDING_INITIAL_PO2 * ALPHA

SCENARIOS = [
    ("normal_zero_flux", "normal", "zero_flux"),
    ("normal_fixed", "normal", "fixed"),
    ("diabetic_zero_flux", "diabetic", "zero_flux"),
    ("diabetic_fixed", "diabetic", "fixed"),
]


def make_beta(condition):
    beta = np.full(
        (f.NX, f.NY),
        f.beta_normal,
        dtype=float,
    )

    if condition == "normal":
        wound_beta = (
            f.beta_normal * f.WOUND_PERFUSION_FRACTION
        )
    else:
        wound_beta = (
            f.beta_diabetic * f.WOUND_PERFUSION_FRACTION
        )

    beta[MASK] = wound_beta
    return beta


def laplacian(C, boundary):
    if boundary == "zero_flux":
        padded = np.pad(C, 1, mode="edge")
    elif boundary == "fixed":
        padded = np.pad(
            C,
            1,
            mode="constant",
            constant_values=FIXED_VALUE,
        )
    else:
        raise ValueError("Unknown boundary type")

    d2x = (
        padded[2:, 1:-1]
        - 2.0 * C
        + padded[:-2, 1:-1]
    ) / f.dx**2

    d2y = (
        padded[1:-1, 2:]
        - 2.0 * C
        + padded[1:-1, :-2]
    ) / f.dy**2

    return d2x + d2y


def enforce_fixed_boundary(C):
    C[0, :] = FIXED_VALUE
    C[-1, :] = FIXED_VALUE
    C[:, 0] = FIXED_VALUE
    C[:, -1] = FIXED_VALUE


def simulate(name, condition, boundary):
    C = INITIAL.copy()
    beta = make_beta(condition)

    if boundary == "fixed":
        enforce_fixed_boundary(C)

    records = []

    for step in range(N_STEPS + 1):
        if step % 10 == 0:
            p = C / ALPHA

            records.append({
                "scenario": name,
                "time_s": step * DT,
                "centre_po2_mmHg": float(p[CENTRE]),
                "mean_wound_po2_mmHg": float(
                    p[MASK].mean()
                ),
                "mean_surrounding_po2_mmHg": float(
                    p[~MASK].mean()
                ),
            })

        if step == N_STEPS:
            break

        diffusion = f.D * laplacian(C, boundary)

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

        if boundary == "fixed":
            enforce_fixed_boundary(C)

        if not np.all(np.isfinite(C)):
            raise RuntimeError(
                f"Non-finite values in {name}"
            )

        if C.min() < -1e-12:
            raise RuntimeError(
                f"Negative concentration in {name}"
            )

    return records, C / ALPHA


print("=" * 70)
print("STEP 31 - OUTER BOUNDARY CONDITION COMPARISON")
print("=" * 70)
print("Fixed boundary pO2:",
      f.SURROUNDING_INITIAL_PO2, "mmHg")
print("Simulation duration:", DURATION, "seconds")
print("Time step:", DT, "seconds")

diffusion_number = (
    f.D * DT * (
        1.0 / f.dx**2 + 1.0 / f.dy**2
    )
)

print("Diffusion number:", diffusion_number)

if diffusion_number > 0.5:
    raise ValueError("Explicit diffusion stability limit exceeded")

all_records = []
final_fields = {}

for name, condition, boundary in SCENARIOS:
    print("Running:", name)

    records, final_field = simulate(
        name, condition, boundary
    )

    all_records.extend(records)
    final_fields[name] = final_field

    np.save(
        OUT / f"{name}_final_po2.npy",
        final_field,
    )

with (OUT / "boundary_comparison.csv").open(
    "w", newline=""
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(all_records[0].keys()),
    )
    writer.writeheader()
    writer.writerows(all_records)


def final_mean(name):
    return float(final_fields[name][MASK].mean())


def final_centre(name):
    return float(final_fields[name][CENTRE])


print()
print("FINAL RESULTS AT 600 SECONDS")
print("-" * 70)

for name, _, _ in SCENARIOS:
    print(
        f"{name:<24} "
        f"Centre={final_centre(name):.4f} mmHg | "
        f"Wound mean={final_mean(name):.4f} mmHg"
    )

print()
print("EFFECT OF CHANGING OUTER BOUNDARY")
print("-" * 70)

for condition in ["normal", "diabetic"]:
    zero = f"{condition}_zero_flux"
    fixed = f"{condition}_fixed"

    print(
        f"{condition.capitalize():<10}"
        f"Centre difference: "
        f"{final_centre(fixed)-final_centre(zero):+.4f} mmHg | "
        f"Mean difference: "
        f"{final_mean(fixed)-final_mean(zero):+.4f} mmHg"
    )

print()
print("CROSS-CHECK AGAINST ORIGINAL BASELINE")
print("-" * 70)

for condition, expected in [
    ("normal", 53.4020),
    ("diabetic", 51.7327),
]:
    name = f"{condition}_zero_flux"
    actual = final_mean(name)
    error = actual - expected

    print(
        f"{name}: actual={actual:.4f}, "
        f"previous={expected:.4f}, "
        f"difference={error:+.4f} mmHg"
    )

    if abs(error) > 0.01:
        print(
            "WARNING: Baseline mismatch. "
            "Do not interpret the comparison yet."
        )


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for condition, ax in zip(
    ["normal", "diabetic"], axes
):
    for boundary, label in [
        ("zero_flux", "Zero flux"),
        ("fixed", "Fixed healthy oxygen"),
    ]:
        name = f"{condition}_{boundary}"

        rows = [
            row for row in all_records
            if row["scenario"] == name
        ]

        ax.plot(
            [row["time_s"] for row in rows],
            [row["centre_po2_mmHg"] for row in rows],
            label=label,
            linewidth=2,
        )

    ax.set_title(condition.capitalize() + " wound")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Wound-centre pO2 (mmHg)")
    ax.grid(alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "boundary_time_comparison.png",
    dpi=200,
)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 10))

for ax, (name, _, _) in zip(
    axes.flat, SCENARIOS
):
    im = ax.imshow(
        final_fields[name].T,
        origin="lower",
        extent=[0, f.Lx * 1000, 0, f.Ly * 1000],
        vmin=0,
        vmax=60,
        cmap="viridis",
    )

    ax.set_title(name.replace("_", " ").title())
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

fig.colorbar(
    im,
    ax=axes,
    label="pO2 (mmHg)",
    shrink=0.8,
)
fig.savefig(
    OUT / "boundary_final_maps.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

print()
print("Results saved to:", OUT.resolve())
print("Original simulation files remain unchanged.")
