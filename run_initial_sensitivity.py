
from pathlib import Path
import csv
import numpy as np

import final_gradient_simulation_corrected as f
from model import full_oxygen_model

OUTPUT = Path("initial_sensitivity_results")
OUTPUT.mkdir(exist_ok=True)

normal_eq = np.load(
    "spatial_equilibrium_results/normal_equilibrium_po2.npy"
)
diabetic_eq = np.load(
    "spatial_equilibrium_results/diabetic_equilibrium_po2.npy"
)

cx, cy = f.NX // 2, f.NY // 2
x, y = np.indices((f.NX, f.NY))
radius = np.sqrt((x - cx)**2 + (y - cy)**2)

profile = np.clip(
    1 - radius / f.WOUND_RADIUS_CELLS, 0, 1
)

def initial_field(eq, centre_target):
    fraction = 1 - centre_target / eq[cx, cy]
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid initial oxygen target")
    return eq * (1 - fraction * profile)

def simulate(name, eq, centre_target, beta_ref):
    C = initial_field(eq, centre_target) * f.alpha_O2

    beta_field = np.full(C.shape, f.beta_normal)
    beta_field[f.WOUND_MASK] = (
        beta_ref * f.WOUND_PERFUSION_FRACTION
    )

    rows = []
    for step in range(f.STEPS + 1):
        if step % f.SAVE_EVERY == 0:
            po2 = C / f.alpha_O2
            rows.append({
                "scenario": name,
                "time_s": round(step * f.DT, 4),
                "centre_po2_mmHg": float(po2[cx, cy]),
                "mean_wound_po2_mmHg": float(
                    np.mean(po2[f.WOUND_MASK])
                ),
                "mean_surrounding_po2_mmHg": float(
                    np.mean(po2[f.SURROUNDING_MASK])
                ),
            })

        if step == f.STEPS:
            break

        dC = full_oxygen_model(
            C, f.D, f.Vmax, f.Km, f.rho_cell,
            beta_field, f.SOURCE_CONCENTRATION,
            f.dx, f.dy, boundary_type="zero_flux"
        )
        C = C + f.DT * dC

        if not np.all(np.isfinite(C)) or np.any(C < 0):
            raise ValueError(f"Invalid solution in {name}")

    final = C / f.alpha_O2

    print(
        f"{name}: centre={final[cx, cy]:.4f}, "
        f"mean wound={np.mean(final[f.WOUND_MASK]):.4f} mmHg"
    )

    return rows

all_rows = []

all_rows += simulate(
    "normal_10", normal_eq, 10, f.beta_normal
)

for target in (9, 7, 5):
    all_rows += simulate(
        f"diabetic_{target}",
        diabetic_eq, target, f.beta_diabetic
    )

with (OUTPUT / "transient_sensitivity.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=list(all_rows[0])
    )
    writer.writeheader()
    writer.writerows(all_rows)

print("Saved:", OUTPUT / "transient_sensitivity.csv")
