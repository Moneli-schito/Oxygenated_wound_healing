
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import final_gradient_simulation_corrected as f

OUTPUT = Path("initial_sensitivity_results")
OUTPUT.mkdir(exist_ok=True)

eq_dir = Path("spatial_equilibrium_results")

normal_eq = np.load(eq_dir / "normal_equilibrium_po2.npy")
diabetic_eq = np.load(eq_dir / "diabetic_equilibrium_po2.npy")

cx, cy = f.NX // 2, f.NY // 2

x, y = np.indices((f.NX, f.NY))
r = np.sqrt((x - cx)**2 + (y - cy)**2)

profile = np.clip(
    1 - r / f.WOUND_RADIUS_CELLS, 0, 1
)

def initial_field(equilibrium, centre_target):
    fraction = 1 - centre_target / equilibrium[cx, cy]
    return equilibrium * (1 - fraction * profile)

normal = initial_field(normal_eq, 10.0)

targets = [9.0, 7.0, 5.0]

plt.figure(figsize=(9, 5))
plt.plot(
    f.X_MM, normal[:, cy],
    label="Normal: 10 mmHg", linewidth=2
)

for target in targets:
    diabetic = initial_field(diabetic_eq, target)

    print(f"\nDiabetic centre target: {target:.1f} mmHg")
    print(f"Actual centre: {diabetic[cx, cy]:.4f} mmHg")
    print(
        f"Mean wound: "
        f"{np.mean(diabetic[f.WOUND_MASK]):.4f} mmHg"
    )

    plt.plot(
        f.X_MM, diabetic[:, cy],
        label=f"Diabetic: {target:g} mmHg",
        linewidth=2
    )

plt.xlabel("Position (mm)")
plt.ylabel("Initial oxygen pressure (mmHg)")
plt.title("Initial-condition sensitivity scenarios")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "initial_sensitivity_profiles.png",
    dpi=250
)
plt.close()

print("\nResults saved to:", OUTPUT)
