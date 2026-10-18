
# Computational Modelling of Perfusion and Oxygenation in Critical Wound Healing

## Model Verification, Results, and Limitations

### 1. Research Objective

This project investigates how coupled oxygen diffusion, cellular oxygen consumption, and spatially varying perfusion influence oxygen distributions in a two-dimensional wound bed.

The central research question is:

**How can computational modelling of coupled oxygen diffusion and perfusion dynamics improve prediction of spatial oxygen gradients in diabetic wound beds under impaired microvascular conditions?**

The model compares normal and impaired-perfusion conditions and evaluates oxygen concentration, spatial gradients, hypoxic area, and transient oxygen recovery.

The work is a mechanistic computational investigation. It does not constitute a clinically validated prediction of diabetic wound healing.

---

## 2. Mathematical Model

The governing equation is:

∂C/∂t = D∇²C − M(C) + P(x,y,t)

where:

- C is dissolved oxygen concentration (mol/m³).
- D is the effective oxygen diffusion coefficient (m²/s).
- M(C) represents cellular oxygen consumption.
- P(x,y,t) represents oxygen supply through effective perfusion.

### 2.1 Oxygen Consumption

Oxygen consumption is represented using Michaelis–Menten kinetics:

M(C) = ρ_cell × Vmax × C / (Km + C)

where:

- ρ_cell is cellular density (cells/m³).
- Vmax is maximum oxygen consumption per cell (mol/cell/s).
- Km is the half-saturation concentration (mol/m³).

### 2.2 Perfusion

Perfusion is represented as:

P(x,y,t) = β(x,y) × (Cb − C)

where:

- β(x,y) is the spatially varying effective perfusion coefficient (s⁻¹).
- Cb is the assumed oxygen source concentration (mol/m³).

The wound interior is assigned a lower effective perfusion coefficient than the surrounding tissue.

The diabetic condition is represented by reduced effective perfusion relative to the normal condition.

Diabetes is not modelled through explicit biochemical, inflammatory, vascular remodelling, or tissue damage pathways.

---

## 3. Computational Implementation

The model was implemented in Python using a two-dimensional finite-difference spatial discretisation.

The principal computational configuration is:

| Property | Value |
|---|---|
| Domain | 10 mm × 10 mm |
| Grid | 101 × 101 |
| Spatial resolution | 0.1 mm |
| Wound geometry | Central circular region |
| Wound radius | Approximately 1.5 mm |
| Boundary condition | Zero-flux |
| Time integration | Explicit time stepping |
| Simulation duration | 600 s |
| Final simulation time step | 0.1 s |

The zero-flux boundary condition prevents diffusive oxygen transport across the outer domain boundary.

Oxygen supply and consumption remain active within the domain, so the total oxygen concentration is not expected to remain constant in the complete model.

The model uses an initially lower oxygen concentration inside the wound than in surrounding tissue.

---

## 4. Numerical Verification

### 4.1 Diffusion and Boundary Conditions

The zero-flux Laplacian was checked for conservative internal diffusion fluxes.

The diffusion-only implementation was assessed separately from the full reaction–diffusion–perfusion model.

### 4.2 Time-Step Comparison

The original normal and diabetic simulations were compared using smaller and larger time steps.

Previous baseline comparisons between dt = 0.1 s and dt = 0.005 s showed small differences in final oxygen metrics.

A separate reduced-supply verification compared:

- dt = 0.1 s
- dt = 0.05 s

For the reduced-supply scenario with source pO₂ = 60 mmHg and perfusion multiplier = 0.25:

| Measurement | dt = 0.1 s | dt = 0.05 s |
|---|---:|---:|
| Normal mean wound pO₂ | 48.4694 | 48.4694 |
| Diabetic mean wound pO₂ | 47.5524 | 47.5523 |
| Normal hypoxic area | 47.7762% | 47.7762% |
| Diabetic hypoxic area | 74.1750% | 74.1750% |

The reported results are consistent at these time-step resolutions.

This supports numerical time-step convergence for the tested scenarios.

It does not establish spatial-grid convergence, physiological validity, or convergence for every possible parameter combination.

---

## 5. Original Baseline Simulations

The original baseline uses the initial source concentration and perfusion parameters specified in `parameters.py`.

### 5.1 Final Results

| Measurement | Normal | Diabetic |
|---|---:|---:|
| Centre pO₂ | 103.20 mmHg | 96.41 mmHg |
| Mean wound pO₂ | 105.00 mmHg | 99.08 mmHg |
| Mean surrounding pO₂ | 109.45 mmHg | 106.00 mmHg |
| Surrounding minus wound pO₂ | 4.45 mmHg | 6.92 mmHg |
| Final hypoxic wound area | 0% | 0% |
| Centre threshold-crossing time | 40.30 s | 60.56 s |

The diabetic condition has lower wound oxygenation, a larger wound-to-surrounding oxygen difference, and delayed centre threshold crossing.

Both conditions have zero hypoxic wound area at 600 seconds under the provisional threshold.

The baseline source oxygen concentration corresponds to approximately 114.29 mmHg using the assumed oxygen solubility coefficient.

The relatively high final oxygenation values should not be interpreted as validated diabetic wound measurements.

---

## 6. Exploratory Physiological Parameter Study

A separate calibration exploration was performed without modifying the original baseline parameters.

The exploration varied:

- Source pO₂: 60, 80, and 100 mmHg.
- Normal effective perfusion multiplier: 0.25, 0.50, and 1.00.
- Diabetic effective perfusion: approximately 69.13% of the corresponding normal value.

Nine parameter combinations were examined, with normal and diabetic simulations for each combination.

These values are exploratory assumptions rather than experimentally calibrated tissue parameters.

### 6.1 Main Observation

Lower assumed oxygen source concentration and reduced effective perfusion generally decreased wound oxygenation and delayed threshold crossing.

Some combinations produced hypoxic wound regions that remained below the provisional threshold after 600 seconds.

A comparison range of 35–50 mmHg was used illustratively during the parameter exploration.

This range is not a universal normal-wound reference, and matching it does not validate the model.

---

## 7. Final Reduced-Supply Simulations

The final reduced-supply scenario used:

- Source pO₂ = 60 mmHg.
- Normal effective perfusion multiplier = 0.25.
- Diabetic-to-normal perfusion ratio ≈ 0.6913.
- Simulation duration = 600 seconds.

### 7.1 Final Results

| Measurement | Normal | Diabetic |
|---|---:|---:|
| Centre pO₂ | 46.85 mmHg | 45.75 mmHg |
| Mean wound pO₂ | 48.47 mmHg | 47.55 mmHg |
| Mean surrounding pO₂ | 53.76 mmHg | 53.73 mmHg |
| Surrounding minus wound pO₂ | 5.29 mmHg | 6.18 mmHg |
| Final hypoxic wound area | 47.78% | 74.18% |
| Centre threshold crossing | Not reached | Not reached |

At 600 seconds, the diabetic wound has approximately 26.40 percentage points more hypoxic area than the normal wound.

Although the mean wound oxygenation differs by less than 1 mmHg, the threshold-defined hypoxic areas differ substantially.

This illustrates the additional information provided by spatially resolved oxygen modelling.

The magnitude of the hypoxic-area difference depends on the assumed threshold and spatial oxygen distribution.

---

## 8. Transient Hypoxic-Area Behaviour

The reduced-supply simulations produced a non-monotonic response in hypoxic wound area.

### 8.1 Normal Condition

| Time (s) | Mean wound pO₂ (mmHg) | Mean surrounding pO₂ (mmHg) | Hypoxic area |
|---|---:|---:|---:|
| 0 | 39.68 | 72.22 | 100.00% |
| 100 | 47.94 | 67.18 | 53.52% |
| 200 | 49.56 | 63.37 | 37.45% |
| 300 | 49.93 | 60.24 | 27.69% |
| 400 | 49.70 | 57.67 | 25.39% |
| 500 | 49.16 | 55.53 | 30.56% |
| 600 | 48.47 | 53.76 | 47.78% |

### 8.2 Diabetic Condition

| Time (s) | Mean wound pO₂ (mmHg) | Mean surrounding pO₂ (mmHg) | Hypoxic area |
|---|---:|---:|---:|
| 0 | 39.68 | 72.22 | 100.00% |
| 100 | 47.67 | 67.65 | 53.52% |
| 200 | 49.17 | 64.03 | 42.04% |
| 300 | 49.45 | 60.90 | 34.58% |
| 400 | 49.11 | 58.19 | 35.72% |
| 500 | 48.42 | 55.81 | 48.92% |
| 600 | 47.55 | 53.73 | 74.18% |

Both conditions initially experience a reduction in hypoxic area, followed by renewed expansion of hypoxic regions.

One plausible explanation is that the initially oxygen-rich surrounding tissue temporarily supplies oxygen to the wound.

The surrounding tissue initially has a pO₂ of approximately 72.22 mmHg, whereas the assumed reduced source pO₂ is 60 mmHg.

As the surrounding tissue oxygenation decreases, the oxygen concentration gradient supporting wound oxygenation weakens.

The diabetic condition shows greater late-stage hypoxic-area expansion.

This is a model-dependent transient response and should not be presented as an experimentally established trajectory of diabetic wound healing.

---

## 9. Sobol Sensitivity Analysis

A global Sobol sensitivity analysis was performed using the refined wound-bed model.

The final analysis used:

- Base sample size N = 256.
- Six uncertain parameters.
- 2,048 model evaluations.
- First-order and total-order Sobol indices.

The investigated parameters were:

1. Oxygen diffusion coefficient D.
2. Maximum cellular consumption Vmax.
3. Michaelis–Menten constant Km.
4. Cell density rho_cell.
5. Surrounding effective perfusion beta.
6. Wound perfusion fraction.

### 9.1 Principal Findings

Within the specified exploratory parameter ranges, surrounding perfusion and wound perfusion fraction had the largest total-order contributions to several oxygenation outcomes.

For example:

| Parameter | Total-order index: centre pO₂ | Total-order index: mean wound pO₂ |
|---|---:|---:|
| Surrounding perfusion | 0.6425 | 0.7329 |
| Wound perfusion fraction | 0.3409 | 0.2453 |
| Diffusion coefficient | 0.0109 | 0.0066 |
| Maximum consumption | 0.0154 | 0.0188 |
| Cell density | 0.0161 | 0.0195 |
| Michaelis–Menten constant | 0.0001 | 0.0001 |

These indices describe sensitivity within the selected input ranges and baseline assumptions.

They do not establish that perfusion is universally more important than diffusion or cellular oxygen consumption in all biological wound conditions.

The Sobol sampling ranges were exploratory rather than independently validated physiological uncertainty distributions.

---

## 10. Model Assumptions

The main modelling assumptions are:

1. Oxygen transport occurs within a two-dimensional tissue domain.
2. The wound has an idealised circular geometry.
3. Effective diffusion is isotropic and spatially uniform.
4. Oxygen consumption follows Michaelis–Menten kinetics.
5. Cellular density remains constant during the simulation.
6. Perfusion is represented by a first-order source term.
7. The wound interior has a fixed fraction of surrounding perfusion.
8. The diabetic condition is represented by reduced perfusion.
9. Outer boundaries have zero diffusive flux.
10. Model parameters remain constant throughout each simulation.
11. Oxygen concentration is converted to pO₂ using a provisional solubility coefficient.
12. Hypoxia is defined using a provisional fixed oxygen threshold.

These assumptions allow controlled computational comparisons but simplify the underlying biological processes.

---

## 11. Limitations

### 11.1 Physiological Parameter Uncertainty

Several parameters are not independently calibrated against diabetic wound-bed measurements.

The oxygen solubility coefficient is used as a provisional concentration-to-pO₂ conversion factor.

The effective perfusion coefficient is a phenomenological model parameter and should not be interpreted directly as a measured blood-flow rate.

### 11.2 Hypoxia Threshold

The hypoxia threshold is provisionally defined as:

C_hyp = 0.061 mol/m³

Using the assumed solubility coefficient, this corresponds to approximately 48.41 mmHg.

Hypoxic-area percentages are sensitive to this threshold.

Transcutaneous oxygen tension measurements and direct wound-bed tissue oxygenation are not interchangeable without appropriate validation.

### 11.3 Biological Simplification

The model does not explicitly represent:

- Angiogenesis.
- Inflammation.
- Fibroblast migration or proliferation.
- Collagen deposition.
- Tissue remodelling.
- Vascular network geometry.
- Blood oxygen binding and haemoglobin transport.
- Dynamic changes in wound geometry.
- Patient-specific diabetic pathophysiology.

Therefore, the model predicts oxygen transport rather than wound closure or healing time.

### 11.4 Numerical Scope

Time-step consistency was demonstrated for selected scenarios.

A complete spatial-grid refinement study has not been established by the checks documented here.

The simulation duration is 600 seconds, so results describe short-term oxygen transport dynamics.

Long-term wound healing occurs over much longer biological timescales.

### 11.5 Validation Scope

The project has completed numerical verification and exploratory physiological comparisons.

Independent experimental or clinical validation against measured wound oxygenation has not been performed.

Consequently, the simulations should be described as mechanistic predictions under assumed conditions.

---

## 12. Contribution to the Research Question

The computational framework couples diffusion, cellular oxygen consumption, and spatially varying perfusion in a two-dimensional wound domain.

It demonstrates that:

- Reduced effective perfusion can decrease local oxygenation.
- Spatial oxygen deficits can differ between normal and impaired-perfusion conditions.
- Reduced oxygen supply can produce persistent threshold-defined hypoxic regions.
- Oxygen recovery can be transient rather than sustained under selected assumptions.
- Spatial hypoxic area can reveal differences not apparent from mean oxygenation alone.
- Global sensitivity analysis can identify influential parameters within defined uncertainty ranges.

The principal contribution is a reproducible computational framework for exploring how perfusion impairment and oxygen transport assumptions affect predicted wound-bed oxygen distributions.

The model does not yet provide a clinically validated or patient-specific prediction of diabetic wound healing.

---

## 13. Reproducibility

The main implementation and analysis files include:

- `parameters.py` — model parameters.
- `model.py` — diffusion, consumption, perfusion, and wound geometry.
- `compare_normal_diabetic.py` — baseline comparison.
- `sobol_wound_refined.py` — refined global sensitivity analysis.
- `physiological_calibration.py` — exploratory parameter study.
- `final_simulations.py` — final comparison simulations and figures.

Results are stored in separate directories:

- `sobol_refined_results/`
- `physiological_calibration_results/`
- `final_simulation_results/`

The final simulations can be reproduced by running:

```powershell
python .\final_simulations.py
```

The exploratory physiological scenarios can be reproduced by running:

```powershell
python .\physiological_calibration.py
```

The simulations use deterministic numerical calculations for fixed input parameters.

---

## 14. Overall Conclusion

The computational model successfully demonstrates how coupled oxygen diffusion, cellular consumption, and spatially varying perfusion can be used to investigate wound-bed oxygen distributions under normal and impaired-perfusion conditions.

Under the tested assumptions, reduced perfusion produces lower oxygenation, larger spatial oxygen deficits, and greater hypoxic wound area when oxygen supply is limited.

The refined Sobol analysis identifies effective perfusion parameters as major contributors to several model outcomes within the investigated parameter ranges.

The reduced-supply scenario additionally demonstrates a transient decrease followed by an increase in hypoxic wound area, highlighting the importance of examining spatial and time-dependent oxygenation rather than relying exclusively on final mean values.

These findings support the usefulness of the computational framework for hypothesis generation and parameter exploration.

Further work should prioritise wound-specific physiological calibration, independent experimental validation, hypoxia-threshold sensitivity, and additional spatial-grid convergence testing before quantitative clinical interpretation.
