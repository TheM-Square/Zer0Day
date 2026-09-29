# OIL TWIN 3D

Interactive 3D advisory dashboard for **Baghewala Field** (heavy oil, Bikaner-Nagaur sub-basin, Rajasthan) covering **Cyclic Steam Stimulation (CSS)** and **Sucker Rod Pumping (SRP)**.

> ## Advisory only
> OIL TWIN is a physics-constrained decision-support prototype. It is **not** a digital twin, **not** autonomous control, and **not** SCADA-integrated. It has **not** been validated against Baghewala field data. Every recommendation needs qualified engineering review.

---

## Run it

No install, no build.

1. Open `oil-twin-3d.html` in a modern browser (Chrome, Edge, Firefox, Safari).
2. An internet connection is needed on first load, because Three.js r128 is fetched from cdnjs.

To host it, upload the single file to any static host (GitHub Pages, Netlify, Vercel).

## What you see

| Area | Purpose |
|---|---|
| **3D scene** | Low-poly pumpjack, wellhead, casing, tubing, rod string, plunger pump with valves, reservoir slab, steam trailer and plume. Drag to orbit, scroll to zoom, hover parts for their governing equation. |
| **Header tiles** | Live temperature, viscosity, SPM, pump rate and guardrail status (solid green PASS / red VETO). |
| **Controls (left)** | All operating inputs from the Streamlit app: SPM, stroke, fillage, T_inj, half-life, rod and plunger geometry, depth, fluid SG, stress limit, gearbox rating, torque factor. |
| **Timeline (bottom)** | CSS cycle scrubber: injection 0-15%, soak 15-25%, production 25-100%. Play button animates it. |
| **Analytics window** | Draggable pop-up with three tabs: *Curves* (μ vs time, μ vs T, rate, stress, torque vs SPM), *Guardrail & optimizer* (loads, utilisation, max safe SPM, recommendation), *ML & provenance*. |
| **Limitations** | Full honest-limitations pop-up. |
| **Camera** | Overview, Surface, Cutaway, Reservoir, Plan presets, plus a ground-cap toggle. |

**Guardrail feedback:** when σ_max or T_peak exceeds its limit, the rod string glows red, the gearbox halo pulses, the pumpjack slows and stutters, and a VETO banner names the violated constraint and the analytic maximum safe SPM.

With the default configuration the veto begins at about **20.6 SPM** (torque-limited). Charts and the SPM input extend to 24-25 so the veto is reachable. Lowering the gearbox rating or stress limit brings it lower.

`prefers-reduced-motion` disables the pumpjack stroke and particles.

## Physics implemented

Ported from the project's Python modules (`config.py`, `srp_physics.py`, `optimizer.py`, `viscosity_model.py`, `trajectory.py`).

**Viscosity** (Arrhenius): `μ(T) = A·exp(B/T_K)`, with B = 60 kJ/mol ÷ R ≈ 7217 K and A calibrated so μ(45 °C) = 11,500 cP. The published 10,000-13,000 cP range is carried as a band. Outside 45 ± 5 °C the result is flagged as **extrapolated**.

**CSS cooling:** `T(t) = T_res + (T_inj − T_res)·exp(−λt)`, where t is days since end of injection, λ = ln2 / half-life, and T_res = 47 °C.

**SRP (API RP 11L, spec form):**
- `Wr = 2.668·d²·L`, `Wf = 0.433·SG·L·Ap`
- `PRLmax/min = Wf − 62.4·SG·(Wr/γs) + Wr ± Wr·S·N²/70471.2`, with **γs = 490 lb/ft³**
- `T_peak = %S·(Wf + 2·S·N²·Wr/707471.2)`
- `σ_max = PRLmax / Ar`, `SLR = Fmin / Fmax`
- VETO if `σ_max > σ_limit` or `T_peak > gearbox rating`
- Analytic max safe SPM from both limits, floored to 0.1

**Optimizer:** grid search over SPM 1-12 (0.1 step) on the guardrail-feasible set only. Objective is `Q = 0.1484·Ap·S·N·(fillage/100)·η(μ)` with `η = 1/(1 + (μ/1000)^0.25)`. If nothing is feasible, no recommendation is returned.

## Assumptions (all from `config.py`, all replaceable)

| Item | Default | Status |
|---|---|---|
| T_res | 47 °C | Source: OIL India tender (46-48 °C) |
| μ at 45 °C | 10,000-13,000 cP (11,500 used) | Source: OIL India tender |
| Arrhenius Ea | 60 kJ/mol | ASSUMPTION (shape only) |
| T_inj | 250 °C | ASSUMPTION |
| Cooling half-life | 45 days | ASSUMPTION |
| Rod / plunger diameter | 1.0 / 2.25 in | ASSUMPTION |
| Stroke | 100 in | ASSUMPTION |
| Pump depth | 1,175 m (midpoint of 1050-1300 m) | ASSUMPTION |
| Fluid SG | 0.96 | From 14-17 °API |
| Rod stress limit | 30,000 psi | CONFIGURATION |
| Gearbox rating | 228,000 in·lbf | CONFIGURATION |
| Torque factor | 30 in | CONFIGURATION |
| Efficiency η(μ) and SPM rule table | see `config.py` | HEURISTIC, not field correlations |

## Known limitations

1. No public Baghewala operational dataset exists. Nothing here is validated against field measurements.
2. NK Field is a conventional field, not thermal heavy oil. It informs SRP mechanics only.
3. μ(T) is anchored at a single temperature; B comes from literature, not from Baghewala data.
4. T_inj and half-life are assumptions. The injection ramp before t = 0 is a linear, illustrative visual only.
5. Pump-rate efficiency and the SPM rule-table band are heuristics. The band is shown for reference and does not constrain the optimizer.
6. Scene depth, stroke and beam angle are schematic, not to scale.
7. **Not ported to the browser:** the Gradient Boosting rod-load model, the autoencoder anomaly detector and the NK data loader. They need Python. The ML tab therefore shows "—" rather than invented metrics and lists the training configuration only.
8. **Parity with the Python code was checked by hand on default inputs, not by an automated fixture suite.** Add one before relying on the numbers.
9. Bloom, chromatic aberration and audio from the original design brief are not implemented.

## Implementation notes

- Single self-contained HTML file: vanilla JavaScript, Three.js r128 (CDN), canvas 2D charts, Arial typography.
- This is not the Vite + React + TypeScript project originally specified. The physics lives in one script block in the file.

## Credits and sources

- API RP 11L, *Recommended Practice for Design Calculations for Sucker Rod Pumping Systems*
- US Patent 11,898,552 B2 (scaled load ratio)
- Bao, Wang & Gates (2016), *Energy* 115:969-985 (CSS cooling methodology)
- OIL India Limited public tender document (Baghewala anchors)
- NK Oil Well Sensor Monitoring dataset, Hugging Face: `Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring` (Apache 2.0)
- Anomaly-detection methodology reference: IEEE OJIM 2026, DOI 10.1109/OJIM.2026.3670416 (cited, not a dataset)
