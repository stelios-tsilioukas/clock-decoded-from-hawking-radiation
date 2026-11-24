# clock-decoded-from-hawking-radiation

[![DOI](https://zenodo.org/badge/1470604111.svg)](https://doi.org/10.5281/zenodo.24511865)

Code and result files for

> S. A. Tsilioukas, *What does a clock decoded from Hawking radiation read? Infalling proper time in quantum Jackiw-Teitelboim gravity* (paper), and the accompanying Letter.

The code computes the reading of a clock that falls into a quantum Jackiw-Teitelboim (JT) black hole, is stopped by an ingoing signal from the boundary, and is decoded from the Hawking radiation. It covers:

- the classical reading in global AdS$_2$;
- the exact Schwarzian disk moments of the tick rate, to all orders in $1/C$;
- the physical-variable map that separates the backreaction of the clock from quantum effects;
- replica (PSSY) decoding of a static black hole, including the threading criterion and a random-matrix model;
- the radiating Hayden-Preskill model with late emission (the main results);
- the non-perturbative corrections from the JT matrix integral (genus one and wormholes).

## Requirements

Python 3.10 or later with NumPy, SciPy, mpmath and Matplotlib:

```bash
pip install -r requirements.txt
```

## How to run

**Run every script from the repository root**, for example

```bash
python 03_phase3_quantum/code/step3e_checks_basic.py
python figures/fig_geometry.py
```

Imports and output paths are relative to the root. Scripts that write results put them in the `results/` folder of their phase. The result files used in the paper are already included. Most scripts run in seconds to minutes; the converged scans (Steps 3b, 4c, 4g and the A7/A8 re-derivation runs) take from tens of minutes to a few hours on a laptop.

## Layout

| Folder | Content |
|---|---|
| `02_phase2_jt/` | Semiclassical JT: classical clock, quantum extremal surfaces, early decoding studies |
| `03_phase3_quantum/` | Exact Schwarzian moments (3a-3b), non-perturbative completion (3c), physical variables and backreaction (3d-3f), static decoding and threading (3g-3i), random-matrix model (3i-3j), stopwatch scan (3k), planar vs threading comparison (3l) |
| `04_phase4_radiating/` | Radiating Hayden-Preskill model: onset (4a), decorated replica sums (4b), main tables (4c), large-$C$ and emission checks (4d-4e), finite brane tension (4f), genus-one decoded reading (4g) |
| `rederivation/` | Independent re-implementations written from a specification only, used as cross-checks (Appendix E of the paper) |
| `figures/` | Scripts and PDFs for Figs. 1 and 2 |

## Where each result comes from

| Paper | Script(s) | Results |
|---|---|---|
| Fig. 1 (geometry) | `figures/fig_geometry.py` | `figures/fig_geometry.pdf` |
| Table I (tick-rate moments) | `rederivation/jt-oneside/a2_exact.py`; cross-check `03_phase3_quantum/code/step3b_*.py` | `rederivation/jt-oneside/exact_*.json` |
| Table II (physical variables) | `03_phase3_quantum/code/step3d_phys.py`, `step3d_run.py` | `03_phase3_quantum/results/step3d_*` |
| Table III (stopwatch) | `03_phase3_quantum/code/step3k_scan.py` | `03_phase3_quantum/results/step3k_scan_*` |
| Table IV (threading vs caps) | `03_phase3_quantum/code/step3i_thread_vs_cap.py`; `rederivation/threading/` | `03_phase3_quantum/results/step3i_thread_vs_cap.json` |
| Table V (random-matrix model) | `03_phase3_quantum/code/step3i_rmt.py`, `step3j_absorption.py`; `rederivation/rmt-absorption/` | `03_phase3_quantum/results/step3i_rmt_*`, `step3j_*` |
| Frozen-resolvent comparison (Sec. V) | `03_phase3_quantum/code/step3l_*.py` | `03_phase3_quantum/results/step3l_*` |
| Table VI (main results) | `04_phase4_radiating/code/step4c_tables.py`; `rederivation/radiating-decoder/` (A7) | `04_phase4_radiating/results/step4c_*` |
| Fig. 2 (decoder correction) | `figures/fig_decoder.py`, data from `rederivation/radiating-decoder/` (A8); `04_phase4_radiating/code/step4e_*.py` | `figures/fig_decoder.pdf` |
| Finite brane tension (Sec. VI E) | `04_phase4_radiating/code/step4f_finite_tension.py` | `04_phase4_radiating/results/step4f_*` |
| Non-perturbative rate, $R_\infty$ (Sec. VII) | `03_phase3_quantum/code/step3c_*.py` | `03_phase3_quantum/results/step3c_*` |
| Table VII (non-perturbative, physical) | `03_phase3_quantum/code/step3d_all_terms_phys.py` | `03_phase3_quantum/results/step3d_all_terms_phys_*` |
| Table VIII (genus-one decoded reading) | `04_phase4_radiating/code/step4g_genus_decoded.py` | `04_phase4_radiating/results/step4g_*` |

## Conventions

Boundary time $u=2\pi t/\beta$, coupling $C=2\pi\phi_r/\beta$, $S-S_0\simeq 2\pi C$. Disk region weight $d\mu(k)\,e^{-s_\beta \ell k^2}$ with $d\mu(k)=2k\sinh(2\pi k)\,dk$ and $s_\beta=1/(2C)$; bilocal vertex $\Gamma(\Delta\pm ia\pm ib)/(2\pi^2\Gamma(2\Delta))$. "Bare" variables refer to the parameters of the disk computation, "physical" variables to those of the black hole with the clock inside (Sec. II D of the paper).

## Citation

Please cite the paper and this software (see `CITATION.cff`). The software is archived on Zenodo:

- all versions (concept DOI): [10.5281/zenodo.24511865](https://doi.org/10.5281/zenodo.24511865)
- v1.0.0: [10.5281/zenodo.24511866](https://doi.org/10.5281/zenodo.24511866)

```bibtex
@misc{Tsilioukas:2026code,
  author       = {Tsilioukas, Stylianos A.},
  title        = {clock-decoded-from-hawking-radiation: code for ``What does a clock decoded from Hawking radiation read?''},
  publisher    = {Zenodo},
  year         = {2026},
  doi          = {10.5281/zenodo.24511865},
  url          = {https://doi.org/10.5281/zenodo.24511865}
}
```

## Licence

MIT, see `LICENSE`.
