# lightmatter

`lightmatter` is a research-oriented Python package for simulating and analyzing the ultrafast optical response of matter with a classical time-dependent Drude--Lorentz description.

The package currently has two goals:

1. **Forward modeling:** run a one-dimensional finite-difference time-domain (FDTD) simulation of an ultrashort optical pulse interacting with a thin material film whose Drude--Lorentz parameters may vary in time.
2. **Inverse analysis:** infer time-dependent Drude parameters of a thin film from the incident/reflected waveform by comparing peak-by-peak changes in the reflected electric field.

This repository is intended as a first-principles, classical electrodynamics sandbox for studying how ultrafast changes in carrier density, scattering rate, and bound-electron response may appear in time-domain optical measurements.

---

## Scientific motivation

Ultrafast pump--probe experiments often measure changes in the reflected or transmitted electric field after a material has been driven out of equilibrium. A useful first description of the optical response is to treat the medium using a Drude--Lorentz dielectric function with time-dependent parameters,

$$
\varepsilon(\omega,t)
=
\varepsilon_{\infty}(t)
-
\frac{\omega_D^2(t)}
{\omega\left[\omega+i\gamma_D(t)\right]}
-
\frac{\Delta\varepsilon(t)\Omega_L^2(t)}
{\omega^2-\Omega_L^2(t)+i\Gamma_L(t)\omega}.
$$

Here, the Drude plasma frequency `omega_D(t)` and damping rate `gamma_D(t)` encode the time-dependent free-carrier response, while the Lorentz oscillator parameters describe bound-electron resonances.

`lightmatter` combines:

- explicit 1D Maxwell time stepping,
- time-dependent Drude and Lorentz material updates,
- reflected/transmitted waveform extraction,
- peak and zero-crossing based phase/amplitude analysis, and
- nonlinear least-squares inference of `omega_D(t)` and `gamma_D(t)` from multifrequency reflected pulses.

---

## Main features

- **1D Yee FDTD solver** with soft-source pulse injection.
- **Time-dependent Drude--Lorentz material model**.
- **Multiple material update schemes**, including:
  - recursive-convolution style update,
  - direct time-domain Drude--Lorentz state update,
  - PML-corrected direct update.
- **Absorbing boundary layers / PML-style damping** for finite simulation domains.
- **Gaussian, boxed, and smoothed-box optical pulses**.
- **Transmission and reflection probe recording**.
- **Snapshot output** for field visualization.
- **Simulation save/load utilities** using `info.json` and compressed `data.npz`.
- **GIF and static-frame visualization** of the electric-field propagation.
- **Peak-by-peak reflected waveform analysis** using matched extrema.
- **Interpolated zero-crossing phase retrieval** for improved timing accuracy.
- **Multifrequency inverse fitting** of time-dependent Drude parameters.
- **Optional uncertainty estimates** from the least-squares Jacobian.

---

## Repository structure

A typical package layout is:

```text
lightmatter/
├── fdtd.py
├── infer_drude_derivative.py
├── peak_analyzer_interpolate.py
├── io.py
├── units.py
└── utils.py
```

### Module summary

| Module | Purpose |
|---|---|
| `fdtd.py` | FDTD parameter classes, time-domain solvers, PML construction, pulse definitions, transmission-model utilities, and refractive-index retrieval. |
| `infer_drude_derivative.py` | Builds multifrequency reflected-waveform datasets and fits `omega_D(t)` and `gamma_D(t)` using a derivative-corrected Drude reflection model. |
| `peak_analyzer_interpolate.py` | Extracts complex reflected-field ratios using signed peak matching and interpolated zero-crossing phase shifts. |
| `io.py` | Saves, loads, plots, and animates simulation outputs. |
| `units.py` | Defines internal code units and SI conversion constants. |
| `utils.py` | Contains analytical Drude--Lorentz dielectric-function utilities. |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/archis02/lightmatter.git
cd lightmatter
```

Create and activate a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy scipy numba matplotlib
```

If the repository contains a `pyproject.toml` or `setup.py`, install it in editable mode:

```bash
pip install -e .
```

Otherwise, run scripts from the repository root or add the parent directory to `PYTHONPATH`.

---

## Dependencies

The current code uses:

- `numpy`
- `scipy`
- `numba`
- `matplotlib`

Optional but recommended for development:

- `pytest`
- `jupyter`
- `ruff` or `black`

---

## Units and conventions

The FDTD solver uses normalized code units:

| Quantity | Code unit | SI conversion |
|---|---:|---:|
| Length | Angstrom | `UNIT_L = 1e-10 m` |
| Time | picosecond | `UNIT_T = 1e-12 s` |
| Frequency | THz | `UNIT_F = 1e12 Hz` |
| Speed of light | Angstrom / ps | `C0 = 2.99792458e6` |
| Speed of light, SI | m/s | `C0_SI = 299792458.0` |

For a stable vacuum Yee update, choose a Courant-like time step approximately satisfying:

$$
dt \lesssim \frac{dz}{C_0}.
$$

---

## Extracting the complex reflected-field ratio

The inverse workflow starts by comparing a reflected waveform from a time-varying material to a reflected waveform from a constant reference material.

`lightmatter` estimates the complex ratio

$$
R_\mathrm{meas}(t)
=
\frac{E_{\mathrm{refl,vary}}(t)}
     {E_{\mathrm{refl,const}}(t)}
\approx
A(t)\exp\left[i\omega_0\Delta t(t)\right].
$$

The amplitude ratio `A(t)` is obtained from matched positive and negative extrema. The phase shift is obtained from linearly interpolated zero crossings associated with each matched extremum.


The returned values are:

| Output | Meaning |
|---|---|
| `t_mid` | Time coordinate associated with each matched peak/zero-crossing pair. |
| `ratio_complex` | Complex reflected-field ratio. |
| `aux` | Diagnostics: matched peaks, zero crossings, timing shifts, amplitude ratios, and raw peak delays. |

---

## Inferring time-dependent Drude parameters

The higher-level inference routine compares multiple reflected-pulse simulations at different carrier frequencies. It constructs a common time grid, interpolates the complex reflected-field ratios onto that grid, and fits `omega_D(t)` and `gamma_D(t)` at each time slice.

The Drude-only dielectric function used in the inverse model is

$$
\varepsilon_D(\omega,t)
=
\varepsilon_{\infty}
-
\frac{\omega_D^2(t)}
{\omega\left[\omega+i\gamma_D(t)\right]}.
$$

The inference model also includes a first-order time-derivative correction,

$$
n_\mathrm{eff}^2(\omega,t)
=
\varepsilon_D(\omega,t)
-
\frac{i}{\omega}
\frac{\partial \varepsilon_D}{\partial t}.
$$

The derivative is approximated from the fitted parameter change between consecutive time slices.


### Peak-by-peak inverse analysis

The reflected field from the material is isolated by subtracting the no-film reference trace,

```math
E_\mathrm{refl}
=
E_r^\mathrm{with\ film}
-
E_r^\mathrm{no\ film}.
```

For each carrier frequency, the algorithm:

1. identifies positive and negative extrema in the constant and time-varying reflected waveforms,
2. performs monotonic sign-preserving peak matching,
3. identifies zero crossings by linear interpolation,
4. assigns a physically consistent zero crossing to each matched peak,
5. computes amplitude ratios from peak heights,
6. computes phase shifts from zero-crossing time delays, and
7. constructs a complex reflected-ratio time series.

The multifrequency inverse step then fits the derivative-corrected Drude reflection model independently at each common time point, with optional smoothness regularization.

---

## Current limitations

This code is actively evolving research software. Important limitations include:

- The FDTD solver is currently **one-dimensional** and assumes normal incidence.
- The material film is represented as a planar slab occupying a contiguous grid region.
- The inverse model currently focuses on fitting time-dependent **Drude** parameters from reflected fields.
- The peak-by-peak analysis assumes the waveform has identifiable, matchable extrema and zero crossings.
- The directory structure expected by the inference helper is currently fixed.
- Internal reflections, finite-thickness effects, PML placement, and pulse bandwidth should be checked carefully for each physical system.
- The covariance estimates from least squares are local linearized estimates, not full Bayesian posterior uncertainties.
- The API may change as the package is cleaned up and generalized.

---

## Development notes

Planned improvements include:

- a cleaner public API,
- noise-aware fitting,
- finite-thickness reflection models beyond the ultrathin-film approximation,
- joint Drude--Lorentz parameter inference, and
- packaging metadata for simple installation.

---

## Disclaimer

`lightmatter` is research code. Results should be verified against analytical limits, grid-convergence studies, known static-material benchmarks, and controlled synthetic inverse problems before being interpreted physically.