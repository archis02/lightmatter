from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit, prange
import sys
import warnings
from scipy.optimize import least_squares

from .units import C0, C0_SI


# Three-pole, lossless Malitson fit for room-temperature fused silica.
# The fitted high-frequency constant is epsilon_inf = 1, which is already
# supplied by MaterialParamsTD.eps_inf in the usual unpumped configuration.
# Therefore only the three dispersive pole contributions are added below.
SIO2_BACKGROUND_DEL_EPS_DEFAULT = (
    0.6961663,
    0.4079426,
    0.8974794,
)
SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT = (
    0.0684043,
    0.1162414,
    9.896161,
)


def _prepare_explicit_pulse_waveform(
    explicit_pulse_waveform,
    *,
    pulsetype,
    Nt,
    dt,
):
    """Return real source samples at the FDTD source-update times.

    For ``pulsetype='explicit'``, ``explicit_pulse_waveform`` may be:

    * a callable evaluated at ``t_n = n*dt``;
    * a one-dimensional array containing exactly ``Nt`` field samples; or
    * a ``(times, values)`` tuple, linearly interpolated onto ``t_n`` and set
      to zero outside the supplied time interval.

    Sampling is deliberately done outside the Numba kernels so ordinary
    Python and SciPy interpolation callables are supported.
    """
    Nt = int(Nt)
    if Nt < 1:
        raise ValueError("Nt must be a positive integer.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")

    if pulsetype != "explicit":
        return np.zeros(Nt, dtype=np.float64)

    if explicit_pulse_waveform is None:
        raise ValueError(
            "explicit_pulse_waveform must be supplied when "
            "pulsetype='explicit'."
        )

    t_grid = np.arange(Nt, dtype=np.float64) * float(dt)

    if callable(explicit_pulse_waveform):
        try:
            values = explicit_pulse_waveform(t_grid)
        except Exception:
            # Some user callables accept only scalar inputs.
            values = [explicit_pulse_waveform(float(t)) for t in t_grid]
        values = np.asarray(values)
        if values.ndim == 0:
            values = np.full(Nt, values.item())

    elif (
        isinstance(explicit_pulse_waveform, tuple)
        and len(explicit_pulse_waveform) == 2
    ):
        source_times = np.asarray(explicit_pulse_waveform[0])
        source_values = np.asarray(explicit_pulse_waveform[1])

        if source_times.ndim != 1 or source_values.ndim != 1:
            raise ValueError(
                "Explicit waveform times and values must be one-dimensional."
            )
        if source_times.size != source_values.size or source_times.size < 2:
            raise ValueError(
                "Explicit waveform times and values must have equal length "
                "and contain at least two samples."
            )
        if np.iscomplexobj(source_times) or np.iscomplexobj(source_values):
            raise ValueError("The explicit pulse waveform must be real-valued.")

        source_times = np.asarray(source_times, dtype=np.float64)
        source_values = np.asarray(source_values, dtype=np.float64)
        if not np.all(np.isfinite(source_times)) or not np.all(
            np.isfinite(source_values)
        ):
            raise ValueError(
                "Explicit waveform times and values must all be finite."
            )
        if np.any(np.diff(source_times) <= 0.0):
            raise ValueError(
                "Explicit waveform times must be strictly increasing."
            )

        values = np.interp(
            t_grid,
            source_times,
            source_values,
            left=0.0,
            right=0.0,
        )
    else:
        values = np.asarray(explicit_pulse_waveform)

    if np.iscomplexobj(values):
        raise ValueError("The explicit pulse waveform must be real-valued.")
    if values.ndim != 1 or values.size != Nt:
        raise ValueError(
            "The explicit pulse waveform must produce exactly Nt samples; "
            f"expected shape ({Nt},), received {values.shape}."
        )

    values = np.ascontiguousarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("The explicit pulse waveform contains NaN or infinity.")
    return values


def _prepare_sio2_background_parameters(
    *,
    dt,
    background_del_eps=None,
    background_omega0=None,
    background_lambda0_um=None,
    background_gamma=None,
):
    """Validate fixed Lorentz poles and convert wavelengths to rad/ps."""
    if background_del_eps is None:
        background_del_eps = SIO2_BACKGROUND_DEL_EPS_DEFAULT

    if background_omega0 is not None and background_lambda0_um is not None:
        raise ValueError(
            "Supply either background_omega0 or background_lambda0_um, not both."
        )

    del_eps = np.atleast_1d(np.asarray(background_del_eps))
    if np.iscomplexobj(del_eps):
        raise ValueError("background_del_eps must be real-valued.")
    del_eps = np.ascontiguousarray(del_eps, dtype=np.float64)

    if background_omega0 is None:
        if background_lambda0_um is None:
            background_lambda0_um = SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT
        lambda0_um = np.atleast_1d(np.asarray(background_lambda0_um))
        if np.iscomplexobj(lambda0_um):
            raise ValueError("background_lambda0_um must be real-valued.")
        lambda0_um = np.ascontiguousarray(lambda0_um, dtype=np.float64)
        if not np.all(np.isfinite(lambda0_um)) or np.any(lambda0_um <= 0.0):
            raise ValueError(
                "All background resonance wavelengths must be finite and positive."
            )

        # C0 is in Angstrom/ps and 1 micrometre = 1e4 Angstrom, hence omega0
        # is returned in rad/ps, matching the rest of this FDTD module.
        omega0 = np.ascontiguousarray(
            2.0 * np.pi * C0 / (lambda0_um * 1.0e4),
            dtype=np.float64,
        )
    else:
        omega0 = np.atleast_1d(np.asarray(background_omega0))
        if np.iscomplexobj(omega0):
            raise ValueError("background_omega0 must be real-valued.")
        omega0 = np.ascontiguousarray(omega0, dtype=np.float64)

    if background_gamma is None:
        gamma = np.zeros_like(omega0)
    else:
        gamma = np.atleast_1d(np.asarray(background_gamma))
        if np.iscomplexobj(gamma):
            raise ValueError("background_gamma must be real-valued.")
        gamma = np.ascontiguousarray(gamma, dtype=np.float64)
        if gamma.size == 1 and omega0.size != 1:
            gamma = np.full(omega0.size, gamma[0], dtype=np.float64)

    if del_eps.ndim != 1 or omega0.ndim != 1 or gamma.ndim != 1:
        raise ValueError("All background Lorentz parameters must be one-dimensional.")
    if del_eps.size == 0:
        raise ValueError("At least one background Lorentz oscillator is required.")
    if not (del_eps.size == omega0.size == gamma.size):
        raise ValueError(
            "background_del_eps, resonance values, and background_gamma "
            "must have the same length."
        )
    if not np.all(np.isfinite(del_eps)) or np.any(del_eps < 0.0):
        raise ValueError("background_del_eps must be finite and non-negative.")
    if not np.all(np.isfinite(omega0)) or np.any(omega0 <= 0.0):
        raise ValueError("background_omega0 must be finite and positive.")
    if not np.all(np.isfinite(gamma)) or np.any(gamma < 0.0):
        raise ValueError("background_gamma must be finite and non-negative.")

    if float(dt) * float(np.max(omega0)) > 0.3:
        warnings.warn(
            "The fastest background Lorentz pole has omega0*dt > 0.3. "
            "The update remains defined, but reduce dt to resolve the UV "
            "silica response accurately.",
            RuntimeWarning,
            stacklevel=3,
        )

    return del_eps, omega0, gamma


@dataclass(frozen=True, slots=True)
class PulseParams:
    E0: float
    f0: float
    t0: float
    tau: float
    phase: float = 0.0
    type: str = "gaussian"


@dataclass(frozen=True, slots=True)
class FDTDParams:
    """
    All simulation, geometry, PML and snapshot parameters
    required for a 1D FDTD run.
    """

    # -----------------
    # Grid / time domain
    # -----------------
    Nz: int
    Nt: int
    dz: float
    dt: float

    # -----------------
    # Geometry / probes
    # -----------------
    src_i: int
    probe_t_i: int
    probe_r_i: int
    metal_i0: int
    metal_i1: int

    # -----------------
    # PML
    # -----------------
    npml: int = 150
    R_target: float = 1e-12
    m: int = 4

    # -----------------
    # Snapshots
    # -----------------
    snap_every: int = 100  # 0 disables snapshots

    # -----------------
    # Validation
    # -----------------
    def __post_init__(self):
        for name in ("Nz", "Nt", "src_i", "probe_t_i", "probe_r_i",
                     "metal_i0", "metal_i1", "npml", "snap_every"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
        if self.Nz < 2 or self.Nt < 1:
            raise ValueError("Nz must be at least 2 and Nt must be positive.")
        if self.npml < 0 or self.snap_every < 0:
            raise ValueError("npml and snap_every must be non-negative.")
        if not (0 <= self.src_i < self.Nz):
            raise ValueError("src_i out of bounds.")

        if not (0 <= self.probe_t_i < self.Nz):
            raise ValueError("probe_t_i out of bounds.")

        if not (0 <= self.probe_r_i < self.Nz):
            raise ValueError("probe_r_i out of bounds.")

        if not (0 <= self.metal_i0 <= self.metal_i1 <= self.Nz):
            raise ValueError("Invalid metal region indices.")

        if self.npml * 2 >= self.Nz:
            raise ValueError("PML too thick relative to Nz.")

        if not np.isfinite(self.dz) or not np.isfinite(self.dt) or self.dz <= 0 or self.dt <= 0:
            raise ValueError("dz and dt must be positive.")


_MATERIAL_PARAMETER_NAMES = (
    "omega_D", "gamma_D", "omega_L", "gamma_L", "del_eps", "eps_inf",
)


@dataclass(frozen=True, slots=True)
class MaterialParamsTD:
    """
    Space- and time-dependent Drude–Lorentz parameters.

    Each parameter accepts (Nt,) for a spatially uniform time series,
    (Nt, 1) for the same broadcast, or (Nt, Nmaterial) for a local value
    at every time and material cell. Axis 1 runs from metal_i0 (inclusive)
    to metal_i1 (exclusive), not over the full simulation grid. Mixed
    uniform and spatial parameters are allowed. For a static spatial profile,
    use np.broadcast_to(profile[None, :], (Nt, profile.size)).

    Samples are at n*dt. The ADE kernels average coefficients between n and
    n+1, holding the final sample for the last update. Arrays are converted to
    real float64; wrappers check their dimensions against the simulation.
    The separate equilibrium SiO2 background poles remain fixed.
    """
    omega_D: np.ndarray
    gamma_D: np.ndarray
    omega_L: np.ndarray
    gamma_L: np.ndarray
    del_eps: np.ndarray
    eps_inf: np.ndarray

    def __post_init__(self):
        arrays = {}
        for name in _MATERIAL_PARAMETER_NAMES:
            value = np.asarray(getattr(self, name))
            if value.ndim not in (1, 2) or value.shape[0] == 0:
                raise ValueError(
                    f"{name} must have shape (Nt,), (Nt, 1), or (Nt, Nmaterial); "
                    f"received {value.shape}."
                )
            if np.iscomplexobj(value):
                raise ValueError(f"{name} must be real-valued.")
            value = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values.")
            if name == "eps_inf" and np.any(value <= 0):
                raise ValueError("eps_inf must be strictly positive.")
            arrays[name] = value
            # Preserve broadcast views, avoiding large copies of static profiles.
            object.__setattr__(self, name, value)
        if len({a.shape[0] for a in arrays.values()}) != 1:
            raise ValueError("All material arrays must have the same time dimension (Nt).")
        spatial_sizes = {a.shape[1] for a in arrays.values()
                         if a.ndim == 2 and a.shape[1] != 1}
        if len(spatial_sizes) > 1:
            raise ValueError("All spatial material arrays must have the same Nmaterial.")

    @classmethod
    def from_static(
        cls,
        *,
        Nt: int,
        omega_D: float,
        gamma_D: float,
        omega_L: float,
        gamma_L: float,
        del_eps: float,
        eps_inf: float,
        dtype=np.float64,
    ) -> "MaterialParamsTD":
        """
        Expand scalar parameters into constant arrays of length Nt.
        """
        return cls(
            omega_D=np.full(Nt, omega_D, dtype=dtype),
            gamma_D=np.full(Nt, gamma_D, dtype=dtype),
            omega_L=np.full(Nt, omega_L, dtype=dtype),
            gamma_L=np.full(Nt, gamma_L, dtype=dtype),
            del_eps=np.full(Nt, del_eps, dtype=dtype),
            eps_inf=np.full(Nt, eps_inf, dtype=dtype),
        )


def _prepare_material_arrays(Nt, nmaterial, values, *, spatial=True):
    """Validate before entering Numba; broadcast uniform arrays without copies."""
    material = MaterialParamsTD(**dict(zip(_MATERIAL_PARAMETER_NAMES, values)))
    prepared = []
    for name in _MATERIAL_PARAMETER_NAMES:
        a = getattr(material, name)
        if a.shape[0] != Nt:
            raise ValueError(f"{name}: expected time dimension Nt={Nt}, received {a.shape}.")
        if not spatial:
            if a.ndim != 1:
                raise ValueError(
                    "The convolution solver only supports (Nt,) material arrays; "
                    "use 'exact pml' or 'sio2_background' for spatial parameters."
                )
            prepared.append(np.ascontiguousarray(a))
            continue
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] not in (1, nmaterial):
            raise ValueError(
                f"{name}: expected shape ({Nt},), ({Nt}, 1), or "
                f"({Nt}, {nmaterial}); received {a.shape}."
            )
        prepared.append(np.broadcast_to(a, (Nt, nmaterial)))
    return tuple(prepared)


def _prepare_material_probes(metal_i0, metal_i1, every, indices):
    """Return global E-node indices, preserving the requested order."""
    if every is not None and indices is not None:
        raise ValueError("Supply material_probe_every or material_probe_indices, not both.")
    if every is not None:
        if isinstance(every, (bool, np.bool_)) or not isinstance(every, (int, np.integer)) or every < 1:
            raise ValueError("material_probe_every must be a positive integer.")
        return np.arange(metal_i0, metal_i1, every, dtype=np.int64)
    if indices is None:
        return np.empty(0, dtype=np.int64)
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError("material_probe_indices must be a one-dimensional integer array.")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if indices.dtype.kind not in "iu":
        raise ValueError("material_probe_indices must contain integers.")
    if np.any(indices < metal_i0) or np.any(indices >= metal_i1):
        raise ValueError("Material probes must lie in [metal_i0, metal_i1).")
    if np.unique(indices).size != indices.size:
        raise ValueError("material_probe_indices must not contain duplicates.")
    return np.ascontiguousarray(indices, dtype=np.int64)


def _prepare_ade_inputs(
    Nz, Nt, dz, dt, src_i, probe_t_i, probe_r_i, metal_i0, metal_i1,
    snap_every, values, sigmaE_by_eps0, bH, cH,
    material_probe_every, material_probe_indices,
):
    # The low-level public wrappers must be safe even without FDTDParams.
    FDTDParams(Nz, Nt, dz, dt, src_i, probe_t_i, probe_r_i,
               metal_i0, metal_i1, npml=0, snap_every=snap_every)
    arrays = _prepare_material_arrays(Nt, metal_i1 - metal_i0, values)
    coeffs = []
    for name, value, size in (("sigmaE_by_eps0", sigmaE_by_eps0, Nz),
                              ("bH", bH, Nz - 1), ("cH", cH, Nz - 1)):
        a = np.asarray(value)
        if a.shape != (size,) or np.iscomplexobj(a):
            raise ValueError(f"{name} must be a real array of shape ({size},).")
        a = np.ascontiguousarray(a, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must contain only finite values.")
        coeffs.append(a)
    indices = _prepare_material_probes(
        metal_i0, metal_i1, material_probe_every, material_probe_indices,
    )
    return arrays, coeffs, indices


def _format_ade_result(result, indices, dz, dt, enabled):
    if not enabled:
        return result[:4]
    return (*result[:4], {
        "indices": indices,
        "z": indices * dz,
        "t": np.arange(result[0].size, dtype=np.float64) * dt,
        "E": result[4],
    })


@njit
def gaussian_sine(t, E0, f0, t0, tau, phase=0.0):
    return E0 * np.cos(2*np.pi*f0*(t-t0) + phase) * np.exp(-0.5*((t - t0)/tau)**2)


@njit
def boxed_pulse(t, E0, f0, t0, tau, phase=0.0):

    check = (t < (t0 + tau)) & (t >= (t0 - tau))

    pulse = 0.0
    if check:
        pulse = E0 * np.cos(2*np.pi*f0*(t-t0) + phase)

    return pulse


@njit
def smooth_boxed_pulse(t, E0, f0, t0, tau, phase=0.0):
    tramp = 0.1 * 2.0*tau
    x = np.abs(t - t0)

    # outside pulse support
    if x > tau:
        return 0.0

    # fully flat central region
    if x <= (tau - tramp):
        env = 1.0

    # raised-cosine edge
    else:
        s = (x - (tau - tramp)) / tramp   # goes 0 -> 1 across edge
        env = 0.5 * (1.0 + np.cos(np.pi * s))

    result = E0 * env * np.cos(2.0 * np.pi * f0 * (t - t0) + phase)

    return result


def build_pml_1d(
    Nz: int,
    dz: float,
    dt: float,
    npml: int = 250,
    R_target: float = 1e-16,
    m: int = 4,
):
    """
    Build 1D PML profiles for E nodes (Nz) and H edges (Nz-1),
    returning multiplicative update coefficients for standard Yee updates.

    Parameters
    ----------
    Nz : int
        Number of E nodes.
    dz : float
        Spatial step.
    dt : float
        Time step.
    npml : int
        PML thickness in cells (applied on both sides).
    R_target : float
        Target reflection (smaller -> stronger PML).
    m : int
        Polynomial grading order (3 or 4 is typical).

    Returns
    -------
    bE, cE : arrays length Nz
        E update coefficients: E = bE*E - cE*(H[i]-H[i-1]) - (dt/eps0)*J (if present)
    bH, cH : arrays length Nz-1
        H update coefficients: H = bH*H - cH*(E[i+1]-E[i])
    sigmaE : array length Nz
        Electric conductivity profile (for diagnostics).
    """
    if npml < 1:
        # No PML
        bE = np.ones(Nz)
        cE = np.full(Nz, dt/dz)
        bH = np.ones(Nz-1)
        cH = np.full(Nz-1, C0*C0*dt/dz)
        sigmaE_by_eps0 = np.zeros(Nz)
        return bE, cE, bH, cH, sigmaE_by_eps0

    # PML physical thickness, in code units
    Lpml = npml * dz

    # Common choice for sigma_max (widely used in FDTD texts)
    # sigma_max ≈ -(m+1) * eps0 * C0 * ln(R) / (2*Lpml)
    sigma_max_by_eps0 = -(m + 1) * C0 * np.log(R_target) / (2 * Lpml) # unit = 1 / UNIT_T

    # sigmaE at E nodes
    sigmaE_by_eps0 = np.zeros(Nz, dtype=np.float64)

    # Left PML: i = 0..npml-1
    for i in range(npml):
        x = (npml - i) / npml  # 1 at boundary, 0 at interface
        sigmaE_by_eps0[i] = sigma_max_by_eps0 * (x ** m)

    # Right PML: i = Nz-npml .. Nz-1
    for i in range(Nz - npml, Nz):
        x = (i - (Nz - npml - 1)) / npml  # 0 at interface, 1 at boundary
        sigmaE_by_eps0[i] = sigma_max_by_eps0 * (x ** m)

    # Match impedances: sigma_m = (mu/eps) * sigma_e = (mu0/eps0) * sigma_e => sigma_h_by_mu0 = sigma_e_by_eps0
    # We need sigma for H cells (edges). Use averaged sigmaE between neighboring E nodes.
    sigmaH_by_mu0 = np.zeros(Nz-1, dtype=np.float64)
    for i in range(Nz-1):
        sigmaH_by_mu0[i] = 0.5 * (sigmaE_by_eps0[i] + sigmaE_by_eps0[i+1])

    # Precompute update coefficients:
    # For E:  (1 + sigmaE*dt/(2eps0)) E^{n+1} = (1 - sigmaE*dt/(2eps0)) E^{n} - (dt/(eps0 dz)) (H-H)
    # => E^{n+1} = bE*E^n - cE*(curlH)
    aE = 1.0 + sigmaE_by_eps0 * dt / 2.0
    bE = (1.0 - sigmaE_by_eps0 * dt / 2.0) / aE
    cE = (dt / dz) / aE

    # For H: (1 + sigmaH*dt/(2mu0)) H^{n+1/2} = (1 - sigmaH*dt/(2mu0)) H^{n-1/2} - (dt/(mu0 dz))(E diff)
    # for my convention: (1 + sigmaH*dt/(2mu0)) H_by_eps0^{n+1/2} = (1 - sigmaH*dt/(2mu0)) H_by_eps0^{n-1/2} - (dt/(mu0 eps0 dz))(E diff)
    aH = 1.0 + sigmaH_by_mu0 * dt / 2.0
    bH = (1.0 - sigmaH_by_mu0 * dt / 2.0) / aH
    cH = ( C0*C0* dt / dz) / aH

    return bE, cE, bH, cH, sigmaE_by_eps0


@njit(parallel=True)
def _run_fdtd_convolution_kernel(
    Nz, Nt, dz, dt,
    src_i,                       # source cell index
    probe_t_i,                   # transmitted probe index
    probe_r_i,                   # reflected probe index
    metal_i0, metal_i1,          # metal region [i0, i1)
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,   # model params, now time-dependent arrays
    E0, f0, t0, tau, phase, pulsetype,      # source waveform params
    bE, cE, bH, cH,              # PML coefficients
    snap_every,
    explicit_pulse_samples,
):
    # Fields
    E = np.zeros(Nz, dtype=np.float64)
    H_by_eps0 = np.zeros(Nz-1, dtype=np.float64)  # between E nodes
    psi_D = np.zeros(Nz,dtype=np.complex64)
    psi_L = np.zeros(Nz,dtype=np.complex64)

    # Probes
    E_t = np.zeros(Nt, dtype=np.float64)
    E_r = np.zeros(Nt, dtype=np.float64)

    # Snapshot storage (preallocate)
    if snap_every > 0:
        nsnaps = (Nt + snap_every - 1) // snap_every  # ceil(Nt/snap_every)
        snaps_E = np.zeros((nsnaps, Nz), dtype=np.float64)
        snaps_t = np.zeros(nsnaps, dtype=np.float64)
    else:
        nsnaps = 0
        snaps_E = np.zeros((1, 1), dtype=np.float64)
        snaps_t = np.zeros(1, dtype=np.float64)
    snap_k = 0


    for n in range(Nt):
        t = n * dt

        # define time-dependent coefficients, 
        # following the recursive accumulator approach in Vial et al., 2005
        # Drude parameters
        delta_eps = -1.0 * (omega_D[n] * omega_D[n]) /  (gamma_D[n] * gamma_D[n])
        chi_0_D = delta_eps * (1.0 - np.exp(-1.0*gamma_D[n] * dt))
        delta_chi_0_D = delta_eps * (1.0 - np.exp(-1.0*gamma_D[n] * dt))**2.0
        sigma_D = omega_D[n]*omega_D[n] / gamma_D[n]
        C_rho_D = np.exp(-1.0*gamma_D[n]*dt)
        C_delta_D = delta_chi_0_D
        # Lorentz parameters
        alpha = gamma_L[n] / 2.0
        beta = np.sqrt(omega_L[n]*omega_L[n] - alpha*alpha)
        gamma = del_eps[n] * omega_L[n]*omega_L[n]/beta
        chi_0_L = -1.0j*gamma/(alpha - 1.0j*beta) * (1.0 - np.exp((-alpha + 1.0j*beta)*dt))
        delta_chi_0_L = -1.0j*gamma/(alpha - 1.0j*beta) * (1.0 - np.exp((-alpha + 1.0j*beta)*dt))**2.0
        C_rho_L = np.exp((-alpha + 1.0j*beta)*dt)
        C_delta_L = delta_chi_0_L
        # overall update coefficients
        chi_0 = chi_0_D + np.real(chi_0_L)
        C_alpha = eps_inf[n] / (eps_inf[n] + chi_0 + sigma_D*dt)
        C_beta = dt / (dz*(eps_inf[n] + chi_0 + sigma_D*dt))
        C_gamma = 1.0 / (eps_inf[n] + chi_0 + sigma_D*dt)

        # --- Update H (n+1/2) from E(n), with PML coefficients
        for i in prange(Nz-1):
            H_by_eps0[i] = bH[i] * H_by_eps0[i] - cH[i] * (E[i+1] - E[i])

        # Interior update
        for i in prange(1, Nz-1):
            curlH = - H_by_eps0[i] + H_by_eps0[i-1]

            # It is assumed that the metal is not near the boundary, where the PML is active
            if metal_i0 <= i < metal_i1:
                psi_D[i] = C_rho_D * psi_D[i] + C_delta_D * E[i]
                psi_L[i] = C_rho_L * psi_L[i] + C_delta_L * E[i]
                E[i] = C_alpha * E[i] + C_beta * curlH + C_gamma * np.real(psi_D[i] + psi_L[i])
            else:
                E[i] = bE[i] * E[i] + cE[i] * curlH

        # --- Source injection (soft source)
        if pulsetype == "gaussian":
            E[src_i] += gaussian_sine(t, E0, f0, t0, tau, phase)
        elif pulsetype == "boxed" or pulsetype=="box":
            E[src_i] += boxed_pulse(t, E0, f0, t0, tau, phase)
        elif pulsetype =="smoothed box":
            E[src_i] += smooth_boxed_pulse(t, E0, f0, t0, tau, phase)
        elif pulsetype == "explicit":
            E[src_i] += explicit_pulse_samples[n]
        else:
            raise ValueError("invalid pulse name")

        # For completeness, update boundaries E[0], E[Nz-1] similarly using one-sided curl.
        # In PML this is usually fine; use nearest curl value.
        E[0] = bE[0] * E[0] - cE[0] * (H_by_eps0[0] - 0.0)
        E[Nz-1] = bE[Nz-1] * E[Nz-1] - cE[Nz-1] * (0.0 - H_by_eps0[Nz-2])

        # --- Record probes
        E_t[n] = E[probe_t_i]
        E_r[n] = E[probe_r_i]

        # --- Snapshot
        if snap_every > 0 and (n % snap_every == 0):
            snaps_t[snap_k] = t
            # copy E into snapshot row
            for i in range(Nz):
                snaps_E[snap_k, i] = E[i]
            snap_k += 1

    return E_t, E_r, snaps_t, snaps_E


@njit(parallel=True)
def _run_fdtd_pml_corrected_kernel(
    Nz, Nt, dz, dt,
    src_i,
    probe_t_i,
    probe_r_i,
    metal_i0, metal_i1,
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
    E0, f0, t0, tau, phase, pulsetype,
    sigmaE_by_eps0,            # NEW
    bH, cH,                    # only H PML coeffs are needed explicitly
    snap_every,
    explicit_pulse_samples,
    material_probe_indices,
):
    """
    1D Yee FDTD with conductivity-graded absorber overlapping a genuinely
    space- and time-dependent Drude-Lorentz medium.

    In cells where the material overlaps the absorber, the E-update uses:

        [eps_inf^{n+1} E^{n+1} - eps_inf^n E^n]/dt
        + sigma_E * (E^{n+1}+E^n)/2
        + J_D^{n+1/2} + J_L^{n+1/2}
        = curlH / dz
    """

    # -------------------------
    # Fields / material states
    # -------------------------
    E = np.zeros(Nz, dtype=np.float64)
    E_old = np.zeros(Nz, dtype=np.float64)

    H_by_eps0 = np.zeros(Nz - 1, dtype=np.float64)

    J_D = np.zeros(Nz, dtype=np.float64)
    P_L = np.zeros(Nz, dtype=np.float64)
    J_L = np.zeros(Nz, dtype=np.float64)

    # -------------------------
    # Probes
    # -------------------------
    E_t = np.zeros(Nt, dtype=np.float64)
    E_r = np.zeros(Nt, dtype=np.float64)
    E_material = np.empty((Nt, material_probe_indices.size), dtype=np.float64)

    # -------------------------
    # Snapshots
    # -------------------------
    if snap_every > 0:
        nsnaps = (Nt + snap_every - 1) // snap_every
        snaps_E = np.zeros((nsnaps, Nz), dtype=np.float64)
        snaps_t = np.zeros(nsnaps, dtype=np.float64)
    else:
        snaps_E = np.zeros((1, 1), dtype=np.float64)
        snaps_t = np.zeros(1, dtype=np.float64)

    snap_k = 0

    # -------------------------
    # Time stepping
    # -------------------------
    for n in range(Nt):
        t = n * dt

        # Save old E
        for i in prange(Nz):
            E_old[i] = E[i]

        # -------------------------
        # Update H^{n+1/2}
        # -------------------------
        for i in prange(Nz - 1):
            H_by_eps0[i] = bH[i] * H_by_eps0[i] - cH[i] * (E_old[i + 1] - E_old[i])

        # Time-slab index
        np1 = n + 1
        if np1 >= Nt:
            np1 = n

        # -------------------------
        # Update E^{n+1} on ALL cells
        # -------------------------
        for i in prange(Nz):

            # One-sided curl at the boundaries, centered in the interior
            if i == 0:
                curlH = -H_by_eps0[0]
            elif i == Nz - 1:
                curlH = H_by_eps0[Nz - 2]
            else:
                curlH = -H_by_eps0[i] + H_by_eps0[i - 1]

            sigma = sigmaE_by_eps0[i]

            # Material region, including overlap with absorber
            if metal_i0 <= i < metal_i1:
                k = i - metal_i0
                # Local coefficients on the n -> n+1 time slab.
                gD = 0.5 * (gamma_D[n, k] + gamma_D[np1, k])
                wD2 = 0.5 * (
                    omega_D[n, k] * omega_D[n, k] +
                    omega_D[np1, k] * omega_D[np1, k]
                )
        
                gL = 0.5 * (gamma_L[n, k] + gamma_L[np1, k])
                wL2 = 0.5 * (
                    omega_L[n, k] * omega_L[n, k] +
                    omega_L[np1, k] * omega_L[np1, k]
                )
        
                sL = 0.5 * (
                    del_eps[n, k] * omega_L[n, k] * omega_L[n, k] +
                    del_eps[np1, k] * omega_L[np1, k] * omega_L[np1, k]
                )
        
                eps_n = eps_inf[n, k]
                eps_np1 = eps_inf[np1, k]

                # ---- Drude current update
                denom_D = 1.0 + 0.5 * gD * dt
                JD_np12 = (
                    (1.0 - 0.5 * gD * dt) * J_D[i]
                    + dt * wD2 * E_old[i]
                ) / denom_D

                # ---- Lorentz current / polarization update
                denom_L = 1.0 + 0.5 * gL * dt + 0.5 * wL2 * dt * dt
                JL_np12 = (
                    (1.0 - 0.5 * gL * dt) * J_L[i]
                    - dt * wL2 * P_L[i]
                    + dt * sL * E_old[i]
                ) / denom_L
                PL_np1 = P_L[i] + dt * JL_np12

                # ---- Ampere update with absorber + material active together
                denom_E = eps_np1 + 0.5 * sigma * dt
                numer_E = (
                    (eps_n - 0.5 * sigma * dt) * E_old[i]
                    + (dt / dz) * curlH
                    - dt * (JD_np12 + JL_np12)
                )
                E[i] = numer_E / denom_E

                # Commit material states
                J_D[i] = JD_np12
                J_L[i] = JL_np12
                P_L[i] = PL_np1

            else:
                # Vacuum / non-dispersive region, including absorber overlap
                denom_E = 1.0 + 0.5 * sigma * dt
                numer_E = (
                    (1.0 - 0.5 * sigma * dt) * E_old[i]
                    + (dt / dz) * curlH
                )
                E[i] = numer_E / denom_E

        # -------------------------
        # Soft source injection
        # -------------------------
        if pulsetype == "gaussian":
            E[src_i] += gaussian_sine(t, E0, f0, t0, tau, phase)
        elif pulsetype == "boxed" or pulsetype == "box":
            E[src_i] += boxed_pulse(t, E0, f0, t0, tau, phase)
        elif pulsetype == "smoothed box":
            E[src_i] += smooth_boxed_pulse(t, E0, f0, t0, tau, phase)
        elif pulsetype == "explicit":
            E[src_i] += explicit_pulse_samples[n]
        else:
            raise ValueError("invalid pulse name")

        # -------------------------
        # Record probes
        # -------------------------
        E_t[n] = E[probe_t_i]
        E_r[n] = E[probe_r_i]
        for j in range(material_probe_indices.size):
            E_material[n, j] = E[material_probe_indices[j]]

        # -------------------------
        # Snapshots
        # -------------------------
        if snap_every > 0 and (n % snap_every == 0):
            snaps_t[snap_k] = t
            for i in range(Nz):
                snaps_E[snap_k, i] = E[i]
            snap_k += 1

    return E_t, E_r, snaps_t, snaps_E, E_material


@njit(parallel=True)
def _run_fdtd_sio2_background_kernel(
    Nz, Nt, dz, dt,
    src_i,
    probe_t_i,
    probe_r_i,
    metal_i0, metal_i1,
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
    E0, f0, t0, tau, phase, pulsetype,
    sigmaE_by_eps0,
    bH, cH,
    snap_every,
    explicit_pulse_samples,
    material_probe_indices,
    background_del_eps,
    background_omega0,
    background_gamma,
):
    """Corrected-PML FDTD plus independent fixed background Lorentz poles.

    The user-supplied space- and time-dependent Drude/Lorentz states are advanced exactly
    as in ``_run_fdtd_pml_corrected_kernel``.  The fixed background uses a
    separate polarization/current pair for every oscillator and contributes
    only its summed current to Ampere's law.
    """

    # -------------------------
    # Fields / material states
    # -------------------------
    E = np.zeros(Nz, dtype=np.float64)
    E_old = np.zeros(Nz, dtype=np.float64)

    H_by_eps0 = np.zeros(Nz - 1, dtype=np.float64)

    # User-supplied, space- and time-dependent material states.
    J_D = np.zeros(Nz, dtype=np.float64)
    P_L = np.zeros(Nz, dtype=np.float64)
    J_L = np.zeros(Nz, dtype=np.float64)

    # Fixed equilibrium-background material states.  The oscillator index is
    # the contiguous dimension because it is the inner loop below.
    n_background = background_omega0.size
    P_background = np.zeros((Nz, n_background), dtype=np.float64)
    J_background = np.zeros((Nz, n_background), dtype=np.float64)

    background_omega02 = background_omega0 * background_omega0
    background_strength = background_del_eps * background_omega02
    background_denom = (
        1.0
        + 0.5 * background_gamma * dt
        + 0.5 * background_omega02 * dt * dt
    )
    background_decay = 1.0 - 0.5 * background_gamma * dt

    # -------------------------
    # Probes
    # -------------------------
    E_t = np.zeros(Nt, dtype=np.float64)
    E_r = np.zeros(Nt, dtype=np.float64)
    E_material = np.empty((Nt, material_probe_indices.size), dtype=np.float64)

    # -------------------------
    # Snapshots
    # -------------------------
    if snap_every > 0:
        nsnaps = (Nt + snap_every - 1) // snap_every
        snaps_E = np.zeros((nsnaps, Nz), dtype=np.float64)
        snaps_t = np.zeros(nsnaps, dtype=np.float64)
    else:
        snaps_E = np.zeros((1, 1), dtype=np.float64)
        snaps_t = np.zeros(1, dtype=np.float64)

    snap_k = 0

    # -------------------------
    # Time stepping
    # -------------------------
    for n in range(Nt):
        t = n * dt

        # Save old E
        for i in prange(Nz):
            E_old[i] = E[i]

        # -------------------------
        # Update H^{n+1/2}
        # -------------------------
        for i in prange(Nz - 1):
            H_by_eps0[i] = (
                bH[i] * H_by_eps0[i]
                - cH[i] * (E_old[i + 1] - E_old[i])
            )

        # Time-slab index
        np1 = n + 1
        if np1 >= Nt:
            np1 = n

        # -------------------------
        # Update E^{n+1} on ALL cells
        # -------------------------
        for i in prange(Nz):

            # One-sided curl at the boundaries, centered in the interior
            if i == 0:
                curlH = -H_by_eps0[0]
            elif i == Nz - 1:
                curlH = H_by_eps0[Nz - 2]
            else:
                curlH = -H_by_eps0[i] + H_by_eps0[i - 1]

            sigma = sigmaE_by_eps0[i]

            # Material region, including overlap with absorber
            if metal_i0 <= i < metal_i1:
                k = i - metal_i0
                # Local coefficients on the n -> n+1 time slab.
                gD = 0.5 * (gamma_D[n, k] + gamma_D[np1, k])
                wD2 = 0.5 * (
                    omega_D[n, k] * omega_D[n, k]
                    + omega_D[np1, k] * omega_D[np1, k]
                )
        
                gL = 0.5 * (gamma_L[n, k] + gamma_L[np1, k])
                wL2 = 0.5 * (
                    omega_L[n, k] * omega_L[n, k]
                    + omega_L[np1, k] * omega_L[np1, k]
                )
        
                sL = 0.5 * (
                    del_eps[n, k] * omega_L[n, k] * omega_L[n, k]
                    + del_eps[np1, k] * omega_L[np1, k] * omega_L[np1, k]
                )
        
                eps_n = eps_inf[n, k]
                eps_np1 = eps_inf[np1, k]

                # ---- User Drude current update
                denom_D = 1.0 + 0.5 * gD * dt
                JD_np12 = (
                    (1.0 - 0.5 * gD * dt) * J_D[i]
                    + dt * wD2 * E_old[i]
                ) / denom_D

                # ---- User Lorentz current / polarization update
                denom_L = 1.0 + 0.5 * gL * dt + 0.5 * wL2 * dt * dt
                JL_np12 = (
                    (1.0 - 0.5 * gL * dt) * J_L[i]
                    - dt * wL2 * P_L[i]
                    + dt * sL * E_old[i]
                ) / denom_L
                PL_np1 = P_L[i] + dt * JL_np12

                # ---- Fixed equilibrium-background Lorentz oscillators
                J_background_total = 0.0
                for j in range(n_background):
                    JB_np12 = (
                        background_decay[j] * J_background[i, j]
                        - dt * background_omega02[j] * P_background[i, j]
                        + dt * background_strength[j] * E_old[i]
                    ) / background_denom[j]
                    PB_np1 = P_background[i, j] + dt * JB_np12

                    J_background[i, j] = JB_np12
                    P_background[i, j] = PB_np1
                    J_background_total += JB_np12

                # ---- Ampere update with the original absorber and user
                # material terms plus the independent background current.
                denom_E = eps_np1 + 0.5 * sigma * dt
                numer_E = (
                    (eps_n - 0.5 * sigma * dt) * E_old[i]
                    + (dt / dz) * curlH
                    - dt * (JD_np12 + JL_np12 + J_background_total)
                )
                E[i] = numer_E / denom_E

                # Commit user material states
                J_D[i] = JD_np12
                J_L[i] = JL_np12
                P_L[i] = PL_np1

            else:
                # Vacuum / non-dispersive region, including absorber overlap
                denom_E = 1.0 + 0.5 * sigma * dt
                numer_E = (
                    (1.0 - 0.5 * sigma * dt) * E_old[i]
                    + (dt / dz) * curlH
                )
                E[i] = numer_E / denom_E

        # -------------------------
        # Soft source injection
        # -------------------------
        if pulsetype == "gaussian":
            E[src_i] += gaussian_sine(t, E0, f0, t0, tau, phase)
        elif pulsetype == "boxed" or pulsetype == "box":
            E[src_i] += boxed_pulse(t, E0, f0, t0, tau, phase)
        elif pulsetype == "smoothed box":
            E[src_i] += smooth_boxed_pulse(t, E0, f0, t0, tau, phase)
        elif pulsetype == "explicit":
            E[src_i] += explicit_pulse_samples[n]
        else:
            raise ValueError("invalid pulse name")

        # -------------------------
        # Record probes
        # -------------------------
        E_t[n] = E[probe_t_i]
        E_r[n] = E[probe_r_i]
        for j in range(material_probe_indices.size):
            E_material[n, j] = E[material_probe_indices[j]]

        # -------------------------
        # Snapshots
        # -------------------------
        if snap_every > 0 and (n % snap_every == 0):
            snaps_t[snap_k] = t
            for i in range(Nz):
                snaps_E[snap_k, i] = E[i]
            snap_k += 1

    return E_t, E_r, snaps_t, snaps_E, E_material


def run_fdtd_convolution(
    Nz, Nt, dz, dt,
    src_i,
    probe_t_i,
    probe_r_i,
    metal_i0, metal_i1,
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
    E0, f0, t0, tau, phase, pulsetype,
    bE, cE, bH, cH,
    snap_every=20,
    explicit_pulse_waveform=None,
):
    """Run the recursive-convolution solver with an optional explicit source."""
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf = _prepare_material_arrays(
        Nt, metal_i1 - metal_i0,
        (omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf), spatial=False,
    )
    explicit_pulse_samples = _prepare_explicit_pulse_waveform(
        explicit_pulse_waveform,
        pulsetype=pulsetype,
        Nt=Nt,
        dt=dt,
    )
    return _run_fdtd_convolution_kernel(
        Nz, Nt, dz, dt,
        src_i, probe_t_i, probe_r_i,
        metal_i0, metal_i1,
        omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
        E0, f0, t0, tau, phase, pulsetype,
        bE, cE, bH, cH,
        snap_every,
        explicit_pulse_samples,
    )


def run_fdtd_pml_corrected(
    Nz, Nt, dz, dt,
    src_i,
    probe_t_i,
    probe_r_i,
    metal_i0, metal_i1,
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
    E0, f0, t0, tau, phase, pulsetype,
    sigmaE_by_eps0,
    bH, cH,
    snap_every=200,
    explicit_pulse_waveform=None,
    *,
    material_probe_every=None,
    material_probe_indices=None,
):
    """Run the corrected-PML ADE solver with local material parameters.

    Material arrays have shape (Nt,), (Nt, 1), or (Nt, metal_i1-metal_i0).
    Use material_probe_every=1 to record every material cell, a larger positive
    integer for a stride, or material_probe_indices for explicit global indices.
    The two probe options are mutually exclusive; None disables recording.

    By default returns (E_t, E_r, snaps_t, snaps_E). When probes are requested,
    appends a dict with 'indices', 'z' (Angstrom), 't' (ps), and 'E' with shape
    (Nt, Nprobes). All traces are recorded after the E update and source
    injection, with the existing source-sample time labels t[n] = n*dt.
    Probe storage is independent of snap_every and costs 8*Nt*Nprobes bytes.
    """
    arrays, coeffs, indices = _prepare_ade_inputs(
        Nz, Nt, dz, dt, src_i, probe_t_i, probe_r_i, metal_i0, metal_i1,
        snap_every, (omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf),
        sigmaE_by_eps0, bH, cH, material_probe_every, material_probe_indices,
    )
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf = arrays
    sigmaE_by_eps0, bH, cH = coeffs
    explicit_pulse_samples = _prepare_explicit_pulse_waveform(
        explicit_pulse_waveform,
        pulsetype=pulsetype,
        Nt=Nt,
        dt=dt,
    )
    result = _run_fdtd_pml_corrected_kernel(
        Nz, Nt, dz, dt,
        src_i, probe_t_i, probe_r_i,
        metal_i0, metal_i1,
        omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
        E0, f0, t0, tau, phase, pulsetype,
        sigmaE_by_eps0,
        bH, cH,
        snap_every,
        explicit_pulse_samples,
        indices,
    )

    return _format_ade_result(
        result, indices, dz, dt,
        material_probe_every is not None or material_probe_indices is not None,
    )


def run_fdtd_sio2_background(
    Nz, Nt, dz, dt,
    src_i,
    probe_t_i,
    probe_r_i,
    metal_i0, metal_i1,
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
    E0, f0, t0, tau, phase, pulsetype,
    sigmaE_by_eps0,
    bH, cH,
    snap_every=200,
    explicit_pulse_waveform=None,
    background_del_eps=None,
    background_omega0=None,
    background_lambda0_um=None,
    background_gamma=None,
    *,
    material_probe_every=None,
    material_probe_indices=None,
):
    """Run corrected-PML FDTD with a fixed fused-silica background.

    The default background is the three-oscillator, lossless Malitson fit:

    ``Delta epsilon = [0.6961663, 0.4079426, 0.8974794]``

    ``lambda0 [um] = [0.0684043, 0.1162414, 9.896161]``

    The fitted background ``epsilon_inf`` is one and is not added separately;
    the caller's ``eps_inf`` array remains the instantaneous term in Ampere's
    law.  To avoid double counting, use ``eps_inf=1`` and do not include these
    equilibrium silica poles again in the user-supplied transient component.

    User Drude-Lorentz arrays and optional material probes follow
    run_fdtd_pml_corrected: (Nt,) or (Nt, 1) broadcasts spatially, and
    (Nt, metal_i1-metal_i0) supplies local coefficients. Request probes with
    material_probe_every (1 for every cell) or material_probe_indices; this
    appends a fifth return value with indices, z, t, and E (Nt, Nprobes).
    The background_* parameters remain fixed one-dimensional pole arrays.

    Custom resonances may be supplied either as ``background_omega0`` in the
    solver's inverse-time units (rad/ps for the standard units) or as
    ``background_lambda0_um``.  Do not supply both.
    """
    arrays, coeffs, indices = _prepare_ade_inputs(
        Nz, Nt, dz, dt, src_i, probe_t_i, probe_r_i, metal_i0, metal_i1,
        snap_every, (omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf),
        sigmaE_by_eps0, bH, cH, material_probe_every, material_probe_indices,
    )
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf = arrays
    sigmaE_by_eps0, bH, cH = coeffs
    explicit_pulse_samples = _prepare_explicit_pulse_waveform(
        explicit_pulse_waveform,
        pulsetype=pulsetype,
        Nt=Nt,
        dt=dt,
    )
    background_del_eps, background_omega0, background_gamma = (
        _prepare_sio2_background_parameters(
            dt=dt,
            background_del_eps=background_del_eps,
            background_omega0=background_omega0,
            background_lambda0_um=background_lambda0_um,
            background_gamma=background_gamma,
        )
    )

    result = _run_fdtd_sio2_background_kernel(
        Nz, Nt, dz, dt,
        src_i, probe_t_i, probe_r_i,
        metal_i0, metal_i1,
        omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
        E0, f0, t0, tau, phase, pulsetype,
        sigmaE_by_eps0,
        bH, cH,
        snap_every,
        explicit_pulse_samples,
        indices,
        background_del_eps,
        background_omega0,
        background_gamma,
    )

    return _format_ade_result(
        result, indices, dz, dt,
        material_probe_every is not None or material_probe_indices is not None,
    )


def check_fdtd_pulse_transition_consistency(
    *,
    params,
    pulse,
    material,
    # src_i,
    # probe_r_i,
    # metal_i0,
    # dz,
    # transition_center,
    # transition_time,
    # f0,
    transition_half_width_factor=1.0,
    min_guard_cycles=5,
    verbose=True,
):
    """
    Consistency checks for a 1D FDTD reflection simulation with a smooth boxed pulse
    and a step-like Drude-parameter transition.

    Parameters
    ----------
    src_i, probe_r_i, metal_i0 : int
        Source, reflection-probe, and metal-interface indices.

    dz : float
        Grid spacing in Angstrom.

    C0 : float
        Speed of light in simulation units, usually Angstrom / ps.

    pulse : PulseParams
        Pulse object with fields t0 and tau in ps.

    transition_center : float
        Center time of Drude transition in ps.

    transition_time : float
        Transition time scale in ps. For your tanh profile this is the `deltat`
        parameter in tanh((t - t0) / deltat).

    f0 : float
        Carrier frequency in THz. Since 1 THz = 1 / ps, the carrier period is 1/f0 ps.

    transition_half_width_factor : float
        Defines the effective transition window as:
            transition_center +/- transition_half_width_factor * transition_time

        Use 1.0 if you want the same window as your fitting MSE calculation.
        Use 2.0 for a stricter tanh-transition window.

    min_guard_cycles : int
        Number of optical cycles desired on both sides of the transition window
        inside the flat part of the pulse.

    verbose : bool
        If True, prints diagnostic information and warnings.

    Returns
    -------
    checks : dict
        Dictionary containing derived timing windows and Boolean pass/fail flags.
    """

    warnings_list = []

    # -------------------------------------------------------------------------
    # Pulse timing at the source
    # -------------------------------------------------------------------------
    tau = pulse.tau
    t0_src = pulse.t0

    pulse_support_duration = 2.0 * tau

    tramp = 0.1 * 2.0 * tau
    flat_half_width = tau - tramp
    flat_duration = 2.0 * flat_half_width

    # -------------------------------------------------------------------------
    # Propagation time from source to metal
    # -------------------------------------------------------------------------
    src_to_metal_dist = (metal_i0 - src_i) * dz
    src_to_metal_time = src_to_metal_dist / C0

    pulse_center_at_metal = t0_src + src_to_metal_time

    pulse_start_at_metal = pulse_center_at_metal - tau
    pulse_end_at_metal = pulse_center_at_metal + tau

    flat_start_at_metal = pulse_center_at_metal - flat_half_width
    flat_end_at_metal = pulse_center_at_metal + flat_half_width

    # -------------------------------------------------------------------------
    # Effective Drude-transition window
    # -------------------------------------------------------------------------
    transition_start = transition_center - transition_half_width_factor * transition_time
    transition_end = transition_center + transition_half_width_factor * transition_time
    transition_effective_duration = transition_end - transition_start

    optical_period = 1.0 / f0
    guard_time = min_guard_cycles * optical_period

    guarded_transition_start = transition_start - guard_time
    guarded_transition_end = transition_end + guard_time

    # -------------------------------------------------------------------------
    # Check 1: pulse must be sufficiently long
    # -------------------------------------------------------------------------
    required_flat_duration = transition_effective_duration + 2.0 * guard_time

    pulse_long_enough = flat_duration >= required_flat_duration

    if not pulse_long_enough:
        warnings_list.append(
            "Pulse flat-top is too short for the requested transition window. "
            f"Flat-top duration = {flat_duration * 1e3:.3f} fs, "
            f"required duration = {required_flat_duration * 1e3:.3f} fs. "
            "Increase tau, reduce the transition-time window, or reduce min_guard_cycles."
        )

    # -------------------------------------------------------------------------
    # Check 2: transition should happen while pulse is reflecting from metal
    # -------------------------------------------------------------------------
    transition_inside_pulse_support = (
        transition_start >= pulse_start_at_metal
        and transition_end <= pulse_end_at_metal
    )

    if not transition_inside_pulse_support:
        warnings_list.append(
            "The Drude transition does not fully occur while the pulse is incident "
            "on the metal surface. "
            f"Transition window = [{transition_start * 1e3:.3f}, "
            f"{transition_end * 1e3:.3f}] fs, "
            f"pulse-at-metal support = [{pulse_start_at_metal * 1e3:.3f}, "
            f"{pulse_end_at_metal * 1e3:.3f}] fs."
        )

    transition_inside_flat_top = (
        guarded_transition_start >= flat_start_at_metal
        and guarded_transition_end <= flat_end_at_metal
    )

    if not transition_inside_flat_top:
        warnings_list.append(
            "The guarded Drude-transition window is not fully inside the flat-top "
            "of the pulse at the metal. "
            f"Guarded transition window = [{guarded_transition_start * 1e3:.3f}, "
            f"{guarded_transition_end * 1e3:.3f}] fs, "
            f"flat-top-at-metal window = [{flat_start_at_metal * 1e3:.3f}, "
            f"{flat_end_at_metal * 1e3:.3f}] fs. "
            "This means the inferred response may mix true material dynamics with "
            "pulse-envelope effects."
        )

    # -------------------------------------------------------------------------
    # Check 3: reflected and incident pulses should not overlap at reflection probe
    # -------------------------------------------------------------------------
    probe_to_metal_dist = (metal_i0 - probe_r_i) * dz

    min_probe_to_metal_dist = 0.5 * pulse_support_duration * C0

    probe_separation_ok = probe_to_metal_dist > min_probe_to_metal_dist

    if not probe_separation_ok:
        warnings_list.append(
            "Reflection probe is too close to the metal surface. Incident and "
            "reflected pulses may overlap at the probe. "
            f"Probe-metal distance = {probe_to_metal_dist:.3e} A, "
            f"minimum recommended distance = {min_probe_to_metal_dist:.3e} A. "
            "Move probe_r farther left from the metal interface or shorten the pulse."
        )

    checks = {
        "pulse_support_duration_ps": pulse_support_duration,
        "pulse_flat_duration_ps": flat_duration,
        "optical_period_ps": optical_period,
        "guard_time_ps": guard_time,
        "required_flat_duration_ps": required_flat_duration,
        "pulse_center_at_metal_ps": pulse_center_at_metal,
        "pulse_start_at_metal_ps": pulse_start_at_metal,
        "pulse_end_at_metal_ps": pulse_end_at_metal,
        "flat_start_at_metal_ps": flat_start_at_metal,
        "flat_end_at_metal_ps": flat_end_at_metal,
        "transition_start_ps": transition_start,
        "transition_end_ps": transition_end,
        "guarded_transition_start_ps": guarded_transition_start,
        "guarded_transition_end_ps": guarded_transition_end,
        "probe_to_metal_distance_A": probe_to_metal_dist,
        "minimum_probe_to_metal_distance_A": min_probe_to_metal_dist,
        "pulse_long_enough": pulse_long_enough,
        "transition_inside_pulse_support": transition_inside_pulse_support,
        "transition_inside_flat_top": transition_inside_flat_top,
        "probe_separation_ok": probe_separation_ok,
        "passed_all_checks": (
            pulse_long_enough
            and transition_inside_pulse_support
            and transition_inside_flat_top
            and probe_separation_ok
        ),
        "warnings": warnings_list,
    }

    if verbose:
        print("\nFDTD pulse/transition consistency check")
        print("---------------------------------------")
        print(f"Pulse center at source      : {t0_src * 1e3:.3f} fs")
        print(f"Pulse center at metal       : {pulse_center_at_metal * 1e3:.3f} fs")
        print(
            f"Pulse support at metal      : "
            f"{pulse_start_at_metal * 1e3:.3f} fs to "
            f"{pulse_end_at_metal * 1e3:.3f} fs"
        )
        print(
            f"Pulse flat-top at metal     : "
            f"{flat_start_at_metal * 1e3:.3f} fs to "
            f"{flat_end_at_metal * 1e3:.3f} fs"
        )
        print(
            f"Transition window           : "
            f"{transition_start * 1e3:.3f} fs to "
            f"{transition_end * 1e3:.3f} fs"
        )
        print(
            f"Guarded transition window   : "
            f"{guarded_transition_start * 1e3:.3f} fs to "
            f"{guarded_transition_end * 1e3:.3f} fs"
        )
        print(f"Pulse support duration      : {pulse_support_duration * 1e3:.3f} fs")
        print(f"Pulse flat-top duration     : {flat_duration * 1e3:.3f} fs")
        print(f"Required flat-top duration  : {required_flat_duration * 1e3:.3f} fs")
        print(f"Optical period              : {optical_period * 1e3:.3f} fs")
        print(f"Guard time                  : {guard_time * 1e3:.3f} fs")
        print(f"Probe-metal distance        : {probe_to_metal_dist:.3e} A")
        print(f"Minimum recommended distance: {min_probe_to_metal_dist:.3e} A")

        if len(warnings_list) == 0:
            print("\nAll consistency checks passed.")
        else:
            print("\nWarnings:")
            for warning in warnings_list:
                print(f"[WARNING] {warning}")

    return checks


def run_simulation(
    params: FDTDParams,
    pulse: PulseParams,
    material: MaterialParamsTD,
    method="convolution",
    explicit_pulse_waveform=None,
    background_del_eps=None,
    background_omega0=None,
    background_lambda0_um=None,
    background_gamma=None,
    *,
    material_probe_every=None,
    material_probe_indices=None,
):
    """Run one of the FDTD material-update methods.

    When ``pulse.type == 'explicit'``, pass a callable, an ``(Nt,)`` sample
    array, or a ``(times, values)`` tuple as ``explicit_pulse_waveform``.
    Select ``method='sio2 background'`` (or ``'sio2_background'``) to add the
    fixed equilibrium fused-silica Lorentz poles.

    The ADE methods accept material arrays shaped (Nt,), (Nt, 1), or
    (Nt, metal_i1-metal_i0). Optional material_probe_every / material_probe_indices
    append a fifth return value as documented in run_fdtd_pml_corrected.
    The legacy 'exact' method has been removed; use 'exact pml' instead.
    """
    valid_methods = (
        "convolution",
        "exact pml",
        "sio2 background",
        "sio2_background",
    )
    if method not in valid_methods:
        raise ValueError(
            "Unknown method; expected 'convolution', 'exact pml', "
            "or 'sio2 background'."
        )
    
    bE, cE, bH, cH, sigmaEbyEpsZero = build_pml_1d(
        Nz=params.Nz,
        dz=params.dz,
        dt=params.dt,
        npml=params.npml,
        R_target=params.R_target,
        m=params.m,
    )

    if method == "convolution":
        if material_probe_every is not None or material_probe_indices is not None:
            raise ValueError("Material probes require 'exact pml' or 'sio2_background'.")
        return run_fdtd_convolution(
            params.Nz, params.Nt, params.dz, params.dt,
            params.src_i, params.probe_t_i, params.probe_r_i,
            params.metal_i0, params.metal_i1,
            material.omega_D, material.gamma_D,
            material.omega_L, material.gamma_L,
            material.del_eps, material.eps_inf,
            pulse.E0, pulse.f0, pulse.t0, pulse.tau, pulse.phase, pulse.type,
            bE, cE, bH, cH,
            params.snap_every,
            explicit_pulse_waveform,
        )
    elif method == "exact pml":
        return run_fdtd_pml_corrected(
            params.Nz, params.Nt, params.dz, params.dt,
            params.src_i, params.probe_t_i, params.probe_r_i,
            params.metal_i0, params.metal_i1,
            material.omega_D, material.gamma_D,
            material.omega_L, material.gamma_L,
            material.del_eps, material.eps_inf,
            pulse.E0, pulse.f0, pulse.t0, pulse.tau, pulse.phase, pulse.type,
            sigmaEbyEpsZero, bH, cH,
            params.snap_every,
            explicit_pulse_waveform,
            material_probe_every=material_probe_every,
            material_probe_indices=material_probe_indices,
        )
    else:
        return run_fdtd_sio2_background(
            params.Nz, params.Nt, params.dz, params.dt,
            params.src_i, params.probe_t_i, params.probe_r_i,
            params.metal_i0, params.metal_i1,
            material.omega_D, material.gamma_D,
            material.omega_L, material.gamma_L,
            material.del_eps, material.eps_inf,
            pulse.E0, pulse.f0, pulse.t0, pulse.tau, pulse.phase, pulse.type,
            sigmaEbyEpsZero, bH, cH,
            params.snap_every,
            explicit_pulse_waveform,
            material_probe_every=material_probe_every,
            material_probe_indices=material_probe_indices,
            background_del_eps=background_del_eps,
            background_omega0=background_omega0,
            background_lambda0_um=background_lambda0_um,
            background_gamma=background_gamma,
        )

# ---------------------------
# Analyze simulation outputs
# ---------------------------
def transmission_coeff_analytical(n, k, d, w):
    # guard against w=0
    w = np.asarray(w, dtype=float)
    w_safe = np.where(w == 0.0, np.finfo(float).tiny, w)

    wavelength = (2.0 * np.pi * C0_SI) / w_safe
    u_2 = n
    v_2 = k

    u_2_sq = u_2**2.0
    v_2_sq = v_2**2.0

    tau_12_sq = 4.0 / ((1.0 + u_2) ** 2.0 + v_2_sq)
    tau_23_sq = 4.0 * (u_2_sq + v_2_sq) / ((1.0 + u_2) ** 2.0 + v_2_sq)
    chi_12 = np.arctan2(-1.0 * v_2, (1.0 + u_2))
    chi_23 = np.arctan2(v_2, (u_2_sq + v_2_sq + u_2))

    rho_sq = ((1.0 - u_2) ** 2.0 + v_2_sq) / ((1.0 + u_2) ** 2.0 + v_2_sq)
    phi = np.arctan((2.0 * v_2) / (u_2_sq + v_2_sq - 1.0))
    eta = 2.0 * np.pi * d / wavelength

    exp_2v_2_eta = np.exp(2.0 * v_2 * eta)
    exp_2v_2_eta_inv = 1.0 / exp_2v_2_eta
    cosine_factor = np.cos(2.0 * phi + 2.0 * u_2 * eta)

    arctan_delta_t_rhs = np.arctan2(
        (exp_2v_2_eta * np.sin(2.0 * u_2 * eta) - rho_sq * np.sin(2.0 * phi)),
        (exp_2v_2_eta * np.cos(2.0 * u_2 * eta) + rho_sq * np.cos(2.0 * phi)),
    )
    delta_t = arctan_delta_t_rhs + chi_12 + chi_23 - u_2 * eta
    phase = np.exp(1.0j * delta_t)

    denom = (
        1.0
        + (rho_sq * exp_2v_2_eta_inv) ** 2.0
        + 2.0 * rho_sq * exp_2v_2_eta_inv * cosine_factor
    )
    t_sq = (tau_12_sq * tau_23_sq * exp_2v_2_eta_inv) / denom

    return np.sqrt(t_sq) * phase


def transmission_finite_reflections(
    n: float,
    k: float,
    d_m: float,
    w: float,
    N_reflections: int,
):
    """
    Parameters
    ----------
    n, k : float
        Real and imaginary parts of film refractive index: n2 = n + i k.
    d_m : float
        Film thickness (meters).
    w : float
        2*pi*frequency
    N_reflections : int
        Number of internal reflections/round-trips to include (N >= 0).
        N=0 -> only the first transmitted pass (no internal round trips)
        N→∞ -> converges to standard thin-film Fresnel result (if |q|<1)

    Returns
    -------
    tN : complex
        Complex field transmission coefficient (E_trans / E_inc).
    """
    if N_reflections < 0:
        raise ValueError("N_reflections must be >= 0")
    
    wavelength_m = 2.0*np.pi * C0_SI / w

    n1 = 1.0 + 0.0j
    n2 = complex(n, k)
    n3 = 1.0 + 0.0j

    # Fresnel amplitude coefficients at normal incidence
    r12 = (n1 - n2) / (n1 + n2)
    t12 = (2.0 * n1) / (n1 + n2)

    r21 = (n2 - n1) / (n2 + n1)
    t23 = (2.0 * n2) / (n2 + n3)

    r23 = (n2 - n3) / (n2 + n3)

    # Propagation phase through the film
    k0 = 2.0 * np.pi / wavelength_m
    delta = k0 * n2 * d_m

    # Truncated geometric series factor
    q = r21 * r23 * np.exp(2.0j * delta)

    # Handle q ~ 1 numerically (avoid division blow-up)
    if np.isclose(q, 1.0 + 0.0j):
        series = (N_reflections + 1)  # sum_{m=0..N} 1 = N+1
    else:
        series = (1.0 - q ** (N_reflections + 1)) / (1.0 - q)

    tN = t12 * t23 * np.exp(1.0j * delta) * series

    return tN


# ---------------------------
# Utilities
# ---------------------------
def _make_window(name: str, N: int) -> np.ndarray:
    name = (name or "").lower()
    if name in ("hann", "hanning"):
        return np.hanning(N)
    if name in ("hamming",):
        return np.hamming(N)
    if name in ("blackman",):
        return np.blackman(N)
    if name in ("rect", "boxcar", "none", ""):
        return np.ones(N)
    raise ValueError(f"Unknown window: {name!r}")


def _fft_transfer_function(E_trans, E_inc, t_ax, d, pad_factor=1, window="hann"):
    E_trans = np.asarray(E_trans, dtype=float)
    E_inc = np.asarray(E_inc, dtype=float)
    t_ax = np.asarray(t_ax, dtype=float)

    if E_trans.shape != E_inc.shape or E_trans.shape != t_ax.shape:
        raise ValueError("E_trans, E_inc, and t_ax must have the same shape.")

    N = t_ax.size
    dt = float(np.mean(np.diff(t_ax)))

    wdw = _make_window(window, N)
    Et = (E_trans - np.mean(E_trans)) * wdw
    Ei = (E_inc - np.mean(E_inc)) * wdw

    Nfft = int(N * max(1, int(pad_factor)))
    Et_w = np.fft.rfft(Et, n=Nfft)
    Ei_w = np.fft.rfft(Ei, n=Nfft)

    f = np.fft.rfftfreq(Nfft, d=dt)  # Hz
    w = 2.0 * np.pi * f              # rad/s

    # transfer function
    with np.errstate(divide="ignore", invalid="ignore"):
        t_meas = Et_w / Ei_w
    t_meas = np.conj(t_meas)

    # correct for vacuum removal
    delta_phi = d*(2*np.pi*f)/C0_SI
    t_meas *= np.exp(1.0j*delta_phi)

    return w, f, t_meas, Ei_w


def retrieve_nk_from_time_traces(
    E_transmitted,
    E_incident,
    t_ax,
    d,
    pad_factor=2,
    window="hann",
    inc_floor_rel=1e-4,
    w_min=None,
    w_max=None,
    n0=1.0,
    k0=0.1,
    bounds=((0.0, 0.0), (50.0, 50.0)),
    max_nfev=10000,
    verbose=False,
    num_reflections=None,
):
    """
    Returns:
        w_fit: angular frequencies (rad/s) where inversion was performed
        n_fit, k_fit: retrieved refractive index components
        t_meas_fit: measured transfer function at w_fit
        t_model_fit: model transfer function at retrieved n,k
        mask: boolean mask into the rfft frequency grid used for fitting
    """

    w, f, t_meas, Ei_w = _fft_transfer_function(
        E_transmitted, E_incident, t_ax, d, pad_factor=pad_factor, window=window,
    )

    # Avoid dividing / fitting where the incident spectrum is tiny
    Ei_mag = np.abs(Ei_w)
    good = Ei_mag > (inc_floor_rel * np.max(Ei_mag))

    if w_min is not None:
        good &= (w >= float(w_min))
    if w_max is not None:
        good &= (w <= float(w_max))

    # also drop DC (often problematic for optics)
    good &= (w > 0.0)

    w_fit = w[good]
    t_meas_fit = t_meas[good]

    n_fit = np.full_like(w_fit, np.nan, dtype=float)
    k_fit = np.full_like(w_fit, np.nan, dtype=float)
    t_model_fit = np.full_like(t_meas_fit, np.nan + 1j * np.nan, dtype=complex)

    # continuation init (helps a lot): start from (n0,k0), then reuse previous solution
    x_prev = np.array([float(n0), float(k0)], dtype=float)

    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)

    for i, (wi, t_target) in enumerate(zip(w_fit, t_meas_fit)):
        # residual in R^2 : [Re, Im]

        if num_reflections is not None:
            def fun(x):
                ni, ki = x
                t_mod = transmission_finite_reflections(ni, ki, d, wi, num_reflections)
                r = t_mod - t_target
                return np.abs(r)
        else:
            def fun(x):
                ni, ki = x
                t_mod = transmission_coeff_analytical(ni, ki, d, wi)
                r = t_mod - t_target
                # return np.array([r.real, r.imag], dtype=float)
                return np.abs(r)

        res = least_squares(
            fun,
            x_prev,
            bounds=(lb, ub),
            max_nfev=max_nfev,
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            verbose=2 if verbose else 0,
        )

        if res.success:
            n_fit[i], k_fit[i] = res.x
            if num_reflections is not None:
                t_model_fit[i] = transmission_finite_reflections(res.x[0], res.x[1], d, wi, num_reflections)
            else:
                t_model_fit[i] = transmission_coeff_analytical(res.x[0], res.x[1], d, wi)
            x_prev = res.x  # continuation
        else:
            # keep previous guess but don't advance it aggressively
            x_prev = np.clip(x_prev, lb, ub)

    return w_fit, n_fit, k_fit, t_meas_fit, t_model_fit, good
