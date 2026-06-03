from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit, prange
import sys
from scipy.optimize import least_squares

from .units import C0, C0_SI


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

        if self.dz <= 0 or self.dt <= 0:
            raise ValueError("dz and dt must be positive.")


@dataclass(frozen=True, slots=True)
class MaterialParamsTD:
    """
    Time-dependent Drude–Lorentz parameters.
    All arrays must have length Nt and dtype float64.
    """
    omega_D: np.ndarray
    gamma_D: np.ndarray
    omega_L: np.ndarray
    gamma_L: np.ndarray
    del_eps: np.ndarray
    eps_inf: np.ndarray

    def __post_init__(self):
        n = len(self.omega_D)
        if not all(len(a) == n for a in (self.gamma_D, self.omega_L, self.gamma_L, self.del_eps, self.eps_inf)):
            raise ValueError("All material arrays must have the same length (Nt).")

        # Ensure contiguous float64 arrays (Numba-friendly)
        object.__setattr__(self, "omega_D", np.ascontiguousarray(self.omega_D, dtype=np.float64))
        object.__setattr__(self, "gamma_D", np.ascontiguousarray(self.gamma_D, dtype=np.float64))
        object.__setattr__(self, "omega_L", np.ascontiguousarray(self.omega_L, dtype=np.float64))
        object.__setattr__(self, "gamma_L", np.ascontiguousarray(self.gamma_L, dtype=np.float64))
        object.__setattr__(self, "del_eps", np.ascontiguousarray(self.del_eps, dtype=np.float64))
        object.__setattr__(self, "eps_inf", np.ascontiguousarray(self.eps_inf, dtype=np.float64))

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
def run_fdtd_convolution(
    Nz, Nt, dz, dt,
    src_i,                       # source cell index
    probe_t_i,                   # transmitted probe index
    probe_r_i,                   # reflected probe index
    metal_i0, metal_i1,          # metal region [i0, i1)
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,   # model params, now time-dependent arrays
    E0, f0, t0, tau, phase, pulsetype,      # source waveform params
    bE, cE, bH, cH,              # PML coefficients
    snap_every=20
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
def run_fdtd_exact(
    Nz, Nt, dz, dt,
    src_i,                       # source cell index
    probe_t_i,                   # transmitted probe index
    probe_r_i,                   # reflected probe index
    metal_i0, metal_i1,          # metal region [i0, i1)
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,   # time-dependent arrays
    E0, f0, t0, tau, phase, pulsetype,                      # source waveform params
    bE, cE, bH, cH,                                         # PML coefficients
    snap_every=20
):
    """
    1D Yee FDTD with PML and a genuinely time-dependent Drude-Lorentz medium.

    This version removes the adiabatic recursive-convolution update and instead
    advances the material dynamics directly:

        dJ_D/dt + gamma_D(t) J_D = omega_D(t)^2 E
        dP_L/dt = J_L
        dJ_L/dt + gamma_L(t) J_L + omega_L(t)^2 P_L
            = del_eps(t) * omega_L(t)^2 * E

    together with Ampere's law in normalized form

        d/dt [eps_inf(t) E + P_L] + J_D = d(H/eps0)/dz .

    Here J_D, J_L, P_L are understood as polarization/current variables divided
    by eps0, consistent with the H_by_eps0 convention already used in this file.
    """

    # -------------------------
    # Fields / material states
    # -------------------------
    E = np.zeros(Nz, dtype=np.float64)
    E_old = np.zeros(Nz, dtype=np.float64)

    H_by_eps0 = np.zeros(Nz - 1, dtype=np.float64)   # H / eps0 on Yee edges

    # Drude current (stored at half steps conceptually; numerically advanced in place)
    J_D = np.zeros(Nz, dtype=np.float64)

    # Lorentz polarization and Lorentz current
    P_L = np.zeros(Nz, dtype=np.float64)
    J_L = np.zeros(Nz, dtype=np.float64)

    # -------------------------
    # Probes
    # -------------------------
    E_t = np.zeros(Nt, dtype=np.float64)
    E_r = np.zeros(Nt, dtype=np.float64)

    # -------------------------
    # Snapshot storage
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

        # Save E^n, because all updates below use the old electric field
        for i in prange(Nz):
            E_old[i] = E[i]

        # -------------------------
        # Update H^{n+1/2} from E^n
        # -------------------------
        for i in prange(Nz - 1):
            H_by_eps0[i] = bH[i] * H_by_eps0[i] - cH[i] * (E_old[i + 1] - E_old[i])

        # Time-slab sampling.
        # Using midpoint-like averages improves robustness for fast parameter changes.
        np1 = n + 1
        if np1 >= Nt:
            np1 = n

        # -------------------------
        # Update E^{n+1}
        # -------------------------
        for i in prange(1, Nz - 1):
            curlH = -H_by_eps0[i] + H_by_eps0[i - 1]

            # Dispersive material region
            if metal_i0 <= i < metal_i1:
                # Midpoint-sampled material coefficients on the time slab [t_n, t_{n+1}]
                gD = 0.5 * (gamma_D[n] + gamma_D[np1])
                wD2 = 0.5 * (
                    omega_D[n] * omega_D[n] +
                    omega_D[np1] * omega_D[np1]
                )

                gL = 0.5 * (gamma_L[n] + gamma_L[np1])
                wL2 = 0.5 * (
                    omega_L[n] * omega_L[n] +
                    omega_L[np1] * omega_L[np1]
                )
                sL = 0.5 * (
                    del_eps[n] * omega_L[n] * omega_L[n] +
                    del_eps[np1] * omega_L[np1] * omega_L[np1]
                )

                eps_n = eps_inf[n]
                eps_np1 = eps_inf[np1]

                # ---- Drude current update
                # Crank-Nicolson for damping, explicit in E^n
                #
                # (J_D^{n+1/2} - J_D^{n-1/2})/dt
                #   + gD * (J_D^{n+1/2} + J_D^{n-1/2})/2
                #   = wD2 * E^n
                denom_D = 1.0 + 0.5 * gD * dt
                JD_np12 = (
                    (1.0 - 0.5 * gD * dt) * J_D[i]
                    + dt * wD2 * E_old[i]
                ) / denom_D

                # ---- Lorentz update
                # First-order form:
                #   dP_L/dt = J_L
                #   dJ_L/dt + gL J_L + wL2 P_L = sL E
                #
                # Semi-implicit/staggered update:
                denom_L = 1.0 + 0.5 * gL * dt + 0.5 * wL2 * dt * dt
                JL_np12 = (
                    (1.0 - 0.5 * gL * dt) * J_L[i]
                    - dt * wL2 * P_L[i]
                    + dt * sL * E_old[i]
                ) / denom_L
                PL_np1 = P_L[i] + dt * JL_np12

                # ---- Ampere update with time-dependent eps_inf
                #
                # [eps_inf^{n+1} E^{n+1} - eps_inf^n E^n]/dt
                #   + J_D^{n+1/2} + J_L^{n+1/2}
                #   = curlH / dz
                E[i] = (
                    eps_n * E_old[i]
                    + (dt / dz) * curlH
                    - dt * (JD_np12 + JL_np12)
                ) / eps_np1

                # Commit updated material states
                J_D[i] = JD_np12
                J_L[i] = JL_np12
                P_L[i] = PL_np1

            else:
                # Vacuum / non-dispersive region with precomputed PML coefficients
                E[i] = bE[i] * E_old[i] + cE[i] * curlH

        # -------------------------
        # Soft source injection
        # -------------------------
        if pulsetype == "gaussian":
            E[src_i] += gaussian_sine(t, E0, f0, t0, tau, phase)
        elif pulsetype == "boxed" or pulsetype == "box":
            E[src_i] += boxed_pulse(t, E0, f0, t0, tau, phase)
        elif pulsetype == "smoothed box":
            E[src_i] += smooth_boxed_pulse(t, E0, f0, t0, tau, phase)
        else:
            raise ValueError("invalid pulse name")

        # -------------------------
        # Boundary E update
        # -------------------------
        E[0] = bE[0] * E[0] - cE[0] * (H_by_eps0[0] - 0.0)
        E[Nz - 1] = bE[Nz - 1] * E[Nz - 1] - cE[Nz - 1] * (0.0 - H_by_eps0[Nz - 2])

        # -------------------------
        # Record probes
        # -------------------------
        E_t[n] = E[probe_t_i]
        E_r[n] = E[probe_r_i]

        # -------------------------
        # Snapshots
        # -------------------------
        if snap_every > 0 and (n % snap_every == 0):
            snaps_t[snap_k] = t
            for i in range(Nz):
                snaps_E[snap_k, i] = E[i]
            snap_k += 1

    return E_t, E_r, snaps_t, snaps_E


@njit(parallel=True)
def run_fdtd_pml_corrected(
    Nz, Nt, dz, dt,
    src_i,
    probe_t_i,
    probe_r_i,
    metal_i0, metal_i1,
    omega_D, gamma_D, omega_L, gamma_L, del_eps, eps_inf,
    E0, f0, t0, tau, phase, pulsetype,
    sigmaE_by_eps0,            # NEW
    bH, cH,                    # only H PML coeffs are needed explicitly
    snap_every=200
):
    """
    1D Yee FDTD with conductivity-graded absorber overlapping a genuinely
    time-dependent Drude-Lorentz medium.

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

        # Precompute time-slab material parameters
        gD = 0.5 * (gamma_D[n] + gamma_D[np1])
        wD2 = 0.5 * (
            omega_D[n] * omega_D[n] +
            omega_D[np1] * omega_D[np1]
        )

        gL = 0.5 * (gamma_L[n] + gamma_L[np1])
        wL2 = 0.5 * (
            omega_L[n] * omega_L[n] +
            omega_L[np1] * omega_L[np1]
        )

        sL = 0.5 * (
            del_eps[n] * omega_L[n] * omega_L[n] +
            del_eps[np1] * omega_L[np1] * omega_L[np1]
        )

        eps_n = eps_inf[n]
        eps_np1 = eps_inf[np1]

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
        else:
            raise ValueError("invalid pulse name")

        # -------------------------
        # Record probes
        # -------------------------
        E_t[n] = E[probe_t_i]
        E_r[n] = E[probe_r_i]

        # -------------------------
        # Snapshots
        # -------------------------
        if snap_every > 0 and (n % snap_every == 0):
            snaps_t[snap_k] = t
            for i in range(Nz):
                snaps_E[snap_k, i] = E[i]
            snap_k += 1

    return E_t, E_r, snaps_t, snaps_E


def run_simulation(
    params: FDTDParams,
    pulse: PulseParams,
    material: MaterialParamsTD,
    method = "convolution"
):
    if method not in ["convolution", "exact", "exact pml"]:
        raise ValueError("Unknown method; must be \'convolution\' or \'exact\' or \'exact pml\'")
    
    bE, cE, bH, cH, sigmaEbyEpsZero = build_pml_1d(
        Nz=params.Nz,
        dz=params.dz,
        dt=params.dt,
        npml=params.npml,
        R_target=params.R_target,
        m=params.m,
    )
    print(f"Maximum correction: {0.5 * sigmaEbyEpsZero[-1] * params.dt}")

    if method=="convolution":
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
        )
    elif method=="exact":
        return run_fdtd_exact(
            params.Nz, params.Nt, params.dz, params.dt,
            params.src_i, params.probe_t_i, params.probe_r_i,
            params.metal_i0, params.metal_i1,
            material.omega_D, material.gamma_D,
            material.omega_L, material.gamma_L,
            material.del_eps, material.eps_inf,
            pulse.E0, pulse.f0, pulse.t0, pulse.tau, pulse.phase, pulse.type,
            bE, cE, bH, cH,
            params.snap_every,
        )
    elif method=="exact pml":
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