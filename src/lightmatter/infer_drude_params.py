from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from lightmatter.io import load_simulation_run
from lightmatter.peak_analyzer import match_peaks_and_transfer, get_transfer_func
from lightmatter.fdtd import transmission_finite_reflections

from scipy.interpolate import interp1d
from scipy.optimize import least_squares


unit_F = 1e12
unit_T = 1e-12
unit_L = 1e-10
c0_SI = 2.9979e8
c0  = 2.9979e6 # in code units

def epsilon_drude(omega, eps_inf, omega_D, gamma_D):
    """
    Drude dielectric function.
    """
    omega = np.asarray(omega, dtype=float)
    return eps_inf - omega_D**2.0 / (omega * (omega + 1.0j * gamma_D))


def complex_n_from_eps(eps: np.ndarray) -> np.ndarray:
    """
    Convert epsilon(ω) to complex refractive index ñ(ω) = n + i k.
    """
    return np.sqrt(np.asarray(eps, dtype=np.complex128))


def t_model_from_params(omega, omega_D, gamma_D, eps_inf, d_m, N_reflections):
    """
    Compute model complex transmission t(ω) for scalar or array omega.
    """
    omega = np.atleast_1d(omega).astype(float)

    eps = epsilon_drude(omega, eps_inf=eps_inf, omega_D=omega_D, gamma_D=gamma_D)
    ntilde = complex_n_from_eps(eps)

    t_out = np.empty_like(ntilde, dtype=np.complex128)

    for i, (w, nt) in enumerate(zip(omega, ntilde)):
        n = float(np.real(nt))
        k = float(np.imag(nt))
        t_out[i] = transmission_finite_reflections(
            n=n,
            k=k,
            d_m=d_m,
            w=float(w),
            N_reflections=N_reflections,
        )

    return t_out if len(t_out) > 1 else t_out[0]


def drude_transfer_residual(
    x,
    omega_arr,
    t_meas_arr,
    eps_inf,
    d_m,
    N_reflections,
    weights=None,
    lam_smooth=0.0,
    x_prev=None,
):
    """
    Residual for fitting one time slice.

    Parameters
    ----------
    x : array-like, shape (2,)
        [omega_D, gamma_D]
    omega_arr : (Nf,) array
        Angular frequencies.
    t_meas_arr : (Nf,) complex array
        Measured complex transfer values at one time.
    weights : (Nf,) array or None
        Optional weights for each frequency point.
    lam_smooth : float
        Optional temporal regularization strength.
    x_prev : (2,) array or None
        Previous time-step fitted parameters for continuation regularization.

    Returns
    -------
    r : real array
        Stacked real and imag residuals, plus optional smoothness penalty.
    """
    omega_D, gamma_D = x

    t_model = t_model_from_params(
        omega=omega_arr,
        omega_D=omega_D,
        gamma_D=gamma_D,
        eps_inf=eps_inf,
        d_m=d_m,
        N_reflections=N_reflections,
    )

    diff = t_model - t_meas_arr

    if weights is None:
        weights = np.ones_like(omega_arr, dtype=float)

    r_real = np.sqrt(weights) * np.real(diff)
    r_imag = np.sqrt(weights) * np.imag(diff)

    r = np.concatenate([r_real, r_imag])

    if lam_smooth > 0.0 and x_prev is not None:
        r_reg = np.sqrt(lam_smooth) * (x - x_prev)
        r = np.concatenate([r, r_reg])

    return r


def fit_drude_one_time(
    omega_arr,
    t_meas_arr,
    eps_inf,
    d_m,
    N_reflections=50,
    x0=(1e16, 1e14),
    bounds=((1e12, 1e11), (1e18, 1e16)),
    weights=None,
    lam_smooth=0.0,
    x_prev=None,
):
    """
    Fit omega_D and gamma_D at one time step from multi-frequency transfer data.
    """
    res = least_squares(
        drude_transfer_residual,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        args=(omega_arr, t_meas_arr, eps_inf, d_m, N_reflections, weights, lam_smooth, x_prev),
        method="trf",
        jac="2-point",
        xtol=1e-12,
        ftol=1e-12,
        # gtol=1e-15,
        max_nfev=20000,
    )

    return {
        "omega_D": res.x[0],
        "gamma_D": res.x[1],
        "success": res.success,
        "cost": res.cost,
        "nfev": res.nfev,
        "status": res.status,
        "message": res.message,
        "result_obj": res,
    }


def extract_transfer_time_series_from_run(
    E_t_withfilm,
    E_t_nofilm,
    t_ax_retrieved,
    f0,
    dt,
    thickness_retrieved,
    env_frac=0.4,
    clip=3,
):
    """
    Run peak matching for one simulation and return corrected times and complex transfer values.

    Returns
    -------
    t_corr : (Np,) array
        Corrected times associated with matched peaks.
    t_complex : (Np,) complex array
        Complex transfer values ratio * exp(i phase).
    aux : dict
        Extra diagnostic outputs.
    """
    ratio, phase, dt_corr, matched = match_peaks_and_transfer(
        E_t_withfilm,
        E_t_nofilm,
        t_ax_retrieved,
        f0,
        dt,
        thickness_retrieved,
        env_frac=env_frac,
    )

    t_complex = ratio * np.exp(1j * phase)

    inc_ids = np.array([a for a, b in matched], dtype=int)
    tr_ids  = np.array([b for a, b in matched], dtype=int)

    t_peaks_inc = t_ax_retrieved[inc_ids]
    t_peaks_tr = t_ax_retrieved[tr_ids]

    t_corr = 0.5 * (t_peaks_inc + t_peaks_tr) ############## This is not yet justified #########   !!!

    # remove the last and first few points
    t_corr = t_corr[clip:int(-1*clip):]
    t_complex = t_complex[clip:int(-1*clip):]
    ratio  = ratio[clip:int(-1*clip):]
    phase = phase[clip:int(-1*clip):]

    if t_corr.shape[0] != t_complex.shape[0]:
        raise ValueError(f"Truncation is not right, f0 = {f0*1e-12}, tcorr:{t_corr.shape[0]}, t_complex:{t_complex.shape[0]}")

    aux = {
        "ratio": ratio,
        "phase": phase,
        "dt_corr": dt_corr,
    }

    return t_corr, t_complex, aux


def build_multifrequency_transfer_dataset(
    f0_arr,
    dirname,
    env_frac=0.4,
    common_time=None,
    kind="linear",
):
    """
    From multiple simulation runs, build t_meas(omega_j, t_k) on a common time grid.

    Parameters
    ----------
    f0_arr : list of carrier frequencies

    common_time : array or None
        If None, a common overlapping time grid is constructed automatically.
    kind : str
        Interpolation kind for interp1d.

    Returns
    -------
    data : dict
        {
            "time": (Nt,) array,
            "omega": (Nf,) array,
            "t_meas": (Nt, Nf) complex array,
            "raw": list of raw per-run dictionaries
        }
    """
    raw = []

    for freq in f0_arr:

        make_dirname = f"{dirname[0]}/f0_{freq}_{dirname[1]}"
        result_nofilm = load_simulation_run(f"../output/{make_dirname}/nofilm",nosnaps=True)
        result_withfilm = load_simulation_run(f"../output/{make_dirname}/with_film",nosnaps=True)

        E_t_withfilm = result_withfilm["E_t"]
        E_t_nofilm = result_nofilm["E_t"]
        Nt = result_withfilm["params"].Nt
        dt = result_withfilm["params"].dt * unit_T
        t_ax_retrieved = np.arange(Nt) * dt
        f0 = result_withfilm["pulse"].f0 * unit_F
        thickness_retrieved = (result_withfilm['params'].metal_i1 - result_withfilm['params'].metal_i0) * result_withfilm['params'].dz * unit_L

        omega0 = 2.0 * np.pi * f0

        t_corr, t_complex, aux = extract_transfer_time_series_from_run(
            E_t_withfilm,
            E_t_nofilm,
            t_ax_retrieved,
            f0,
            dt,
            thickness_retrieved,
            env_frac=env_frac,
        )

        # Sort by time in case needed
        # idx = np.argsort(t_corr)
        # t_corr = np.asarray(t_corr)[idx]
        # t_complex = np.asarray(t_complex)[idx]

        raw.append({
            "f0": f0,
            "omega0": omega0,
            "time": t_corr,
            "t_complex": t_complex,
            "aux": aux,
            "thickness_retrieved": thickness_retrieved,
        })

    # Check thickness consistency
    d_vals = np.array([r["thickness_retrieved"] for r in raw], dtype=float)
    if not np.allclose(d_vals, d_vals[0], rtol=1e-8, atol=0.0):
        raise ValueError("Film thickness is not identical across runs.")

    # Construct common overlapping time grid if not supplied
    if common_time is None:
        t_min = max(np.min(r["time"]) for r in raw)
        t_max = min(np.max(r["time"]) for r in raw)

        if t_max <= t_min:
            raise ValueError("No overlapping time interval across runs.")

        # Conservative choice: use the smallest number of samples among runs
        Nt_common = min(len(r["time"]) for r in raw)
        common_time = np.linspace(t_min, t_max, Nt_common)

    omega_arr = np.array([r["omega0"] for r in raw], dtype=float)
    sortf = np.argsort(omega_arr)
    omega_arr = omega_arr[sortf]
    raw = [raw[i] for i in sortf]

    Nt = len(common_time)
    Nf = len(raw)
    t_meas = np.empty((Nt, Nf), dtype=np.complex128)

    for j, r in enumerate(raw):
        fr = interp1d(
            r["time"],
            np.real(r["t_complex"]),
            kind=kind,
            bounds_error=True,
        )
        fi = interp1d(
            r["time"],
            np.imag(r["t_complex"]),
            kind=kind,
            bounds_error=True,
        )
        t_meas[:, j] = fr(common_time) + 1j * fi(common_time)

    return {
        "time": np.asarray(common_time),
        "omega": omega_arr,
        "t_meas": t_meas,
        "raw": raw,
        "d_m": raw[0]["thickness_retrieved"],
    }


def covariance_from_least_squares(res):
    """
    Approximate covariance matrix from scipy.optimize.least_squares result.
    Returns None if covariance cannot be estimated.
    """
    J = res.jac
    if J is None:
        return None

    n_res, n_par = J.shape
    dof = n_res - n_par
    if dof <= 0:
        return None

    try:
        JTJ = J.T @ J
        JTJ_inv = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        return None

    s_sq = 2.0 * res.cost / dof
    cov = s_sq * JTJ_inv
    return cov


def infer_drude_time_series_from_multifrequency_runs(
    f0_arr,
    dirname,
    eps_inf,
    N_reflections=50,
    env_frac=0.4,
    common_time=None,
    x0=(1e16, 1e14),
    bounds=((1e12, 1e11), (1e18, 1e16)),
    weights=None,
    lam_smooth=0.0,
    kind="linear",
):
    dataset = build_multifrequency_transfer_dataset(
        f0_arr=f0_arr,
        dirname=dirname,
        env_frac=env_frac,
        common_time=common_time,
        kind=kind,
    )

    time_arr = dataset["time"]
    omega_arr = dataset["omega"]
    t_meas = dataset["t_meas"]
    d_m = dataset["d_m"]

    Nt, Nf = t_meas.shape

    omegaD_t = np.empty(Nt, dtype=float)
    gammaD_t = np.empty(Nt, dtype=float)

    omegaD_err = np.full(Nt, np.nan, dtype=float)
    gammaD_err = np.full(Nt, np.nan, dtype=float)

    cost_t = np.empty(Nt, dtype=float)
    success_t = np.empty(Nt, dtype=bool)

    fit_objects = []

    x_prev = np.asarray(x0, dtype=float)

    for k in range(Nt):
        fitk = fit_drude_one_time(
            omega_arr=omega_arr,
            t_meas_arr=t_meas[k, :],
            eps_inf=eps_inf,
            d_m=d_m,
            N_reflections=N_reflections,
            x0=x_prev,
            bounds=bounds,
            weights=weights,
            lam_smooth=lam_smooth,
            x_prev=x_prev if k > 0 else None,
        )

        omegaD_t[k] = fitk["omega_D"]
        gammaD_t[k] = fitk["gamma_D"]
        cost_t[k] = fitk["cost"]
        success_t[k] = fitk["success"]
        fit_objects.append(fitk)

        # extract 1-sigma uncertainties
        cov = covariance_from_least_squares(fitk["result_obj"])
        if cov is not None:
            if cov[0, 0] >= 0:
                omegaD_err[k] = np.sqrt(cov[0, 0])
            if cov[1, 1] >= 0:
                gammaD_err[k] = np.sqrt(cov[1, 1])

        x_prev = np.array([fitk["omega_D"], fitk["gamma_D"]], dtype=float)

    return {
        "time": time_arr,
        "omega": omega_arr,
        "t_meas": t_meas,
        "omegaD_t": omegaD_t,
        "gammaD_t": gammaD_t,
        "omegaD_err": omegaD_err,
        "gammaD_err": gammaD_err,
        "cost_t": cost_t,
        "success_t": success_t,
        "fit_objects": fit_objects,
        "dataset": dataset,
    }