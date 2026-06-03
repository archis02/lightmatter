from __future__ import annotations
import numpy as np

from .infer_drude_reflection import r_model_from_params
from .io import load_simulation_run
from .units import UNIT_T, UNIT_F, UNIT_L
from .infer_drude_comp_const_interpolate import extract_peak_ratio_time_series_from_reflections as extract_peak_ratio_timeseries_interp

from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def _signed_peaks(x, t, f0, peak_prom_frac=0.05, min_dist_frac=0.35):
    """
    Find both positive and negative extrema and return them in time order.

    Returns
    -------
    peak_idx : ndarray[int]
        Indices of extrema sorted by time.
    peak_sign : ndarray[int]
        +1 for maxima, -1 for minima.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    T0 = 1.0 / f0

    min_dist = max(1, int(np.ceil(min_dist_frac * T0 / dt)))
    prom = peak_prom_frac * np.max(np.abs(x))

    idx_pos, _ = find_peaks(x, prominence=prom, distance=min_dist)
    idx_neg, _ = find_peaks(-x, prominence=prom, distance=min_dist)

    idx = np.r_[idx_pos, idx_neg]
    sgn = np.r_[np.ones(len(idx_pos), dtype=int), -np.ones(len(idx_neg), dtype=int)]

    order = np.argsort(idx)
    return idx[order], sgn[order]


def _match_signed_peaks(idx_const, sgn_const, idx_vary, sgn_vary, t, tol):
    """
    Monotonic one-to-one matching of extrema with the same sign.

    Parameters
    ----------
    tol : float
        Maximum allowed time mismatch.
    """
    matched = []
    j0 = 0

    # time opints of extrema
    t_const = t[idx_const]
    t_vary = t[idx_vary]

    # loop over extrema in constant case
    for i, (ic, sc, tc) in enumerate(zip(idx_const, sgn_const, t_const)):
        
        # search over the extrema array in varying case
        while j0 < len(idx_vary) and t_vary[j0] < tc - tol:
            j0 += 1
        
        # pick candidates
        candidates = []
        for jj in (j0 - 1, j0, j0 + 1, j0 + 2):
            if 0 <= jj < len(idx_vary):
                if sgn_vary[jj] == sc and abs(t_vary[jj] - tc) <= tol:
                    candidates.append(jj)

        if not candidates:
            continue
        
        # finded the closest match among candidates
        jj_best = min(candidates, key=lambda jj: abs(t_vary[jj] - tc))
        matched.append((ic, idx_vary[jj_best]))

        # increment the counter variable
        j0 = jj_best + 1

    return matched


def extract_peak_ratio_time_series_from_reflections(
    E_refl_const,
    E_refl_vary,
    t_ax,
    f0,
    peak_prom_frac=0.05,
    peak_tol_frac=0.45,
    clip=5,
):
    """
    Build a complex ratio time series from matched extrema of the reflected-only traces.

    Measured quantity:
        ratio_meas = (E_vary_peak / E_const_peak) * exp(i * omega0 * dt_peak)

    where dt_peak = t_vary_peak - t_const_peak.

    Returns
    -------
    t_mid : ndarray
        Midpoint times of the matched extrema.
    ratio_complex : ndarray[complex]
        Complex measured ratio at the matched times.
    aux : dict
        Diagnostics.
    """
    t_ax = np.asarray(t_ax, dtype=float)
    omega0 = 2.0 * np.pi * f0
    T0 = 1.0 / f0
    tol = peak_tol_frac * T0

    # identify signed peaks (or extrema)
    idx_c, sgn_c = _signed_peaks(E_refl_const, t_ax, f0, peak_prom_frac=peak_prom_frac)
    idx_v, sgn_v = _signed_peaks(E_refl_vary, t_ax, f0, peak_prom_frac=peak_prom_frac)

    matched = _match_signed_peaks(idx_c, sgn_c, idx_v, sgn_v, t_ax, tol=tol)

    if len(matched) == 0:
        return np.array([]), np.array([], dtype=np.complex128), {
            "matched": [],
            "idx_const": idx_c,
            "idx_vary": idx_v,
        }

    ic = np.array([m[0] for m in matched], dtype=int)
    iv = np.array([m[1] for m in matched], dtype=int)

    # clip some peaks from the beginning and end, to remove the rising and falling edge
    if clip > 0 and len(ic) > 2 * clip:
        sl = slice(clip, -clip)
        ic = ic[sl]
        iv = iv[sl]

    t_const = t_ax[ic]
    t_vary = t_ax[iv]
    t_mid = 0.5 * (t_const + t_vary)

    amp_ratio = E_refl_vary[iv] / E_refl_const[ic]
    dt_peak = t_vary - t_const

    # phase from time shift between matched extrema
    ratio_complex = amp_ratio * np.exp(1.0j * omega0 * dt_peak)

    aux = {
        "matched": list(zip(ic.tolist(), iv.tolist())),
        "idx_const": idx_c,
        "idx_vary": idx_v,
        "t_const": t_const,
        "t_vary": t_vary,
        "dt_peak": dt_peak,
        "amp_ratio": amp_ratio,
    }

    return t_mid, ratio_complex, aux


def drude_delta_reflection_residual(
    x,
    omega_arr,
    ratio_r_meas_arr,
    eps_inf,
    omega_D0,
    gamma_D0,
    d_m,
    N_reflections=0,
    weights=None,
    lam_smooth=0.0,
    x_prev=None,
    floor=1e-5,
):
    """
    Residual for one time slice.

    Uses the complex peak-ratio data:
        ratio_meas(omega, t) ~ r_dyn(omega, t) / r_const(omega)

    Residual is stacked [Re(diff), Im(diff)].
    """
    omega_D, gamma_D = x

    r_dyn = r_model_from_params(
        omega=omega_arr,
        omega_D=omega_D,
        gamma_D=gamma_D,
        eps_inf=eps_inf,
        d_m=d_m,
        N_reflections=N_reflections,
    )

    r_const = r_model_from_params(
        omega=omega_arr,
        omega_D=omega_D0,
        gamma_D=gamma_D0,
        eps_inf=eps_inf,
        d_m=d_m,
        N_reflections=N_reflections,
    )

    ratio_model = r_dyn / r_const
    
    diff = ratio_model - ratio_r_meas_arr

    if weights is None:
        w = np.ones_like(omega_arr, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    scale = np.maximum(np.abs(ratio_r_meas_arr), floor)

    r_data = np.concatenate([
        np.sqrt(w) * diff.real / scale,
        np.sqrt(w) * diff.imag / scale,
    ])

    if lam_smooth > 0.0 and x_prev is not None:
        x_prev = np.asarray(x_prev, dtype=float)
        r_reg = np.sqrt(lam_smooth) * (np.asarray(x, dtype=float) - x_prev)
        return np.concatenate([r_data, r_reg])

    return r_data


def fit_drude_one_time(
    omega_arr,
    ratio_r_meas_arr,
    eps_inf,
    omega_D0,
    gamma_D0,
    d_m,
    N_reflections,
    x0=(1e16, 1e14),
    bounds=((1e12, 1e11), (1e18, 1e16)),
    weights=None,
    lam_smooth=0.0,
    x_prev=None,
):
    res = least_squares(
        drude_delta_reflection_residual,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        args=(
            omega_arr,
            ratio_r_meas_arr,
            eps_inf,
            omega_D0,
            gamma_D0,
            d_m,
            N_reflections,
            weights,
            lam_smooth,
            x_prev,
        ),
        # loss="soft_l1",
        # f_scale=1.0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
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


def build_multifrequency_reflection_dataset(
    f0_arr,
    dirname,
    interpolation=True,
    peak_prom_frac=0.05,
    peak_tol_frac=0.45,
    common_time=None,
    kind="linear",
    clip=2,
):
    """
    Build ratio_meas(omega_j, t_k) from matched peaks of reflected-only traces.

    Folder structure assumed:
      ../output/{dirname[0]}/constant/f0_{freq}_{dirname[1]}/with_film
      ../output/{dirname[0]}/constant/f0_{freq}_{dirname[1]}/nofilm
      ../output/{dirname[0]}/vary/f0_{freq}_{dirname[1]}/with_film
      ../output/{dirname[0]}/vary/f0_{freq}_{dirname[1]}/nofilm
    """
    raw = []

    for freq in f0_arr:
        result_constant = load_simulation_run(
            f"../output/{dirname[0]}/constant/f0_{freq}_{dirname[1]}/with_film",
            nosnaps=True
        )
        result_constant_nofilm = load_simulation_run(
            f"../output/{dirname[0]}/constant/f0_{freq}_{dirname[1]}/nofilm",
            nosnaps=True
        )
        result_variable = load_simulation_run(
            f"../output/{dirname[0]}/vary/f0_{freq}_{dirname[1]}/with_film",
            nosnaps=True
        )
        result_variable_nofilm = load_simulation_run(
            f"../output/{dirname[0]}/vary/f0_{freq}_{dirname[1]}/nofilm",
            nosnaps=True
        )

        # get only reflected pulse
        E_refl_const = result_constant["E_r"] - result_constant_nofilm["E_r"]
        E_refl_vary = result_variable["E_r"] - result_variable_nofilm["E_r"]

        Nt = result_constant["params"].Nt
        dt = result_constant["params"].dt * UNIT_T
        t_ax = np.arange(Nt) * dt

        f0 = result_constant["pulse"].f0 * UNIT_F
        omega0 = 2.0 * np.pi * f0

        thickness_retrieved = (
            (result_constant["params"].metal_i1 - result_constant["params"].metal_i0)
            * result_constant["params"].dz
            * UNIT_L
        )

        # extract peaks
        if interpolation:
            t_peak, ratio_complex, aux = extract_peak_ratio_timeseries_interp(
                E_refl_const=E_refl_const,
                E_refl_vary=E_refl_vary,
                t_ax=t_ax,
                f0=f0,
                peak_prom_frac=peak_prom_frac,
                peak_tol_frac=peak_tol_frac,
                clip=clip,
            )
        else:
            t_peak, ratio_complex, aux = extract_peak_ratio_time_series_from_reflections(
                E_refl_const=E_refl_const,
                E_refl_vary=E_refl_vary,
                t_ax=t_ax,
                f0=f0,
                peak_prom_frac=peak_prom_frac,
                peak_tol_frac=peak_tol_frac,
                clip=clip,
            )

        if len(t_peak) == 0:
            raise ValueError(f"No matched peaks found for frequency {freq}")

        raw.append({
            "f0": f0,
            "omega0": omega0,
            "time": t_peak,
            "ratio_r_complex": ratio_complex,
            "aux": aux,
            "thickness_retrieved": thickness_retrieved,
        })

    d_vals = np.array([r["thickness_retrieved"] for r in raw], dtype=float)
    if not np.allclose(d_vals, d_vals[0], rtol=1e-8, atol=0.0):
        raise ValueError(f"Film thickness is not identical across runs. {d_vals}")
    
    # build a common time axis, if not supplied
    if common_time is None:
        t_min = max(np.min(r["time"]) for r in raw)
        t_max = min(np.max(r["time"]) for r in raw)

        if t_max <= t_min:
            raise ValueError("No overlapping time interval across runs.")

        Nt_common = min(len(r["time"]) for r in raw)
        common_time = np.linspace(t_min, t_max, Nt_common)

    omega_arr = np.array([r["omega0"] for r in raw], dtype=float)
    sortf = np.argsort(omega_arr)
    omega_arr = omega_arr[sortf]
    raw = [raw[i] for i in sortf]

    Nt_common = len(common_time)
    Nf = len(raw)
    ratio_r_meas = np.empty((Nt_common, Nf), dtype=np.complex128)

    # interpolate the complex ratio on the common time axis
    for j, r in enumerate(raw):
        fr = interp1d(
            r["time"],
            np.real(r["ratio_r_complex"]),
            kind=kind,
            bounds_error=True,
        )
        fi = interp1d(
            r["time"],
            np.imag(r["ratio_r_complex"]),
            kind=kind,
            bounds_error=True,
        )
        ratio_r_meas[:, j] = fr(common_time) + 1j * fi(common_time)

    return {
        "time": np.asarray(common_time),
        "omega": omega_arr,
        "ratio_r_meas": ratio_r_meas,
        "raw": raw,
        "d_m": raw[0]["thickness_retrieved"],
    }


def covariance_from_least_squares(res):
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
    return s_sq * JTJ_inv


def infer_drude_time_series_from_multifrequency_runs(
    f0_arr,
    dirname,
    interpolation,
    eps_inf,
    omega_D0,
    gamma_D0,
    N_reflections,
    x0,
    bounds,
    peak_prom_frac=0.05,
    peak_tol_frac=0.45,
    common_time=None,
    weights=None,
    lam_smooth=0.0,
    kind="linear",
    clip=2,
):
    dataset = build_multifrequency_reflection_dataset(
        f0_arr=f0_arr,
        dirname=dirname,
        interpolation=interpolation,
        peak_prom_frac=peak_prom_frac,
        peak_tol_frac=peak_tol_frac,
        common_time=common_time,
        kind=kind,
        clip=clip,
    )

    time_arr = dataset["time"]
    omega_arr = dataset["omega"]
    ratio_r_meas = dataset["ratio_r_meas"]
    d_m = dataset["d_m"]

    Nt, Nf = ratio_r_meas.shape

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
            ratio_r_meas_arr=ratio_r_meas[k, :],
            eps_inf=eps_inf,
            omega_D0=omega_D0,
            gamma_D0=gamma_D0,
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
        "ratio_r_meas": ratio_r_meas,
        "omegaD_t": omegaD_t,
        "gammaD_t": gammaD_t,
        "omegaD_err": omegaD_err,
        "gammaD_err": gammaD_err,
        "cost_t": cost_t,
        "success_t": success_t,
        "fit_objects": fit_objects,
        "dataset": dataset,
    }