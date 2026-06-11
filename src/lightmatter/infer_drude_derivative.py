from __future__ import annotations
import numpy as np

from .io import load_simulation_run
from .units import UNIT_T, UNIT_F, UNIT_L, C0_SI
from .peak_analyzer_interpolate import extract_peak_ratio_time_series_from_reflections as extract_peak_ratio_timeseries_interp
from .peak_analyzer_interpolate import covariance_from_least_squares,epsilon_drude

from scipy.interpolate import interp1d
from scipy.optimize import least_squares


def get_epsilon_dot(omega_arr, omega_D, gamma_D, del_omega_del_t, del_gamma_del_t):
    epsilon_dot = -2.0*omega_D / (omega_arr * (omega_arr + 1.0j*gamma_D)) * del_omega_del_t + 1.0j*omega_D**2.0 / (omega_arr * (omega_arr + 1.0j*gamma_D)**2.0) * del_gamma_del_t
    return epsilon_dot

def drude_delta_reflection_residual_with_derivative(
    x,
    dt,
    omega_arr,
    ratio_r_meas_arr,
    eps_inf,
    omega_D0,
    gamma_D0,
    d_m,
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
    omega_D_prev, gamma_D_prev = x_prev

    del_omegaD_del_t = (omega_D - omega_D_prev) / dt
    del_gammaD_del_t = (gamma_D - gamma_D_prev) / dt

    # calculate del_eps_del_t
    del_eps_del_t = get_epsilon_dot(omega_arr,omega_D,gamma_D,del_omegaD_del_t,del_gammaD_del_t)

    eps_drude = epsilon_drude(
        omega=omega_arr,
        eps_inf=eps_inf,
        omega_D=omega_D,
        gamma_D=gamma_D
    )

    n_sqr = eps_drude - 1.0j/omega_arr * del_eps_del_t

    t_complex_vary = 1.0 + 1.0j * omega_arr * d_m / (2.0 * C0_SI) * (1.0 + n_sqr)
    r_complex_vary = (0.5j*omega_arr*d_m / C0_SI)*(n_sqr - 1.0) * t_complex_vary

    eps_drude_const = epsilon_drude(omega_arr, eps_inf=eps_inf, omega_D=omega_D0, gamma_D=gamma_D0, )
    t_complex_const = 1.0 + 1.0j * omega_arr * d_m / (2.0 * C0_SI) * (1.0 + eps_drude_const)
    r_complex_const = (0.5j*omega_arr*d_m / C0_SI)*(eps_drude_const - 1.0) * t_complex_const

    ratio_model = r_complex_vary / r_complex_const
    
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
    dt,
    omega_arr,
    ratio_r_meas_arr,
    eps_inf,
    omega_D0,
    gamma_D0,
    d_m,
    x0=(1e16, 1e14),
    bounds=((1e12, 1e11), (1e18, 1e16)),
    weights=None,
    lam_smooth=0.0,
    x_prev=None,
):
    res = least_squares(
        drude_delta_reflection_residual_with_derivative,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        args=(
            dt,
            omega_arr,
            ratio_r_meas_arr,
            eps_inf,
            omega_D0,
            gamma_D0,
            d_m,
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
        "del_omegaD_del_t": (res.x[0]-x_prev[0])/dt,
        "del_gammaD_del_t": (res.x[1]-x_prev[1])/dt,
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

        t_peak, ratio_complex, aux = extract_peak_ratio_timeseries_interp(
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


def infer_drude_time_series_from_multifrequency_runs(
    f0_arr,
    dirname,
    eps_inf,
    omega_D0,
    gamma_D0,
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
        peak_prom_frac=peak_prom_frac,
        peak_tol_frac=peak_tol_frac,
        common_time=common_time,
        kind=kind,
        clip=clip,
    )

    time_arr = dataset["time"]
    dt = time_arr[1]-time_arr[0]
    omega_arr = dataset["omega"]
    ratio_r_meas = dataset["ratio_r_meas"]
    d_m = dataset["d_m"]

    Nt, Nf = ratio_r_meas.shape

    omegaD_t = np.empty(Nt, dtype=float)
    gammaD_t = np.empty(Nt, dtype=float)
    del_omegaD_del_t = np.empty(Nt, dtype=float)
    del_gammaD_del_t = np.empty(Nt, dtype=float)

    omegaD_err = np.full(Nt, np.nan, dtype=float)
    gammaD_err = np.full(Nt, np.nan, dtype=float)
    # del_omegaD_del_t_err = np.full(Nt, np.nan, dtype=float)
    # del_gammaD_del_t_err = np.full(Nt, np.nan, dtype=float)

    cost_t = np.empty(Nt, dtype=float)
    success_t = np.empty(Nt, dtype=bool)

    fit_objects = []
    x_prev = np.asarray(x0, dtype=float)

    for k in range(Nt):
        fitk = fit_drude_one_time(
            dt=dt,
            omega_arr=omega_arr,
            ratio_r_meas_arr=ratio_r_meas[k, :],
            eps_inf=eps_inf,
            omega_D0=omega_D0,
            gamma_D0=gamma_D0,
            d_m=d_m,
            x0=x_prev,
            bounds=bounds,
            weights=weights,
            lam_smooth=lam_smooth,
            x_prev=x_prev if k > 0 else [omega_D0,gamma_D0],
        )

        omegaD_t[k] = fitk["omega_D"]
        gammaD_t[k] = fitk["gamma_D"]
        del_omegaD_del_t[k] = fitk["del_omegaD_del_t"]
        del_gammaD_del_t[k] = fitk["del_gammaD_del_t"]
        cost_t[k] = fitk["cost"]
        success_t[k] = fitk["success"]
        fit_objects.append(fitk)

        cov = covariance_from_least_squares(fitk["result_obj"])
        if cov is not None:
            if cov[0, 0] >= 0:
                omegaD_err[k] = np.sqrt(cov[0, 0])
            if cov[1, 1] >= 0:
                gammaD_err[k] = np.sqrt(cov[1, 1])
            # if cov[2, 2] >= 0:
            #     del_omegaD_del_t_err[k] = np.sqrt(cov[2, 2])
            # if cov[3, 3] >= 0:
            #     del_gammaD_del_t_err[k] = np.sqrt(cov[3, 3])

        x_prev = np.array([fitk["omega_D"], fitk["gamma_D"]], dtype=float)

    return {
        "time": time_arr,
        "omega": omega_arr,
        "ratio_r_meas": ratio_r_meas,
        "omegaD_t": omegaD_t,
        "gammaD_t": gammaD_t,
        "del_omegaD_del_t": del_omegaD_del_t,
        "del_gammaD_del_t": del_gammaD_del_t,
        "omegaD_err": omegaD_err,
        "gammaD_err": gammaD_err,
        # "del_omegaD_del_t_err": del_gammaD_del_t_err,
        # "del_gammaD_del_t_err": del_gammaD_del_t_err,
        "cost_t": cost_t,
        "success_t": success_t,
        "fit_objects": fit_objects,
        "dataset": dataset,
    }