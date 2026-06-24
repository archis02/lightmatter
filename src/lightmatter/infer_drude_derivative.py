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

    Calculates residual as total_diff = diff_angle / scale_angle + diff_abs / scale_abs

    This gives approximately similar weightage to the absolute value and argument of the complex ratio of reflectivities,
    yielding an improved fit during inference.
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
    
    diff_angle = np.unwrap(np.angle(ratio_model)) - np.unwrap(np.angle(ratio_r_meas_arr))
    diff_abs = np.abs(ratio_model) - np.abs(ratio_r_meas_arr)

    scale_abs = np.maximum(np.abs(ratio_r_meas_arr), floor)
    scale_angle = np.maximum(np.abs(np.angle(ratio_r_meas_arr)), floor)

    total_diff = diff_angle / scale_angle + diff_abs / scale_abs

    if weights is None:
        w = np.ones_like(omega_arr, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    r_data = np.concatenate([
        np.sqrt(w) * total_diff,
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
    time_choices=["zv","v","v"],
    shift_delta=None,
    shift_freq=None,
    peak_prom_frac=0.05,
    peak_tol_frac=0.45,
    common_time=None,
    common_time_sampling="highest",
    kind="linear",
    clip=2,
):
    """
    Build ratio_meas(omega_j, t_k) from matched peaks of reflected-only traces.

    Folder structure
    ----------------
    Unshifted simulations:

        ../output/{dirname[0]}/constant/f0_{freq}_{dirname[1]}/with_film
        ../output/{dirname[0]}/constant/f0_{freq}_{dirname[1]}/nofilm
        ../output/{dirname[0]}/vary/f0_{freq}_{dirname[1]}/with_film
        ../output/{dirname[0]}/vary/f0_{freq}_{dirname[1]}/nofilm

    Shifted simulations:

        ../output/{dirname[0]}/constant/
            f0_{freq}_{dirname[1]}_shift_{timeshift}/with_film

        ../output/{dirname[0]}/constant/
            f0_{freq}_{dirname[1]}_shift_{timeshift}/nofilm

        ../output/{dirname[0]}/vary/
            f0_{freq}_{dirname[1]}_shift_{timeshift}/with_film

        ../output/{dirname[0]}/vary/
            f0_{freq}_{dirname[1]}_shift_{timeshift}/nofilm

    Parameters
    ----------
    f0_arr : array-like
        Frequency labels used in the simulation-directory names.

    dirname : tuple or list of str
        Directory-name components. The expected form is

            dirname = (main_directory, run_suffix)

    shift_freq : dict, optional
        Dictionary containing the shifted-transition simulations associated
        with each frequency. For example,

            shift_freq = {
                1.0: [10e-15, 20e-15],
                1.5: [-10e-15, 10e-15],
            }

        The values of ``timeshift`` must use the same time units as ``t_peak``.
        Since ``t_peak`` is converted using ``UNIT_T``, this normally means
        that the shifts should be supplied in seconds.

        The unshifted simulation is always included. A shifted simulation is
        aligned to the unshifted transition-time convention using

            t_aligned = t_peak - timeshift

    peak_prom_frac : float, optional
        Relative peak-prominence threshold.

    peak_tol_frac : float, optional
        Peak-matching tolerance as a fraction of the optical period.

    common_time : array-like, optional
        User-supplied common time axis. If omitted, it is constructed from the
        overlapping interval of all frequencies.

    common_time_sampling : {"highest", "lowest"}, optional
        Rule used to determine the number of samples in an automatically
        generated common time axis.

    kind : str, optional
        Interpolation type passed to ``scipy.interpolate.interp1d``.

    clip : int, optional
        Number of matched samples removed at each end by the peak-extraction
        function.

    Returns
    -------
    dataset : dict
        Dictionary containing

        ``time``
            Common time axis, shape ``(Nt,)``.

        ``omega``
            Sorted angular frequencies, shape ``(Nf,)``.

        ``ratio_r_meas``
            Complex reflected-field ratio, shape ``(Nt, Nf)``.

        ``raw``
            Per-frequency merged data and the individual shifted segments.

        ``d_m``
            Retrieved film thickness.
    """

    if shift_freq is None:
        shift_freq = {}

    raw = []

    def _get_shifts_for_frequency(freq):
        """
        Return shifts associated with a frequency.

        Exact dictionary lookup is attempted first. A close floating-point
        match is attempted afterward to avoid minor float-key mismatches.
        """
        if freq in shift_freq:
            shifts = shift_freq[freq]
        else:
            shifts = []

            for key, value in shift_freq.items():
                try:
                    keys_match = np.isclose(
                        float(key),
                        float(freq),
                        rtol=1e-12,
                        atol=0.0,
                    )
                except (TypeError, ValueError):
                    keys_match = False

                if keys_match:
                    shifts = value
                    break

        if shifts is None:
            return []

        if np.isscalar(shifts):
            shifts = [shifts]

        return list(shifts)

    def _load_frequency_run(freq, timeshift=None):
        """
        Load one unshifted or shifted simulation and extract its complex ratio.
        timeshift is supplied in fs
        """
        run_name = f"f0_{freq}_{dirname[1]}"

        if timeshift is not None:
            run_name += f"_shift_{timeshift}"

        constant_root = (
            f"../output/{dirname[0]}/constant/{run_name}"
        )
        variable_root = (
            f"../output/{dirname[0]}/vary/{run_name}"
        )

        result_constant = load_simulation_run(
            f"{constant_root}/with_film",
            nosnaps=True,
        )
        result_constant_nofilm = load_simulation_run(
            f"{constant_root}/nofilm",
            nosnaps=True,
        )
        result_variable = load_simulation_run(
            f"{variable_root}/with_film",
            nosnaps=True,
        )
        result_variable_nofilm = load_simulation_run(
            f"{variable_root}/nofilm",
            nosnaps=True,
        )

        # Isolate the reflected field from the film.
        E_refl_const = (
            result_constant["E_r"]
            - result_constant_nofilm["E_r"]
        )
        E_refl_vary = (
            result_variable["E_r"]
            - result_variable_nofilm["E_r"]
        )

        Nt = result_constant["params"].Nt
        dt = result_constant["params"].dt * UNIT_T
        t_ax = np.arange(Nt, dtype=float) * dt

        f0_loaded = result_constant["pulse"].f0 * UNIT_F
        omega0_loaded = 2.0 * np.pi * f0_loaded

        thickness_retrieved = (
            (
                result_constant["params"].metal_i1
                - result_constant["params"].metal_i0
            )
            * result_constant["params"].dz
            * UNIT_L
        )

        t_peak, ratio_complex, aux = (
            extract_peak_ratio_timeseries_interp(
                E_refl_const=E_refl_const,
                E_refl_vary=E_refl_vary,
                t_ax=t_ax,
                f0=f0_loaded,
                time_choices=time_choices,
                shift_delta=shift_delta,
                peak_prom_frac=peak_prom_frac,
                peak_tol_frac=peak_tol_frac,
                clip=clip,
            )
        )

        t_peak = np.asarray(t_peak, dtype=float)
        ratio_complex = np.asarray(
            ratio_complex,
            dtype=np.complex128,
        )

        if len(t_peak) == 0:
            shift_description = (
                "unshifted"
                if timeshift is None
                else f"shift={timeshift}"
            )

            raise ValueError(
                "No matched peaks found for "
                f"frequency {freq}, {shift_description}."
            )

        if len(t_peak) != len(ratio_complex):
            raise ValueError(
                "The extracted time and ratio arrays have different "
                f"lengths for frequency {freq}, shift={timeshift}."
            )

        numerical_shift = (
            0.0 if timeshift is None else float(timeshift * 1e-15) # convert to SI units, since t_ax is in SI
        )

        # Transform the shifted simulation back to the common physical
        # transition-time convention.
        t_aligned = t_peak - numerical_shift

        return {
            "f0": f0_loaded,
            "omega0": omega0_loaded,
            "time_original": t_peak,
            "time_aligned": t_aligned,
            "ratio_r_complex": ratio_complex,
            "timeshift": numerical_shift,
            "shift_label": timeshift,
            "aux": aux,
            "thickness_retrieved": thickness_retrieved,
        }

    for freq in f0_arr:
        # Always include the unshifted simulation.
        segments = [_load_frequency_run(freq, timeshift=None)]

        # Add all shifted-transition simulations for this frequency.
        shifts = _get_shifts_for_frequency(freq)

        for timeshift in shifts:
            numerical_shift = float(timeshift)

            # Avoid loading the unshifted simulation twice if zero appears
            # explicitly in shift_freq.
            if np.isclose(numerical_shift, 0.0):
                continue

            segments.append(
                _load_frequency_run(
                    freq,
                    timeshift=timeshift,
                )
            )

        # Check that all shifted runs correspond to the same actual
        # frequency.
        segment_frequencies = np.array(
            [segment["f0"] for segment in segments],
            dtype=float,
        )

        if not np.allclose(
            segment_frequencies,
            segment_frequencies[0],
            rtol=1e-10,
            atol=0.0,
        ):
            raise ValueError(
                "Loaded frequency is inconsistent among shifted runs "
                f"for directory-frequency label {freq}: "
                f"{segment_frequencies}"
            )

        # Check film-thickness consistency among the shifted runs of this
        # frequency.
        segment_thicknesses = np.array(
            [
                segment["thickness_retrieved"]
                for segment in segments
            ],
            dtype=float,
        )

        if not np.allclose(
            segment_thicknesses,
            segment_thicknesses[0],
            rtol=1e-8,
            atol=0.0,
        ):
            raise ValueError(
                "Film thickness is inconsistent among shifted runs "
                f"for frequency {freq}: {segment_thicknesses}"
            )

        # Merge all aligned samples for this physical frequency.
        time_merged = np.concatenate(
            [segment["time_aligned"] for segment in segments]
        )
        ratio_merged = np.concatenate(
            [segment["ratio_r_complex"] for segment in segments]
        )

        sort_time = np.argsort(time_merged)
        time_merged = time_merged[sort_time]
        ratio_merged = ratio_merged[sort_time]

        # interp1d requires a strictly increasing x axis. Average ratio
        # values when two simulations produce exactly the same aligned time.
        unique_time, inverse, counts = np.unique(
            time_merged,
            return_inverse=True,
            return_counts=True,
        )

        if len(unique_time) != len(time_merged):
            ratio_sum = np.zeros(
                len(unique_time),
                dtype=np.complex128,
            )
            np.add.at(ratio_sum, inverse, ratio_merged)
            ratio_merged = ratio_sum / counts
            time_merged = unique_time

        if len(time_merged) < 2:
            raise ValueError(
                "At least two distinct aligned time samples are required "
                f"for frequency {freq}."
            )

        raw.append({
            "f0": segments[0]["f0"],
            "omega0": segments[0]["omega0"],
            "time": time_merged,
            "ratio_r_complex": ratio_merged,
            "segments": segments,
            "timeshifts": np.array(
                [segment["timeshift"] for segment in segments],
                dtype=float,
            ),
            "aux": [segment["aux"] for segment in segments],
            "thickness_retrieved": segment_thicknesses[0],
        })

    if len(raw) == 0:
        raise ValueError("No frequencies were supplied in f0_arr.")

    # Verify that the film thickness is identical across all frequencies.
    d_vals = np.array(
        [r["thickness_retrieved"] for r in raw],
        dtype=float,
    )

    if not np.allclose(
        d_vals,
        d_vals[0],
        rtol=1e-8,
        atol=0.0,
    ):
        raise ValueError(
            "Film thickness is not identical across frequencies. "
            f"Retrieved thicknesses: {d_vals}"
        )

    # Sort the physical frequencies before constructing the matrix.
    omega_arr = np.array(
        [r["omega0"] for r in raw],
        dtype=float,
    )

    sortf = np.argsort(omega_arr)
    omega_arr = omega_arr[sortf]
    raw = [raw[i] for i in sortf]

    # Construct a common time axis if one was not supplied.
    if common_time is None:
        t_min = max(np.min(r["time"]) for r in raw)
        t_max = min(np.max(r["time"]) for r in raw)

        if t_max <= t_min:
            time_ranges = [
                (np.min(r["time"]), np.max(r["time"]))
                for r in raw
            ]

            raise ValueError(
                "No overlapping aligned time interval exists across "
                f"frequencies. Time ranges: {time_ranges}"
            )

        if common_time_sampling == "highest":
            Nt_common = int(
                np.ceil(
                    max(len(r["time"]) for r in raw)
                )
            )
        elif common_time_sampling == "lowest":
            Nt_common = min(len(r["time"]) for r in raw)
        else:
            raise ValueError(
                "common_time_sampling must be either "
                "'highest' or 'lowest'."
            )

        Nt_common = max(Nt_common, 2)
        common_time = np.linspace(t_min, t_max, Nt_common)

    else:
        common_time = np.asarray(common_time, dtype=float)

        if common_time.ndim != 1:
            raise ValueError("common_time must be one-dimensional.")

        if len(common_time) < 2:
            raise ValueError(
                "common_time must contain at least two samples."
            )

        if np.any(np.diff(common_time) <= 0.0):
            raise ValueError(
                "common_time must be strictly increasing."
            )

        # Give a clearer error than the one produced later by interp1d.
        for r in raw:
            if (
                common_time[0] < r["time"][0]
                or common_time[-1] > r["time"][-1]
            ):
                raise ValueError(
                    "The supplied common_time lies outside the available "
                    "aligned interval for angular frequency "
                    f"{r['omega0']:.6e} rad/s. Available interval: "
                    f"[{r['time'][0]:.6e}, {r['time'][-1]:.6e}] s."
                )

    Nt_common = len(common_time)
    Nf = len(raw)

    ratio_r_meas = np.empty(
        (Nt_common, Nf),
        dtype=np.complex128,
    )

    # Interpolate each physical frequency onto the common time axis.
    # The shift correction has already been included in r["time"].
    for j, r in enumerate(raw):
        fr = interp1d(
            r["time"],
            np.real(r["ratio_r_complex"]),
            kind=kind,
            bounds_error=True,
            assume_sorted=True,
        )

        fi = interp1d(
            r["time"],
            np.imag(r["ratio_r_complex"]),
            kind=kind,
            bounds_error=True,
            assume_sorted=True,
        )

        ratio_r_meas[:, j] = (
            fr(common_time)
            + 1j * fi(common_time)
        )

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
    shift_freq,
    eps_inf,
    omega_D0,
    gamma_D0,
    x0,
    bounds,
    time_choices,
    shift_delta=None,
    peak_prom_frac=0.05,
    peak_tol_frac=0.45,
    common_time=None,
    common_time_sampling="highest",
    weights=None,
    lam_smooth=0.0,
    kind="linear",
    clip=2,
):
    dataset = build_multifrequency_reflection_dataset(
        f0_arr=f0_arr,
        dirname=dirname,
        time_choices=time_choices,
        shift_delta=shift_delta,
        shift_freq=shift_freq,
        peak_prom_frac=peak_prom_frac,
        peak_tol_frac=peak_tol_frac,
        common_time_sampling=common_time_sampling,
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
        "cost_t": cost_t,
        "success_t": success_t,
        "fit_objects": fit_objects,
        "dataset": dataset,
    }