from __future__ import annotations
import numpy as np

from .io import load_simulation_run
from .units import UNIT_T, UNIT_F, UNIT_L

from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

def epsilon_drude(omega, eps_inf, omega_D, gamma_D):
    omega = np.asarray(omega, dtype=float)
    return eps_inf - omega_D**2.0 / (omega * (omega + 1.0j * gamma_D))

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

    if len(t) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

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

    t_const = t[idx_const]
    t_vary = t[idx_vary]

    for ic, sc, tc in zip(idx_const, sgn_const, t_const):
        while j0 < len(idx_vary) and t_vary[j0] < tc - tol:
            j0 += 1

        candidates = []
        for jj in (j0 - 1, j0, j0 + 1, j0 + 2):
            if 0 <= jj < len(idx_vary):
                if sgn_vary[jj] == sc and abs(t_vary[jj] - tc) <= tol:
                    candidates.append(jj)

        if not candidates:
            continue

        jj_best = min(candidates, key=lambda jj: abs(t_vary[jj] - tc))
        matched.append((ic, idx_vary[jj_best]))

        j0 = jj_best + 1

    return matched


def _zero_crossings_linear(x, t):
    """
    Find zero-crossing times by linear interpolation.

    Returns
    -------
    z_times : ndarray[float]
        Interpolated zero-crossing times.

    z_types : ndarray[int]
        +1 for rising crossing, i.e. x goes - to +.
        -1 for falling crossing, i.e. x goes + to -.

    z_brackets : ndarray[int]
        Index i such that the zero crossing lies between i and i+1.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    z_times = []
    z_types = []
    z_brackets = []

    for i in range(len(x) - 1):
        x0 = x[i]
        x1 = x[i + 1]

        if x0 == 0.0 and x1 == 0.0:
            continue

        if x0 * x1 < 0.0:
            tz = t[i] - x0 * (t[i + 1] - t[i]) / (x1 - x0)
            ctype = +1 if x0 < 0.0 and x1 > 0.0 else -1

            z_times.append(tz)
            z_types.append(ctype)
            z_brackets.append(i)

        elif x0 == 0.0 and x1 != 0.0:
            tz = t[i]
            ctype = +1 if x1 > 0.0 else -1

            z_times.append(tz)
            z_types.append(ctype)
            z_brackets.append(i)

    return (
        np.asarray(z_times, dtype=float),
        np.asarray(z_types, dtype=int),
        np.asarray(z_brackets, dtype=int),
    )


def _expected_preceding_zero_type(peak_sign):
    """
    For a positive maximum, the preceding zero is rising.
    For a negative minimum, the preceding zero is falling.
    """
    return +1 if peak_sign > 0 else -1


def _pick_zero_for_peak(
    peak_idx,
    peak_sign,
    z_times,
    z_types,
    t,
    T0,
    side="previous",
    max_sep_frac=0.45,
):
    """
    Pick the zero crossing associated with a given peak.

    Parameters
    ----------
    peak_idx : int
        Index of the peak.

    peak_sign : int
        +1 for positive maximum, -1 for negative minimum.

    side : {"previous", "next", "nearest"}
        Which zero crossing around the peak to use.

    max_sep_frac : float
        Maximum allowed separation from the peak in units of the period.
        For a clean sinusoid, peak-zero separation is approximately T0/4.
    """
    if len(z_times) == 0:
        return None

    t_peak = t[peak_idx]
    expected_type = _expected_preceding_zero_type(peak_sign)

    max_sep = max_sep_frac * T0

    if side == "previous":
        valid = np.where(
            (z_times < t_peak)
            & (t_peak - z_times <= max_sep)
            & (z_types == expected_type)
        )[0]

        if len(valid) == 0:
            return None

        k = valid[-1]
        return z_times[k], z_types[k]

    elif side == "next":
        expected_type_next = -expected_type

        valid = np.where(
            (z_times > t_peak)
            & (z_times - t_peak <= max_sep)
            & (z_types == expected_type_next)
        )[0]

        if len(valid) == 0:
            return None

        k = valid[0]
        return z_times[k], z_types[k]

    elif side == "nearest":
        valid = np.where(np.abs(z_times - t_peak) <= max_sep)[0]

        if len(valid) == 0:
            return None

        k = valid[np.argmin(np.abs(z_times[valid] - t_peak))]
        return z_times[k], z_types[k]

    else:
        raise ValueError("side must be one of: 'previous', 'next', or 'nearest'")


def extract_peak_ratio_time_series_from_reflections(
    E_refl_const,
    E_refl_vary,
    t_ax,
    f0,
    peak_prom_frac=0.05,
    peak_tol_frac=0.45,
    zero_side="previous",
    zero_max_sep_frac=0.45,
    clip=5,
):
    """
    Build a complex ratio time series from matched extrema of the reflected-only traces.

    The amplitude ratio is obtained from matched peaks:
        amp_ratio = E_vary_peak / E_const_peak

    The phase shift is obtained from matched zero crossings:
        dt_zero = t_zero_vary - t_zero_const

    Measured quantity:
        ratio_meas = amp_ratio * exp(i * omega0 * dt_zero)

    Notes
    -----
    The sign in exp(+i omega0 dt_zero) follows your current convention.
    If your model phase uses the opposite convention, change this to
    exp(-i * omega0 * dt_zero).

    Returns
    -------
    t_mid : ndarray
        Midpoint times of the matched zero crossings.

    ratio_complex : ndarray[complex]
        Complex measured ratio at the matched times.

    aux : dict
        Diagnostics.
    """
    E_refl_const = np.asarray(E_refl_const, dtype=float)
    E_refl_vary = np.asarray(E_refl_vary, dtype=float)
    t_ax = np.asarray(t_ax, dtype=float)

    omega0 = 2.0 * np.pi * f0
    T0 = 1.0 / f0
    tol_peak = peak_tol_frac * T0

    # 1. Identify signed peaks.
    idx_c, sgn_c = _signed_peaks(
        E_refl_const,
        t_ax,
        f0,
        peak_prom_frac=peak_prom_frac,
    )

    idx_v, sgn_v = _signed_peaks(
        E_refl_vary,
        t_ax,
        f0,
        peak_prom_frac=peak_prom_frac,
    )

    # 2. Match peaks. These matched peaks are used for amplitude ratios.
    matched_peak_pairs = _match_signed_peaks(
        idx_const=idx_c,
        sgn_const=sgn_c,
        idx_vary=idx_v,
        sgn_vary=sgn_v,
        t=t_ax,
        tol=tol_peak,
    )

    if len(matched_peak_pairs) == 0:
        return np.array([]), np.array([], dtype=np.complex128), {
            "matched_peak_pairs": [],
            "matched_zero_pairs": [],
            "idx_const": idx_c,
            "idx_vary": idx_v,
        }

    ic_all = np.array([m[0] for m in matched_peak_pairs], dtype=int)
    iv_all = np.array([m[1] for m in matched_peak_pairs], dtype=int)

    # Map peak index -> sign for convenience.
    sign_const_by_idx = {int(i): int(s) for i, s in zip(idx_c, sgn_c)}
    sign_vary_by_idx = {int(i): int(s) for i, s in zip(idx_v, sgn_v)}

    # 3. Identify interpolated zero crossings in both traces.
    zc_t, zc_type, zc_bracket = _zero_crossings_linear(E_refl_const, t_ax)
    zv_t, zv_type, zv_bracket = _zero_crossings_linear(E_refl_vary, t_ax)

    accepted_ic = []
    accepted_iv = []

    t_zero_const_list = []
    t_zero_vary_list = []
    zero_type_const_list = []
    zero_type_vary_list = []

    amp_ratio_list = []
    dt_zero_list = []
    dt_peak_raw_list = []
    t_mid_list = []

    matched_zero_pairs = []

    # 4. For each matched peak pair, associate a zero crossing.
    for ic, iv in zip(ic_all, iv_all):
        sc = sign_const_by_idx[int(ic)]
        sv = sign_vary_by_idx[int(iv)]

        # This should already hold because peaks are sign-matched.
        if sc != sv:
            continue

        zc = _pick_zero_for_peak(
            peak_idx=ic,
            peak_sign=sc,
            z_times=zc_t,
            z_types=zc_type,
            t=t_ax,
            T0=T0,
            side=zero_side,
            max_sep_frac=zero_max_sep_frac,
        )

        zv = _pick_zero_for_peak(
            peak_idx=iv,
            peak_sign=sv,
            z_times=zv_t,
            z_types=zv_type,
            t=t_ax,
            T0=T0,
            side=zero_side,
            max_sep_frac=zero_max_sep_frac,
        )

        if zc is None or zv is None:
            continue

        tz_c, type_c = zc
        tz_v, type_v = zv

        # Require the same zero-crossing type.
        if type_c != type_v:
            continue

        if E_refl_const[ic] == 0.0:
            continue

        amp_ratio = E_refl_vary[iv] / E_refl_const[ic]

        dt_zero = tz_v - tz_c
        dt_peak_raw = t_ax[iv] - t_ax[ic]

        accepted_ic.append(ic)
        accepted_iv.append(iv)

        t_zero_const_list.append(tz_c)
        t_zero_vary_list.append(tz_v)
        zero_type_const_list.append(type_c)
        zero_type_vary_list.append(type_v)

        amp_ratio_list.append(amp_ratio)
        dt_zero_list.append(dt_zero)
        dt_peak_raw_list.append(dt_peak_raw)

        t_mid_list.append(0.5 * (tz_c + tz_v))
        matched_zero_pairs.append((int(ic), int(iv), float(tz_c), float(tz_v)))

    if len(accepted_ic) == 0:
        return np.array([]), np.array([], dtype=np.complex128), {
            "matched_peak_pairs": list(zip(ic_all.tolist(), iv_all.tolist())),
            "matched_zero_pairs": [],
            "idx_const": idx_c,
            "idx_vary": idx_v,
            "z_const_times": zc_t,
            "z_const_types": zc_type,
            "z_vary_times": zv_t,
            "z_vary_types": zv_type,
        }

    accepted_ic = np.asarray(accepted_ic, dtype=int)
    accepted_iv = np.asarray(accepted_iv, dtype=int)

    t_mid = np.asarray(t_mid_list, dtype=float)
    amp_ratio = np.asarray(amp_ratio_list, dtype=float)
    dt_zero = np.asarray(dt_zero_list, dtype=float)
    dt_peak_raw = np.asarray(dt_peak_raw_list, dtype=float)

    t_zero_const = np.asarray(t_zero_const_list, dtype=float)
    t_zero_vary = np.asarray(t_zero_vary_list, dtype=float)

    # 5. Clip rising/falling edge samples after zero-crossing filtering.
    if clip > 0 and len(t_mid) > 2 * clip:
        sl = slice(clip, -clip)

        accepted_ic = accepted_ic[sl]
        accepted_iv = accepted_iv[sl]

        t_mid = t_mid[sl]
        amp_ratio = amp_ratio[sl]
        dt_zero = dt_zero[sl]
        dt_peak_raw = dt_peak_raw[sl]

        t_zero_const = t_zero_const[sl]
        t_zero_vary = t_zero_vary[sl]

        matched_zero_pairs = matched_zero_pairs[clip:-clip]

    # 6. Use zero-crossing delay for the phase.
    ratio_complex = amp_ratio * np.exp(1.0j * omega0 * dt_zero)

    aux = {
        "matched_peak_pairs": list(zip(accepted_ic.tolist(), accepted_iv.tolist())),
        "matched_zero_pairs": matched_zero_pairs,

        "idx_const": idx_c,
        "idx_vary": idx_v,

        "z_const_times": zc_t,
        "z_const_types": zc_type,
        "z_const_brackets": zc_bracket,

        "z_vary_times": zv_t,
        "z_vary_types": zv_type,
        "z_vary_brackets": zv_bracket,

        "t_zero_const": t_zero_const,
        "t_zero_vary": t_zero_vary,

        "dt_zero": dt_zero,
        "dt_peak_raw": dt_peak_raw,

        "amp_ratio": amp_ratio,
    }

    return t_mid, ratio_complex, aux


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
