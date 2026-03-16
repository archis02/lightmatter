from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, find_peaks, correlate
from .io import load_simulation_run

# Units, time = ps => frequency = THz, 
# mass = m_e, q = e, x = A
# => E = 5.686e2 V / m

# unit_M = 9.109e-31
# unit_Q = 1.602e-19
unit_F = 1e12
unit_T = 1e-12
unit_L = 1e-10
# unit_E = 5.686e2

c0  = 2.9979e6 # in code units
c0_SI = 2.9979e8
# mu0 = 4e-7 * np.pi
# eps0 = 1.0 / (mu0 * c0**2)
# Z0 = np.sqrt(mu0 / eps0)

def _main_lobe_mask(x, frac=0.15): # 0.15 is too conservative
    env = np.abs(hilbert(x))
    thr = frac * env.max()
    idx = np.where(env >= thr)[0]
    if len(idx) == 0:
        return np.zeros_like(x, dtype=bool)
    # take the largest contiguous block (usually main pulse)
    # find breaks
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks, len(idx) - 1]
    blocks = [(idx[s], idx[e]) for s, e in zip(starts, ends)]
    # choose widest block
    i0, i1 = max(blocks, key=lambda p: (p[1] - p[0]))
    mask = np.zeros_like(x, dtype=bool)
    mask[i0:i1+1] = True
    return mask


def _xcorr_delay(x, y, dt):
    # delay that best aligns y to x (y delayed -> positive lag)
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    c = correlate(y0, x0, mode="full")  # correlate(y, x)
    lags = np.arange(-len(x0)+1, len(x0)) * dt
    return lags[np.argmax(c)]


def match_peaks_and_transfer(E_trans, E_inc, t, f0, dt, thickness,
                             env_frac=0.05, peak_prom_frac=0.05):
    """
    Returns matched ratios and phases (arrays) without hand-tuned t_lims.
    """
    # 1) auto window: intersection of main-lobe masks
    m_inc = _main_lobe_mask(E_inc, frac=env_frac)
    m_tr  = _main_lobe_mask(E_trans, frac=env_frac)

    # 2) estimate delay by xcorr (on masked region helps)
    #    use masked signals but keep length
    E_inc_m  = E_inc * m_inc
    E_tr_m   = E_trans * m_tr
    dt_corr = _xcorr_delay(E_inc_m, E_tr_m, dt)

    # 3) peak finding settings
    T0 = 1.0 / f0
    min_dist = max(1, int(0.45 * T0 / dt))  # ~half-period
    prom_inc = peak_prom_frac * np.max(np.abs(E_inc[m_inc])) if np.any(m_inc) else None
    prom_tr  = peak_prom_frac * np.max(np.abs(E_trans[m_tr])) if np.any(m_tr) else None

    # detect peaks on incident ONLY within mask
    peaks_inc, _ = find_peaks(E_inc, distance=min_dist, prominence=prom_inc)
    peaks_inc = peaks_inc[m_inc[peaks_inc]]

    # detect peaks on transmitted within mask (we’ll match to these)
    peaks_tr, _ = find_peaks(E_trans, distance=min_dist, prominence=prom_tr)
    peaks_tr = peaks_tr[m_tr[peaks_tr]]

    t_inc = t[peaks_inc]
    t_tr  = t[peaks_tr]

    # 4) match: for each incident peak, find nearest transmitted peak near t_i + dt_corr
    tol = 0.25 * T0
    matched = []
    j = 0
    for i, ti in enumerate(t_inc):
        target = ti + dt_corr

        # advance pointer to the first t_tr >= target - tol
        while j < len(t_tr) and t_tr[j] < target - tol:
            j += 1

        candidates = []
        if j < len(t_tr) and abs(t_tr[j] - target) <= tol:
            candidates.append(j)
        if j-1 >= 0 and abs(t_tr[j-1] - target) <= tol:
            candidates.append(j-1)
        if j+1 < len(t_tr) and abs(t_tr[j+1] - target) <= tol:
            candidates.append(j+1)

        if not candidates:
            continue

        k = min(set(candidates), key=lambda kk: abs(t_tr[kk] - target))
        matched.append((peaks_inc[i], peaks_tr[k]))

    if len(matched) < 3:
        # too few matches -> either threshold too strict, or peak settings too aggressive
        return np.array([]), np.array([]), dt_corr, matched

    inc_ids = np.array([a for a, b in matched], dtype=int)
    tr_ids  = np.array([b for a, b in matched], dtype=int)

    ratio = E_trans[tr_ids] / E_inc[inc_ids]
    delta_phi = thickness * (2.0 * np.pi * f0) / c0_SI
    phase = (t[tr_ids] - t[inc_ids]) * f0 * 2*np.pi  + delta_phi

    return ratio, phase, dt_corr, matched


def get_transfer_func(freq_scan_arr, tau_scan, return_errorbars: bool = True):
    """
    Compute time-domain transfer function estimates by matched peak analysis.

    Parameters
    ----------
    freq_scan_arr : array-like
        Frequencies used for scan (whatever units your folder naming expects).
    tau_scan : float/int
        Pulse parameter used in folder naming.
    return_errorbars : bool, default True
        If True, also compute and return 10th/90th percentile bounds
        for |t| and phase. If False, bounds are returned as None.

    Returns
    -------
    abs_t : np.ndarray
        Median of amplitude ratio for each frequency.
    abs_t_l : np.ndarray or None
        10th percentile of amplitude ratio (if return_errorbars).
    abs_t_u : np.ndarray or None
        90th percentile of amplitude ratio (if return_errorbars).
    angle_t : np.ndarray
        Median of phase for each frequency.
    angle_t_l : np.ndarray or None
        10th percentile of phase (if return_errorbars).
    angle_t_u : np.ndarray or None
        90th percentile of phase (if return_errorbars).
    """
    abs_t_arr = []
    abs_t_arr_lower = []
    abs_t_arr_upper = []
    angle_t_arr = []
    angle_t_arr_lower = []
    angle_t_arr_upper = []

    for i, freq in enumerate(freq_scan_arr):
        dirname = f"../output/frequency_scan/tau_{tau_scan}_f0_{freq}"

        result_nofilm = load_simulation_run(f"../output/{dirname}/nofilm", nosnaps=True)
        result_withfilm = load_simulation_run(f"../output/{dirname}/with_film", nosnaps=True)

        E_t_withfilm = result_withfilm["E_t"]
        E_t_nofilm = result_nofilm["E_t"]
        Nt = result_withfilm["params"].Nt

        dt = result_withfilm["params"].dt * unit_T
        f0 = result_withfilm["pulse"].f0 * unit_F
        t_ax_retrieved = np.arange(Nt) * dt

        thickness_retrieved = (
            (result_withfilm["params"].metal_i1 - result_withfilm["params"].metal_i0)
            * result_withfilm["params"].dz
            * unit_L
        )

        # Detect/match peaks and get per-peak ratio and phase samples
        ratio, phase, dt_corr, matched = match_peaks_and_transfer(
            E_t_withfilm, E_t_nofilm, t_ax_retrieved, f0, dt, thickness_retrieved,
        )

        # Append medians
        abs_t_arr.append(np.median(ratio))
        angle_t_arr.append(np.median(phase))

        # Append percentile bounds only if requested
        if return_errorbars:
            abs_t_arr_lower.append(np.quantile(ratio, 0.1))
            abs_t_arr_upper.append(np.quantile(ratio, 0.9))
            angle_t_arr_lower.append(np.quantile(phase, 0.1))
            angle_t_arr_upper.append(np.quantile(phase, 0.9))

    abs_t = np.asarray(abs_t_arr)
    angle_t = np.asarray(angle_t_arr)

    if return_errorbars:
        abs_t_l = np.asarray(abs_t_arr_lower)
        abs_t_u = np.asarray(abs_t_arr_upper)
        angle_t_l = np.asarray(angle_t_arr_lower)
        angle_t_u = np.asarray(angle_t_arr_upper)
    else:
        abs_t_l = abs_t_u = None
        angle_t_l = angle_t_u = None

    return abs_t, abs_t_l, abs_t_u, angle_t, angle_t_l, angle_t_u