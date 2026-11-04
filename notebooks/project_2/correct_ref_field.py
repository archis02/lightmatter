#!/usr/bin/env python3
"""Recover the incident probe from one measured first-pass silica transmission.

Run with the lightmatter environment, e.g. ``python correct_ref_field.py`` or
``python correct_ref_field.py --trace-index 3``. Paths default to this directory.
By default both FDTD and analytical corrections are calculated. Use
``--method analytical`` for Fourier-domain correction without running FDTD, or
``--method fdtd`` for only the original method. Analytical results have separate
filenames and do not replace the notebook's FDTD-corrected incident source.
No notebook or pump-driven simulation is executed.

Reversing the direction of propagation alone applies the slab transfer again.
Instead, two right-to-left FDTD runs measure its complex first-pass transfer,
with/without unpumped silica. Their ratio removes the soft-source response and
external vacuum paths. We invert this transfer, including its amplitude, over
the optical measurement band. An independent left-to-right FDTD run checks the
reconstruction. The separate analytical method divides the measured spectrum
by the first-pass Fresnel transmission and propagation phase, using the same
lossless three-pole silica index as the core solver. Its comparison figure also
shows the existing FDTD result when matching saved artifacts are available.
For visual comparison, a separate forward run sends the measured reference
through silica again. This doubly propagated pulse uses the same optical taper
and is shown only in the first plot panel; its numerical data are not saved.

The 80 fs experimental record cannot constrain the later (~125 fs) internal
echo. We therefore gate the calibration before that echo and reconstruct only
the first pass, rather than assuming the unmeasured echo is zero. A constant
group transit time is removed from the transfer to keep the corrected pulse on
the original time axis; this chooses an arbitrary incident-source time origin.
The baseline, spectral taper, time convention and errors are saved in metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

# Allow direct invocation without requiring an editable package installation.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
from lightmatter.fdtd_revised_sio2 import (  # noqa: E402
    FDTDParams, MaterialParamsTD, PulseParams, run_simulation,
    SIO2_BACKGROUND_DEL_EPS_DEFAULT, SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT,
)
from lightmatter.units import C0  # noqa: E402


def silica_indices(frequency_thz):
    """Phase/group indices used to schedule runs and choose the common epoch."""
    wavelength = C0 * 1e-4 / np.asarray(frequency_thz)
    b = np.asarray(SIO2_BACKGROUND_DEL_EPS_DEFAULT)
    pole2 = np.asarray(SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT)**2
    eps = 1 + np.sum(b * wavelength[..., None]**2 /
                     (wavelength[..., None]**2 - pole2), axis=-1)
    n = np.sqrt(eps)
    derivative = np.sum(-2 * b * wavelength[..., None] * pole2 /
                        (wavelength[..., None]**2 - pole2)**2, axis=-1)
    return n, n - wavelength * derivative / (2 * n)


def silica_refractive_index(frequency_thz):
    """Continuum n(f) from the core solver's lossless SiO2 Lorentz poles.

    eps_inf=1 and all carrier/extra Lorentz contributions vanish. Equivalently,
    n²=1+sum(B_j*lambda²/(lambda²-lambda_j²)), with the imported Malitson
    constants (https://doi.org/10.1364/JOSA.55.001205). The Lorentz form also
    defines DC without dividing by f. For the exp(+i*omega*t) inverse-FFT
    convention, the passive branch has Im(n)<=0, so exp(-i*omega*n*d/c) decays.
    Only the transparent optical band is used in the actual inverse.
    """
    frequency = np.asarray(frequency_thz, dtype=float)
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0):
        raise ValueError("Frequencies must be finite and nonnegative (THz).")
    omega = 2*np.pi*frequency
    omega0 = 2*np.pi*C0/(np.asarray(SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT)*1e4)
    denominator = omega0**2-omega[..., None]**2
    if np.any(abs(denominator) <= 8*np.finfo(float).eps*omega0**2):
        raise ValueError("The lossless refractive index is singular at a material resonance.")
    epsilon = 1 + np.sum(np.asarray(SIO2_BACKGROUND_DEL_EPS_DEFAULT)*omega0**2/denominator, axis=-1)
    return np.sqrt(epsilon.astype(complex)).conj()


def analytical_transmission(frequency_thz, thickness_A, *, internal_reflections=False, group_delay_ps=0.0):
    """Complex air/SiO2/air FIELD transmission at normal incidence.

    Units: f in cycles/ps (THz), thickness in Angstrom, C0 in Angstrom/ps.
    NumPy/SciPy's forward FFT uses exp(-i*omega*t); a positive physical delay
    therefore has a negative spectral phase. With p=exp(-i*omega*n*d/C0),

        T_first = 4*n/(1+n)**2 * p
        T_full  = T_first / (1-((n-1)/(n+1))**2 * p**2).

    The default matches the short first-pass measurement. The full Fabry-Perot
    form is available for records containing the internal echoes; blindly
    inverting it on this 80 fs record would constrain unmeasured echoes to zero.
    Multiplying by exp(+i*omega*group_delay_ps) removes a constant flight time
    for plotting/incident-source epoch selection, without removing dispersion.
    """
    if not np.isfinite(thickness_A) or thickness_A < 0 or not np.isfinite(group_delay_ps):
        raise ValueError("Require finite nonnegative thickness and a finite group delay.")
    frequency = np.asarray(frequency_thz, dtype=float)
    index = silica_refractive_index(frequency)
    omega = 2*np.pi*frequency
    propagation = np.exp(-1j*omega*index*thickness_A/C0)
    transmission = 4*index/(1+index)**2 * propagation
    if internal_reflections:
        reflection = (index-1)/(index+1)
        transmission /= 1-reflection**2*propagation**2
    return transmission*np.exp(1j*omega*group_delay_ps)


def load_reference(field_path, time_path, trace_index):
    if trace_index < 0:
        raise ValueError("trace-index must be nonnegative.")
    times = np.loadtxt(time_path, delimiter=",")
    field = np.loadtxt(field_path, delimiter=",", usecols=trace_index)
    if (times.ndim != 1 or field.shape != times.shape or times.size < 3
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(field))):
        raise ValueError("Require equally sized, finite, one-dimensional time/field arrays.")
    spacing = np.diff(times)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-6, atol=1e-12):
        raise ValueError("Experimental times must be increasing and uniformly sampled (fs).")
    count = max(1, int(0.05 * field.size))
    baseline = float(np.mean(np.r_[field[:count], field[-count:]]))
    field = field - baseline
    if not np.any(field):
        raise ValueError("The baseline-corrected reference is zero.")
    return times, field, baseline


def optical_band(frequency_thz):
    """Raised-cosine 40--60 / 320--400 THz edges; unity at 60--320 THz.

    Excludes the silica IR pole (~30 THz) and high-frequency noise. Inverting
    an unmeasured/near-zero transfer there would amplify noise without bound.
    The discarded measured spectral energy is explicitly reported.
    """
    f = np.asarray(frequency_thz)
    weight = np.zeros_like(f)
    weight[(f >= 60) & (f <= 320)] = 1
    lo = (f > 40) & (f < 60)
    hi = (f > 320) & (f < 400)
    weight[lo] = 0.5 * (1 - np.cos(np.pi * (f[lo] - 40) / 20))
    weight[hi] = 0.5 * (1 + np.cos(np.pi * (f[hi] - 320) / 80))
    return weight


def regularized_inverse(transfer, weight, relative_floor=1e-6):
    """Tikhonov complex inverse; unsupported bins are exactly zero."""
    support = weight > 0
    if not np.any(support) or not np.all(np.isfinite(transfer[support])):
        raise ValueError("Invalid calibrated transfer within the correction band.")
    scale = float(np.max(abs(transfer[support])))
    if scale == 0:
        raise ValueError("Zero transmission in the correction band.")
    result = np.zeros_like(transfer, dtype=complex)
    result[support] = (weight[support] * transfer[support].conj() /
                       (abs(transfer[support])**2 + (relative_floor * scale)**2))
    return result


def geometry(nt):
    # Numerical defaults in simulate_experimental_setup.ipynb (2 nm, not 1 nm).
    dz = 20.0
    thickness_cells = int(round(12.82e4 / dz))
    front = int(round(15e4 / dz)) - thickness_cells // 2
    back = front + thickness_cells
    return FDTDParams(
        Nz=int(round(30e4 / dz)), Nt=nt, dz=dz, dt=0.99 * dz / C0,
        src_i=int(round(2.5e4 / dz)), probe_t_i=back + int(round(3e4 / dz)),
        probe_r_i=front - int(round(3e4 / dz)), metal_i0=front, metal_i1=back,
        npml=int(round(1e4 / dz)), snap_every=0,
    )


def propagate(params, samples, label, *, vacuum=False):
    started = time.monotonic()
    print(f"{label}: {params.Nz:,} cells, {params.Nt:,} steps, "
          f"{(params.Nt-1)*params.dt*1e3:.2f} fs", flush=True)
    material = MaterialParamsTD.from_static(
        Nt=params.Nt, omega_D=0, gamma_D=0, omega_L=0, gamma_L=0, del_eps=0, eps_inf=1,
    )
    if vacuum:
        params = replace(params, metal_i1=params.metal_i0)
    result = run_simulation(
        params, PulseParams(E0=1, f0=166, t0=0, tau=0.001, type="explicit"), material,
        method="sio2_background", explicit_pulse_waveform=samples,
    )[0]
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"Nonfinite field from {label}.")
    print(f"  finished in {time.monotonic()-started:.1f} s", flush=True)
    return result


def first_pass_gate(times_ps, arrival_ps, round_trip_ps):
    # The calibration is ~6 fs wide. A generous post-arrival interval retains
    # its dispersive tail; smoothly close well before the first internal echo.
    start = arrival_ps + 0.45 * round_trip_ps
    end = arrival_ps + 0.65 * round_trip_ps
    gate = np.ones_like(times_ps)
    transition = (times_ps > start) & (times_ps < end)
    gate[transition] = 0.5 * (1 + np.cos(np.pi * (times_ps[transition] - start) / (end-start)))
    gate[times_ps >= end] = 0
    return gate


def relative_l2(actual, target):
    return float(np.linalg.norm(actual-target) / np.linalg.norm(target))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correct_reference(field_path, time_path, output_dir, trace_index=0):
    times_fs, measured, baseline = load_reference(field_path, time_path, trace_index)
    times_rel_ps = (times_fs - times_fs[0]) * 1e-3
    params = geometry(1)
    dt = params.dt
    thickness = (params.metal_i1-params.metal_i0) * params.dz
    distance = (params.probe_t_i-params.src_i) * params.dz
    group_index = float(silica_indices(166.0)[1])
    transit = group_index * thickness / C0
    total_transit = (distance-thickness) / C0 + transit
    round_trip = 2 * transit
    if times_rel_ps[-1] >= 0.9 * round_trip:
        raise ValueError("Record can contain internal echoes; this first-pass correction is not appropriate.")

    # Calibrate the reciprocal path: source at the old right transmission
    # probe, receiver at the old left source, both outside silica and PML.
    calibration_t0 = 0.010
    calibration_duration = calibration_t0 + total_transit + 0.8*round_trip
    reverse = replace(params, Nt=int(np.ceil(calibration_duration/dt))+1,
                      src_i=params.probe_t_i, probe_t_i=params.src_i)
    calibration_times = np.arange(reverse.Nt)*dt
    u = (calibration_times-calibration_t0)/0.001
    calibration_source = (1-u*u)*np.exp(-0.5*u*u)  # zero-DC broadband Ricker pulse
    silica = propagate(reverse, calibration_source, "Right-to-left unpumped calibration")
    vacuum = propagate(reverse, calibration_source, "Right-to-left vacuum calibration", vacuum=True)
    silica_gate = first_pass_gate(calibration_times, calibration_t0+total_transit, round_trip)
    vacuum_gate = first_pass_gate(calibration_times, calibration_t0+distance/C0, round_trip)

    # A long FFT buffer prevents the negative-time inverse kernel from wrapping
    # into the pulse. Only the original experimental window is exported.
    pad_ps = 2 * round_trip
    nfft = next_fast_len(int(np.ceil((times_rel_ps[-1]+2*pad_ps)/dt))+1)
    fft_times = np.arange(nfft)*dt
    frequency = rfftfreq(nfft, dt)
    omega = 2*np.pi*frequency
    weight = optical_band(frequency)
    support = weight > 0
    slab_fft = rfft(silica*silica_gate, nfft)
    vacuum_fft = rfft(vacuum*vacuum_gate, nfft)
    source_fft = rfft(calibration_source, nfft)
    if np.min(abs(vacuum_fft[support])) < 1e-5*np.max(abs(vacuum_fft)):
        raise RuntimeError("Calibration does not cover the requested optical band.")
    transfer = np.zeros_like(slab_fft)
    # Ratio gives silica relative to equal-length vacuum; restore slab vacuum
    # phase and then remove ONLY a constant group delay (arbitrary source epoch).
    transfer[support] = (slab_fft[support]/vacuum_fft[support] *
                         np.exp(1j*omega[support]*(transit-thickness/C0)))
    inverse = regularized_inverse(transfer, weight)
    measured_padded = np.interp(fft_times-pad_ps, times_rel_ps, measured, left=0, right=0)
    measured_fft = rfft(measured_padded)
    corrected_padded = irfft(measured_fft*inverse, nfft)
    filtered_padded = irfft(measured_fft*weight, nfft)
    corrected = np.interp(pad_ps+times_rel_ps, fft_times, corrected_padded)
    filtered = np.interp(pad_ps+times_rel_ps, fft_times, filtered_padded)
    inside = (fft_times >= pad_ps) & (fft_times <= pad_ps+times_rel_ps[-1])
    outside_energy = float(np.sum(corrected_padded[~inside]**2)/np.sum(corrected_padded**2))
    if outside_energy > 1e-3:
        raise RuntimeError(f"Corrected field loses {outside_energy:.3%} energy outside its time window; extend the measured record.")

    # Independent forward check of the EXPORTED waveform, not the inverse FFT
    # product. De-embed the calibrated soft-source response before launching a
    # physical incident field. The travelling field is ~half the source addition.
    source_response = np.zeros_like(transfer)
    source_response[support] = (vacuum_fft[support]/source_fft[support] *
                                np.exp(1j*omega[support]*distance/C0))
    source_prepad = 0.020
    physical_incident = np.interp(fft_times-source_prepad, times_rel_ps, corrected, left=0, right=0)
    source_inverse = regularized_inverse(source_response, support.astype(float))
    forward_source = irfft(rfft(physical_incident)*source_inverse, nfft)
    forward_duration = source_prepad+times_rel_ps[-1]+total_transit+0.75*round_trip
    forward = replace(params, Nt=int(np.ceil(forward_duration/dt))+1)
    if forward.Nt > nfft:
        raise RuntimeError("FFT buffer is too short for the forward check.")
    transmitted = propagate(forward, forward_source[:forward.Nt], "Left-to-right reconstruction check")
    forward_times = np.arange(forward.Nt)*dt
    reconstructed = np.interp(times_rel_ps+source_prepad+total_transit, forward_times, transmitted)
    raw_error = relative_l2(reconstructed, measured)
    filtered_error = relative_l2(reconstructed, filtered)
    if filtered_error > 0.02:
        raise RuntimeError(f"Forward FDTD reconstruction error {filtered_error:.3%} exceeds 2%.")

    # Visual comparison only: the measured pulse has already traversed silica.
    # Send it forward once more with the same optical taper and calibrated
    # physical-field injection. Remove only the added pass's group transit for
    # the common plot clock; retain its amplitude and dispersive reshaping.
    measured_incident = np.interp(fft_times-source_prepad, times_rel_ps, measured, left=0, right=0)
    doubly_propagated_source = irfft(rfft(measured_incident)*weight*source_inverse, nfft)
    doubly_propagated_output = propagate(
        forward, doubly_propagated_source[:forward.Nt], "Doubly propagated pulse (measured reference forward again)",
    )
    doubly_propagated = np.interp(
        times_rel_ps+source_prepad+total_transit, forward_times, doubly_propagated_output,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    field_output = output_dir/"ref_field_corrected.txt"
    time_output = output_dir/"time_axis_corrected.txt"
    np.savetxt(field_output, corrected, fmt="%.15e")
    np.savetxt(time_output, times_fs, fmt="%.15e")
    metadata = {
        "method": "regularized inverse of vacuum-calibrated first-pass unpumped SiO2 FDTD transmission",
        "reference_trace_index": int(trace_index),
        "input_field": str(field_path.resolve()), "input_time": str(time_path.resolve()),
        "input_field_sha256": sha256(field_path), "input_time_sha256": sha256(time_path),
        "corrected_field_sha256": sha256(field_output), "corrected_time_sha256": sha256(time_output),
        "baseline_subtracted": baseline, "baseline_edge_fraction": 0.05,
        "field_units": "original experimental arbitrary units; no peak normalization",
        "time_units": "fs", "sample_count": int(corrected.size),
        "time_convention": "Original time axis retained. H_shape=H_slab*exp(+i*omega*group_transit_ps); incident epoch shifted later by group_transit_ps.",
        "group_transit_ps": transit, "group_index_at_166_THz": group_index,
        "source_to_probe_group_transit_ps": total_transit,
        "sample_thickness_um": thickness*1e-4, "first_internal_echo_delay_fs": round_trip*1e3,
        "finite_record_assumption": "Measured record contains the first pass only; later internal echoes are unmeasured, not constrained to zero.",
        "spectral_taper_THz": [40, 60, 320, 400], "inverse_relative_regularization": 1e-6,
        "background_del_eps": list(SIO2_BACKGROUND_DEL_EPS_DEFAULT),
        "background_lambda0_um": list(SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT),
        "background_gamma_per_ps": [0., 0., 0.],
        "reverse_fdtd_params": asdict(reverse), "forward_validation_fdtd_params": asdict(forward),
        "calibration_gate_after_arrival_round_trips": [0.45, 0.65],
        "fft_size": int(nfft), "fft_padding_each_side_ps": pad_ps,
        "correction_outside_saved_window_energy_fraction": outside_energy,
        "band_filter_relative_L2": relative_l2(filtered_padded, measured_padded),
        "band_filter_removed_energy_fraction": float(1-np.sum(filtered_padded**2)/np.sum(measured_padded**2)),
        "forward_vs_measured_relative_L2": raw_error,
        "forward_vs_bandlimited_measured_relative_L2": filtered_error,
        "corrected_vs_measured_relative_L2": relative_l2(corrected, measured),
        "corrected_to_measured_peak_ratio": float(np.max(abs(corrected))/np.max(abs(measured))),
        "validation_source_prepad_ps": source_prepad,
        "validation_note": "Independent forward FDTD of saved-window corrected waveform with calibrated soft-source de-embedding; no fitted amplitude, phase or delay.",
    }
    (output_dir/"ref_field_corrected_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    np.savez_compressed(
        output_dir/"ref_field_correction_diagnostics.npz", time_fs=times_fs,
        measured_baseline_removed=measured, corrected=corrected, deviation=corrected-measured,
        measured_bandlimited=filtered, forward_reconstructed=reconstructed,
        frequency_THz=frequency, first_pass_shape_transfer=transfer, spectral_taper=weight,
        calibration_time_ps=calibration_times, calibration_source=calibration_source,
        calibration_silica=silica, calibration_vacuum=vacuum, calibration_gate=silica_gate,
    )
    # The extra forward trace is deliberately absent from the saved data above.
    save_plot(output_dir, times_fs, measured, corrected, reconstructed, metadata,
              doubly_propagated=doubly_propagated)
    print(json.dumps({key: metadata[key] for key in (
        "reference_trace_index", "forward_vs_measured_relative_L2",
        "forward_vs_bandlimited_measured_relative_L2", "band_filter_removed_energy_fraction",
        "corrected_to_measured_peak_ratio", "correction_outside_saved_window_energy_fraction",
    )}, indent=2), flush=True)
    print(f"Saved corrected field, time axis, metadata, diagnostics and plot in {output_dir.resolve()}")
    return metadata


def save_plot(output_dir, times_fs, measured, corrected, reconstructed, metadata, *, doubly_propagated):
    # Noninteractive: the script never starts a notebook or waits for a GUI.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/lightmatter-correct-ref-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    stride = max(1, times_fs.size//6000)
    t = times_fs[::stride]
    axes[0].plot(t, measured[::stride], label="Measured transmitted reference (baseline removed)", lw=1.5)
    axes[0].plot(t, corrected[::stride], label="Corrected incident reference", lw=1.3)
    axes[0].plot(t, doubly_propagated[::stride], "-.", color="tab:green", lw=1.3,
                 label="Doubly propagated pulse (same optical taper)")
    axes[0].set_ylabel("Field (experimental arb. units)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"Reference trace {metadata['reference_trace_index']}: undo first-pass propagation through 12.82 µm SiO₂\n"
                      f"Constant group transit ({metadata['group_transit_ps']*1e3:.2f} fs) removed for comparison; no amplitude rescaling",
                      fontsize=11)
    axes[1].plot(t, (corrected-measured)[::stride], color="tab:purple", label="Corrected − measured")
    axes[1].set_ylabel("Field deviation (arb. units)")
    axes[1].legend(loc="upper right")
    axes[2].plot(t, measured[::stride], label="Measured", lw=1.6)
    axes[2].plot(t, reconstructed[::stride], "--", label="Corrected field propagated forward by FDTD", lw=1.2)
    axes[2].set(xlabel="Reference time (fs; common group-delay-compensated origin)", ylabel="Field (arb. units)")
    axes[2].set_title(f"Independent forward check: relative L2 error {metadata['forward_vs_measured_relative_L2']:.3%}", fontsize=11)
    axes[2].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="0.65", lw=0.5)
        ax.grid(alpha=0.2)
    fig.savefig(output_dir/"ref_field_correction.png", dpi=180)
    plt.close(fig)


def _load_fdtd_comparison(output_dir, times_fs, trace_index, input_hashes):
    """Load a compatible saved comparison; never rerun or overwrite FDTD here."""
    paths = [output_dir/name for name in (
        "ref_field_corrected_metadata.json", "ref_field_corrected.txt", "time_axis_corrected.txt",
    )]
    if not all(path.is_file() for path in paths):
        return None
    metadata = json.loads(paths[0].read_text())
    if (metadata.get("reference_trace_index") != trace_index or
            any(metadata.get(key) != value for key, value in input_hashes.items())):
        print("Saved FDTD correction belongs to different input data; omitting it from the analytical plot.")
        return None
    if (sha256(paths[1]) != metadata.get("corrected_field_sha256") or
            sha256(paths[2]) != metadata.get("corrected_time_sha256")):
        raise ValueError("Saved FDTD field/time hashes disagree with its metadata.")
    field = np.loadtxt(paths[1])
    saved_times = np.loadtxt(paths[2])
    if (field.shape != times_fs.shape or saved_times.shape != times_fs.shape or
            not np.all(np.isfinite(field)) or not np.allclose(saved_times, times_fs, rtol=0, atol=1e-10)):
        raise ValueError("Saved FDTD comparison has an incompatible time axis or invalid field.")
    comparison = {"field": field, "metadata": metadata}
    diagnostics_path = output_dir/"ref_field_correction_diagnostics.npz"
    if diagnostics_path.is_file():
        with np.load(diagnostics_path, allow_pickle=False) as archive:
            if not np.allclose(archive["corrected"], field, rtol=1e-12, atol=1e-12):
                raise ValueError("Saved FDTD diagnostics disagree with its corrected field.")
            comparison.update(frequency_THz=archive["frequency_THz"],
                              transfer=archive["first_pass_shape_transfer"],
                              taper=archive["spectral_taper"])
    return comparison


def analytical_correct_reference(field_path, time_path, output_dir, trace_index=0):
    """FFT -> divide by analytic T(f) -> inverse FFT, with separate artifacts.

    The optical taper and constant group-delay convention match the FDTD method.
    The FFT uses the measured sampling interval directly; zero padding prevents
    the time-advanced inverse response from wrapping into the saved pulse.
    There is no FDTD run, fitted amplitude, or normalization of either waveform.
    """
    field_path, time_path, output_dir = map(Path, (field_path, time_path, output_dir))
    times_fs, measured, baseline = load_reference(field_path, time_path, trace_index)
    dt_ps = float(times_fs[1]-times_fs[0])*1e-3
    params = geometry(1)
    thickness_A = (params.metal_i1-params.metal_i0)*params.dz
    transit_ps = float(silica_indices(166.0)[1])*thickness_A/C0
    round_trip_ps = 2*transit_ps
    if (times_fs[-1]-times_fs[0])*1e-3 >= 0.9*round_trip_ps:
        raise ValueError("Record can contain internal echoes; use a full-slab transmission model for that record.")
    padding = int(np.ceil(2*round_trip_ps/dt_ps))
    nfft = next_fast_len(measured.size+2*padding)
    window = slice(padding, padding+measured.size)
    measured_padded = np.zeros(nfft)
    measured_padded[window] = measured
    measured_spectrum = rfft(measured_padded)
    frequency = rfftfreq(nfft, dt_ps)
    weight = optical_band(frequency)
    supported = weight > 0
    transfer = np.zeros(frequency.size, dtype=complex)
    transfer[supported] = analytical_transmission(
        frequency[supported], thickness_A, group_delay_ps=transit_ps,
    )
    if (not np.all(np.isfinite(transfer[supported])) or np.min(abs(transfer[supported])) < 1e-6):
        raise ValueError("Analytical transmission is singular or too small to invert in the optical band.")
    incident_spectrum = np.zeros_like(measured_spectrum)
    # Divide the COMPLEX field spectrum, retaining both phase and amplitude.
    # Unsupported frequencies stay zero; avoid evaluating the lossless poles.
    incident_spectrum[supported] = measured_spectrum[supported]*weight[supported]/transfer[supported]
    incident_padded = irfft(incident_spectrum, nfft)
    incident = incident_padded[window].copy()
    bandlimited = irfft(measured_spectrum*weight, nfft)
    reconstructed = irfft(incident_spectrum*transfer, nfft)
    outside_energy = float((np.sum(incident_padded[:padding]**2) +
                            np.sum(incident_padded[padding+measured.size:]**2))/np.sum(incident_padded**2))
    if outside_energy > 1e-3:
        raise RuntimeError(f"Analytical incident field loses {outside_energy:.3%} energy outside the saved window.")
    input_hashes = {"input_field_sha256": sha256(field_path), "input_time_sha256": sha256(time_path)}
    comparison = _load_fdtd_comparison(output_dir, times_fs, trace_index, input_hashes)
    if comparison is not None and not np.isclose(
            comparison["metadata"]["group_transit_ps"], transit_ps, rtol=0, atol=1e-12):
        raise ValueError("FDTD and analytical corrections use different time origins.")
    output_dir.mkdir(parents=True, exist_ok=True)
    field_output = output_dir/"ref_field_corrected_analytical.txt"
    np.savetxt(field_output, incident, fmt="%.15e")
    metadata = {
        "method": "direct complex Fourier-spectrum division by analytical first-pass air/SiO2/air field transmission",
        "reference_trace_index": int(trace_index),
        "input_field": str(field_path.resolve()), "input_time": str(time_path.resolve()), **input_hashes,
        "corrected_field_sha256": sha256(field_output), "time_axis": str(time_path.resolve()),
        "field_units": "original experimental arbitrary units; no peak normalization",
        "time_units": "fs", "sample_count": int(incident.size), "sample_dt_ps": dt_ps,
        "baseline_subtracted": baseline, "baseline_edge_fraction": 0.05,
        "sample_thickness_um": thickness_A*1e-4, "group_transit_ps": transit_ps,
        "index_model": "n(omega)^2 = 1 + sum_j B_j*omega_j^2/(omega_j^2-omega^2); eps_inf=1; no carriers or damping",
        "background_del_eps": list(SIO2_BACKGROUND_DEL_EPS_DEFAULT),
        "background_lambda0_um": list(SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT),
        "index_reference": "https://doi.org/10.1364/JOSA.55.001205",
        "transmission_formula": "T_first=4*n/(1+n)^2*exp(-i*omega*n*d/C0); T_shape=T_first*exp(+i*omega*group_transit_ps)",
        "fft_convention": "forward exp(-i*omega*t); positive time delay has negative spectral phase",
        "inverse_formula": "E_incident=irfft(rfft(E_measured_padded)*optical_taper/T_shape)",
        "time_convention": "Original experimental axis retained; physical incident epoch shifted later by group_transit_ps, matching the FDTD comparison.",
        "internal_reflections": False,
        "finite_record_assumption": "First transmitted pass only; the ~125 fs internal echo is outside this 80 fs measurement.",
        "spectral_taper_THz": [40, 60, 320, 400], "fft_size": int(nfft), "padding_samples_before_record": padding,
        "analytic_roundtrip_relative_L2": relative_l2(reconstructed, bandlimited),
        "band_filter_relative_L2": relative_l2(bandlimited, measured_padded),
        "correction_outside_saved_window_energy_fraction": outside_energy,
        "corrected_to_measured_peak_ratio": float(np.max(abs(incident))/np.max(abs(measured))),
        "comparison_note": "Analytical transmission uses the continuum lossless refractive index; the finite-timestep FDTD material update can introduce numerical attenuation.",
    }
    if comparison is not None:
        metadata["analytical_vs_fdtd_relative_L2"] = relative_l2(incident, comparison["field"])
        metadata["compared_fdtd_field_sha256"] = comparison["metadata"]["corrected_field_sha256"]
    (output_dir/"ref_field_corrected_analytical_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    save_analytical_plot(output_dir, times_fs, measured, incident, frequency, transfer, weight, metadata, comparison)
    print(f"Analytical spectral round-trip relative L2: {metadata['analytic_roundtrip_relative_L2']:.3g}")
    if comparison is not None:
        print(f"Analytical vs FDTD incident relative L2: {metadata['analytical_vs_fdtd_relative_L2']:.3%}")
    print(f"Saved analytical field, metadata and comparison plot in {output_dir.resolve()}")
    return metadata


def save_analytical_plot(output_dir, times_fs, measured, incident, frequency, transfer, weight, metadata, comparison):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/lightmatter-correct-ref-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    stride = max(1, times_fs.size//6000)
    t = times_fs[::stride]
    axes[0].plot(t, measured[::stride], color="0.5", lw=1.3, label="Measured transmitted reference (baseline removed)")
    if comparison is not None:
        axes[0].plot(t, comparison["field"][::stride], color="tab:orange", lw=1.4, label="FDTD-corrected incident pulse")
    axes[0].plot(t, incident[::stride], "--", color="tab:blue", lw=1.5, label="Analytically corrected incident pulse")
    methods = "analytical and FDTD first-pass corrections" if comparison is not None else "analytical first-pass correction"
    axes[0].set(xlabel="Reference time (fs; common group-delay-compensated origin)", ylabel="Field (experimental arb. units)",
                title=f"Reference trace {metadata['reference_trace_index']}: {methods}\n"
                      f"12.82 µm SiO₂; same optical taper and {metadata['group_transit_ps']*1e3:.2f} fs group-transit compensation")
    axes[0].legend(fontsize=9, loc="upper right")
    if comparison is not None:
        difference = incident-comparison["field"]
        label = "Analytical − FDTD"
        title = f"Incident-waveform difference: relative L2 = {metadata['analytical_vs_fdtd_relative_L2']:.3%}"
    else:
        difference = incident-measured
        label, title = "Analytical − measured", "No compatible saved FDTD correction available"
    axes[1].plot(t, difference[::stride], color="tab:purple", label=label)
    axes[1].set(xlabel="Reference time (fs)", ylabel="Field difference (arb. units)", title=title)
    axes[1].legend()
    band = weight > 0
    axes[2].plot(frequency[band], abs(transfer[band]), color="tab:blue", label="Analytical first-pass |T| (continuum silica)")
    if comparison is not None and "transfer" in comparison:
        band_fdtd = comparison["taper"] > 0
        axes[2].plot(comparison["frequency_THz"][band_fdtd], abs(comparison["transfer"][band_fdtd]),
                     color="tab:orange", label="Calibrated FDTD first-pass |T|")
    axes[2].set(xlabel="Frequency (THz)", ylabel="Field transmission magnitude", xlim=(40, 400),
                title="Transmission amplitude used in the inverse; plotted before the common spectral taper")
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output_dir/"ref_field_correction_analytical.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-index", type=int, default=0, help="Zero-based column of ref_field.txt (default: 0).")
    parser.add_argument("--input", type=Path, default=directory/"ref_field.txt")
    parser.add_argument("--time-axis", type=Path, default=directory/"time_axis.txt", help="Experimental times in fs.")
    parser.add_argument("--output-dir", type=Path, default=directory)
    parser.add_argument("--method", choices=("fdtd", "analytical", "both"), default="both",
                        help="Correction method (default: both). Analytical mode runs no FDTD and uses separate output files.")
    args = parser.parse_args(argv)
    if args.method in ("fdtd", "both"):
        correct_reference(args.input, args.time_axis, args.output_dir, args.trace_index)
    if args.method in ("analytical", "both"):
        analytical_correct_reference(args.input, args.time_axis, args.output_dir, args.trace_index)


if __name__ == "__main__":
    main()
