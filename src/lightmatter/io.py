from __future__ import annotations
from .fdtd import MaterialParamsTD,PulseParams,FDTDParams
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import json
from datetime import datetime
from dataclasses import asdict


def make_fdtd_gif(
    gif_path: str,
    dz: float,
    snaps_t: np.ndarray,
    snaps_E: np.ndarray,
    metal_i0: int,
    metal_i1: int,
    every_frame: int = 1,
    fps: int = 20,
    dpi: int = 100,
):
    """
    Create a GIF of E(z,t) snapshots using matplotlib FuncAnimation.

    Parameters
    ----------
    gif_path : str
        Output GIF path.
    dz : float
        Spatial step (m).
    snaps_t : (Nsnap,) array
        Snapshot times.
    snaps_E : (Nsnap, Nz) array
        Snapshot electric fields.
    metal_i0, metal_i1 : int
        Metal region indices [i0, i1).
    every_frame : int
        Use every Nth snapshot in the animation.
    fps : int
        GIF frame rate.
    dpi : int
        Output resolution.
    """
    snaps_t = np.asarray(snaps_t)
    snaps_E = np.asarray(snaps_E)

    if snaps_E.ndim != 2:
        raise ValueError("snaps_E must be 2D with shape (Nsnap, Nz).")
    if snaps_t.shape[0] != snaps_E.shape[0]:
        raise ValueError("snaps_t and snaps_E must have the same length.")

    # Subsample frames if requested
    frame_idx = np.arange(0, snaps_t.shape[0], every_frame, dtype=int)
    if frame_idx.size == 0:
        raise ValueError("No frames selected. Check every_frame and snapshot arrays.")

    Nz = snaps_E.shape[1]
    z_um = np.arange(Nz) * dz * 1e-4 # change units to microns
    dz_um = dz * 1e-4

    # Fixed y-limits for stable visualization
    Emax = float(np.max(np.abs(snaps_E[frame_idx, :])))
    if Emax == 0.0:
        Emax = 1.0

    fig, ax = plt.subplots()
    ax.set_xlabel("z (um)")
    ax.set_ylabel("E (code units)")
    ax.set_ylim(-1.05 * Emax, 1.05 * Emax)

    # Shade metal region once (static artists)
    ax.axvspan(metal_i0 * dz_um, metal_i1 * dz_um, alpha=0.2)
    metal_xc = 0.5 * (metal_i0 + metal_i1) * dz_um
    metal_label = ax.text(metal_xc, 0.9 * Emax, "metal", ha="center", va="center")

    # Line artist updated each frame
    (line,) = ax.plot(z_um, snaps_E[frame_idx[0], :], lw=1.5)

    title = ax.set_title(f"Electric field, t = {snaps_t[frame_idx[0]]:.3e} ps")

    def init():
        line.set_ydata(snaps_E[frame_idx[0], :])
        title.set_text(f"Electric field, t = {snaps_t[frame_idx[0]]:.3e} ps")
        return line, title, metal_label

    def update(frame_number: int):
        k = frame_idx[frame_number]
        line.set_ydata(snaps_E[k, :])
        title.set_text(f"Electric field, t = {snaps_t[k]:.3e} ps")
        return line, title, metal_label

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_idx.size,
        init_func=init,
        blit=True,
        interval=1000.0 / fps,
    )

    anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved GIF to: {gif_path}")


def plot_fdtd_frame(
    dz: float,
    frame_num: int,
    snaps_t: np.ndarray,
    snaps_E: np.ndarray,
    metal_i0: int,
    metal_i1: int,
    every_frame: int = 1,

):

    snaps_t = np.asarray(snaps_t)
    snaps_E = np.asarray(snaps_E)

    if snaps_E.ndim != 2:
        raise ValueError("snaps_E must be 2D with shape (Nsnap, Nz).")
    if snaps_t.shape[0] != snaps_E.shape[0]:
        raise ValueError("snaps_t and snaps_E must have the same length.")

    # Subsample frames if requested
    frame_idx = np.arange(0, snaps_t.shape[0], every_frame, dtype=int)
    if frame_idx.size == 0:
        raise ValueError("No frames selected. Check every_frame and snapshot arrays.")

    Nz = snaps_E.shape[1]
    z_um = np.arange(Nz) * dz * 1e-4 # change units to microns
    dz_um = dz * 1e-4

    # Fixed y-limits for stable visualization
    Emax = float(np.max(np.abs(snaps_E[frame_idx, :])))
    if Emax == 0.0:
        Emax = 1.0

    fig, ax = plt.subplots(figsize=(13,8))
    ax.set_xlabel("z (um)")
    ax.set_ylabel("E (code units)")
    ax.set_ylim(-1.05 * Emax, 1.05 * Emax)

    # Shade metal region once (static artists)
    ax.axvspan(metal_i0 * dz_um, metal_i1 * dz_um, alpha=0.2)
    metal_xc = 0.5 * (metal_i0 + metal_i1) * dz_um
    metal_label = ax.text(metal_xc, 0.9 * Emax, "metal", ha="center", va="center")

    # Line artist updated each frame
    (line,) = ax.plot(z_um, snaps_E[frame_idx[frame_num], :], lw=1.5)

    title = ax.set_title(f"Electric field, t = {snaps_t[frame_idx[frame_num]]:.3e} ps")

    plt.show()


def save_simulation_run(
    folder: str,
    params,              # FDTDParams
    pulse,               # PulseParams
    material,            # MaterialParamsTD
    E_t: np.ndarray,
    E_r: np.ndarray,
    snaps_t: np.ndarray,
    snaps_E: np.ndarray,
):
    """
    Save simulation results and metadata in a structured format.

    Parameters
    ----------
    folder : str
        Output directory.
    params : FDTDParams
        Full simulation configuration.
    pulse : PulseParams
        Source pulse parameters.
    material : MaterialParamsTD
        Time-dependent material parameters.
    E_t, E_r : np.ndarray
        Transmission and reflection probe time series.
    snaps_t : np.ndarray
        Snapshot times.
    snaps_E : np.ndarray
        Snapshot electric fields.
    """

    os.makedirs(folder, exist_ok=True)

    # -----------------------
    # Metadata (JSON)
    # -----------------------

    info = {
        "date": datetime.now().isoformat(),
        "fdtd_params": asdict(params),
        "pulse_params": asdict(pulse),
        "material_info": {
            "type": "time_dependent",
            "Nt": len(material.omega_D),
        }
    }

    with open(os.path.join(folder, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # -----------------------
    # Numerical Data (.npz)
    # -----------------------

    np.savez_compressed(
        os.path.join(folder, "data.npz"),
        E_t=E_t,
        E_r=E_r,
        snaps_t=snaps_t,
        snaps_E=snaps_E,
        omega_D=material.omega_D,
        gamma_D=material.gamma_D,
        omega_L=material.omega_L,
        gamma_L=material.gamma_L,
        del_eps=material.del_eps,
        eps_inf=material.eps_inf,
    )

    print(f"Simulation saved to: {folder}")


def load_simulation_run(folder: str, nosnaps = False):
    """
    Load simulation results and reconstruct parameter classes.

    Returns
    -------
    dict containing:
        params : FDTDParams
        pulse : PulseParams
        material : MaterialParamsTD
        E_t, E_r, snaps_t, snaps_E
    """

    # -----------------------
    # Load metadata
    # -----------------------

    with open(os.path.join(folder, "info.json"), "r") as f:
        info = json.load(f)

    # Reconstruct parameter classes
    params = FDTDParams(**info["fdtd_params"])
    pulse = PulseParams(**info["pulse_params"])

    # -----------------------
    # Load numerical arrays
    # -----------------------

    data = np.load(os.path.join(folder, "data.npz"))

    material = MaterialParamsTD(
        omega_D=data["omega_D"],
        gamma_D=data["gamma_D"],
        omega_L=data["omega_L"],
        gamma_L=data["gamma_L"],
        del_eps=data["del_eps"],
        eps_inf=data["eps_inf"],
    )

    if nosnaps:
        result = {
            "params": params,
            "pulse": pulse,
            "material": material,
            "E_t": data["E_t"],
            "E_r": data["E_r"],
        }
    else:
        result = {
            "params": params,
            "pulse": pulse,
            "material": material,
            "E_t": data["E_t"],
            "E_r": data["E_r"],
            "snaps_t": data["snaps_t"],
            "snaps_E": data["snaps_E"],
        }

    return result