import numpy as np
import matplotlib.pyplot as plt


def epsilon_analytical_drude_lorentz(
        eps_inf = 5.9673,
        wD_over_2pi_THz = 2113.6,
        gD_over_2pi_THz = 15.92,
        OL_over_2pi_THz = 650.07,
        GL_over_2pi_THz = 104.86,
        Delta_eps = 1.09,
        plot = False,
        convert_to_photon=True):

    THz = 1e12
    wD = 2 * np.pi * (wD_over_2pi_THz * THz)
    gD = 2 * np.pi * (gD_over_2pi_THz * THz)
    OL = 2 * np.pi * (OL_over_2pi_THz * THz)
    GL = 2 * np.pi * (GL_over_2pi_THz * THz)

    # Frequency axis
    f_THz_min, f_THz_max = 1.0, 3000.0  # avoid w=0 singularity
    N = 6000
    f_THz = np.linspace(f_THz_min, f_THz_max, N)
    w = 2 * np.pi * (f_THz * THz)

    eps_DL = (
        eps_inf
        - (wD**2) / (w * (w + 1j * gD))
        - (Delta_eps * OL**2) / ((w**2 - OL**2) + 1j * GL * w)
    )

    # --- Plot real and imaginary parts ---
    if plot:
        plt.figure()
        if convert_to_photon:
            phot_ev = w/(2*np.pi) * 4.136e-15
            plt.plot(phot_ev, np.real(eps_DL), label=r"Re{$\epsilon(\omega)$}")
            plt.plot(phot_ev, np.imag(eps_DL), label=r"Im{$\epsilon(\omega)$}")
            plt.xlabel("Energy (eV)")
            plt.xlim((1.24,2.48))
            plt.ylim((-40,10))
        else:
            plt.plot(f_THz, np.real(eps_DL), label=r"Re{$\epsilon(\omega)$}")
            plt.plot(f_THz, np.imag(eps_DL), label=r"Im{$\epsilon(\omega)$}")
            plt.xlabel("Frequency [THz]")

        plt.ylabel("Dielectric constant")
        plt.title("Extended Drude–Lorentz model (Eq. 3)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if convert_to_photon:
        phot_ev = w/(2*np.pi) * 4.136e-15
        return phot_ev, eps_DL

    return f_THz,eps_DL