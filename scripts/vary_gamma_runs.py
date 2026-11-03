from lightmatter.fdtd import run_simulation
from lightmatter.fdtd import MaterialParamsTD, PulseParams, FDTDParams
from lightmatter.io import save_simulation_run
import numpy as np
from lightmatter.units import C0, C0_SI, UNIT_F, UNIT_L, UNIT_T, UNIT_E


# Grid + time
L = 60 * 1e4 # A
dz= 10.0 # A
Nz = int(L/dz)
dt = 0.99 * dz / C0

# Source
src_pos = 5.0 * 1e4
src_i = int(np.round(src_pos/dz))

#------------------ very thin metal film, but at the same position----------------#
thicknesses = [10.0,20.0,40.0,80.0,160.0,320.0,640.0,1280.0,2560.0,80000.0] #

for thickness in thicknesses:
        
    metal_i0_pos = 50.0 * 1e4
    metal_cells = int(np.round(thickness / dz))
    metal_i0 = int(np.round( metal_i0_pos/ dz))
    metal_i1 = metal_i0 + metal_cells

    #probes, for thin film
    probe_r_pos = 10.0 * 1e4
    probe_t_pos = metal_i0_pos + 2.0e4
    probe_r_i = int(np.round(probe_r_pos/dz))
    probe_t_i = int(np.round(probe_t_pos/dz))

    #------------------Pulse parameters---------------#
    E0_SI = 0.1 * 1e10   #  E0 in V / m (0.1 V / A)
    E0 = E0_SI / UNIT_E  #  in code units
    # f0 = 370.0           #  THz
    tau = 0.2 * 0.5      #  200 fs
    t0 = 3*tau
    pulsetype = "smoothed box"
    # pulse = PulseParams(E0=E0, f0=f0, t0=t0, tau=tau, phase=0.0, type=pulsetype)

    T_estimate_reflection = (metal_i0_pos - src_pos + metal_i0_pos - probe_r_pos) / C0 + t0*2.0

    T = np.round(T_estimate_reflection + 0.005, 3) #ps
    Nt = int(T/dt)
    # num_snaps = 5000

    #------------------Simulation box parameters (with film)---------------#
    simulation_params_withfilm = FDTDParams(
        Nz=Nz,
        Nt=Nt,
        dz=dz,
        dt=dt,
        src_i=src_i,
        probe_t_i=probe_t_i,
        probe_r_i=probe_r_i,
        metal_i0=metal_i0,
        metal_i1=metal_i1,
        npml=150,
        snap_every= 0,  # 0 disables snapshots
    )

    #------------------Simulation box parameters (without film)---------------#
    simulation_params_nofilm = FDTDParams(
        Nz=Nz,
        Nt=Nt,
        dz=dz,
        dt=dt,
        src_i=src_i,
        probe_t_i=probe_t_i,
        probe_r_i=probe_r_i,
        metal_i0=metal_i0,
        metal_i1=metal_i0, # make metal thickness=0.0
        npml=150,
        snap_every= 0,   # 0 disables snapshots
    )

    #------------------Material parameters (time-dependent)---------------#
    def omega_D_smoothstep(t, omega0, delta, t0, deltat):
        step = 0.5 * (1.0 + np.tanh((t - t0) / deltat))
        return omega0 + delta * step

    def omega_D_smoothbump(t, omega0, delta, t0, deltat):
        bump = 1./np.cosh((t - t0) / deltat)
        return omega0 + delta * bump

    t_ax = np.linspace(0.0,T,Nt)
    omega_D0 = 2 * np.pi * 2113.6
    delta_omega = 0.2 * omega_D0 # 20% increase
    timestep_transition = 1 * 1e-3 # 20 fs
    # omega_D_timedep = omega_D_smoothstep(t_ax, omega_D0, delta_omega, 0.45, timestep_transition)
    omega_D_const = np.ones(Nt) * omega_D0

    gamma_D0 = 2.0 * np.pi * 15.92
    delta_gamma = 0.2 * gamma_D0 # also 20% increase
    gamma_D_timedep = omega_D_smoothstep(t_ax, gamma_D0, delta_gamma, 0.45, timestep_transition)
    gamma_D_const = np.ones(Nt) * gamma_D0

    # material = MaterialParamsTD(
    #     omega_D=omega_D_const, 
    #     gamma_D=gamma_D_const,
    #     omega_L=np.ones(Nt) * 2.0*np.pi*650.07, 
    #     gamma_L=np.ones(Nt) * 2.0*np.pi*104.86,
    #     del_eps=np.ones(Nt) * 1.09 * 0.0,        #lorentz off
    #     eps_inf=np.ones(Nt) * 5.9673,
    # )

    freq_list1 = np.arange(500.0,900.0,100.0)
    freq_list2 = np.arange(900.0,1000.0,25.0) # ------------  since the resonance position does not shift, the higer frequency resolution is not required upto 1200.0 THz
    freq_list3 = np.arange(1000.0,1600.1,100.0)
    freqs = np.concatenate((freq_list1,freq_list2,freq_list3))

    for freq in freqs:
        pulse = PulseParams(E0=E0, f0=freq, t0=t0, tau=tau, phase=0.0, type=pulsetype)

        # filename vars
        dz_filename = int(dz)
        root_dirname = f"tests_varygamma_17-07-2026_t_{thickness}_transition_{timestep_transition * 1e3:.1f}"
        print(f"test vary gamma with smooth step - 17-07-2026 t {thickness}, dz {dz}")
        
        #------------------constant parameters------------------#
        dirname = f"{root_dirname}/constant/f0_{freq}_step_dz_{dz_filename}_dt_flat"
        material = MaterialParamsTD(
            omega_D=omega_D_const, 
            gamma_D=gamma_D_const,
            omega_L=np.ones(Nt) * 2.0*np.pi*650.07, 
            gamma_L=np.ones(Nt) * 2.0*np.pi*104.86,
            del_eps=np.ones(Nt) * 1.09 * 0.0,        #lorentz off
            eps_inf=np.ones(Nt) * 5.9673,
        )

        E_t, E_r, snaps_t, snaps_E = run_simulation(simulation_params_withfilm, pulse, material, method="exact pml")

        save_simulation_run(
            folder=f"../output/{dirname}/with_film",
            params=simulation_params_withfilm, 
            pulse=pulse,
            material=material,
            E_t=E_t,
            E_r=E_r,
            snaps_t=snaps_t,
            snaps_E=snaps_E,
        )

        E_t, E_r, snaps_t, snaps_E = run_simulation(simulation_params_nofilm, pulse, material, method="exact pml")

        save_simulation_run(
            folder=f"../output/{dirname}/nofilm",
            params=simulation_params_nofilm,
            pulse=pulse,
            material=material,
            E_t=E_t,
            E_r=E_r,
            snaps_t=snaps_t,
            snaps_E=snaps_E,
        )

        #------------------time-varying parameters------------------#
        dirname = f"{root_dirname}/vary/f0_{freq}_step_dz_{dz_filename}_dt_flat"
        material = MaterialParamsTD(
            omega_D=omega_D_const, 
            gamma_D=gamma_D_timedep,#    ----------------------   varying gamma -------- #####
            omega_L=np.ones(Nt) * 2.0*np.pi*650.07, 
            gamma_L=np.ones(Nt) * 2.0*np.pi*104.86,
            del_eps=np.ones(Nt) * 1.09 * 0.0,        #lorentz off
            eps_inf=np.ones(Nt) * 5.9673,
        )

        E_t, E_r, snaps_t, snaps_E = run_simulation(simulation_params_withfilm, pulse, material, method="exact pml")

        save_simulation_run(
            folder=f"../output/{dirname}/with_film",
            params=simulation_params_withfilm, 
            pulse=pulse,
            material=material,
            E_t=E_t,
            E_r=E_r,
            snaps_t=snaps_t,
            snaps_E=snaps_E,
        )

        E_t, E_r, snaps_t, snaps_E = run_simulation(simulation_params_nofilm, pulse, material, method="exact pml")

        save_simulation_run(
            folder=f"../output/{dirname}/nofilm",
            params=simulation_params_nofilm,
            pulse=pulse,
            material=material,
            E_t=E_t,
            E_r=E_r,
            snaps_t=snaps_t,
            snaps_E=snaps_E,
        )
