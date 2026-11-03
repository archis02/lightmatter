"""Regression checks for local ADE materials and time-resolved material probes.

Run with PYTHONPATH=src NUMBA_NUM_THREADS=2 python -m pytest tests/.
"""
from dataclasses import replace

import numpy as np
import pytest

from lightmatter import fdtd_revised_sio2 as fdtd


NAMES = ("omega_D", "gamma_D", "omega_L", "gamma_L", "del_eps", "eps_inf")
METHODS = ("exact pml", "sio2_background")


@pytest.fixture
def setup():
    params = fdtd.FDTDParams(
        Nz=64, Nt=120, dz=20.0, dt=0.7 * 20 / fdtd.C0,
        src_i=12, probe_r_i=30, probe_t_i=38,
        metal_i0=28, metal_i1=40, npml=8, snap_every=3,
    )
    pulse = fdtd.PulseParams(1.0, 360.0, 15 * params.dt, 5 * params.dt)
    time_index = np.arange(params.Nt)
    material = fdtd.MaterialParamsTD(
        omega_D=1000 + 3 * time_index, gamma_D=200 + time_index,
        omega_L=2000 + 5 * time_index, gamma_L=100 + time_index,
        del_eps=0.2 + 0.001 * time_index, eps_inf=1 + 0.001 * time_index,
    )
    return params, pulse, material


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("spatial_columns", [1, 12])
def test_broadcast_matches_uniform_and_probes_match_snapshots(setup, method, spatial_columns):
    params, pulse, material = setup
    uniform = fdtd.run_simulation(params, pulse, material, method=method)
    spatial = fdtd.MaterialParamsTD(**{
        name: np.broadcast_to(getattr(material, name)[:, None], (params.Nt, spatial_columns))
        for name in NAMES
    })
    result = fdtd.run_simulation(params, pulse, spatial, method=method, material_probe_every=1)
    assert len(uniform) == 4
    assert len(result) == 5
    for old, new in zip(uniform, result[:4]):
        np.testing.assert_allclose(old, new, atol=1e-14, rtol=1e-13)
    probes = result[4]
    assert probes["E"].shape == (params.Nt, 12)
    np.testing.assert_array_equal(probes["indices"], np.arange(28, 40))
    np.testing.assert_array_equal(probes["t"], np.arange(params.Nt) * params.dt)
    np.testing.assert_array_equal(probes["z"], np.arange(28, 40) * params.dz)
    np.testing.assert_array_equal(probes["E"][::3], result[3][:, 28:40])
    np.testing.assert_array_equal(probes["E"][:, 2], result[1])
    np.testing.assert_array_equal(probes["E"][:, 10], result[0])


@pytest.mark.parametrize("method", METHODS)
def test_strided_and_explicit_probes_without_snapshots(setup, method):
    params, pulse, material = setup
    params = replace(params, snap_every=0)
    all_probes = fdtd.run_simulation(params, pulse, material, method=method, material_probe_every=1)[4]
    stride = fdtd.run_simulation(params, pulse, material, method=method, material_probe_every=5)[4]
    np.testing.assert_array_equal(stride["indices"], [28, 33, 38])
    np.testing.assert_array_equal(stride["E"], all_probes["E"][:, ::5])
    selected = fdtd.run_simulation(params, pulse, material, method=method, material_probe_indices=[39, 28, 33])[4]
    np.testing.assert_array_equal(selected["E"], all_probes["E"][:, [11, 0, 5]])
    empty = fdtd.run_simulation(params, pulse, material, method=method, material_probe_indices=[])[4]
    assert empty["E"].shape == (params.Nt, 0)


def isolated_run(material, method):
    """Turn off inter-cell coupling to check local parameter indexing independently."""
    nt = material.omega_D.shape[0]
    fn = fdtd.run_fdtd_pml_corrected if method == "exact pml" else fdtd.run_fdtd_sio2_background
    extra = {} if method == "exact pml" else dict(
        background_del_eps=[0.0], background_omega0=[1.0], background_gamma=[0.0],
    )
    return fn(
        12, nt, 20., 1e-5, 5, 5, 5, 2, 9,
        *(getattr(material, name) for name in NAMES),
        1., 1., 0., 1., 0., "explicit",
        np.zeros(12), np.ones(11), np.zeros(11),
        snap_every=1, explicit_pulse_waveform=np.r_[1., np.zeros(nt - 1)],
        material_probe_every=1, **extra,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", NAMES)
def test_each_parameter_uses_local_time_series(name, method):
    nt, nm, source_local_index = 20, 7, 3
    base = fdtd.MaterialParamsTD.from_static(
        Nt=nt, omega_D=8000., gamma_D=2000., omega_L=12000.,
        gamma_L=4000., del_eps=0.5, eps_inf=1.2,
    )
    # Vary both axes, mixing this spatial parameter with five uniform ones.
    values = getattr(base, name)[:, None] * (
        1 + np.arange(nt)[:, None] / nt + np.arange(nm)[None, :] / nm
    )
    actual = isolated_run(replace(base, **{name: values}), method)
    expected = isolated_run(replace(base, **{name: values[:, source_local_index]}), method)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-14, rtol=1e-13)
    # A spatially ignored parameter must fail this check.
    other_cell = isolated_run(replace(base, **{name: values[:, 0]}), method)
    assert not np.allclose(actual[0], other_cell[0], atol=1e-12, rtol=1e-10)
    np.testing.assert_array_equal(actual[4]["E"][:, source_local_index], actual[0])


@pytest.mark.parametrize("method", METHODS)
def test_time_varying_eps_conserves_displacement_and_holds_last_sample(method):
    nt = 10
    eps = 1 + np.arange(nt)[:, None] * np.arange(1, 8)[None, :] / nt
    material = fdtd.MaterialParamsTD.from_static(
        Nt=nt, omega_D=0., gamma_D=0., omega_L=0., gamma_L=0., del_eps=0., eps_inf=1.,
    )
    result = isolated_run(replace(material, eps_inf=eps), method)
    # Initial impulse is injected after update 0; subsequent epsilon*E stays fixed.
    expected = eps[1, 3] / eps[np.minimum(np.arange(nt) + 1, nt - 1), 3]
    np.testing.assert_allclose(result[0], expected, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [(119,), (120, 64), (120, 11), (120, 0)])
def test_wrappers_reject_wrong_material_grid(setup, method, shape):
    params, pulse, _ = setup
    material = fdtd.MaterialParamsTD(**{name: np.ones(shape) for name in NAMES})
    with pytest.raises(ValueError, match="expected"):
        fdtd.run_simulation(params, pulse, material, method=method)


@pytest.mark.parametrize("bad", [1., np.ones((2, 3, 4)), [], [1., np.nan], [1., np.inf], [1 + 1j, 2]])
def test_material_rejects_invalid_arrays(bad):
    with pytest.raises(ValueError):
        fdtd.MaterialParamsTD(**{name: bad for name in NAMES})


def test_material_rejects_mismatched_axes_and_nonpositive_eps(setup):
    _, _, material = setup
    for updates in (
        {"gamma_D": np.ones(119)},
        {"omega_D": np.ones((120, 12)), "gamma_D": np.ones((120, 13))},
        {"eps_inf": np.zeros(120)},
    ):
        with pytest.raises(ValueError):
            replace(material, **updates)


@pytest.mark.parametrize("options", [
    {"material_probe_every": 0}, {"material_probe_every": -1},
    {"material_probe_every": 1.5}, {"material_probe_every": True},
    {"material_probe_indices": [27]}, {"material_probe_indices": [40]},
    {"material_probe_indices": [28., 29.]}, {"material_probe_indices": [[28]]},
    {"material_probe_indices": [28, 28]}, {"material_probe_indices": [True]},
    {"material_probe_every": 1, "material_probe_indices": [28]},
])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_probes_fail_before_kernel(setup, options, method):
    params, pulse, material = setup
    with pytest.raises(ValueError):
        fdtd.run_simulation(params, pulse, material, method=method, **options)


@pytest.mark.parametrize("method", METHODS)
def test_empty_material_region(setup, method):
    params, pulse, material = setup
    params = replace(params, metal_i1=params.metal_i0)
    result = fdtd.run_simulation(params, pulse, material, method=method, material_probe_every=1)
    assert result[4]["E"].shape == (params.Nt, 0)
    assert np.all(np.isfinite(result[0]))


def test_removed_exact_and_unsupported_convolution_options(setup):
    params, pulse, material = setup
    assert not hasattr(fdtd, "run_fdtd_exact")
    assert not hasattr(fdtd, "_run_fdtd_exact_kernel")
    with pytest.raises(ValueError, match="Unknown method"):
        fdtd.run_simulation(params, pulse, material, method="exact")
    with pytest.raises(ValueError, match="Material probes require"):
        fdtd.run_simulation(params, pulse, material, material_probe_every=1)
    material = replace(material, eps_inf=material.eps_inf[:, None])
    with pytest.raises(ValueError, match="convolution solver only supports"):
        fdtd.run_simulation(params, pulse, material)
