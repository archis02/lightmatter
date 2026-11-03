"""Exercise the corrected single case and real analyses without running FDTD.

Only synthetic compact traces are produced, exclusively under pytest's tmp_path.
The notebook's final invocation is removed before its definitions are loaded.
Run in the notebook's spacelab environment, from the workspace root:
    python -m pytest -q lightmatter/tests/test_single_lobe_matching_notebook.py
"""

import ast
import builtins
import dis
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/project_2/simulate_experimental_setup.ipynb"
GAMMAS = [250.0]
DELTA = 130.0
DELAYS_FS = np.array([-10.0, 0.0, 10.0])
CORRECTED_PEAK_FS = 29.4
SIMULATION_STAGES = ("helpers", "geometry", "cache_and_duration", "persistence", "controls", "delay_scan",
                     "timing_diagnostic", "geometry_plot")
RUN_FIGURES = (
    "transmitted_energy_vs_peak_delay.png",
    "transmitted_waveforms.png",
    "selected_lobe_previews.png",
    "lobe_energy_ratio_vs_delay.png",
    "selected_peak_time_vs_delay.png",
)
COMPARISON_FIGURES = (
    "reference_alignment.png",
    "experimental_simulated_energy_overplots.png",
    "experimental_simulated_lobe_timing.png",
)


@pytest.fixture
def runner(monkeypatch):
    """Load definitions and compile source cells, forbidding the solver import."""
    original_import = builtins.__import__

    def no_solver_import(name, *args, **kwargs):
        if name.startswith("lightmatter.fdtd"):
            raise AssertionError("These tests must never import an FDTD solver.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_solver_import)
    monkeypatch.chdir(NOTEBOOK.parents[2])
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][-1]["source"])
    tree = ast.parse(source)
    invocation = tree.body.pop()
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "run_corrected_single_case"
    assert invocation.value.args == []
    assert invocation.value.keywords == []
    namespace = {"__name__": "__notebook_test__"}
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    yield namespace
    plt.close("all")


def _literal_config(runner, name):
    node = runner["_matching_assignment"](runner["matching_setup_tree"], name)
    return ast.literal_eval(node.value)


def _synthetic_run_source(run_directory, pump_directory, gamma, complete, provenance, settings,
                          correction_metadata, experiment_dir):
    """Replace every producer stage; retain the real compact-file contract."""
    return textwrap.dedent(
        f"""
        from pathlib import Path
        import json
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import trapezoid, cumulative_trapezoid

        def tqdm(values, **kwargs):
            return values

        def load_probe_run(folder):
            with np.load(Path(folder) / "data.npz", allow_pickle=False) as data:
                arrays = {{key: data[key] for key in data.files}}
            return arrays, json.loads((Path(folder) / "info.json").read_text())

        UNIT_T = 1e-12
        base_dir = Path({str(run_directory)!r})
        base_dir.mkdir(parents=True)
        with (base_dir.parent / "synthetic_producer_calls.txt").open("a") as stream:
            stream.write({str(gamma)!r} + "\\n")
        time_fs = np.arange(1001) * 0.1
        t_ps = time_fs * 1e-3
        reference = np.exp(-((time_fs - 50.0) / 6.0)**2) * np.cos(2*np.pi*(time_fs - 50.0)/6.0)
        source_directory = Path({str(experiment_dir)!r})
        source_time_fs = np.loadtxt(source_directory / {provenance['time_filename']!r}, delimiter=",")
        waveform_time_rel_ps = (source_time_fs - source_time_fs[0]) * 1e-3
        reference_field = np.loadtxt(source_directory / {provenance['field_filename']!r}, delimiter=",")
        # The real producer peak-normalizes its corrected source without a
        # second baseline subtraction; transmitted references remain raw-like.
        reference_field *= {settings['probe_source_peak_E0']!r} / np.max(np.abs(reference_field))
        probe_incident_peak_rel_ps = waveform_time_rel_ps[np.argmax(abs(reference_field))]
        fdtd_params = dict(Nt=time_fs.size, dt=0.0001)
        pulse_params = {{"synthetic": True}}
        common = dict(fdtd_params=fdtd_params, pulse_params=pulse_params,
                      gamma_D_per_ps={gamma!r}, del_omega_D_rad_per_ps={DELTA!r}, omega_D0_rad_per_ps=0.0,
                      last_window_transmitted_energy_fraction=0.0)

        def save(folder, field, material_state, delay_fs=0.0):
            directory = base_dir / folder
            directory.mkdir()
            np.savez_compressed(directory / "data.npz", t_ps=t_ps, E_t=field, E_r=np.zeros_like(field))
            info = dict(common, material_state=material_state, delay_fs=delay_fs)
            (directory / "info.json").write_text(json.dumps(info))

        save("unpumped_reference", reference, "unpumped")
        save("fully_injected_reference", 1.3*reference, "fully_injected")
        rows = []
        for index, delay in enumerate((-10.0, 0.0, 10.0)):
            folder = f"delay_{{index:04d}}"
            scale = 1.0 + ({gamma!r} - 100.0)/1000.0 + 0.05*index
            save(folder, scale*reference, "pump_driven", delay)
            rows.append(dict(index=index, delay_fs=delay, folder=folder))
        manifest = dict(complete={complete!r}, completed_indices=[0, 1, 2] if {complete!r} else [0, 1], runs=rows)
        (base_dir / "scan_manifest.json").write_text(json.dumps(manifest))
        config = dict(common, experimental_trace_index=0, shared_axes="shared_axes.npz",
                      shared_profile="shared_profile.npy", probe_source_provenance={provenance!r},
                      probe_source_correction_metadata={correction_metadata!r},
                      probe_source_baseline_subtracted_in_notebook=False,
                      pump_run_directory={str(pump_directory)!r},
                      delay_definition="probe incident abs(E) peak minus pump incident abs(E) peak at the sample front; positive=pump first")
        config.update({settings!r})
        (base_dir / "scan_configuration.json").write_text(json.dumps(config))
        np.savez_compressed(base_dir / "shared_axes.npz", probe_source_t_rel_ps=waveform_time_rel_ps,
                            probe_source_E=reference_field)
        np.save(base_dir / "shared_profile.npy", np.zeros((2, 2)))
        """
    )


def _configure_synthetic_run(runner, tmp_path):
    output = tmp_path / runner["matching_output_dir"].name
    pump_directory = tmp_path / "saved_pump"
    runner.update(
        matching_output_dir=output,
        matching_pump_dir=pump_directory,
        matching_figure_dir=output / "figures",
        matching_comparison_dir=output / "experimental_simulation_comparison_sqrt_E8",
    )
    pump_directory.mkdir()
    for filename in ("info.json", "pump_metadata.json", "pump_material_probes.npz", "pump_propagation.png"):
        (pump_directory / filename).write_bytes(b"synthetic pump placeholder; no FDTD")
    # Pump compatibility is tested separately with the real saved metadata and
    # compact archive headers. Synthetic producers never consume pump fields.
    runner["_matching_validate_pump_inputs"] = lambda: None
    experiment_dir = tmp_path / "experimental"
    experiment_dir.mkdir()
    time_fs = np.arange(671) * 0.1
    field = np.exp(-((time_fs - 33.5) / 6.0)**2) * np.cos(2*np.pi*(time_fs - 33.5)/6.0)
    references = field[:, None] * np.array([1.0, 2.0, 0.5])[None, :]
    modified = references * np.array([1.0, 0.8, 1.2])[None, :]
    for filename, array in (
        ("time_axis.txt", time_fs), ("ref_field.txt", references),
        ("mod_field.txt", modified), ("inj_time.txt", [43.5, 33.5, 23.5]),
    ):
        np.savetxt(experiment_dir / filename, array, delimiter=",")
    # Corrected incident and measured transmitted pulses must differ. A distinct
    # sampling grid also catches comparisons against raw data before shape checks.
    corrected_time_fs = np.arange(801) * 0.1 - 6.5
    corrected_field = (np.exp(-((corrected_time_fs - CORRECTED_PEAK_FS) / 5.0)**2)
                       * np.cos(2*np.pi*(corrected_time_fs - CORRECTED_PEAK_FS)/5.5) + 0.04)
    np.savetxt(experiment_dir / "time_axis_corrected.txt", corrected_time_fs, delimiter=",")
    np.savetxt(experiment_dir / "ref_field_corrected.txt", corrected_field, delimiter=",")
    correction_metadata = {
        "method": "synthetic inverse-propagation fixture; no FDTD",
        "reference_trace_index": 0,
        "time_units": "fs",
        "input_field": str(experiment_dir / "ref_field.txt"),
        "input_time": str(experiment_dir / "time_axis.txt"),
        "sample_count": int(corrected_time_fs.size),
    }
    for key, filename in (
        ("input_field_sha256", "ref_field.txt"), ("input_time_sha256", "time_axis.txt"),
        ("corrected_field_sha256", "ref_field_corrected.txt"),
        ("corrected_time_sha256", "time_axis_corrected.txt"),
    ):
        correction_metadata[key] = hashlib.sha256((experiment_dir / filename).read_bytes()).hexdigest()
    (experiment_dir / "ref_field_corrected_metadata.json").write_text(json.dumps(correction_metadata))
    actual_provenance_loader = runner["load_corrected_probe_provenance"]
    runner["load_corrected_probe_provenance"] = lambda _: actual_provenance_loader(experiment_dir)
    runner["matching_source_metadata"], runner["matching_source_provenance"] = actual_provenance_loader(experiment_dir)
    runner["synthetic_experiment_dir"] = experiment_dir
    runner["matching_experiment_code"]["paths"] = compile(
        f"from pathlib import Path\nimport json\nimport numpy as np\n"
        f"EXPERIMENTAL_DATA_DIR = Path({str(experiment_dir)!r})\n"
        f"SIMULATION_OUTPUT_ROOT = Path({str(output)!r})\n",
        "synthetic-experimental-paths", "exec",
    )
    for stage in SIMULATION_STAGES:
        runner["matching_simulation_code"][stage] = compile("pass", f"disabled-{stage}", "exec")
    for stage in ("pump_setup", "pump_run", "pump_plot"):
        runner["matching_simulation_code"][stage] = compile(
            'raise AssertionError("The single case must reuse the saved pump.")', stage, "exec",
        )
    for gamma in runner["matching_gammas"]:
        runner["matching_probe_code"][gamma] = compile(
            _synthetic_run_source(runner["_matching_scan_directory"](gamma), pump_directory,
                                  gamma, True, runner["matching_source_provenance"],
                                  runner["_matching_expected_scan_settings"](),
                                  runner["matching_source_metadata"], experiment_dir),
            "synthetic-compact-traces", "exec",
        )
    return output


def _file_snapshot(root):
    """Contents and modification times detect any preflight mutation."""
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob("*") if path.is_file()}


def _make_saved_scan(runner, *, complete=True):
    directory = runner["_matching_scan_directory"](GAMMAS[0])
    source = _synthetic_run_source(directory, runner["matching_pump_dir"], GAMMAS[0], complete,
                                   runner["matching_source_provenance"], runner["_matching_expected_scan_settings"](),
                                   runner["matching_source_metadata"], runner["synthetic_experiment_dir"])
    exec(compile(source, "synthetic-saved-scan", "exec"), {})
    return directory


def _comparison_namespace(runner):
    """Load source-validation helpers only; never run analysis or a producer."""
    namespace = {"__name__": "__test_comparison__"}
    exec(runner["matching_experiment_code"]["paths"], namespace)
    for variable, filename in (("time_ax", "time_axis.txt"), ("ref_waveforms", "ref_field.txt"),
                               ("modified_waveforms", "mod_field.txt"), ("injection_times", "inj_time.txt")):
        namespace[variable] = np.genfromtxt(namespace["EXPERIMENTAL_DATA_DIR"] / filename, delimiter=",")
    for stage in ("validation", "plot_helpers", "pulse_helpers", "lobe_helpers", "settings", "units",
                  "comparison_settings", "comparison_helpers"):
        exec(runner["matching_experiment_code"][stage], namespace)
    return namespace


def test_corrected_source_uses_incident_clock_without_a_second_baseline(runner, tmp_path):
    _configure_synthetic_run(runner, tmp_path)
    directory = _make_saved_scan(runner)
    namespace = _comparison_namespace(runner)
    config = json.loads((directory / "scan_configuration.json").read_text())
    before = _file_snapshot(tmp_path)
    column, peak_fs = namespace["_simulation_source_peak"](directory, config)
    assert column == 0
    assert peak_fs == pytest.approx(CORRECTED_PEAK_FS)
    assert _file_snapshot(tmp_path) == before
    # Transmission alignment still compares transmitted output to the measured
    # reference, not to the deconvolved incident source.
    sim_time, sim_field, _ = namespace["_load_simulation_trace"](directory / "unpumped_reference")
    shift, score = namespace["_simulation_reference_shift"](
        sim_time, sim_field, namespace["time_ax"] * 1e-15, namespace["ref_waveforms"][:, column],
    )
    assert shift * 1e15 == pytest.approx(-16.5, abs=0.01)
    assert score > 0.999


def test_legacy_uncorrected_source_preserves_raw_peak_and_baseline_handling(runner, tmp_path):
    _configure_synthetic_run(runner, tmp_path)
    directory = _make_saved_scan(runner)
    namespace = _comparison_namespace(runner)
    config = json.loads((directory / "scan_configuration.json").read_text())
    for key in ("probe_source_provenance", "probe_source_correction_metadata",
                "probe_source_baseline_subtracted_in_notebook"):
        config.pop(key)
    raw = namespace["ref_waveforms"][:, 0]
    edge = max(1, int(0.05 * raw.size))
    baseline_corrected = raw - np.mean(np.r_[raw[:edge], raw[-edge:]])
    np.savez_compressed(directory / "shared_axes.npz",
                        probe_source_t_rel_ps=(namespace["time_ax"] - namespace["time_ax"][0]) * 1e-3,
                        probe_source_E=2.5 * baseline_corrected)
    column, peak_fs = namespace["_simulation_source_peak"](directory, config)
    assert column == 0
    assert peak_fs == pytest.approx(33.5)


@pytest.mark.parametrize("artifact", ("ref_field_corrected.txt", "time_axis_corrected.txt",
                                      "ref_field_corrected_metadata.json", "ref_field.txt", "time_axis.txt"))
def test_corrected_comparison_rejects_changed_artifacts_read_only(runner, tmp_path, artifact):
    _configure_synthetic_run(runner, tmp_path)
    directory = _make_saved_scan(runner)
    namespace = _comparison_namespace(runner)
    config = json.loads((directory / "scan_configuration.json").read_text())
    path = runner["synthetic_experiment_dir"] / artifact
    # Even a numerically harmless file edit must fail the provenance check.
    path.write_text(path.read_text() + "\n")
    before = _file_snapshot(tmp_path)
    with pytest.raises(ValueError, match="hash|metadata|changed|differ|provenance|input"):
        namespace["_simulation_source_peak"](directory, config)
    assert _file_snapshot(tmp_path) == before


@pytest.mark.parametrize("alteration", ("field", "amplitude", "time", "second_baseline", "shape"))
def test_corrected_comparison_rejects_saved_source_changes(runner, tmp_path, alteration):
    _configure_synthetic_run(runner, tmp_path)
    directory = _make_saved_scan(runner)
    namespace = _comparison_namespace(runner)
    config = json.loads((directory / "scan_configuration.json").read_text())
    path = directory / "shared_axes.npz"
    with np.load(path) as data:
        time, field = data["probe_source_t_rel_ps"], data["probe_source_E"]
    if alteration == "field":
        field = field.copy()
        field[len(field) // 2] += 0.1
    elif alteration == "amplitude":
        field = 2 * field
    elif alteration == "time":
        time = time + 0.001
    elif alteration == "second_baseline":
        edge = max(1, int(0.05 * field.size))
        field = field - np.mean(np.r_[field[:edge], field[-edge:]])
    else:
        time, field = time[:-1], field[:-1]
    np.savez_compressed(path, probe_source_t_rel_ps=time, probe_source_E=field)
    with pytest.raises(ValueError, match="source|differ|time"):
        namespace["_simulation_source_peak"](directory, config)


@pytest.mark.parametrize("alteration", ("missing_provenance", "missing_metadata", "trace_index",
                                       "provenance_index", "embedded_metadata", "second_baseline"))
def test_corrected_comparison_rejects_inconsistent_configuration(runner, tmp_path, alteration):
    _configure_synthetic_run(runner, tmp_path)
    directory = _make_saved_scan(runner)
    namespace = _comparison_namespace(runner)
    config = json.loads((directory / "scan_configuration.json").read_text())
    if alteration == "missing_provenance":
        config.pop("probe_source_provenance")
    elif alteration == "missing_metadata":
        config.pop("probe_source_correction_metadata")
    elif alteration == "trace_index":
        config["experimental_trace_index"] = 1
    elif alteration == "provenance_index":
        config["probe_source_provenance"]["reference_trace_index"] = 1
    elif alteration == "embedded_metadata":
        config["probe_source_correction_metadata"]["method"] = "changed method"
    else:
        config["probe_source_baseline_subtracted_in_notebook"] = True
    with pytest.raises(ValueError, match="source|metadata|index|trace|provenance|baseline"):
        namespace["_simulation_source_peak"](directory, config)


def test_single_case_configuration_and_all_stage_compilation(runner):
    assert runner["matching_gammas"] == GAMMAS
    assert runner["matching_delta"] == DELTA
    assert _literal_config(runner, "N_DELAYS") == 151
    assert _literal_config(runner, "ECHO_DELAY_STEP_FS") == 30.0
    assert _literal_config(runner, "SCAN_MARGIN_FS") == 20.0
    assert _literal_config(runner, "PROBE_DZ_A") is None
    assert runner["matching_source_metadata"]["reverse_fdtd_params"]["dz"] == 20.0
    assert runner["matching_output_dir"].name == "corrected"
    assert runner["matching_output_dir"].parent == NOTEBOOK.parents[2] / "output/project2"
    assert set(runner["matching_probe_code"]) == {250.0}
    settings = runner["matching_settings"]
    assert settings["MATCHING_GAMMA_D0"] == GAMMAS
    assert settings["MATCHING_DEL_OMEGA_D"] == DELTA
    assert settings["MATCHING_RUN_IN_FINAL_CELL"] is True
    assert "MATCHING_EXTEND_EXISTING" not in settings
    assert runner["matching_pump_dir"] == NOTEBOOK.parents[2] / settings["MATCHING_PUMP_INPUT_RELATIVE"]
    assert runner["matching_pump_dir"].name == "pump_3fs_1p5eV_sio2_12p82um"
    assert runner["matching_pump_dir"].parent.name == "SINGLE_LOBE_MATCHING_del_omega_130"
    assert not runner["matching_pump_dir"].is_relative_to(runner["matching_output_dir"])
    assert runner["_matching_scan_directory"](250.0).name == "probe_pump_sqrt_E8_3fs_1p5eV_gamma_D0_250.0000_ps-1_del_omega_D_130.0000_rad-ps-1_cubic"
    setup = runner["matching_simulation_sources"]["probe_setup"]
    assert '"ref_field_corrected.txt"' in setup
    assert '"time_axis_corrected.txt"' in setup
    assert "REFERENCE_TRACE_INDEX" not in setup
    assert '"ref_field.txt"' not in setup
    for name, expected in (("gamma_D0", GAMMAS[0]), ("del_omega_D", DELTA)):
        assignment = runner["_matching_assignment"](runner["matching_setup_tree"], name)
        expression = ast.Expression(body=assignment.value)
        assert eval(compile(expression, "safe-single-case-setting", "eval"),
                    {"__builtins__": {}}, settings) == expected
    for gamma, code in runner["matching_probe_code"].items():
        instructions = list(dis.get_instructions(code))
        constants = {
            instruction.argval: instructions[index - 1].argval
            for index, instruction in enumerate(instructions)
            if instruction.opname == "STORE_NAME" and index > 0
            and instructions[index - 1].opname == "LOAD_CONST"
        }
        assert constants["gamma_D0"] == gamma
        assert constants["del_omega_D"] == DELTA
        assert constants["N_DELAYS"] == 151
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_run_all_skips_intermediate_stages_before_final_runner(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    notebook = json.loads(NOTEBOOK.read_text())
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    namespace = {"__builtins__": {}, "MATCHING_RUN_IN_FINAL_CELL": True}
    for source in code_cells[1:-1]:
        tree = ast.parse(source)
        if source.startswith("# Corrected probe source provenance (read-only)."):
            assert all(isinstance(node, ast.FunctionDef) and not node.decorator_list for node in tree.body)
            # A helper definition is safe even with all builtins/imports removed.
            exec(compile(tree, "pure-corrected-source-helper", "exec"), namespace)
            continue
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
        guard = tree.body[0]
        assert isinstance(guard.test, ast.UnaryOp) and isinstance(guard.test.op, ast.Not)
        assert isinstance(guard.test.operand, ast.Name)
        assert guard.test.operand.id == "MATCHING_RUN_IN_FINAL_CELL"
        assert guard.orelse == []
        before = namespace.copy()
        exec(compile(tree, "guarded-intermediate-cell", "exec"), namespace)
        assert namespace == before
    assert list(tmp_path.iterdir()) == []


def test_actual_saved_pump_and_corrected_geometry_pass_read_only_preflight(runner):
    # The validator inspects JSON and archive headers; it must not decompress the
    # large stored pump array or import the FDTD solver.
    before = _file_snapshot(runner["matching_output_dir"])
    runner["_matching_validate_pump_inputs"]()
    runner["_matching_preflight"]()
    assert _file_snapshot(runner["matching_output_dir"]) == before


def test_fresh_single_case_needs_no_existing_probe_scan(runner, tmp_path):
    output = _configure_synthetic_run(runner, tmp_path)
    assert not output.exists()
    plan = runner["_matching_preflight"]()
    assert plan["gamma_D0_per_ps"] == 250.0
    assert plan["reuse"] is False
    assert Path(plan["scan_directory"]) == runner["_matching_scan_directory"](250.0)
    assert not output.exists()


def test_single_case_exports_real_energy_lobe_and_experimental_analyses(runner, tmp_path):
    output = _configure_synthetic_run(runner, tmp_path)
    pump_before = _file_snapshot(runner["matching_pump_dir"])
    runner["run_corrected_single_case"]()
    assert _file_snapshot(runner["matching_pump_dir"]) == pump_before
    assert [float(value) for value in (output / "synthetic_producer_calls.txt").read_text().splitlines()] == GAMMAS
    manifest = json.loads((output / "grid_manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["comparison_status"] == "complete"
    assert manifest["dense_main_overlap_samples"] == 151
    assert manifest["pump_status"] == "reused"
    assert Path(manifest["pump_run_directory"]) == runner["matching_pump_dir"]
    assert manifest["gamma_D0_per_ps"] == GAMMAS
    assert manifest["del_omega_D_rad_per_ps"] == DELTA
    assert len(manifest["saved_source_sha256"]) == 64
    assert (output / "executed_cell_sources.json").is_file()
    comparison = runner["matching_comparison_dir"]
    metadata = json.loads((comparison / "comparison_metadata.json").read_text())
    assert len(metadata["datasets"]) == 1
    assert len(manifest["runs"]) == 1
    record = manifest["runs"][0]
    dataset = metadata["datasets"][0]
    assert record["status"] == "complete"
    assert record["reused_completed_scan"] is False
    assert record["gamma_D0_per_ps"] == 250.0
    assert record["del_omega_D_rad_per_ps"] == DELTA
    assert record["number_of_delays"] == 3
    directory = Path(record["scan_directory"])
    assert directory.is_relative_to(tmp_path)
    expected = (1.15 + 0.05*np.arange(3))**2
    with np.load(directory / "delay_scan_analysis.npz") as analysis:
        np.testing.assert_allclose(analysis["transmitted_energy_ratio"], expected)
    with np.load(directory / "single_lobe_analysis/lobe_energy_analysis.npz") as analysis:
        np.testing.assert_allclose(analysis["ratio"], expected)
    with np.load(comparison / dataset["analysis_file"]) as analysis:
        np.testing.assert_allclose(analysis["pulse_ratio"], expected)
        np.testing.assert_allclose(analysis["lobe_ratio"], expected)
        np.testing.assert_allclose(analysis["injection_time_fs"], CORRECTED_PEAK_FS - DELAYS_FS)
        assert analysis["lobe_valid"].all()
    assert dataset["alignment_score"] > 0.999
    assert dataset["waveform_shift_fs"] == pytest.approx(-16.5, abs=0.01)
    for filename in RUN_FIGURES:
        assert (Path(record["figure_directory"]) / filename).stat().st_size > 1000
    experimental = np.genfromtxt(comparison / "experimental_energy_analysis.csv", names=True, delimiter=",")
    np.testing.assert_allclose(experimental["pulse_ratio"], [1.0, 0.64, 1.44])
    np.testing.assert_allclose(experimental["lobe_ratio"], [1.0, 0.64, 1.44])
    for filename in COMPARISON_FIGURES:
        original = comparison / filename
        copied = runner["matching_figure_dir"] / comparison.name / filename
        assert original.stat().st_size > 1000
        assert copied.read_bytes() == original.read_bytes()
    preserved = {folder: _file_snapshot(directory / folder)
                 for folder in ("unpumped_reference", "fully_injected_reference", "delay_0000", "delay_0001", "delay_0002")}
    producer_log = (output / "synthetic_producer_calls.txt").read_bytes()
    sentinel = compile('raise AssertionError("A complete corrected case must not rerun any producer")',
                       "completed-case-sentinel", "exec")
    runner["matching_probe_code"][250.0] = sentinel
    for stage in SIMULATION_STAGES:
        runner["matching_simulation_code"][stage] = sentinel
    runner["run_corrected_single_case"]()
    assert {folder: _file_snapshot(directory / folder) for folder in preserved} == preserved
    assert _file_snapshot(runner["matching_pump_dir"]) == pump_before
    assert (output / "synthetic_producer_calls.txt").read_bytes() == producer_log
    second_manifest = json.loads((output / "grid_manifest.json").read_text())
    assert second_manifest["complete"] is True
    assert second_manifest["runs"][0]["reused_completed_scan"] is True


def test_incomplete_case_fails_preflight_without_writes(runner, tmp_path):
    output = _configure_synthetic_run(runner, tmp_path)
    _make_saved_scan(runner, complete=False)
    before = _file_snapshot(output)
    with pytest.raises(RuntimeError, match="Incomplete saved scan"):
        runner["run_corrected_single_case"]()
    assert _file_snapshot(output) == before


@pytest.mark.parametrize("key,value", (("gamma_D_per_ps", 99.0), ("del_omega_D_rad_per_ps", 140.0),
                                       ("pump_run_directory", "/tmp/unrelated-pump")))
def test_existing_case_must_match_material_and_pump(runner, tmp_path, key, value):
    output = _configure_synthetic_run(runner, tmp_path)
    config_path = _make_saved_scan(runner) / "scan_configuration.json"
    config = json.loads(config_path.read_text())
    config[key] = value
    config_path.write_text(json.dumps(config))
    before = _file_snapshot(output)
    with pytest.raises(ValueError, match="parameters or pump directory"):
        runner["run_corrected_single_case"]()
    assert _file_snapshot(output) == before


@pytest.mark.parametrize("provenance", (None, {"corrected_field_sha256": "different-source"}))
def test_existing_case_must_match_corrected_source_before_writes(runner, tmp_path, provenance):
    output = _configure_synthetic_run(runner, tmp_path)
    config_path = _make_saved_scan(runner) / "scan_configuration.json"
    config = json.loads(config_path.read_text())
    config["probe_source_provenance"] = provenance
    config_path.write_text(json.dumps(config))
    before = _file_snapshot(output)
    with pytest.raises(RuntimeError, match="corrected-source provenance"):
        runner["run_corrected_single_case"]()
    assert _file_snapshot(output) == before


@pytest.mark.parametrize("alteration", ("saved_source_amplitude", "raw_reference"))
def test_reused_case_validates_incident_source_before_analysis_writes(runner, tmp_path, alteration):
    _configure_synthetic_run(runner, tmp_path)
    directory = _make_saved_scan(runner)
    if alteration == "saved_source_amplitude":
        path = directory / "shared_axes.npz"
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["probe_source_E"] *= 2
        np.savez_compressed(path, **arrays)
    else:
        path = runner["synthetic_experiment_dir"] / "ref_field.txt"
        path.write_text(path.read_text() + "\n")
    before = _file_snapshot(tmp_path)
    with pytest.raises(ValueError, match="source|hash|input"):
        runner["run_corrected_single_case"]()
    assert _file_snapshot(tmp_path) == before


@pytest.mark.parametrize("relative", ("delay_0001/data.npz", "shared_profile.npy", "shared_axes.npz"))
def test_missing_completed_case_artifacts_fail_without_writes(runner, tmp_path, relative):
    output = _configure_synthetic_run(runner, tmp_path)
    (_make_saved_scan(runner) / relative).unlink()
    before = _file_snapshot(output)
    with pytest.raises(FileNotFoundError):
        runner["run_corrected_single_case"]()
    assert _file_snapshot(output) == before


def test_changed_source_since_runner_setup_fails_before_writes(runner, tmp_path):
    output = _configure_synthetic_run(runner, tmp_path)
    runner["matching_source_provenance"] = {"corrected_field_sha256": "stale-runner-source"}
    with pytest.raises(RuntimeError, match="Corrected source changed"):
        runner["run_corrected_single_case"]()
    assert not output.exists()


def _small_pump_inputs(runner, tmp_path):
    """Keep authentic pump geometry metadata with a three-step header fixture."""
    actual = runner["matching_pump_dir"]
    info = json.loads((actual / "info.json").read_text())
    info["fdtd_params"]["Nt"] = 3
    metadata = json.loads((actual / "pump_metadata.json").read_text())
    directory = tmp_path / "pump_inputs"
    directory.mkdir()
    (directory / "info.json").write_text(json.dumps(info))
    (directory / "pump_metadata.json").write_text(json.dumps(metadata))
    np.savez(directory / "pump_material_probes.npz", E=np.ones((3, 2)), t_ps=np.arange(3),
             z_A=np.arange(2), source_E=np.ones(3), cell_indices=np.arange(2))
    runner["matching_pump_dir"] = directory
    return directory


@pytest.mark.parametrize("filename", ("info.json", "pump_metadata.json", "pump_material_probes.npz"))
def test_missing_pump_inputs_fail_read_only(runner, tmp_path, filename):
    directory = _small_pump_inputs(runner, tmp_path)
    (directory / filename).unlink()
    before = _file_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError, match="requires the saved pump"):
        runner["_matching_preflight"]()
    assert _file_snapshot(tmp_path) == before


@pytest.mark.parametrize("bad_input", ("grid", "pulse", "background", "array_shape"))
def test_incompatible_pump_inputs_fail_read_only(runner, tmp_path, bad_input):
    directory = _small_pump_inputs(runner, tmp_path)
    runner["_matching_validate_pump_inputs"]()
    if bad_input == "grid":
        path = directory / "info.json"
        info = json.loads(path.read_text())
        info["fdtd_params"]["dz"] *= 2
        path.write_text(json.dumps(info))
    elif bad_input in ("pulse", "background"):
        path = directory / "pump_metadata.json"
        metadata = json.loads(path.read_text())
        if bad_input == "pulse":
            metadata["carrier_energy_eV"] = 4.0
        else:
            metadata["background_del_eps"][0] *= 2
        path.write_text(json.dumps(metadata))
    else:
        np.savez(directory / "pump_material_probes.npz", E=np.ones((2, 3)), t_ps=np.arange(3),
                 z_A=np.arange(2), source_E=np.ones(3), cell_indices=np.arange(2))
    before = _file_snapshot(tmp_path)
    with pytest.raises(ValueError, match="disagree|configured"):
        runner["_matching_preflight"]()
    assert _file_snapshot(tmp_path) == before


def test_changed_scan_settings_fail_before_writes(runner, tmp_path):
    output = _configure_synthetic_run(runner, tmp_path)
    config_path = _make_saved_scan(runner) / "scan_configuration.json"
    config = json.loads(config_path.read_text())
    config["dense_main_overlap_samples"] = 19
    config_path.write_text(json.dumps(config))
    before = _file_snapshot(output)
    with pytest.raises(ValueError, match="setting|configuration"):
        runner["run_corrected_single_case"]()
    assert _file_snapshot(output) == before
