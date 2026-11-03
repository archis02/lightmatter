"""Small regression tests for reference-waveform inversion; no FDTD runs.

Run with ``python -m unittest discover -s lightmatter/tests -p test_correct_ref_field.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "notebooks/project_2/correct_ref_field.py"
SPEC = importlib.util.spec_from_file_location("correct_ref_field", SCRIPT)
correction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(correction)


class ReferenceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.field_path = root / "ref_field.txt"
        self.time_path = root / "time_axis.txt"
        self.times = np.arange(80, dtype=float) * 0.1 - 4
        self.signal = np.zeros(80)
        self.signal[20:60] = np.sin(np.arange(40) * 0.4)
        np.savetxt(self.time_path, self.times, delimiter=",")

    def test_single_column_retains_amplitude_and_subtracts_edge_baseline(self):
        np.savetxt(self.field_path, self.signal + 2.75, delimiter=",")
        times, field, baseline = correction.load_reference(self.field_path, self.time_path, 0)
        np.testing.assert_array_equal(times, self.times)
        np.testing.assert_allclose(field, self.signal, atol=5e-16, rtol=0)
        self.assertEqual(baseline, 2.75)

    def test_requested_column_is_the_only_trace_loaded(self):
        traces = np.column_stack((self.signal + 2, -3 * self.signal - 0.5))
        np.savetxt(self.field_path, traces, delimiter=",")
        _, field, baseline = correction.load_reference(self.field_path, self.time_path, 1)
        np.testing.assert_allclose(field, -3 * self.signal, atol=5e-16, rtol=0)
        self.assertEqual(baseline, -0.5)

    def test_negative_and_out_of_range_columns_are_rejected(self):
        np.savetxt(self.field_path, np.column_stack((self.signal, self.signal)), delimiter=",")
        for trace_index in (-1, 2):
            with self.subTest(trace_index=trace_index), self.assertRaises(ValueError):
                correction.load_reference(self.field_path, self.time_path, trace_index)

    def test_nonuniform_descending_and_nonfinite_times_are_rejected(self):
        np.savetxt(self.field_path, self.signal, delimiter=",")
        nonuniform = self.times.copy()
        nonuniform[30] += 0.01
        nonfinite = self.times.copy()
        nonfinite[30] = np.nan
        for times in (nonuniform, self.times[::-1], nonfinite):
            with self.subTest(times=times[30]), self.assertRaises(ValueError):
                np.savetxt(self.time_path, times, delimiter=",")
                correction.load_reference(self.field_path, self.time_path, 0)

    def test_zero_after_baseline_removal_and_nonfinite_field_are_rejected(self):
        nonfinite = self.signal.copy()
        nonfinite[30] = np.inf
        for field in (np.full_like(self.signal, 2.0), nonfinite):
            with self.subTest(field=field[30]), self.assertRaises(ValueError):
                np.savetxt(self.field_path, field, delimiter=",")
                correction.load_reference(self.field_path, self.time_path, 0)


class InverseTransferTests(unittest.TestCase):
    def test_attenuating_dispersive_transfer_recovers_incident_waveform(self):
        # Known real, band-limited incident pulse and a lossy dispersive slab.
        # Conjugation alone removes phase but leaves |H|^2 attenuation behind.
        count, dt_ps = 2048, 0.001
        frequency = np.fft.rfftfreq(count, dt_ps)
        spectrum = np.zeros(frequency.size, dtype=complex)
        supported = (frequency >= 80) & (frequency <= 280)
        spectrum[supported] = (
            np.exp(-0.5 * ((frequency[supported] - 166) / 35) ** 2)
            * np.exp(-2j * np.pi * frequency[supported] * 0.45)
        )
        incident = np.fft.irfft(spectrum, count)
        transfer = (0.65 + 0.1 * np.cos(frequency / 90)) * np.exp(
            -2j * np.pi * frequency * 0.062 + 0.00008j * (frequency - 166) ** 2
        )
        measured_spectrum = spectrum * transfer
        inverse = correction.regularized_inverse(transfer, correction.optical_band(frequency))
        reconstructed = np.fft.irfft(measured_spectrum * inverse, count)
        np.testing.assert_allclose(reconstructed, incident, atol=1e-12, rtol=1e-10)
        conjugate_only = np.fft.irfft(measured_spectrum * transfer.conj(), count)
        self.assertGreater(np.linalg.norm(conjugate_only - incident) / np.linalg.norm(incident), 0.4)

    def test_optical_taper_boundaries_and_midpoints(self):
        frequencies = np.array([0, 40, 50, 60, 166, 320, 360, 400, 500], dtype=float)
        np.testing.assert_allclose(
            correction.optical_band(frequencies), [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0],
            atol=1e-15, rtol=0,
        )

    def test_unsupported_bins_are_zero_even_with_invalid_transfer(self):
        transfer = np.array([np.nan, 0.0, 1e-9j, 0.5 + 0.5j, np.inf], dtype=complex)
        weight = np.array([0, 1, 1, 1, 0], dtype=float)
        inverse = correction.regularized_inverse(transfer, weight, relative_floor=0.01)
        np.testing.assert_array_equal(inverse[[0, 1, 4]], 0)
        self.assertTrue(np.all(np.isfinite(inverse)))
        # At near-zero supported transmission, regularization suppresses the
        # result instead of the unbounded gain of direct division by H.
        self.assertLess(abs(inverse[2]), 1e-3)

    def test_missing_or_invalid_supported_calibration_is_rejected(self):
        cases = (
            (np.array([1 + 0j]), np.array([0.0])),
            (np.array([0 + 0j]), np.array([1.0])),
            (np.array([np.nan + 0j]), np.array([1.0])),
        )
        for transfer, weight in cases:
            with self.subTest(transfer=transfer, weight=weight), self.assertRaises(ValueError):
                correction.regularized_inverse(transfer, weight)

    def test_geometry_uses_notebook_resolution_and_external_probes(self):
        geometry = correction.geometry(123)
        self.assertEqual(geometry.Nt, 123)
        self.assertEqual(geometry.Nz, 15000)
        self.assertEqual(geometry.dz, 20.0)
        self.assertAlmostEqual(geometry.dt, 0.99 * 20 / correction.C0)
        self.assertEqual((geometry.metal_i1 - geometry.metal_i0) * geometry.dz, 128200)
        self.assertLess(geometry.npml, geometry.src_i)
        self.assertLess(geometry.src_i, geometry.metal_i0)
        self.assertLess(geometry.metal_i1, geometry.probe_t_i)
        self.assertLess(geometry.probe_t_i, geometry.Nz - geometry.npml)
        self.assertEqual(geometry.snap_every, 0)


class AnalyticalTransmissionTests(unittest.TestCase):
    def test_refractive_index_matches_known_silica_value_and_sellmeier_form(self):
        # Independent wavelength-domain expression for the same Malitson model.
        wavelengths_um = np.array([0.8, 1.0, 1.55, 2.0])
        strengths = np.array([0.6961663, 0.4079426, 0.8974794])
        poles_um = np.array([0.0684043, 0.1162414, 9.896161])
        wavelength_squared = wavelengths_um[:, None] ** 2
        expected = np.sqrt(1 + np.sum(
            strengths * wavelength_squared / (wavelength_squared - poles_um ** 2),
            axis=1,
        ))
        actual = correction.silica_refractive_index(correction.C0 / (wavelengths_um * 1e4))
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
        self.assertAlmostEqual(actual[2].real, 1.4440236217, places=9)

    def test_dc_limit_and_evanescent_branch_are_finite_and_passive(self):
        indices = correction.silica_refractive_index(np.array([0.0, 31.0]))
        self.assertAlmostEqual(indices[0].real ** 2, 1 + 0.6961663 + 0.4079426 + 0.8974794)
        self.assertEqual(indices[0].imag, 0.0)
        # Just above the infrared pole epsilon is negative. For numpy's FFT
        # convention exp(-i*w*n*d/c), the decaying branch has Im(n) < 0.
        self.assertAlmostEqual(indices[1].real, 0.0)
        self.assertLess(indices[1].imag, 0.0)
        propagation = np.exp(-2j * np.pi * 31 * indices[1] * 1000 / correction.C0)
        self.assertLess(abs(propagation), 1.0)

    def test_invalid_frequencies_and_exact_resonances_are_rejected(self):
        pole_frequency = correction.C0 / (
            np.asarray(correction.SIO2_BACKGROUND_LAMBDA0_UM_DEFAULT) * 1e4
        )
        invalid = [np.array([-1.0]), np.array([np.nan]), np.array([np.inf])]
        invalid.extend(np.array([frequency]) for frequency in pole_frequency)
        for frequency in invalid:
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                correction.silica_refractive_index(frequency)

    def test_first_pass_fresnel_loss_and_fft_delay_sign(self):
        count, dt_ps = 256, 0.002
        frequencies = np.fft.rfftfreq(count, dt_ps)
        refractive_index = 1.5
        delay_samples = 17
        delay_ps = delay_samples * dt_ps
        thickness = correction.C0 * delay_ps / refractive_index
        with patch.object(correction, "silica_refractive_index", return_value=np.full(
            frequencies.shape, refractive_index, dtype=complex
        )):
            transfer = correction.analytical_transmission(frequencies, thickness)
            aligned = correction.analytical_transmission(
                frequencies, thickness, group_delay_ps=delay_ps,
            )
        fresnel = 4 * refractive_index / (1 + refractive_index) ** 2
        np.testing.assert_allclose(np.abs(transfer), fresnel, atol=2e-15, rtol=0)
        incident = np.zeros(count)
        incident[50] = 1
        transmitted = np.fft.irfft(np.fft.rfft(incident) * transfer, count)
        np.testing.assert_allclose(
            transmitted, fresnel * np.roll(incident, delay_samples), atol=4e-15, rtol=0,
        )
        np.testing.assert_allclose(aligned, fresnel, atol=2e-14, rtol=0)

    def test_complete_slab_zero_thickness_identity_and_energy_conservation(self):
        frequencies = np.linspace(60, 320, 2048)
        identity = correction.analytical_transmission(
            frequencies, 0.0, internal_reflections=True,
        )
        np.testing.assert_allclose(identity, 1.0, atol=5e-16, rtol=0)
        transfer = correction.analytical_transmission(
            frequencies, 128200.0, internal_reflections=True,
        )
        self.assertTrue(np.all(np.isfinite(transfer)))
        self.assertLessEqual(np.max(np.abs(transfer)), 1 + 2e-15)
        self.assertGreater(np.ptp(np.abs(transfer)), 0.05)

    def test_complete_slab_resonance_and_antiresonance(self):
        refractive_index, thickness = 1.5, 128200.0
        half_wave_frequency = correction.C0 / (2 * refractive_index * thickness)
        frequencies = half_wave_frequency * np.array([3.0, 3.5])
        with patch.object(correction, "silica_refractive_index", return_value=np.full(
            frequencies.shape, refractive_index, dtype=complex
        )):
            transfer = correction.analytical_transmission(
                frequencies, thickness, internal_reflections=True,
            )
        reflectivity = ((refractive_index - 1) / (refractive_index + 1)) ** 2
        np.testing.assert_allclose(
            np.abs(transfer), [1, (1 - reflectivity) / (1 + reflectivity)],
            atol=2e-15, rtol=0,
        )

    def test_narrowband_analytical_forward_and_inverse_recovers_incident(self):
        count, dt_ps = 4096, 0.001
        frequencies = np.fft.rfftfreq(count, dt_ps)
        supported = (frequencies >= 80) & (frequencies <= 280)
        spectrum = np.zeros(frequencies.size, dtype=complex)
        spectrum[supported] = (
            np.exp(-0.5 * ((frequencies[supported] - 166) / 30) ** 2)
            * np.exp(-2j * np.pi * frequencies[supported] * 1.5)
        )
        incident = np.fft.irfft(spectrum, count)
        for internal_reflections in (False, True):
            with self.subTest(internal_reflections=internal_reflections):
                transfer = correction.analytical_transmission(
                    frequencies[supported], 128200.0,
                    internal_reflections=internal_reflections,
                )
                measured_spectrum = np.zeros_like(spectrum)
                measured_spectrum[supported] = spectrum[supported] * transfer
                measured = np.fft.irfft(measured_spectrum, count)
                reconstructed_spectrum = np.zeros_like(spectrum)
                reconstructed_spectrum[supported] = np.fft.rfft(measured)[supported] / transfer
                reconstructed = np.fft.irfft(reconstructed_spectrum, count)
                np.testing.assert_allclose(reconstructed, incident, atol=2e-16, rtol=2e-12)
                self.assertGreater(np.linalg.norm(measured - incident) / np.linalg.norm(incident), 0.1)


class AnalyticalWorkflowTests(unittest.TestCase):
    def test_cli_analytical_only_and_default_both_dispatch_selected_trace(self):
        common = [
            "--input", "/tmp/input-reference.txt", "--time-axis", "/tmp/input-time.txt",
            "--output-dir", "/tmp/analytical-test-output", "--trace-index", "2",
        ]
        expected = (
            Path("/tmp/input-reference.txt"), Path("/tmp/input-time.txt"),
            Path("/tmp/analytical-test-output"), 2,
        )
        for mode_args in (["--method", "analytical"], []):
            with self.subTest(mode_args=mode_args), \
                    patch.object(correction, "correct_reference") as fdtd, \
                    patch.object(correction, "analytical_correct_reference") as analytical:
                correction.main(common + mode_args)
                analytical.assert_called_once_with(*expected)
                if mode_args:
                    fdtd.assert_not_called()
                else:
                    fdtd.assert_called_once_with(*expected)

    def test_analytical_file_workflow_is_independent_and_preserves_original_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field_path, time_path = root / "ref_field.txt", root / "time_axis.txt"
            output_dir = root / "output"
            times_fs = np.linspace(0, 80, 1601)
            pulse = np.exp(-0.5 * ((times_fs - 40) / 6) ** 2) * np.sin(
                2 * np.pi * 0.166 * (times_fs - 40)
            )
            np.savetxt(field_path, np.column_stack((pulse * 0.5, pulse)), delimiter=",")
            np.savetxt(time_path, times_fs, delimiter=",")
            original_field, original_times = field_path.read_bytes(), time_path.read_bytes()
            with patch.object(correction, "save_analytical_plot") as plot, \
                    patch.object(correction, "propagate", side_effect=AssertionError("FDTD forbidden")) as propagate, \
                    patch.object(correction, "run_simulation", side_effect=AssertionError("FDTD forbidden")) as simulate, \
                    patch("builtins.print"):
                metadata = correction.analytical_correct_reference(
                    field_path, time_path, output_dir, trace_index=1,
                )
            plot.assert_called_once()
            self.assertIsNone(plot.call_args.args[-1])
            propagate.assert_not_called()
            simulate.assert_not_called()
            result = np.loadtxt(output_dir / "ref_field_corrected_analytical.txt")
            self.assertEqual(result.shape, times_fs.shape)
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertGreater(np.linalg.norm(result), 0)
            self.assertEqual(metadata["reference_trace_index"], 1)
            self.assertLess(metadata["analytic_roundtrip_relative_L2"], 1e-12)
            self.assertTrue((output_dir / "ref_field_corrected_analytical_metadata.json").is_file())
            self.assertFalse((output_dir / "ref_field_corrected.txt").exists())
            self.assertFalse((output_dir / "time_axis_corrected.txt").exists())
            self.assertEqual(field_path.read_bytes(), original_field)
            self.assertEqual(time_path.read_bytes(), original_times)


if __name__ == "__main__":
    unittest.main()
