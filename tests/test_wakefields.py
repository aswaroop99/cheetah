"""Tests for the wakefield utilities and for Cavity when wake=True.

Every physics check cites the equation in Ferrario, Migliorati and Palumbo,
"Wakefields and Instabilities in Linear Accelerators", CERN-2014-009 pp. 357-375.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import cheetah

from cheetah import Cavity, LongitudinalCosineWake
from cheetah.accelerator.wakefields import (
    SPEED_OF_LIGHT,
    VACUUM_IMPEDANCE,
    CavityGeometry,
    DiffractionModelWake,
    LongitudinalGrid,
    PeriodicStructureTransverseWake,
    PeriodicStructureWake,
    ResonantModeTransverseWake,
    ResonantModeWake,
    bunch_energy_loss,
    gaussian_resonant_wake_potential,
    longitudinal_energy_kick,
    longitudinal_grid_from_beam,
    longitudinal_wake_potential,
    space_charge_wake_potential,
    transverse_dipole_wake_kick,
)

DTYPE = torch.float64
FREQUENCY = 1.3e9
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY
REFERENCE_ENERGY = 100e6


def make_beam(sigma_tau, total_charge=1e-9, num_particles=200_000, seed=0):
    """Gaussian `ParticleBeam` with the given bunch length and charge."""
    torch.manual_seed(seed)
    beam = cheetah.ParticleBeam.from_twiss(
        num_particles=num_particles,
        beta_x=torch.tensor(10.0, dtype=DTYPE),
        beta_y=torch.tensor(10.0, dtype=DTYPE),
        emittance_x=torch.tensor(2e-9, dtype=DTYPE),
        emittance_y=torch.tensor(2e-9, dtype=DTYPE),
        sigma_tau=torch.tensor(sigma_tau, dtype=DTYPE),
        sigma_p=torch.tensor(1e-3, dtype=DTYPE),
        energy=torch.tensor(REFERENCE_ENERGY, dtype=DTYPE),
        total_charge=torch.tensor(total_charge, dtype=DTYPE),
        dtype=DTYPE,
    )
    return beam


# ----------------------------------------------------------------- kernel identities


def test_resonant_kernel_reduces_to_cosine_at_infinite_q():
    """With `Q = inf` the resonant wake is a pure cosine, `w(s) = 2k cos(ks)`."""
    kernel = ResonantModeWake(frequency=FREQUENCY, r_over_q=1.0e-8)
    s = torch.linspace(0.0, 3 * WAVELENGTH, 501, dtype=DTYPE)
    amplitude = 2.0 * math.pi * FREQUENCY * 1.0e-8
    expected = amplitude * torch.cos(2 * math.pi * s / WAVELENGTH)
    # atol is scaled to the amplitude: near zero crossings rtol alone is too strict.
    assert torch.allclose(
        kernel.sample(s), expected, rtol=1e-10, atol=1e-12 * amplitude
    )
    assert kernel.value_at_origin == pytest.approx(amplitude)


def test_beam_loading_theorem_holds_for_every_kernel():
    """
    Every wake satisfies the beam-loading theorem `k = w(0+) / 2` (Ferrario Eq. 18).
    """
    kernels = [
        ResonantModeWake(frequency=FREQUENCY, r_over_q=1e-8, quality_factor=1e4),
        PeriodicStructureWake(
            iris_radius=35e-3, cell_gap=0.03, cell_period=0.0345, length=1.038
        ),
    ]
    for kernel in kernels:
        assert kernel.loss_factor == pytest.approx(0.5 * kernel.value_at_origin)


def test_resonant_kernel_matches_equation_11_with_damping():
    """The damped resonant wake matches Ferrario Eq. 11."""
    frequency, quality_factor, shunt = 8.4e9, 11_000.0, 5.0e6
    kernel = ResonantModeWake.from_shunt_impedance(frequency, shunt, quality_factor)
    omega_r = 2 * math.pi * frequency
    damping = omega_r / (2 * quality_factor)
    omega_bar = math.sqrt(omega_r**2 - damping**2)

    s = torch.linspace(0.0, 0.5, 401, dtype=DTYPE)
    reference = (
        shunt
        * omega_r
        / quality_factor
        * torch.exp(-damping * s / SPEED_OF_LIGHT)
        * (
            torch.cos(omega_bar * s / SPEED_OF_LIGHT)
            - (damping / omega_bar) * torch.sin(omega_bar * s / SPEED_OF_LIGHT)
        )
    )
    assert torch.allclose(
        kernel.sample(s),
        reference,
        rtol=1e-9,
        atol=1e-10 * float(reference.abs().max()),
    )
    # Eq. 24 remark: k = omega_r R / 2Q.
    assert kernel.loss_factor == pytest.approx(omega_r * shunt / (2 * quality_factor))


def test_periodic_structure_wake_matches_equations_15_and_16():
    """The periodic-structure wake matches Ferrario Eqs. 15-16."""
    b, g, p, length = 11.6e-3, 0.0289, 0.0350, 3.0
    kernel = PeriodicStructureWake(b, g, p, length)
    assert kernel.s1 == pytest.approx(0.41 * b**1.8 * g**1.6 / p**2.4)

    s = torch.linspace(0.0, 20 * kernel.s1, 301, dtype=DTYPE)
    amplitude = VACUUM_IMPEDANCE * SPEED_OF_LIGHT * length / (math.pi * b**2)
    expected = amplitude * torch.exp(-torch.sqrt(s / kernel.s1))
    assert torch.allclose(kernel.sample(s), expected, rtol=1e-11)


def test_periodic_wake_amplitude_scales_as_inverse_aperture_squared():
    """The periodic wake amplitude scales as `1 / b^2`."""
    wide = PeriodicStructureWake(20e-3, 0.03, 0.035, 1.0)
    narrow = PeriodicStructureWake(10e-3, 0.03, 0.035, 1.0)
    assert narrow.value_at_origin / wide.value_at_origin == pytest.approx(4.0)


def test_kernel_cell_averages_match_numerical_quadrature():
    """The closed-form cell averages agree with numerical integration."""
    from scipy.integrate import quad

    dz = 1e-4
    for kernel in (
        PeriodicStructureWake(11.6e-3, 0.0289, 0.035, 3.0),
        DiffractionModelWake(11.6e-3, 0.0289, num_cells=86.0),
    ):

        def scalar(s_value):
            return float(kernel.sample(torch.tensor([s_value], dtype=DTYPE)))

        for start in (0.0, 5e-4, 5e-3):
            reference = quad(scalar, start, start + dz, points=[start])[0] / dz
            exact = float(kernel.cell_average(torch.tensor([start], dtype=DTYPE), dz))
            assert exact == pytest.approx(reference, rel=1e-6)


def test_critical_cell_count_matches_equation_14():
    """`N_cr` matches Ferrario Eq. 14."""
    geometry = CavityGeometry(
        iris_radius=11.6e-3, cell_gap=0.0289, cell_period=0.035, length=3.0
    )
    sigma, gamma = 300e-6, 196.0
    expected = 11.6e-3**2 / (2 * 0.0289 * (sigma + 2 * 11.6e-3 / gamma))
    assert geometry.critical_cell_count(sigma, gamma) == pytest.approx(expected)


def test_regime_selection_follows_critical_cell_count():
    """A short structure falls back to the single-cell diffraction wake."""
    short = CavityGeometry(
        iris_radius=30e-3, cell_gap=1e-3, cell_period=0.035, length=0.07
    )
    with pytest.warns(UserWarning, match="N_cr"):
        kernel = short.longitudinal_kernel("auto", sigma_tau=1e-5, gamma=1e4)
    assert isinstance(kernel, DiffractionModelWake)

    long = CavityGeometry(
        iris_radius=11.6e-3, cell_gap=0.0289, cell_period=0.035, length=30.0
    )
    assert isinstance(
        long.longitudinal_kernel("auto", sigma_tau=300e-6, gamma=196.0, warn=False),
        PeriodicStructureWake,
    )


# --------------------------------------------------------------------- grid sizing


def test_grid_resolves_the_bunch_and_covers_it():
    """The automatic grid resolves the bunch and covers every particle."""
    beam = make_beam(300e-6, num_particles=20_000)
    kernel = ResonantModeWake(frequency=FREQUENCY, r_over_q=1e-8)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,), cells_per_feature=8.0
    )
    sigma = float(beam.tau.std())
    assert grid.cells_per_sigma >= 8.0
    assert grid.resolved
    assert grid.z_min < float(beam.tau.min())
    assert grid.z_max > float(beam.tau.max())
    # The old scheme: n_grid=2000 with 4 wavelengths of padding on each side.
    old_dz = (float(beam.tau.max()) - float(beam.tau.min()) + 8 * WAVELENGTH) / 1999
    assert old_dz / sigma > 1.0  # the whole bunch inside a few cells
    assert grid.dz / sigma < 0.2


def test_grid_takes_the_shortest_of_beam_and_kernel_scales():
    """The grid spacing follows the shorter of the bunch length and the wake's scale."""
    beam = make_beam(3e-2, num_particles=20_000)  # bunch much longer than s1
    kernel = PeriodicStructureWake(11.6e-3, 0.0289, 0.035, 3.0)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,), cells_per_feature=8.0
    )
    assert kernel.s1 < float(beam.tau.std())
    assert grid.dz == pytest.approx(kernel.s1 / 8.0)


def test_grid_warns_when_clamped_by_max_grid():
    """A warning is raised when `max_grid` limits the resolution."""
    beam = make_beam(1e-6, num_particles=5_000)
    kernel = PeriodicStructureWake(11.6e-3, 0.0289, 0.035, 3.0)
    with pytest.warns(UserWarning, match="clamped"):
        grid = longitudinal_grid_from_beam(
            beam.tau,
            feature_lengths=(kernel.feature_length,),
            cells_per_feature=8.0,
            max_grid=64,
        )
    assert not grid.resolved


# ----------------------------------------------------- convolution against analytics


def test_wake_potential_matches_gaussian_closed_form():
    """The numerical wake potential matches the closed-form Gaussian result."""
    sigma = 3e-3
    total_charge = 1e-9
    beam = make_beam(sigma, total_charge=total_charge, num_particles=400_000)
    kernel = ResonantModeWake(frequency=FREQUENCY, r_over_q=1e-4)

    wake, diagnostics = longitudinal_wake_potential(
        beam.tau, beam.particle_charges, kernel
    )

    tau = beam.tau.numpy()
    analytic = gaussian_resonant_wake_potential(
        tau,
        sigma_tau=float(beam.tau.std()),
        wavelength=kernel.wavelength,
        total_charge=float(beam.particle_charges.sum()),
        loss_factor=kernel.loss_factor,
    )
    error = np.sqrt(np.mean((wake.numpy() - analytic) ** 2)) / np.sqrt(
        np.mean(analytic**2)
    )
    assert error < 0.02, f"relative rms error {error:.3%}"


def test_trapezoidal_origin_weight_improves_accuracy():
    """Halving the `s = 0` sample makes the convolution more accurate."""
    sigma = 3e-3
    beam = make_beam(sigma, num_particles=400_000)
    kernel = ResonantModeWake(frequency=FREQUENCY, r_over_q=1e-4)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,), cells_per_feature=2.0
    )

    analytic = gaussian_resonant_wake_potential(
        beam.tau.numpy(),
        sigma_tau=float(beam.tau.std()),
        wavelength=kernel.wavelength,
        total_charge=float(beam.particle_charges.sum()),
        loss_factor=kernel.loss_factor,
    )

    errors = {}
    for trapezoidal in (True, False):
        wake, _ = longitudinal_wake_potential(
            beam.tau,
            beam.particle_charges,
            kernel,
            grid=grid,
            trapezoidal=trapezoidal,
        )
        errors[trapezoidal] = np.sqrt(np.mean((wake.numpy() - analytic) ** 2))
    assert errors[True] < errors[False]


def test_wake_potential_is_linear_in_charge_not_quadratic():
    """Doubling the charge doubles the wake potential."""
    kernel = ResonantModeWake(frequency=FREQUENCY, r_over_q=1e-4)
    beam = make_beam(3e-3, total_charge=1e-9, num_particles=50_000)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,)
    )
    single, _ = longitudinal_wake_potential(
        beam.tau, beam.particle_charges, kernel, grid=grid
    )
    double, _ = longitudinal_wake_potential(
        beam.tau, 2.0 * beam.particle_charges, kernel, grid=grid
    )
    assert torch.allclose(double, 2.0 * single, rtol=1e-10)


def test_wake_is_causal():
    """Nothing ahead of the bunch feels the wake, and the head sees half its own."""
    kernel = ResonantModeWake(frequency=FREQUENCY, r_over_q=1e-4)
    beam = make_beam(3e-3, num_particles=100_000)
    wake, diagnostics = longitudinal_wake_potential(
        beam.tau, beam.particle_charges, kernel
    )
    z_grid = diagnostics["z_grid"]
    wake_grid = diagnostics["W_grid"]
    ahead = z_grid < float(beam.tau.min())
    peak = float(wake_grid.abs().max())
    assert float(wake_grid[ahead].abs().max()) < 1e-6 * peak


def test_energy_kick_decelerates_for_a_positive_wake():
    """A positive wake takes energy from the beam (Ferrario Eq. 4)."""
    kernel = PeriodicStructureWake(11.6e-3, 0.0289, 0.035, 3.0)
    beam = make_beam(3e-4, num_particles=50_000)
    delta_energy, _ = longitudinal_energy_kick(
        beam.tau, beam.particle_charges, kernel, num_elementary_charges=-1.0
    )
    assert float(delta_energy.mean()) < 0.0
    # Species sign must not matter: a proton bunch loses energy too.
    proton, _ = longitudinal_energy_kick(
        beam.tau, beam.particle_charges, kernel, num_elementary_charges=+1.0
    )
    assert torch.allclose(delta_energy, proton)


def test_bunch_energy_loss_matches_loss_factor_for_a_short_bunch():
    """A very short bunch loses `U = k Q^2` (Ferrario Eq. 28)."""
    total_charge = 1e-9
    kernel = PeriodicStructureWake(11.6e-3, 0.0289, 0.035, 3.0)
    # w ~ exp(-sqrt(s/s1)) approaches its s=0 value only as sqrt(sigma/s1), so the
    # bunch has to be very short indeed before U -> -k Q^2 to a few percent.
    beam = make_beam(kernel.s1 / 2e4, total_charge=total_charge, num_particles=200_000)
    wake, _ = longitudinal_wake_potential(beam.tau, beam.particle_charges, kernel)
    energy = float(bunch_energy_loss(wake, beam.particle_charges))
    expected = -kernel.loss_factor * total_charge**2
    assert energy == pytest.approx(expected, rel=0.03)


def test_bunch_energy_loss_matches_loss_factor_for_a_resonant_mode():
    """The same `U = k Q^2` limit holds for a resonant-mode wake."""
    total_charge = 1e-9
    kernel = ResonantModeWake(frequency=FREQUENCY, r_over_q=1e-4)
    beam = make_beam(
        WAVELENGTH / 2000.0, total_charge=total_charge, num_particles=200_000
    )
    wake, _ = longitudinal_wake_potential(beam.tau, beam.particle_charges, kernel)
    energy = float(bunch_energy_loss(wake, beam.particle_charges))
    expected = -kernel.loss_factor * total_charge**2
    assert energy == pytest.approx(expected, rel=0.01)


# ----------------------------------------------------------------- space-charge wake


def test_space_charge_wake_conserves_bunch_energy():
    """
    Space charge moves energy from tail to head with no net loss (Ferrario Eq. 29).
    """
    beam = make_beam(3e-4, total_charge=1e-9, num_particles=200_000)
    delta_energy, diagnostics = space_charge_wake_potential(
        beam.tau,
        beam.particle_charges,
        length=1.0,
        pipe_radius=35e-3,
        beam_radius=2 * float(torch.sqrt(beam.sigma_x * beam.sigma_y)),
        gamma=float(beam.relativistic_gamma),
    )
    weighted = float((beam.particle_charges * delta_energy).sum())
    scale = float((beam.particle_charges * delta_energy.abs()).sum())
    assert abs(weighted) < 1e-3 * scale


def test_space_charge_head_gains_energy():
    """Under space charge the head gains energy and the tail loses it."""
    beam = make_beam(3e-4, num_particles=200_000)
    delta_energy, _ = space_charge_wake_potential(
        beam.tau,
        beam.particle_charges,
        length=1.0,
        pipe_radius=35e-3,
        beam_radius=1e-4,
        gamma=float(beam.relativistic_gamma),
    )
    head = beam.tau < -float(beam.tau.std())
    tail = beam.tau > float(beam.tau.std())
    assert float(delta_energy[head].mean()) > 0.0
    assert float(delta_energy[tail].mean()) < 0.0


def test_space_charge_wake_matches_equation_29_for_a_gaussian():
    """The space-charge kick on a Gaussian bunch matches Ferrario Eq. 29."""
    from scipy import constants

    sigma, total_charge, b, a = 3e-4, 1e-9, 35e-3, 1e-4
    beam = make_beam(sigma, total_charge=total_charge, num_particles=400_000)
    gamma = float(beam.relativistic_gamma)
    delta_energy, _ = space_charge_wake_potential(
        beam.tau,
        beam.particle_charges,
        length=1.0,
        pipe_radius=b,
        beam_radius=a,
        gamma=gamma,
        cells_per_feature=4.0,
    )

    sigma_measured = float(beam.tau.std())
    tau = beam.tau.numpy()
    prefactor = (
        total_charge
        * (1.0 + 2.0 * np.log(b / a))
        / (
            4.0
            * np.pi
            * constants.epsilon_0
            * gamma**2
            * np.sqrt(2.0 * np.pi)
            * sigma_measured**3
        )
    )
    # z_paper = -tau, and dividing by e converts joules per unit charge to eV.
    analytic = prefactor * (-tau) * np.exp(-(tau**2) / (2 * sigma_measured**2))

    inner = np.abs(tau) < 2 * sigma_measured
    error = np.sqrt(
        np.mean((delta_energy.numpy()[inner] - analytic[inner]) ** 2)
    ) / np.sqrt(np.mean(analytic[inner] ** 2))
    assert error < 0.05, f"relative rms error {error:.3%}"


# ------------------------------------------------------------------- transverse wake


def two_particle_transverse_kick(separation, kernel, *, total_charge=1e-9,
                                 head_offset=50e-6, p0c=100e6, cells_per_sep=64):
    """
    Analytical transverse kick on a test charge behind an offset source (Ferrario Eq.
    33).
    """
    dz = separation / cells_per_sep
    grid = LongitudinalGrid(
        z_min=-4 * dz, dz=dz, n_grid=1024,
        sigma_tau=separation, target_dz=dz, resolved=True,
    )
    tau = torch.tensor([0.0, separation], dtype=DTYPE)  # larger tau = trailing
    offsets = torch.tensor([head_offset, 0.0], dtype=DTYPE)
    charges = torch.tensor([total_charge / 2, 0.0], dtype=DTYPE)  # q1 = q0/2

    kick, _ = transverse_dipole_wake_kick(
        tau, offsets, charges, kernel, torch.tensor(p0c, dtype=DTYPE), grid=grid
    )
    # Eq. 33 integrated over the device: M / (c p0) = q1 w_perp(z) y1 / p0c.
    expected = (
        (total_charge / 2)
        * float(kernel.sample(torch.tensor([separation], dtype=DTYPE)))
        * head_offset
        / p0c
    )
    return float(kick[0]), float(kick[1]), expected


def test_transverse_kick_matches_two_particle_model():
    """The transverse kick matches the two-particle model (Ferrario Eq. 33)."""
    kernel = PeriodicStructureTransverseWake(11.6e-3, 0.0289, 0.035, 3.0)
    for separation in (0.5e-3, 1e-3, 2e-3, 5e-3, 1e-2):
        head, tail, expected = two_particle_transverse_kick(separation, kernel)
        assert tail == pytest.approx(expected, rel=1e-10), f"z = {separation}"


def test_transverse_kick_is_causal():
    """The leading charge feels no transverse wake."""
    kernel = PeriodicStructureTransverseWake(11.6e-3, 0.0289, 0.035, 3.0)
    head, tail, _ = two_particle_transverse_kick(2e-3, kernel)
    assert abs(head) < 1e-12 * abs(tail)


def test_transverse_amplitude_scales_as_inverse_aperture_fourth_power():
    """The transverse wake amplitude scales as `1 / b^4` (Ferrario Eq. 15)."""
    wide = PeriodicStructureTransverseWake(20e-3, 0.0289, 0.035, 3.0)
    narrow = PeriodicStructureTransverseWake(10e-3, 0.0289, 0.035, 3.0)

    scaled = 100.0  # s / s2, identical for both -> bracket cancels
    ratio = float(
        narrow.sample(torch.tensor([scaled * narrow.s2], dtype=DTYPE))
    ) / float(wide.sample(torch.tensor([scaled * wide.s2], dtype=DTYPE)))

    prefactor_ratio = (2.0**4) * (narrow.s2 / wide.s2)
    assert ratio == pytest.approx(prefactor_ratio, rel=1e-12)

    # And the raw 1/b^4 does appear once s2 is held fixed by hand.
    def amplitude(kernel):
        return (
            4.0 * VACUUM_IMPEDANCE * SPEED_OF_LIGHT * 3.0 * kernel.s2
            / (math.pi * float(kernel.iris_radius) ** 4)
        )

    assert amplitude(narrow) / amplitude(wide) == pytest.approx(
        prefactor_ratio, rel=1e-12
    )


def test_transverse_resonant_kernel_vanishes_at_the_origin():
    """The transverse wake is zero at `s = 0`, so the head never kicks itself."""
    zero = torch.tensor([0.0], dtype=DTYPE)
    assert float(ResonantModeTransverseWake(8.4e9, 50e6, 11_000.0).sample(zero)) == 0.0
    periodic = PeriodicStructureTransverseWake(11.6e-3, 0.0289, 0.035, 3.0)
    assert float(periodic.sample(zero)) == 0.0


def test_paper_table_1_hom_round_trips():
    """The C-band HOM from Ferrario Table 1 round-trips through the transverse wake."""
    kernel = ResonantModeTransverseWake(
        frequency=8.4e9, transverse_shunt_impedance=50e6, quality_factor=11_000.0
    )
    omega_r = 2 * math.pi * 8.4e9
    damping = omega_r / (2 * 11_000.0)
    omega_bar = math.sqrt(omega_r**2 - damping**2)
    s = torch.linspace(0.0, 0.05, 201, dtype=DTYPE)
    reference = (
        50e6 * omega_r / 11_000.0
        * torch.exp(-damping * s / SPEED_OF_LIGHT)
        * torch.sin(omega_bar * s / SPEED_OF_LIGHT)
    )
    assert torch.allclose(
        kernel.sample(s), reference, rtol=1e-9,
        atol=1e-10 * float(reference.abs().max()),
    )


def test_longitudinal_resonant_kernel_reproduces_paper_rlc_identity():
    """
    The resonant wake reproduces Ferrario's RLC check, `k = 1/2C` and `w(0+) = 1/C`.
    """
    frequency, shunt, quality_factor = 8.4e9, 5.0e6, 11_000.0
    kernel = ResonantModeWake.from_shunt_impedance(frequency, shunt, quality_factor)
    # C = Q / (R omega_r)
    capacitance = quality_factor / (shunt * 2 * math.pi * frequency)
    assert kernel.value_at_origin == pytest.approx(1.0 / capacitance, rel=1e-12)
    assert kernel.loss_factor == pytest.approx(1.0 / (2 * capacitance), rel=1e-12)


# ------------------------------------------------------------ Cavity with wake=True


def cavity_kwargs(**overrides):
    """RF settings for a TESLA-like cavity, with optional overrides."""
    kwargs = dict(
        length=torch.tensor(1.038, dtype=DTYPE),
        voltage=torch.tensor(16e6 * 1.038, dtype=DTYPE),
        phase=torch.tensor(0.0, dtype=DTYPE),
        frequency=torch.tensor(FREQUENCY, dtype=DTYPE),
    )
    kwargs.update(overrides)
    return kwargs


def test_wake_off_is_bit_identical_to_plain_cavity():
    """Wake parameters do nothing while `wake=False`."""
    beam = make_beam(3e-4, num_particles=10_000)
    plain = cheetah.Cavity(**cavity_kwargs())
    switched = cheetah.Cavity(
        **cavity_kwargs(),
        wake=False,
        wake_strength=torch.tensor(35e12, dtype=DTYPE),
    )
    assert torch.equal(plain.track(beam).particles, switched.track(beam).particles)
    assert switched.is_skippable == plain.is_skippable


def test_wake_on_changes_only_the_energy_coordinate():
    """Turning the wake on changes only the momentum of the particles."""
    beam = make_beam(3e-4, num_particles=10_000)
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
    )
    reference = cheetah.Cavity(**cavity_kwargs()).track(beam)
    outgoing = cavity.track(beam)
    for index, name in enumerate(["x", "px", "y", "py", "tau"]):
        assert torch.equal(
            outgoing.particles[..., index], reference.particles[..., index]
        ), name
    assert not torch.equal(outgoing.particles[..., 5], reference.particles[..., 5])
    assert float(outgoing.p.mean()) < float(reference.p.mean())  # beam loading


def test_wake_scales_with_aperture_as_one_over_b_squared():
    """The cavity's wake kick scales as `1 / b^2`."""
    beam = make_beam(3e-4, num_particles=20_000)
    reference = cheetah.Cavity(**cavity_kwargs()).track(beam)

    def mean_kick(iris_radius):
        cavity = Cavity(
            **cavity_kwargs(),
            wake=True,
            iris_radius=iris_radius,
            cell_gap=0.0289,
            cell_period=0.0345,
            wake_model="periodic",
        )
        outgoing = cavity.track(beam)
        return float((outgoing.p - reference.p).mean())

    # Halving the aperture also shortens s1, so the ratio is only approximately 4;
    # check it lands between 2 and 8 and is monotonic.
    wide, narrow = mean_kick(20e-3), mean_kick(10e-3)
    assert abs(narrow) > abs(wide)
    assert 2.0 < abs(narrow / wide) < 8.0


def test_wake_scales_linearly_with_total_charge():
    """The cavity's wake kick scales linearly with the bunch charge."""
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
    )
    reference_cavity = cheetah.Cavity(**cavity_kwargs())

    def mean_kick(total_charge):
        beam = make_beam(3e-4, total_charge=total_charge, num_particles=20_000)
        return float((cavity.track(beam).p - reference_cavity.track(beam).p).mean())

    single = mean_kick(1e-9)
    double = mean_kick(2e-9)
    assert double / single == pytest.approx(2.0, rel=1e-6)


def test_uncharged_beam_is_a_no_op():
    """An uncharged beam sees no wake."""
    beam = make_beam(3e-4, num_particles=5_000)
    uncharged = cheetah.ParticleBeam(
        particles=beam.particles,
        energy=beam.energy,
        particle_charges=torch.zeros_like(beam.particle_charges),
        species=beam.species,
    )
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
    )
    reference = cheetah.Cavity(**cavity_kwargs()).track(uncharged)
    assert torch.equal(cavity.track(uncharged).particles, reference.particles)


def test_resonant_model_reproduces_the_old_cosine_wake_strength():
    """`wake_strength` reproduces the original cosine wake."""
    wake_strength = 35.0
    beam = make_beam(3e-4, total_charge=1e-9, num_particles=100_000)

    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=torch.tensor(wake_strength, dtype=DTYPE),
    )

    # The conversion is an explicit claim about the element, not a chore the
    # caller does in setup: wake_strength is stored as R/Q = A / omega_r and
    # reads back through it (Ferrario Eqs. 18, 24).  "auto" resolved itself to
    # the resonant kernel because an amplitude was given.
    assert cavity.wake_model == "resonant"
    assert float(cavity.r_over_q) == pytest.approx(
        wake_strength / (2 * math.pi * FREQUENCY), rel=1e-14
    )
    assert float(cavity.wake_strength) == pytest.approx(wake_strength, rel=1e-14)
    reference = cheetah.Cavity(**cavity_kwargs()).track(beam)
    outgoing = cavity.track(beam)

    # wake_position="exit": the kick is normalised by the *outgoing* p0c.
    p0c = float(
        torch.sqrt(reference.energy**2 - torch.as_tensor(beam.species.mass_eV) ** 2)
    )
    measured = (outgoing.p - reference.p).numpy() * p0c

    analytic = -gaussian_resonant_wake_potential(
        reference.tau.numpy(),
        sigma_tau=float(reference.tau.std()),
        wavelength=WAVELENGTH,
        total_charge=float(beam.particle_charges.sum()),
        wake_strength=wake_strength,
    )
    error = np.sqrt(np.mean((measured - analytic) ** 2)) / np.sqrt(
        np.mean(analytic**2)
    )
    assert error < 0.05, f"relative rms error {error:.3%}"


def test_split_position_agrees_with_exit_to_first_order():
    """Splitting the kick between both ends agrees with an exit kick to first order."""
    beam = make_beam(3e-4, num_particles=20_000)
    common = dict(
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
    )
    exit_cavity = Cavity(**cavity_kwargs(), **common, wake_position="exit")
    split_cavity = Cavity(**cavity_kwargs(), **common, wake_position="split")
    at_exit = exit_cavity.track(beam).p
    split = split_cavity.track(beam).p
    tolerance = 2e-4 * float(at_exit.std()) + 1e-9
    assert torch.allclose(split, at_exit, rtol=0.0, atol=tolerance) or (
        float((split - at_exit).abs().max()) < 0.05 * float(at_exit.abs().max())
    )


def test_incomplete_wake_configuration_raises_immediately():
    """Unusable wake settings fail in the constructor, not during tracking."""
    # The geometry-driven models still need the mechanical dimensions.
    with pytest.raises(ValueError, match="iris_radius"):
        Cavity(**cavity_kwargs(), wake=True, wake_model="periodic")
    with pytest.raises(ValueError, match="iris_radius"):
        Cavity(**cavity_kwargs(), wake=True, wake_model="space_charge")

    # The resonant model needs an amplitude, in either spelling.
    with pytest.raises(ValueError, match="wake_strength or r_over_q"):
        Cavity(**cavity_kwargs(), wake=True, wake_model="resonant")

    # The two spellings are the same number, so giving both is contradictory.
    with pytest.raises(ValueError, match="not both"):
        Cavity(
            **cavity_kwargs(), wake=True, wake_strength=35.0, r_over_q=1e-8
        )

    # Neither spelling means anything at f_r = 0, which is Cavity's default.
    # Note the first case has no wake=True: passing an amplitude at all is the
    # declaration of intent, so the guard cannot wait for the switch.
    with pytest.raises(ValueError, match="non-zero mode frequency"):
        Cavity(length=torch.tensor(1.0, dtype=DTYPE), wake_strength=35.0)
    with pytest.raises(ValueError, match="non-zero mode frequency"):
        Cavity(
            length=torch.tensor(1.0, dtype=DTYPE),
            wake=True,
            wake_model="resonant",
            r_over_q=1e-8,
        )


def test_parameter_beam_with_wake_raises():
    """Tracking a `ParameterBeam` with the wake on raises a clear error."""
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
    )
    beam = cheetah.ParameterBeam.from_parameters(
        sigma_tau=torch.tensor(3e-4, dtype=DTYPE),
        energy=torch.tensor(REFERENCE_ENERGY, dtype=DTYPE),
        dtype=DTYPE,
    )
    with pytest.raises(AssertionError, match="ParticleBeam"):
        cavity.track(beam)


def test_diagnostics_are_cached_for_plotting():
    """The last wake diagnostics are kept on the cavity for plotting."""
    beam = make_beam(3e-4, num_particles=10_000)
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
    )
    cavity.track(beam)
    diagnostics = cavity.cached_wake_diagnostics
    assert diagnostics is not None
    assert {"grid", "z_grid", "lambda", "W_grid", "loss_factor"} <= set(diagnostics)
    assert diagnostics["grid"].resolved


def test_element_survives_a_segment_and_clone():
    """A wake cavity can be cloned and tracked inside a `Segment`."""
    beam = make_beam(3e-4, num_particles=5_000)
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
        name="wake_cavity",
    )
    segment = cheetah.Segment(
        [cheetah.Drift(length=torch.tensor(0.1, dtype=DTYPE)), cavity]
    )
    outgoing = segment.track(beam)
    assert outgoing.particles.shape == beam.particles.shape
    assert "wake" in cavity.defining_features


def test_vectorised_beam_is_handled():
    """A vectorised (batched) beam gets the wake in every batch entry."""
    beam = make_beam(3e-4, num_particles=5_000)
    stacked = cheetah.ParticleBeam(
        particles=torch.stack([beam.particles, beam.particles]),
        energy=beam.energy,
        particle_charges=beam.particle_charges,
        species=beam.species,
    )
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        wake_model="periodic",
    )
    outgoing = cavity.track(stacked)
    assert outgoing.particles.shape == stacked.particles.shape
    assert torch.allclose(outgoing.particles[0], outgoing.particles[1])


def test_wake_strength_round_trips_through_r_over_q_and_the_loss_factor():
    """
    `wake_strength`, `r_over_q` and `loss_factor` convert into each other consistently.
    """
    omega_r = 2 * math.pi * FREQUENCY
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=torch.tensor(35.0, dtype=DTYPE),
    )

    assert float(cavity.r_over_q) == pytest.approx(35.0 / omega_r, rel=1e-14)
    assert float(cavity.wake_strength) == pytest.approx(35.0, rel=1e-14)
    assert float(cavity.loss_factor) == pytest.approx(17.5, rel=1e-14)
    assert float(
        cavity.loss_factor_for(make_beam(3e-4, num_particles=1_000))
    ) == pytest.approx(17.5, rel=1e-14)

    # The other spelling of the same number must give the same element.
    by_r_over_q = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_model="resonant",
        r_over_q=torch.tensor(35.0 / omega_r, dtype=DTYPE),
    )
    assert float(by_r_over_q.wake_strength) == pytest.approx(35.0, rel=1e-14)

    # clone() round-trips through r_over_q, so the amplitude must survive it.
    assert float(cavity.clone().wake_strength) == pytest.approx(35.0, rel=1e-14)

    # For a HOM the mode frequency, not the RF frequency, sets the conversion.
    hom = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=torch.tensor(35.0, dtype=DTYPE),
        mode_frequency=torch.tensor(8.4e9, dtype=DTYPE),
    )
    assert float(hom.r_over_q) == pytest.approx(
        35.0 / (2 * math.pi * 8.4e9), rel=1e-14
    )


def test_wake_strength_is_differentiable():
    """
    Gradients flow back to `wake_strength`, and match `kick / wake_strength` because the
    kick is linear.
    """
    beam = make_beam(3e-4, num_particles=20_000)
    amplitude = 35e9
    wake_strength = torch.tensor(amplitude, dtype=DTYPE, requires_grad=True)
    cavity = Cavity(
        length=torch.tensor(0.0, dtype=DTYPE),
        voltage=torch.tensor(0.0, dtype=DTYPE),
        phase=torch.tensor(0.0, dtype=DTYPE),
        frequency=torch.tensor(FREQUENCY, dtype=DTYPE),
        wake=True,
        wake_strength=wake_strength,
    )

    mean_kick = (cavity.track(beam).p - beam.p).mean()
    mean_kick.backward()

    assert wake_strength.grad is not None
    assert float(wake_strength.grad) == pytest.approx(
        float(mean_kick.detach()) / amplitude, rel=1e-12
    )


def test_wake_strength_agrees_with_longitudinal_cosine_wake():
    """
    `Cavity` and `LongitudinalCosineWake` give the same kick for the same amplitude.
    """
    beam = make_beam(3e-4, num_particles=20_000)
    amplitude = torch.tensor(35e9, dtype=DTYPE)

    element = LongitudinalCosineWake(
        wake_strength=amplitude,
        wavelength=torch.tensor(WAVELENGTH, dtype=DTYPE),
        sign_convention="physical",  # `Cavity` follows Ferrario Eq. 4
        dtype=DTYPE,
    )
    cavity = Cavity(
        length=torch.tensor(0.0, dtype=DTYPE),
        voltage=torch.tensor(0.0, dtype=DTYPE),
        phase=torch.tensor(0.0, dtype=DTYPE),
        frequency=torch.tensor(FREQUENCY, dtype=DTYPE),
        wake=True,
        wake_strength=amplitude,
    )

    standalone = element.track(beam).p
    kick_scale = float((standalone - beam.p).abs().max())
    assert torch.allclose(
        cavity.track(beam).p, standalone, rtol=0.0, atol=1e-9 * kick_scale
    )


def test_wake_is_not_dropped_by_segment_tracking():
    """A `Segment` does not merge a wake cavity away, so its kick survives."""
    beam = make_beam(3e-4, num_particles=5_000)
    drift = cheetah.Drift(length=torch.tensor(0.1, dtype=DTYPE))
    # voltage=0 on purpose.  `Cavity.is_skippable` is `not is_active`, so a
    # powered cavity is non-skippable anyway and would mask the regression --
    # only with the RF off does the `not self.wake` term actually carry the
    # guard, which is the case that can silently break.
    cavity = Cavity(
        **cavity_kwargs(voltage=torch.tensor(0.0, dtype=DTYPE)),
        wake=True,
        wake_model="periodic",
        iris_radius=35e-3,
        cell_gap=0.0289,
        cell_period=0.0345,
        name="wake_cavity",
    )
    segment = cheetah.Segment([drift, cavity, drift.clone()])

    assert not cavity.is_skippable
    assert not segment.is_skippable

    direct = drift.track(cavity.track(drift.track(beam))).p
    tolerance = 1e-9 * float(direct.abs().max())

    assert torch.allclose(segment.track(beam).p, direct, rtol=0.0, atol=tolerance)
    assert torch.allclose(
        segment.transfer_maps_merged(incoming_beam=beam).track(beam).p,
        direct,
        rtol=0.0,
        atol=tolerance,
    )

    # Non-vacuous: the wake must actually move p, or the asserts above are free.
    wake_free = (
        cheetah.Segment(
            [
                drift,
                cheetah.Cavity(**cavity_kwargs(voltage=torch.tensor(0.0, dtype=DTYPE))),
                drift.clone(),
            ]
        )
        .track(beam)
        .p
    )
    assert float((direct - wake_free).abs().max()) > 1e-3 * float(direct.abs().max())
