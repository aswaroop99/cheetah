"""
Wakefield physics used by Cavity and LongitudinalCosineWake.

The wake function w(s) depends only on the structure. The beam enters through its line
charge density lambda, in the wake potential W = w * lambda (Ferrario Eq. 27), and
through the grid, which is sized from the bunch length.

Signs: w_parallel > 0 means a trailing particle loses energy  and w_perp > 0 is defocusing via transverse.
The separation is s = tau_test - tau_source >= 0, so only particles ahead contribute by causality.

Reference: M. Ferrario, M. Migliorati and L. Palumbo, "Wakefields and Instabilities in
Linear Accelerators", CERN-2014-009, pp. 357-375.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
from scipy import constants

from cheetah.utils import cloud_in_cell_charge_deposition

__all__ = [
    "SPEED_OF_LIGHT",
    "VACUUM_IMPEDANCE",
    "LongitudinalWakeKernel",
    "TransverseWakeKernel",
    "ResonantModeWake",
    "ResonantModeTransverseWake",
    "PeriodicStructureWake",
    "PeriodicStructureTransverseWake",
    "DiffractionModelWake",
    "DiffractionModelTransverseWake",
    "CavityGeometry",
    "LongitudinalGrid",
    "longitudinal_grid_from_beam",
    "deposit_line_density",
    "convolve_causal_wake",
    "longitudinal_wake_potential",
    "longitudinal_energy_kick",
    "transverse_dipole_wake_kick",
    "space_charge_wake_potential",
    "bunch_energy_loss",
    "reference_momentum",
    "gaussian_resonant_wake_potential",
]

SPEED_OF_LIGHT = constants.speed_of_light
VACUUM_IMPEDANCE = float(constants.mu_0 * constants.speed_of_light)  # Z_0 [ohm]
_EPSILON_0 = constants.epsilon_0


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _like(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """Cast 'value' to the device and dtype of 'reference', keeping gradients."""
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _scalar(value: torch.Tensor | float) -> float:
    """Detached Python float, for grid decisions only."""
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def _next_power_of_two(n: int) -> int:
    """Smallest power of two that is at least 'n'."""
    return 1 << max(0, int(n - 1)).bit_length()


def reference_momentum(
    energy_eV: torch.Tensor, mass_eV: torch.Tensor | float
) -> torch.Tensor:
    """Reference momentum 'p0c' in eV from total energy and rest mass."""
    mass_eV = _like(mass_eV, energy_eV)
    return torch.sqrt(torch.clamp(energy_eV**2 - mass_eV**2, min=0.0))


# --------------------------------------------------------------------------------------
# Wake Green functions
# --------------------------------------------------------------------------------------


class LongitudinalWakeKernel(ABC):
    """Base class for a longitudinal wake function 'w_parallel(s)' in V/C."""

    @abstractmethod
    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_parallel' at separations 's >= 0' behind the source."""

    @abstractmethod
    def cell_average(self, separations: torch.Tensor, dz: float) -> torch.Tensor:
        """
        Average of 'w_parallel' over '[s, s + dz)', for the first grid cell where some
        wakes diverge.
        """

    @property
    @abstractmethod
    def feature_length(self) -> float:
        """Shortest length scale the grid must resolve, in meters."""

    @property
    @abstractmethod
    def value_at_origin(self) -> float:
        """'w_parallel(0+)' in V/C."""

    @property
    def loss_factor(self) -> float:
        """Loss factor 'k = w_parallel(0+) / 2' (Ferrario Eq. 18)."""
        return 0.5 * self.value_at_origin


class TransverseWakeKernel(ABC):
    """Base class for a transverse dipole wake function 'w_perp(s)' in V/(C m)."""

    @abstractmethod
    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_perp' at separations 's >= 0' behind the source."""

    def cell_average(self, separations: torch.Tensor, dz: float) -> torch.Tensor:
        """Average over a grid cell. Point sampling unless a wake overrides it."""
        return self.sample(separations)

    @property
    @abstractmethod
    def feature_length(self) -> float:
        """Shortest length scale the grid must resolve, in meters."""


@dataclass(frozen=True)
class ResonantModeWake(LongitudinalWakeKernel):
    """
    Wake of a single resonant mode (Ferrario Eq. 11).

    'w(s) = omega_r (R/Q) exp(-Gamma s / c) [cos(omega_bar s / c) - (Gamma / omega_bar)
    sin(omega_bar s / c)]' with 'Gamma = omega_r / 2Q'. 'R/Q' uses the loss-factor
    convention, 'w(0+) = omega_r (R/Q) = 2k'; the circuit convention differs by a factor
    of two. With 'Q = inf' this is the cosine wake of 'LongitudinalCosineWake'.

    :param frequency: Mode frequency in Hz.
    :param r_over_q: Mode 'R/Q' in ohm.
    :param quality_factor: Mode quality factor 'Q'. Infinite means no damping.
    """

    frequency: float | torch.Tensor
    r_over_q: float | torch.Tensor
    quality_factor: float | torch.Tensor = math.inf

    @classmethod
    def from_loss_factor(
        cls,
        frequency: float | torch.Tensor,
        loss_factor: float | torch.Tensor,
        quality_factor: float | torch.Tensor = math.inf,
    ) -> "ResonantModeWake":
        """Create from a loss factor 'k' in V/C instead of 'R/Q'."""
        omega_r = 2.0 * math.pi * _scalar(frequency)
        r_over_q = (2.0 * loss_factor) / omega_r
        return cls(
            frequency=frequency, r_over_q=r_over_q, quality_factor=quality_factor
        )

    @classmethod
    def from_shunt_impedance(
        cls,
        frequency: float | torch.Tensor,
        shunt_impedance: float | torch.Tensor,
        quality_factor: float | torch.Tensor,
    ) -> "ResonantModeWake":
        """Create from a shunt impedance 'R' in ohm and 'Q', as tabulated for HOMs."""
        return cls(
            frequency=frequency,
            r_over_q=shunt_impedance / quality_factor,
            quality_factor=quality_factor,
        )

    # -- geometry-free derived quantities ------------------------------------------

    @property
    def angular_frequency(self) -> float:
        """Mode angular frequency 'omega_r' in rad/s."""
        return 2.0 * math.pi * _scalar(self.frequency)

    @property
    def wavelength(self) -> float:
        """Mode wavelength in meters."""
        return SPEED_OF_LIGHT / _scalar(self.frequency)

    @property
    def _damping_rate(self) -> float:
        """Damping rate 'Gamma = omega_r / 2Q', zero without damping."""
        q = _scalar(self.quality_factor)
        return 0.0 if math.isinf(q) else self.angular_frequency / (2.0 * q)

    @property
    def decay_length(self) -> float:
        """Distance 'c / Gamma' over which the mode rings down."""
        gamma = self._damping_rate
        return math.inf if gamma == 0.0 else SPEED_OF_LIGHT / gamma

    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_parallel' at separations 's', zero ahead of the source."""
        omega_r = _like(2.0 * math.pi * _like(self.frequency, separations), separations)
        r_over_q = _like(self.r_over_q, separations)
        q = _scalar(self.quality_factor)

        if math.isinf(q):
            phase = omega_r * separations / SPEED_OF_LIGHT
            wake = omega_r * r_over_q * torch.cos(phase)
        else:
            gamma = omega_r / (2.0 * _like(q, separations))
            omega_bar = torch.sqrt(torch.clamp(omega_r**2 - gamma**2, min=1e-30))
            phase = omega_bar * separations / SPEED_OF_LIGHT
            envelope = torch.exp(-gamma * separations / SPEED_OF_LIGHT)
            wake = (
                omega_r
                * r_over_q
                * envelope
                * (torch.cos(phase) - (gamma / omega_bar) * torch.sin(phase))
            )
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    def cell_average(self, separations: torch.Tensor, dz: float) -> torch.Tensor:
        """Midpoint sample, which is enough since this wake is smooth at 's = 0'."""
        return self.sample(separations + 0.5 * dz)

    @property
    def feature_length(self) -> float:
        """One radian of phase, 'lambda / 2 pi'."""
        return self.wavelength / (2.0 * math.pi)

    @property
    def value_at_origin(self) -> float:
        """'w(0+) = omega_r (R/Q)' in V/C."""
        return self.angular_frequency * _scalar(self.r_over_q)


@dataclass(frozen=True)
class ResonantModeTransverseWake(TransverseWakeKernel):
    """
    Transverse dipole wake of a single resonant mode (Ferrario Eq. 12).

    :param frequency: Mode frequency in Hz.
    :param transverse_shunt_impedance: Transverse shunt impedance 'R_perp' in ohm/m.
    :param quality_factor: Mode quality factor 'Q'.
    """

    frequency: float | torch.Tensor
    transverse_shunt_impedance: float | torch.Tensor
    quality_factor: float | torch.Tensor

    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_perp' at separations 's', zero ahead of the source."""
        omega_r = 2.0 * math.pi * _like(self.frequency, separations)
        q = _like(self.quality_factor, separations)
        r_perp = _like(self.transverse_shunt_impedance, separations)

        gamma = omega_r / (2.0 * q)
        omega_bar = torch.sqrt(torch.clamp(omega_r**2 - gamma**2, min=1e-30))
        wake = (
            r_perp
            * omega_r
            / q
            * torch.exp(-gamma * separations / SPEED_OF_LIGHT)
            * torch.sin(omega_bar * separations / SPEED_OF_LIGHT)
        )
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    @property
    def feature_length(self) -> float:
        """One radian of phase, 'lambda / 2 pi'."""
        return SPEED_OF_LIGHT / (2.0 * math.pi * _scalar(self.frequency))


@dataclass(frozen=True)
class PeriodicStructureWake(LongitudinalWakeKernel):
    """
    Short-range longitudinal wake of a periodic structure (Ferrario Eqs. 15-16).

    'w(s) = Z_0 c L / (pi b^2) exp(-sqrt(s / s1))' with 's1 = 0.41 b^1.8 g^1.6 / p^2.4'.
    Ferrario writes it per period; here 'L' is the total length. It scales as '1 / b^2'.

    :param iris_radius: Iris radius 'b' in meters.
    :param cell_gap: Cell gap 'g' in meters.
    :param cell_period: Cell period 'p' in meters.
    :param length: Total structure length 'L' in meters.
    """

    iris_radius: float | torch.Tensor
    cell_gap: float | torch.Tensor
    cell_period: float | torch.Tensor
    length: float | torch.Tensor

    @property
    def s1(self) -> float:
        """Scale length 's1' in meters (Ferrario Eq. 16)."""
        b = _scalar(self.iris_radius)
        g = _scalar(self.cell_gap)
        p = _scalar(self.cell_period)
        return 0.41 * b**1.8 * g**1.6 / p**2.4

    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_parallel' at separations 's', zero ahead of the source."""
        amplitude = self._amplitude(separations)
        s1 = _like(self.s1, separations)
        wake = amplitude * torch.exp(
            -torch.sqrt(torch.clamp(separations, min=0.0) / s1)
        )
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    def cell_average(self, separations: torch.Tensor, dz: float) -> torch.Tensor:
        """
        Exact average over a grid cell, needed because the slope is infinite at 's = 0'.
        """
        s1 = self.s1
        lo = torch.clamp(separations, min=0.0)
        hi = lo + dz

        def antiderivative(s: torch.Tensor) -> torch.Tensor:
            u = torch.sqrt(s / _like(s1, s))
            return -2.0 * _like(s1, s) * (u + 1.0) * torch.exp(-u)

        mean = (antiderivative(hi) - antiderivative(lo)) / dz
        wake = self._amplitude(separations) * mean
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    def _amplitude(self, reference: torch.Tensor) -> torch.Tensor:
        """'w(0+) = Z_0 c L / (pi b^2)' as a tensor."""
        b = _like(self.iris_radius, reference)
        length = _like(self.length, reference)
        return VACUUM_IMPEDANCE * SPEED_OF_LIGHT * length / (math.pi * b**2)

    @property
    def feature_length(self) -> float:
        """The scale length 's1'."""
        return self.s1

    @property
    def value_at_origin(self) -> float:
        """'w(0+) = Z_0 c L / (pi b^2)' in V/C."""
        b = _scalar(self.iris_radius)
        length = _scalar(self.length)
        return VACUUM_IMPEDANCE * SPEED_OF_LIGHT * length / (math.pi * b**2)


@dataclass(frozen=True)
class PeriodicStructureTransverseWake(TransverseWakeKernel):
    """
    Short-range transverse dipole wake of a periodic structure (Ferrario Eq. 15).

    'w(s) = 4 Z_0 c L s2 / (pi b^4) [1 - (1 + sqrt(s / s2)) exp(-sqrt(s / s2))]' with
    's2 = 0.17 b^1.79 g^0.38 / p^1.17'. It scales as '1 / b^4', so it is far more
    sensitive to the aperture than the longitudinal wake.
    """

    iris_radius: float | torch.Tensor
    cell_gap: float | torch.Tensor
    cell_period: float | torch.Tensor
    length: float | torch.Tensor

    @property
    def s2(self) -> float:
        """Scale length 's2' in meters (Ferrario Eq. 16)."""
        b = _scalar(self.iris_radius)
        g = _scalar(self.cell_gap)
        p = _scalar(self.cell_period)
        return 0.17 * b**1.79 * g**0.38 / p**1.17

    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_perp' at separations 's', zero ahead of the source."""
        b = _like(self.iris_radius, separations)
        length = _like(self.length, separations)
        s2 = _like(self.s2, separations)
        amplitude = 4.0 * VACUUM_IMPEDANCE * SPEED_OF_LIGHT * length * s2 / (
            math.pi * b**4
        )
        u = torch.sqrt(torch.clamp(separations, min=0.0) / s2)
        wake = amplitude * (1.0 - (1.0 + u) * torch.exp(-u))
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    @property
    def feature_length(self) -> float:
        """The scale length 's2'."""
        return self.s2


@dataclass(frozen=True)
class DiffractionModelWake(LongitudinalWakeKernel):
    """
    Single-cell longitudinal wake from diffraction theory (Ferrario Eq. 13).

    'w(s) = Z_0 c / (sqrt(2) pi^2 b) sqrt(g / s)' per cell, times 'num_cells'. Valid for
    short bunches, 'sigma << b'. It diverges at 's = 0', so the first grid cell uses
    'cell_average'.
    """

    iris_radius: float | torch.Tensor
    cell_gap: float | torch.Tensor
    num_cells: float | torch.Tensor = 1.0

    def _amplitude(self, reference: torch.Tensor) -> torch.Tensor:
        """Prefactor of '1 / sqrt(s)' as a tensor."""
        b = _like(self.iris_radius, reference)
        g = _like(self.cell_gap, reference)
        n = _like(self.num_cells, reference)
        return (
            n
            * VACUUM_IMPEDANCE
            * SPEED_OF_LIGHT
            * torch.sqrt(g)
            / (math.sqrt(2.0) * math.pi**2 * b)
        )

    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_parallel' at separations 's', zero ahead of the source."""
        s = torch.clamp(separations, min=1e-30)
        wake = self._amplitude(separations) / torch.sqrt(s)
        return torch.where(separations > 0.0, wake, torch.zeros_like(wake))

    def cell_average(self, separations: torch.Tensor, dz: float) -> torch.Tensor:
        """Exact average of '1 / sqrt(s)' over a grid cell."""
        lo = torch.clamp(separations, min=0.0)
        hi = lo + dz
        mean = 2.0 * (torch.sqrt(hi) - torch.sqrt(lo)) / dz
        wake = self._amplitude(separations) * mean
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    @property
    def feature_length(self) -> float:
        """The cell gap, since the model has no length scale of its own."""
        return _scalar(self.cell_gap)

    @property
    def value_at_origin(self) -> float:
        """Infinite, since the wake diverges at 's = 0'."""
        return math.inf


@dataclass(frozen=True)
class DiffractionModelTransverseWake(TransverseWakeKernel):
    """
    Single-cell transverse wake from diffraction theory, 'w(s) = 2^1.5 Z_0 c / (pi^2
    b^3) sqrt(g s)' (Ferrario Eq. 13).
    """

    iris_radius: float | torch.Tensor
    cell_gap: float | torch.Tensor
    num_cells: float | torch.Tensor = 1.0

    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_perp' at separations 's', zero ahead of the source."""
        b = _like(self.iris_radius, separations)
        g = _like(self.cell_gap, separations)
        n = _like(self.num_cells, separations)
        amplitude = (
            n * 2.0**1.5 * VACUUM_IMPEDANCE * SPEED_OF_LIGHT / (math.pi**2 * b**3)
        )
        wake = amplitude * torch.sqrt(g * torch.clamp(separations, min=0.0))
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    @property
    def feature_length(self) -> float:
        """The cell gap."""
        return _scalar(self.cell_gap)


# --------------------------------------------------------------------------------------
# Geometry -> kernel
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CavityGeometry:
    """
    Geometry and mode parameters of a structure, from which its wake is built.

    :param iris_radius: Iris radius 'b' in meters.
    :param cell_gap: Cell gap 'g' in meters.
    :param cell_period: Cell period 'p' in meters.
    :param length: Total structure length 'L' in meters.
    :param cell_radius: Cell radius 'd' in meters, only to check the diffraction model.
    :param r_over_q: Mode 'R/Q' in ohm, for the resonant wake.
    :param mode_frequency: Mode frequency in Hz.
    :param quality_factor: Mode quality factor 'Q'.
    :param transverse_shunt_impedance: Transverse shunt impedance in ohm/m.
    """

    iris_radius: float | torch.Tensor | None = None
    cell_gap: float | torch.Tensor | None = None
    cell_period: float | torch.Tensor | None = None
    length: float | torch.Tensor | None = None
    cell_radius: float | torch.Tensor | None = None
    r_over_q: float | torch.Tensor | None = None
    mode_frequency: float | torch.Tensor | None = None
    quality_factor: float | torch.Tensor = math.inf
    transverse_shunt_impedance: float | torch.Tensor | None = None
    _warned: set = field(default_factory=set, repr=False, compare=False)

    # -- derived scales -------------------------------------------------------------

    @property
    def num_cells(self) -> float:
        """Number of cells, 'L / p'."""
        return _scalar(self.length) / _scalar(self.cell_period)

    @property
    def s1(self) -> float:
        """Longitudinal scale length (Ferrario Eq. 16)."""
        return PeriodicStructureWake(
            self.iris_radius, self.cell_gap, self.cell_period, self.length
        ).s1

    @property
    def s2(self) -> float:
        """Transverse scale length (Ferrario Eq. 16)."""
        return PeriodicStructureTransverseWake(
            self.iris_radius, self.cell_gap, self.cell_period, self.length
        ).s2

    def critical_cell_count(self, sigma_tau: float, gamma: float) -> float:
        """
        Critical cell count 'N_cr = b^2 / (2 g (sigma + 2 b / gamma))' (Ferrario Eq.
        14).

        The periodic wake only applies after 'N_cr' cells. Before that the single-cell
        diffraction wake does.
        """
        b = _scalar(self.iris_radius)
        g = _scalar(self.cell_gap)
        return b**2 / (2.0 * g * (sigma_tau + 2.0 * b / gamma))

    def diffraction_model_is_valid(self, sigma_tau: float) -> bool:
        """Whether 'g < (d - b)^2 / (2 sigma)' holds (Ferrario Eq. 13)."""
        if self.cell_radius is None:
            return True
        b = _scalar(self.iris_radius)
        d = _scalar(self.cell_radius)
        return _scalar(self.cell_gap) < (d - b) ** 2 / (2.0 * sigma_tau)

    # -- kernel factories -----------------------------------------------------------

    def longitudinal_kernel(
        self,
        regime: Literal["auto", "periodic", "diffraction", "resonant"] = "auto",
        *,
        sigma_tau: float | None = None,
        gamma: float | None = None,
        warn: bool = True,
    ) -> LongitudinalWakeKernel:
        """
        Longitudinal wake for the given regime.

        '"auto"' picks periodic or diffraction from 'N_cr' (Ferrario Eq. 14) and needs
        'sigma_tau' and 'gamma'.
        """
        if regime == "resonant":
            if self.r_over_q is None or self.mode_frequency is None:
                raise ValueError(
                    "regime='resonant' needs both r_over_q and mode_frequency."
                )
            return ResonantModeWake(
                frequency=self.mode_frequency,
                r_over_q=self.r_over_q,
                quality_factor=self.quality_factor,
            )

        self._require_mechanical()

        if regime == "auto":
            if sigma_tau is None or gamma is None:
                raise ValueError(
                    "regime='auto' needs sigma_tau and gamma to evaluate Eq. 14."
                )
            n_cr = self.critical_cell_count(sigma_tau, gamma)
            regime = "periodic" if self.num_cells >= n_cr else "diffraction"
            if warn and regime == "diffraction" and "n_cr" not in self._warned:
                self._warned.add("n_cr")
                warnings.warn(
                    f"Structure has {self.num_cells:.1f} cells but N_cr = "
                    f"{n_cr:.1f} (Ferrario Eq. 14); using the single-cell "
                    "diffraction wake (Eq. 13) rather than the periodic "
                    "asymptotic form (Eq. 15).",
                    stacklevel=3,
                )

        if regime == "periodic":
            if (
                warn
                and sigma_tau is not None
                and sigma_tau < self.s1
                and "s1" not in self._warned
            ):
                self._warned.add("s1")
                warnings.warn(
                    f"sigma_tau = {sigma_tau:.3e} m is shorter than s1 = "
                    f"{self.s1:.3e} m; the bunch only samples the flat top of "
                    "the periodic wake, so the induced energy spread will be "
                    "small and sensitive to grid resolution.",
                    stacklevel=3,
                )
            return PeriodicStructureWake(
                iris_radius=self.iris_radius,
                cell_gap=self.cell_gap,
                cell_period=self.cell_period,
                length=self.length,
            )

        if regime == "diffraction":
            if warn and sigma_tau is not None:
                if not self.diffraction_model_is_valid(sigma_tau):
                    warnings.warn(
                        "Diffraction-model validity condition "
                        "g < (d - b)^2 / (2 sigma) is violated (Ferrario Eq. 13).",
                        stacklevel=3,
                    )
                #0.3 b is a judgment call for "sigma << b"
                if sigma_tau > 0.3 * _scalar(self.iris_radius):
                    warnings.warn(
                        f"sigma_tau = {sigma_tau:.3e} m is not << b = "
                        f"{_scalar(self.iris_radius):.3e} m; the diffraction "
                        "model assumes a short bunch.",
                        stacklevel=3,
                    )
            return DiffractionModelWake(
                iris_radius=self.iris_radius,
                cell_gap=self.cell_gap,
                num_cells=self.num_cells,
            )

        raise ValueError(f"Unknown regime {regime!r}.")

    def transverse_kernel(
        self,
        regime: Literal["auto", "periodic", "diffraction", "resonant"] = "periodic",
        *,
        sigma_tau: float | None = None,
        gamma: float | None = None,
    ) -> TransverseWakeKernel:
        """Transverse wake for the given regime, chosen as in 'longitudinal_kernel'."""
        if regime == "resonant":
            if self.transverse_shunt_impedance is None or self.mode_frequency is None:
                raise ValueError(
                    "regime='resonant' needs transverse_shunt_impedance and "
                    "mode_frequency."
                )
            return ResonantModeTransverseWake(
                frequency=self.mode_frequency,
                transverse_shunt_impedance=self.transverse_shunt_impedance,
                quality_factor=self.quality_factor,
            )

        self._require_mechanical()

        if regime == "auto":
            if sigma_tau is None or gamma is None:
                raise ValueError("regime='auto' needs sigma_tau and gamma.")
            regime = (
                "periodic"
                if self.num_cells >= self.critical_cell_count(sigma_tau, gamma)
                else "diffraction"
            )

        if regime == "periodic":
            return PeriodicStructureTransverseWake(
                iris_radius=self.iris_radius,
                cell_gap=self.cell_gap,
                cell_period=self.cell_period,
                length=self.length,
            )
        if regime == "diffraction":
            return DiffractionModelTransverseWake(
                iris_radius=self.iris_radius,
                cell_gap=self.cell_gap,
                num_cells=self.num_cells,
            )
        raise ValueError(f"Unknown regime {regime!r}.")

    def _require_mechanical(self) -> None:
        """Raise if any of 'b', 'g', 'p' or 'L' is missing."""
        missing = [
            name
            for name in ("iris_radius", "cell_gap", "cell_period", "length")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "Geometry-derived wakes need " + ", ".join(missing) + "."
            )


# --------------------------------------------------------------------------------------
# Grid: chosen from the beam and from the kernel
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LongitudinalGrid:
    """Uniform grid along 'tau', with the numbers needed to check it."""

    z_min: float
    dz: float
    n_grid: int
    sigma_tau: float
    target_dz: float
    resolved: bool

    @property
    def z_max(self) -> float:
        """Position of the last node."""
        return self.z_min + (self.n_grid - 1) * self.dz

    @property
    def cells_per_sigma(self) -> float:
        """Grid cells per rms bunch length."""
        return self.sigma_tau / self.dz if self.dz > 0 else math.inf

    def nodes(self, device=None, dtype=None) -> torch.Tensor:
        """Node positions as a tensor."""
        return torch.linspace(
            self.z_min, self.z_max, self.n_grid, device=device, dtype=dtype
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        """Short summary of the grid."""
        return (
            f"LongitudinalGrid(n_grid={self.n_grid}, dz={self.dz:.3e} m, "
            f"extent=[{self.z_min:.3e}, {self.z_max:.3e}] m, "
            f"cells_per_sigma={self.cells_per_sigma:.1f}, resolved={self.resolved})"
        )


def longitudinal_grid_from_beam(
    tau: torch.Tensor,
    *,
    feature_lengths: tuple[float, ...] = (),
    cells_per_feature: float = 8.0,
    margin_cells: int = 4,
    min_grid: int = 256,
    max_grid: int = 1 << 17,
    warn: bool = True,
) -> LongitudinalGrid:
    """
    Size the grid from the bunch and the wake's shortest length scale.

    'dz' resolves the shorter of the rms bunch length and the wake's feature lengths
    with 'cells_per_feature' cells. The wake is causal, so there is no padding ahead of
    the bunch, and spare cells from rounding up go behind it.
    """
    tau_flat = tau.detach().reshape(-1)
    tau_min = float(tau_flat.min())
    tau_max = float(tau_flat.max())
    sigma_tau = float(tau_flat.std()) if tau_flat.numel() > 1 else 0.0

    candidates = [sigma_tau] + [f for f in feature_lengths]
    candidates = [
        c for c in candidates if c is not None and math.isfinite(c) and c > 0.0
    ]
    if not candidates:
        raise ValueError(
            "Cannot size the grid: no finite length scale from the beam or kernel."
        )
    target_dz = min(candidates) / cells_per_feature

    span = tau_max - tau_min
    if span <= 0.0:
        span = 2.0 * margin_cells * target_dz

    n_required = int(math.ceil(span / target_dz)) + 2 * margin_cells + 1
    n_grid = min(max(_next_power_of_two(n_required), min_grid), max_grid)

    if n_grid >= n_required:
        dz = target_dz
        resolved = True
    else:
        # Clamped by 'max_grid', so resolution is lost and the user is warned
        dz = (span + 2 * margin_cells * target_dz) / (n_grid - 1)
        resolved = False
        if warn:
            warnings.warn(
                f"Wake grid clamped to max_grid={max_grid}: dz = {dz:.3e} m gives "
                f"only {sigma_tau / dz:.2f} cells per sigma_tau (target "
                f"{cells_per_feature:.0f}). Raise max_grid or lower "
                "cells_per_feature.",
                stacklevel=2,
            )

    return LongitudinalGrid(
        z_min=tau_min - margin_cells * dz,
        dz=dz,
        n_grid=n_grid,
        sigma_tau=sigma_tau,
        target_dz=target_dz,
        resolved=resolved,
    )


# --------------------------------------------------------------------------------------
# Deposition, convolution, gather
# --------------------------------------------------------------------------------------


def deposit_line_density(
    tau: torch.Tensor,
    charges: torch.Tensor,
    grid: LongitudinalGrid,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Deposit charges onto the grid with cloud-in-cell, as a line density in C/m.

    :param weights: Optional per-particle weights, e.g. offsets for the dipole density
        of the transverse wake.
    """
    tau = tau.reshape(-1)
    charges = charges.reshape(-1).to(device=tau.device, dtype=tau.dtype)
    if weights is not None:
        charges = charges * weights.reshape(-1).to(device=tau.device, dtype=tau.dtype)

    extent = torch.tensor(
        [[grid.z_min - 0.5 * grid.dz, grid.z_max + 0.5 * grid.dz]],
        device=tau.device,
        dtype=tau.dtype,
    )
    deposited = cloud_in_cell_charge_deposition(
        positions=tau.reshape(-1, 1),
        bins=grid.n_grid,
        extent=extent,
        charges=charges,
    )
    return deposited / grid.dz


def convolve_causal_wake(
    line_density: torch.Tensor,
    kernel_samples: torch.Tensor,
    dz: float,
    *,
    trapezoidal: bool = True,
) -> torch.Tensor:
    """
    Causal convolution 'W_i = sum_j w_j lambda_(i-j) dz', using FFTs.

    :param trapezoidal: Halve the 'j = 0' sample, so a particle sees half of its own
        slice's wake (Ferrario Eq. 18).
    """
    if trapezoidal:
        kernel_samples = kernel_samples.clone()
        kernel_samples[0] = 0.5 * kernel_samples[0]

    n = line_density.numel()
    m = kernel_samples.numel()
    length = n + m - 1
    n_fft = _next_power_of_two(length)
    convolution = torch.fft.irfft(
        torch.fft.rfft(line_density, n=n_fft) * torch.fft.rfft(kernel_samples, n=n_fft),
        n=n_fft,
    )
    return convolution[:n] * dz


def _gather_linear(
    values_on_grid: torch.Tensor, tau: torch.Tensor, grid: LongitudinalGrid
) -> torch.Tensor:
    """Linearly interpolate a grid quantity back to the particles."""
    u = (tau.reshape(-1) - grid.z_min) / grid.dz
    i0 = torch.clamp(torch.floor(u.detach()).long(), 0, grid.n_grid - 2)
    frac = (u - i0).to(values_on_grid.dtype)
    return (1.0 - frac) * values_on_grid[i0] + frac * values_on_grid[i0 + 1]


# --------------------------------------------------------------------------------------
# Wake potentials and kicks
# --------------------------------------------------------------------------------------


def longitudinal_wake_potential(
    tau: torch.Tensor,
    particle_charges: torch.Tensor,
    kernel: LongitudinalWakeKernel,
    *,
    grid: LongitudinalGrid | None = None,
    cells_per_feature: float = 8.0,
    max_grid: int = 1 << 17,
    trapezoidal: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Longitudinal wake potential 'W' in volts at each particle (Ferrario Eq. 27).

    :return: 'W' per particle, and a dict of diagnostics (grid, line density, kernel and
        'W' on the grid) for plotting.
    """
    tau = tau.reshape(-1)
    charges = particle_charges.reshape(-1).abs()

    if grid is None:
        grid = longitudinal_grid_from_beam(
            tau,
            feature_lengths=(kernel.feature_length,),
            cells_per_feature=cells_per_feature,
            max_grid=max_grid,
        )

    line_density = deposit_line_density(tau, charges, grid)

    separations = torch.arange(
        grid.n_grid, device=tau.device, dtype=tau.dtype
    ) * grid.dz
    kernel_samples = kernel.sample(separations)
    # Some wakes diverge at s = 0, so use the exact cell average for the first cell
    first = kernel.cell_average(separations[:1], grid.dz)
    if torch.isfinite(first).all():
        kernel_samples = torch.cat([first, kernel_samples[1:]])

    wake_on_grid = convolve_causal_wake(
        line_density, kernel_samples, grid.dz, trapezoidal=trapezoidal
    )
    wake_per_particle = _gather_linear(wake_on_grid, tau, grid)

    diagnostics = {
        "grid": grid,
        "z_grid": grid.nodes(device=tau.device, dtype=tau.dtype),
        "lambda": line_density,
        "kernel": kernel_samples,
        "W_grid": wake_on_grid,
        "loss_factor": kernel.loss_factor,
    }
    return wake_per_particle, diagnostics


def longitudinal_energy_kick(
    tau: torch.Tensor,
    particle_charges: torch.Tensor,
    kernel: LongitudinalWakeKernel,
    *,
    num_elementary_charges: float = -1.0,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Energy change in eV of each particle from the longitudinal wake.

    'dE = -|Z| W': a positive wake decelerates (Ferrario Eq. 4) for any species, because
    the source and test charges share a sign.
    """
    wake, diagnostics = longitudinal_wake_potential(
        tau, particle_charges, kernel, **kwargs
    )
    return -abs(num_elementary_charges) * wake, diagnostics


def transverse_dipole_wake_kick(
    tau: torch.Tensor,
    offsets: torch.Tensor,
    particle_charges: torch.Tensor,
    kernel: TransverseWakeKernel,
    p0c: torch.Tensor,
    *,
    num_elementary_charges: float = -1.0,
    grid: LongitudinalGrid | None = None,
    cells_per_feature: float = 8.0,
    max_grid: int = 1 << 17,
    trapezoidal: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Transverse kick 'd(p_y / p0)' from the dipole wake (Ferrario Eqs. 5, 33).

    Convolves the dipole density 'lambda <y>' with 'w_perp'. A positive wake pushes the
    tail further along the head's offset.
    """
    # Only checked against the analytical two-particle model, not yet against a Gaussian bunch or another code
    tau = tau.reshape(-1)
    charges = particle_charges.reshape(-1).abs()

    if grid is None:
        grid = longitudinal_grid_from_beam(
            tau,
            feature_lengths=(kernel.feature_length,),
            cells_per_feature=cells_per_feature,
            max_grid=max_grid,
        )

    dipole_density = deposit_line_density(tau, charges, grid, weights=offsets)
    separations = torch.arange(
        grid.n_grid, device=tau.device, dtype=tau.dtype
    ) * grid.dz
    kernel_samples = kernel.sample(separations)

    voltage_on_grid = convolve_causal_wake(
        dipole_density, kernel_samples, grid.dz, trapezoidal=trapezoidal
    )
    voltage = _gather_linear(voltage_on_grid, tau, grid)

    kick = abs(num_elementary_charges) * voltage / _like(p0c, voltage)
    return kick, {
        "grid": grid,
        "z_grid": grid.nodes(device=tau.device, dtype=tau.dtype),
        "dipole_density": dipole_density,
        "V_grid": voltage_on_grid,
    }


def space_charge_wake_potential(
    tau: torch.Tensor,
    particle_charges: torch.Tensor,
    *,
    length: float | torch.Tensor,
    pipe_radius: float | torch.Tensor,
    beam_radius: float | torch.Tensor,
    gamma: float | torch.Tensor,
    grid: LongitudinalGrid | None = None,
    cells_per_feature: float = 8.0,
    max_grid: int = 1 << 17,
) -> tuple[torch.Tensor, dict]:
    """
    Space-charge energy kick in eV (Ferrario Eqs. 6, 7, 29).

    'dE = L / (4 pi eps_0 gamma^2) (1 + 2 ln(b / a)) d lambda / d tau', so the head
    gains and the tail loses with no net change. Unlike the other wakes this one depends
    on the beam, through its radius 'a' and 'gamma'.
    """
    tau = tau.reshape(-1)
    charges = particle_charges.reshape(-1).abs()

    if grid is None:
        grid = longitudinal_grid_from_beam(
            tau, cells_per_feature=cells_per_feature, max_grid=max_grid
        )

    line_density = deposit_line_density(tau, charges, grid)

    prefactor = (
        _like(length, line_density)
        / (4.0 * math.pi * _EPSILON_0 * _like(gamma, line_density) ** 2)
        * (
            1.0
            + 2.0
            * torch.log(
                _like(pipe_radius, line_density) / _like(beam_radius, line_density)
            )
        )
    )

    # Differentiating the deposited density amplifies shot noise by about sigma_tau / dz, so this model wants a coarser grid
    gradient = torch.gradient(line_density, spacing=grid.dz)[0]
    # [m^2 V/C] * [C/m^2] = V, which is already eV per elementary charge
    energy_on_grid = prefactor * gradient

    return _gather_linear(energy_on_grid, tau, grid), {
        "grid": grid,
        "z_grid": grid.nodes(device=tau.device, dtype=tau.dtype),
        "lambda": line_density,
        "dE_grid": energy_on_grid,
        "prefactor": prefactor,
    }


def bunch_energy_loss(
    wake_potential: torch.Tensor, particle_charges: torch.Tensor
) -> torch.Tensor:
    """
    Energy lost by the whole bunch, 'U = -sum q_i W_i', in joules (Ferrario Eq. 28).
    """
    charges = particle_charges.reshape(-1).abs()
    return -(charges * wake_potential.reshape(-1)).sum()


# --------------------------------------------------------------------------------------
# Analytic reference, for tests and for the validation notebook
# --------------------------------------------------------------------------------------


def gaussian_resonant_wake_potential(
    tau,
    *,
    sigma_tau: float,
    wavelength: float,
    total_charge: float,
    loss_factor: float | None = None,
    wake_strength: float | None = None,
):
    """
    Closed-form 'W(tau)' in volts for a Gaussian bunch in an undamped cosine wake.

    Used as the analytical reference in the tests. Give either 'loss_factor' or
    'wake_strength = 2 * loss_factor'.
    """
    import numpy as np
    from scipy.special import erf

    if (loss_factor is None) == (wake_strength is None):
        raise ValueError("Give exactly one of loss_factor or wake_strength.")
    amplitude = wake_strength if wake_strength is not None else 2.0 * loss_factor

    tau = np.asarray(tau, dtype=float)
    k = 2.0 * np.pi / wavelength
    argument = (tau + 1j * k * sigma_tau**2) / (sigma_tau * np.sqrt(2.0))
    return (
        (amplitude * total_charge / 2.0)
        * np.exp(-0.5 * k**2 * sigma_tau**2)
        * np.real(np.exp(1j * k * tau) * (1.0 + erf(argument)))
    )
