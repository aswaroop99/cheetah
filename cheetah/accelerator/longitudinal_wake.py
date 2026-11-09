"""
Standalone longitudinal wake element, not tied to a cavity.

Use it for a wake on its own (e.g. on a drift) or to scan the wake amplitude. For an
accelerating structure use `Cavity` with `wake=True` instead.
"""

from __future__ import annotations

import math
from typing import Literal

import matplotlib.pyplot as plt
import torch

from cheetah.accelerator.element import Element
from cheetah.accelerator.wakefields import (
    SPEED_OF_LIGHT,
    LongitudinalGrid,
    ResonantModeWake,
    longitudinal_wake_potential,
    reference_momentum,
)
from cheetah.particles import ParticleBeam

__all__ = ["LongitudinalCosineWake"]


class LongitudinalCosineWake(Element):
    """
    Single-mode (cosine) longitudinal wake, applied as a thin momentum kick.

    :param wake_strength: Wake amplitude `w(0+)` in V/C, twice the loss factor.
    :param wavelength: Mode wavelength in meters.
    :param quality_factor: Mode quality factor `Q`. Infinite gives an undamped cosine.
    :param n_grid: Fixed number of grid cells. `None` sizes the grid from the beam,
        which is recommended because a fixed grid can under-resolve short bunches.
    :param pad_wavelengths: Padding on each side of the bunch in wavelengths, only
        used with a fixed `n_grid`.
    :param cells_per_feature: Grid cells per shortest length scale when `n_grid` is
        `None`.
    :param max_grid: Maximum number of grid cells.
    :param sign_convention: `"legacy"` means a positive `wake_strength` adds energy,
        as in the original element. `"physical"` means it removes energy (Ferrario
        Eq. 4), as `Cavity` does.
    :param trapezoidal: Halve the kernel's `s = 0` sample (Ferrario Eq. 18). `False`
        gives the original rectangle rule.
    """

    def __init__(
        self,
        wake_strength: torch.Tensor | float,
        wavelength: torch.Tensor | float,
        quality_factor: torch.Tensor | float = math.inf,
        n_grid: int | None = None,
        pad_wavelengths: float = 4.0,
        cells_per_feature: float = 8.0,
        max_grid: int = 1 << 17,
        # Default kept as "legacy" for the original behaviour; consider switching to "physical" to match `Cavity` for ferrario stuff
        sign_convention: Literal["legacy", "physical"] = "legacy",
        trapezoidal: bool = True,
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            name=name,
            sanitize_name=sanitize_name,
            metadata=metadata,
            **factory_kwargs,
        )

        self.n_grid = None if n_grid is None else int(n_grid)
        self.pad_wavelengths = float(pad_wavelengths)
        self.cells_per_feature = float(cells_per_feature)
        self.max_grid = int(max_grid)
        self.sign_convention = sign_convention
        self.trapezoidal = bool(trapezoidal)
        self.quality_factor = quality_factor

        self.register_buffer_or_parameter(
            "wake_strength", torch.as_tensor(wake_strength, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "wavelength", torch.as_tensor(wavelength, **factory_kwargs)
        )

        self.cached_diagnostics: list[dict] | None = None

    @classmethod
    def from_cavity(
        cls,
        cavity: Element,
        wake_strength: torch.Tensor | float,
        n_grid: int | None = None,
        **kwargs,
    ) -> "LongitudinalCosineWake":
        """Create the wake with its wavelength taken from a cavity's RF frequency."""
        wavelength = SPEED_OF_LIGHT / float(cavity.frequency.detach().cpu())
        return cls(
            wake_strength=wake_strength,
            wavelength=wavelength,
            n_grid=n_grid,
            **kwargs,
        )

    @classmethod
    def from_loss_factor(
        cls,
        loss_factor: torch.Tensor | float,
        wavelength: torch.Tensor | float,
        **kwargs,
    ) -> "LongitudinalCosineWake":
        """Create the wake from a loss factor `k`, using `wake_strength = 2k`."""
        return cls(wake_strength=2.0 * loss_factor, wavelength=wavelength, **kwargs)

    @property
    def loss_factor(self) -> float:
        """Loss factor `k = w(0+) / 2` in V/C."""
        return 0.5 * float(self.wake_strength.detach().cpu())

    @property
    def r_over_q(self) -> float:
        """Equivalent `R/Q` in ohm, `wake_strength / omega_r`."""
        omega_r = 2.0 * math.pi * SPEED_OF_LIGHT / float(self.wavelength.detach().cpu())
        return float(self.wake_strength.detach().cpu()) / omega_r

    def kernel(self) -> ResonantModeWake:
        """The resonant-mode wake function this element applies."""
        wavelength = float(self.wavelength.detach().cpu())
        frequency = SPEED_OF_LIGHT / wavelength
        omega_r = 2.0 * math.pi * frequency
        return ResonantModeWake(
            frequency=frequency,
            r_over_q=self.wake_strength / omega_r,
            quality_factor=self.quality_factor,
        )

    def track(self, incoming: ParticleBeam) -> ParticleBeam:
        """Apply the wake kick to the momentum of `incoming`."""
        # A wake needs the longitudinal charge distribution, which a `ParameterBeam`
        # does not have
        if not isinstance(incoming, ParticleBeam):
            raise AssertionError(
                "Wakefield tracking is currently only supported for `ParticleBeam`, "
                "because a wake needs the longitudinal charge distribution."
            )

        if (
            float(self.wake_strength.detach().cpu()) == 0.0
            or bool(torch.all(incoming.particle_charges == 0))
            # NaN in, NaN out, as in `Cavity._apply_wake`
            or not bool(torch.isfinite(incoming.particles[..., 4]).all())
        ):
            self.cached_diagnostics = None
            return incoming

        kernel = self.kernel()
        sign = -1.0 if self.sign_convention == "physical" else +1.0

        particles = incoming.particles
        batch_shape = particles.shape[:-2]
        num_particles = particles.shape[-2]
        n_batch = math.prod(batch_shape) if batch_shape else 1

        flat = particles.reshape(n_batch, num_particles, 7)
        charges = torch.broadcast_to(
            incoming.particle_charges, (*batch_shape, num_particles)
        ).reshape(n_batch, num_particles)

        energy = incoming.energy
        if energy.shape == batch_shape and energy.numel() == n_batch:
            energy_flat = energy.reshape(n_batch)
        else:
            energy_flat = energy.detach().reshape(-1).mean().expand(n_batch)
        p0c = reference_momentum(energy_flat, incoming.species.mass_eV)

        out = flat.clone()
        diagnostics: list[dict] = []

        for index in range(n_batch):
            tau = flat[index, :, 4]
            grid = self._fixed_grid(tau) if self.n_grid is not None else None
            wake, diagnostic = longitudinal_wake_potential(
                tau,
                charges[index],
                kernel,
                grid=grid,
                cells_per_feature=self.cells_per_feature,
                max_grid=self.max_grid,
                trapezoidal=self.trapezoidal,
            )
            out[index, :, 5] = flat[index, :, 5] + sign * wake / p0c[index]
            diagnostics.append(diagnostic)

        self.cached_diagnostics = diagnostics

        return ParticleBeam(
            particles=out.reshape(*batch_shape, num_particles, 7),
            energy=incoming.energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s,
            species=incoming.species,
        )

    def _fixed_grid(self, tau: torch.Tensor) -> LongitudinalGrid:
        """The original fixed grid: `n_grid` cells, padded by `pad_wavelengths`."""
        wavelength = float(self.wavelength.detach().cpu())
        tau_detached = tau.detach().reshape(-1)
        z_min = float(tau_detached.min()) - self.pad_wavelengths * wavelength
        z_max = float(tau_detached.max()) + self.pad_wavelengths * wavelength
        if z_max - z_min < 1e-9:
            z_max = z_min + wavelength
        dz = (z_max - z_min) / (self.n_grid - 1)
        sigma_tau = float(tau_detached.std()) if tau_detached.numel() > 1 else 0.0
        return LongitudinalGrid(
            z_min=z_min,
            dz=dz,
            n_grid=self.n_grid,
            sigma_tau=sigma_tau,
            target_dz=dz,
            resolved=(sigma_tau == 0.0 or dz <= 0.2 * sigma_tau),
        )

    @property
    def is_skippable(self) -> bool:
        """Never skippable, so `Segment` never merges the wake away."""
        return False

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        """Draw the wake as a vertical line at its position."""
        ax = ax or plt.subplot(111)
        plot_s = s[vector_idx] if torch.is_tensor(s) and s.dim() > 0 else s
        ax.axvline(plot_s, ymin=0.01, ymax=0.99, color="teal", linestyle="-")
        return ax

    @property
    def defining_features(self) -> list[str]:
        """Constructor arguments needed to rebuild this element."""
        return super().defining_features + [
            "wake_strength",
            "wavelength",
            "quality_factor",
            "n_grid",
            "pad_wavelengths",
            "cells_per_feature",
            "sign_convention",
            "trapezoidal",
            "max_grid",
        ]

    def __repr__(self) -> str:
        """Short summary of the element."""
        return (
            f"{self.__class__.__name__}("
            f"wake_strength={self.wake_strength}, "
            f"wavelength={self.wavelength}, "
            f"n_grid={self.n_grid}, "
            f'sign_convention="{self.sign_convention}", '
            f'name="{self.name}")'
        )
