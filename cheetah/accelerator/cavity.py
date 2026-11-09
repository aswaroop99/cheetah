import math
from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy import constants

from cheetah.accelerator.element import Element

#wake stuff
from cheetah.accelerator.wakefields import (
    CavityGeometry,
    LongitudinalWakeKernel,
    TransverseWakeKernel,
    longitudinal_energy_kick,
    reference_momentum,
    space_charge_wake_potential,
    transverse_dipole_wake_kick,
)
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import (
    UniqueNameGenerator,
    cache_transfer_map,
    compute_relativistic_factors,
)
from cheetah.utils.autograd import log1pdiv

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

WakeModel = Literal["auto", "periodic", "diffraction", "resonant", "space_charge"]
WakePosition = Literal["entrance", "exit", "split"]


class Cavity(Element):
    """
    Accelerating cavity in a particle accelerator.

    NOTE: When computing the gradient on cavities with exactly zero energy gain, it is
        possible that some gradient components are lost. This is unlikely to cause
        (therefore do not hesitate to have cavities with zero voltage in your lattice),
        but this note is here to make you aware of this possibility.

    :param length: Length in meters.
    :param voltage: Voltage of the cavity in volts. NOTE: This assumes the physical
        voltage. Positive voltage will accelerate electron-like particles.
        For particle with charge `n * e`, the energy gain on crest will be
        `n * voltage`.
    :param phase: Phase of the cavity in degrees.
    :param frequency: Frequency of the cavity in Hz.
    :param cavity_type: Type of the cavity.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element (e.g. control-system addresses or PVs). This information is *not* used
        in simulation and may contain any extra data the user wants to store along with
        the lattice. See :doc:`/getting_started/control_systems` for more information.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.

    The parameters below only matter when wake=True, which adds a longitudinal
    wakefield kick to the RF tracking. With wake=False the cavity behaves exactly
    as without them.

    :param wake: Turn the wakefield on or off.
    :param iris_radius: Iris radius `b` in meters.
    :param cell_gap: Cell gap `g` in meters.
    :param cell_period: Cell period `p` in meters.
    :param cell_radius: Cell radius `d` in meters, only used to check the model applies.
    :param wake_strength: Wake amplitude `w(0+) = 2 pi f_r (R/Q) = 2k` in V/C, e.g.
        from a measurement. Stored as `r_over_q`, so give one or the other.
    :param r_over_q: Mode `R/Q` in ohm, for the resonant model.
    :param quality_factor: Mode quality factor `Q`. Infinite means no damping.
    :param mode_frequency: Mode frequency in Hz. Defaults to `frequency`.
    :param transverse_shunt_impedance: Transverse shunt impedance in ohm/m, for the
        resonant transverse wake.
    :param wake_model: Which wake to use. `"auto"` picks between the periodic and
        diffraction models from the bunch (Ferrario Eq. 14), or `"resonant"` if
        `wake_strength`/`r_over_q` is given.
    :param include_transverse: Also apply the transverse dipole wake (experimental).
    :param wake_position: Where the kick is applied: `"entrance"`, `"exit"` or
        `"split"` (half at each end).
    :param cells_per_feature: Grid cells per shortest length scale of the wake.
    :param max_grid: Maximum number of grid cells.
    :param beam_radius: Beam radius for the space-charge model. Defaults to the
        uniform-disc equivalent of the incoming beam.
    :param cache_diagnostics: Keep the last wake diagnostics for plotting.
    """

    def __init__(
        self,
        length: torch.Tensor,
        voltage: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        frequency: torch.Tensor | None = None,
        cavity_type: Literal["standing_wave", "traveling_wave"] = "standing_wave",
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        wake: bool = False,
        iris_radius: torch.Tensor | float | None = None,
        cell_gap: torch.Tensor | float | None = None,
        cell_period: torch.Tensor | float | None = None,
        cell_radius: torch.Tensor | float | None = None,
        wake_strength: torch.Tensor | float | None = None,
        r_over_q: torch.Tensor | float | None = None,
        quality_factor: torch.Tensor | float = math.inf,
        mode_frequency: torch.Tensor | float | None = None,
        transverse_shunt_impedance: torch.Tensor | float | None = None,
        wake_model: WakeModel = "auto",
        include_transverse: bool = False,
        wake_position: WakePosition = "exit",
        #Grid defaults to try to get reasonable compute time
        cells_per_feature: float = 8.0,
        max_grid: int = 1 << 17,
        beam_radius: torch.Tensor | float | None = None,
        cache_diagnostics: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            name=name, sanitize_name=sanitize_name, metadata=metadata, **factory_kwargs
        )

        self.length = length

        self.register_buffer_or_parameter(
            "voltage",
            voltage if voltage is not None else torch.tensor(0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "phase",
            phase if phase is not None else torch.tensor(0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "frequency",
            frequency if frequency is not None else torch.tensor(0.0, **factory_kwargs),
        )

        self.cavity_type = cavity_type

        # wake_strength = r/q in diff units
        # Uses (Ferrario Eqs. 18, 24) we only store r/q
        # Todo: calc wakestrength directly form geometry via Ferrario eq 18
        if wake_strength is not None:
            if r_over_q is not None:
                raise ValueError(
                    "Give either wake_strength or r_over_q, not both: they are "
                    "the same quantity in different units, "
                    "wake_strength = 2 pi f_r (R/Q) = 2 k."
                )
            mode = mode_frequency if mode_frequency is not None else self.frequency
            # parameter; pass a float64 tensor if you need full precision
            amplitude = torch.as_tensor(wake_strength, **factory_kwargs)
            omega_r = 2.0 * math.pi * torch.as_tensor(
                mode, dtype=amplitude.dtype, device=amplitude.device
            )
            # Checked even with the wake off, so an infinite R/Q can never be stored
            if not bool((omega_r != 0.0).all()):
                raise ValueError(
                    "wake_strength needs a non-zero mode frequency, since "
                    "wake_strength = 2 pi f_r (R/Q). Pass frequency=... or "
                    "mode_frequency=... ."
                )
            # Tensor ops, so gradients flow back to `wake_strength`
            r_over_q = amplitude / omega_r

        self.wake = bool(wake)
        # "auto" means resonant when the amplitude is known, else Ferrario Eq. 14
        self.wake_model: WakeModel = (
            "resonant" if wake_model == "auto" and r_over_q is not None else wake_model
        )
        self.include_transverse = bool(include_transverse)
        self.wake_position: WakePosition = wake_position
        self.cells_per_feature = float(cells_per_feature)
        self.max_grid = int(max_grid)
        # loops
        self.cache_diagnostics = bool(cache_diagnostics)
        self.quality_factor = quality_factor

        for attribute, value in (
            ("iris_radius", iris_radius),
            ("cell_gap", cell_gap),
            ("cell_period", cell_period),
            ("cell_radius", cell_radius),
            ("r_over_q", r_over_q),
            ("transverse_shunt_impedance", transverse_shunt_impedance),
            ("mode_frequency", mode_frequency),
            ("beam_radius", beam_radius),
        ):
            if value is None:
                setattr(self, attribute, None)
            else:
                self.register_buffer_or_parameter(
                    attribute, torch.as_tensor(value, **factory_kwargs)
                )

        self.cached_wake_diagnostics: dict | None = None

        if self.wake:
            self._validate_wake_configuration()

    @property
    def is_active(self) -> bool:
        return (self.voltage != 0).any().item()

    @property
    def is_skippable(self) -> bool:
        # With the wake on, the cavity must never be merged into a transfer map, because that would silently drop the wake
        return not self.is_active and not self.wake

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        return self._cavity_rmatrix(energy, species)

    def track(self, incoming: Beam) -> Beam:
        """Track through the cavity, including the wake if wake=True."""
        if not self.wake:
            return self._track_rf(incoming)
        return self.track_with_wake(incoming)

    #original rf tracking code
    def _track_rf(self, incoming: Beam) -> Beam:
        """RF tracking only, without the wake."""
        gamma0, igamma2, beta0 = compute_relativistic_factors(
            incoming.energy, incoming.species.mass_eV
        )

        phi = self.phase.deg2rad()

        tm = self.first_order_transfer_map(incoming.energy, incoming.species)
        if isinstance(incoming, ParameterBeam):
            outgoing_mu = (tm @ incoming.mu.unsqueeze(-1)).squeeze(-1)
            outgoing_cov = tm @ incoming.cov @ tm.mT
        else:  # ParticleBeam
            outgoing_particles = incoming.particles @ tm.mT
        delta_energy = (
            self.voltage * phi.cos() * incoming.species.num_elementary_charges * -1
        )

        T566 = 1.5 * self.length * igamma2 / beta0.pow(3)
        T556 = self.length.new_zeros(())
        T555 = self.length.new_zeros(())

        k = 2.0 * torch.pi * self.frequency / constants.speed_of_light
        outgoing_energy = incoming.energy + delta_energy
        gamma1, _, beta1 = compute_relativistic_factors(
            outgoing_energy, incoming.species.mass_eV
        )

        if isinstance(incoming, ParameterBeam):
            outgoing_mu[..., 5] = incoming.mu[..., 5] * incoming.energy * beta0 / (
                outgoing_energy * beta1
            ) + self.voltage * beta0 / (outgoing_energy * beta1) * (
                (-incoming.mu[..., 4] * beta0 * k + phi).cos() - phi.cos()
            )
            outgoing_cov[..., 5, 5] = incoming.cov[..., 5, 5]
        else:  # ParticleBeam
            outgoing_particles[..., 5] = incoming.particles[
                ..., 5
            ] * incoming.energy.unsqueeze(-1) * beta0.unsqueeze(-1) / (
                outgoing_energy.unsqueeze(-1) * beta1.unsqueeze(-1)
            ) + self.voltage.unsqueeze(
                -1
            ) * beta0.unsqueeze(
                -1
            ) / (
                outgoing_energy.unsqueeze(-1) * beta1.unsqueeze(-1)
            ) * (
                (
                    -incoming.particles[..., 4] * beta0.unsqueeze(-1) * k.unsqueeze(-1)
                    + phi.unsqueeze(-1)
                ).cos()
                - phi.cos().unsqueeze(-1)
            )

        dgamma = self.voltage / incoming.species.mass_eV

        # TODO: Branch stays in for now because no proper limit exists and would
        # probably require a different model altogether.
        if (delta_energy > 0).any():
            T566 = (
                self.length
                * (beta0.pow(3) * gamma0.pow(3) - beta1.pow(3) * gamma1.pow(3))
                / (
                    2.0
                    * beta0
                    * beta1.pow(3)
                    * gamma0
                    * (gamma0 - gamma1)
                    * gamma1.pow(3)
                )
            )
            T556 = (
                beta0
                * k
                * self.length
                * dgamma
                * gamma0
                * (beta1.pow(3) * gamma1.pow(3) + beta0 * (gamma0 - gamma1.pow(3)))
                * phi.sin()
                / (beta1.pow(3) * gamma1.pow(3) * (gamma0 - gamma1).square())
            )
            T555 = (
                beta0.square()
                * k.square()
                * self.length
                * dgamma
                / 2.0
                * (
                    dgamma
                    * (
                        2.0 * gamma0 * gamma1.pow(3) * (beta0 * beta1.pow(3) - 1.0)
                        + gamma0.square()
                        + 3.0 * gamma1.square()
                        - 2.0
                    )
                    / (beta1.pow(3) * gamma1.pow(3) * (gamma0 - gamma1).pow(3))
                    * phi.sin().square()
                    - (gamma1 * gamma0 * (beta1 * beta0 - 1.0) + 1.0)
                    / (beta1 * gamma1 * (gamma0 - gamma1).square())
                    * phi.cos()
                )
            )

        if isinstance(incoming, ParameterBeam):
            outgoing_mu[..., 4] = outgoing_mu[..., 4] + (
                T566 * incoming.mu[..., 5].square()
                + T556 * incoming.mu[..., 4] * incoming.mu[..., 5]
                + T555 * incoming.mu[..., 4].square()
            )
            outgoing_cov[..., 4, 4] = (
                T566 * incoming.cov[..., 5, 5].square()
                + T556 * incoming.cov[..., 4, 5] * incoming.cov[..., 5, 5]
                + T555 * incoming.cov[..., 4, 4].square()
            )
            outgoing_cov[..., 4, 5] = (
                T566 * incoming.cov[..., 5, 5].square()
                + T556 * incoming.cov[..., 4, 5] * incoming.cov[..., 5, 5]
                + T555 * incoming.cov[..., 4, 4].square()
            )
            outgoing_cov[..., 5, 4] = outgoing_cov[..., 4, 5]
        else:  # ParticleBeam
            outgoing_particles[..., 4] = outgoing_particles[..., 4] + (
                T566.unsqueeze(-1) * incoming.particles[..., 5].square()
                + T556.unsqueeze(-1)
                * incoming.particles[..., 4]
                * incoming.particles[..., 5]
                + T555.unsqueeze(-1) * incoming.particles[..., 4].square()
            )

        if isinstance(incoming, ParameterBeam):
            outgoing = ParameterBeam(
                mu=outgoing_mu,
                cov=outgoing_cov,
                energy=outgoing_energy,
                total_charge=incoming.total_charge,
                s=incoming.s + self.length,
                species=incoming.species,
                device=outgoing_mu.device,
                dtype=outgoing_mu.dtype,
            )
            return outgoing
        else:  # ParticleBeam
            outgoing = ParticleBeam(
                particles=outgoing_particles,
                energy=outgoing_energy,
                particle_charges=incoming.particle_charges,
                survival_probabilities=incoming.survival_probabilities,
                s=incoming.s + self.length,
                species=incoming.species,
                device=outgoing_particles.device,
                dtype=outgoing_particles.dtype,
            )
            return outgoing

    def _cavity_rmatrix(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        """Produces an R-matrix for a cavity when it is on, i.e. voltage > 0.0."""
        assert torch.all(energy > 0), "Initial energy must be larger than 0"
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        phi = self.phase.deg2rad()
        effective_voltage = -self.voltage * species.num_elementary_charges
        delta_energy = effective_voltage * phi.cos()

        Ei = energy / species.mass_eV
        dE = delta_energy / species.mass_eV
        Ef = Ei + dE
        Ep = dE / self.length  # Derivative of the energy

        k = 2 * torch.pi * self.frequency / constants.speed_of_light

        if self.cavity_type == "standing_wave":
            alpha = (
                math.sqrt(0.125)
                * effective_voltage
                / energy
                * log1pdiv(delta_energy / energy)
            )
            beta0 = (1 - Ei.square().reciprocal()).sqrt()
            beta1 = (1 - Ef.square().reciprocal()).sqrt()

            r11 = alpha.cos() - math.sqrt(2.0) * phi.cos() * alpha.sin()
            r12 = (
                (alpha / torch.pi).sinc()
                * log1pdiv(delta_energy / energy)
                * self.length
            )
            r21 = -(
                effective_voltage
                / ((energy + delta_energy) * math.sqrt(2.0) * self.length)
                * (0.5 + phi.cos().square())
                * alpha.sin()
            )

            r22 = Ei / Ef * (alpha.cos() + math.sqrt(2.0) * phi.cos() * alpha.sin())

            # TODO: Branch stays in for now because no proper limit exists and would
            # probably require a different model altogether.
            r55 = 1.0 + (
                k
                * self.length
                * beta0
                * phi.tan()
                * (Ei * Ef * (beta0 * beta1 - 1) + 1)
                / (beta1 * Ef * dE)
            ).where(dE != 0.0, 0.0)
            r56 = (
                -self.length / (Ef.square() * Ei * beta1) * (Ef + Ei) / (beta1 + beta0)
            )
            r65 = k * phi.sin() * effective_voltage / (beta1 * (energy + delta_energy))
            r66 = Ei / Ef * beta0 / beta1

        elif self.cavity_type == "traveling_wave":
            # Reference paper: Rosenzweig and Serafini, PhysRevE, Vol.49, p.1599,(1994)
            vector_shape = torch.broadcast_shapes(
                self.length.shape, dE.shape, Ei.shape, Ef.shape
            )

            M_body = torch.eye(2, **factory_kwargs).repeat((*vector_shape, 1, 1))
            M_body[..., 0, 1] = self.length * log1pdiv(dE / Ei)
            M_body[..., 1, 1] = Ei / Ef

            M_f_entry = torch.eye(2, **factory_kwargs).repeat((*vector_shape, 1, 1))
            M_f_entry[..., 1, 0] = -Ep / (2 * Ei)

            M_f_exit = torch.eye(2, **factory_kwargs).repeat((*vector_shape, 1, 1))
            M_f_exit[..., 1, 0] = Ep / (2 * Ef)

            M_combined = M_f_exit @ M_body @ M_f_entry

            r11 = M_combined[..., 0, 0]
            r12 = M_combined[..., 0, 1]
            r21 = M_combined[..., 1, 0]
            r22 = M_combined[..., 1, 1]
            r55 = self.length.new_ones(())
            r56 = self.length.new_zeros(())
            r65 = k * torch.sin(phi) * effective_voltage / (energy + delta_energy)
            r66 = r22
        else:
            raise ValueError(f"Invalid cavity type: {self.cavity_type}")

        # Make sure that all matrix elements have the same shape
        r11, r12, r21, r22, r55, r56, r65, r66 = torch.broadcast_tensors(
            r11, r12, r21, r22, r55, r56, r65, r66
        )

        R = torch.eye(7, **factory_kwargs).repeat((*r11.shape, 1, 1))
        R[..., 0, 0] = r11
        R[..., 0, 1] = r12
        R[..., 1, 0] = r21
        R[..., 1, 1] = r22
        R[..., 2, 2] = r11
        R[..., 2, 3] = r12
        R[..., 3, 2] = r21
        R[..., 3, 3] = r22
        R[..., 4, 4] = r55
        R[..., 4, 5] = r56
        R[..., 5, 4] = r65
        R[..., 5, 5] = r66

        return R

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="gold", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

        # Mark cavities with the wake on, and return the axes
        if self.wake:
            ax.axvline(plot_s, ymin=0.01, ymax=0.99, color="teal", linestyle=":")
        return ax

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "voltage",
            "phase",
            "frequency",
            "cavity_type",
            #wake_strength is deliberately absent: it is stored as r_over_q, and clone() would otherwise pass both and fail
            "wake",
            "iris_radius",
            "cell_gap",
            "cell_period",
            "cell_radius",
            "r_over_q",
            "quality_factor",
            "mode_frequency",
            "transverse_shunt_impedance",
            "wake_model",
            "include_transverse",
            "wake_position",
            "cells_per_feature",
            "max_grid",
            "beam_radius",
            "cache_diagnostics",
        ]

    # ---------------------------------------------------------------- wakefields

    def _validate_wake_configuration(self) -> None:
        """Fail at construction if the wake is on but under-specified."""
        if self.wake_model == "resonant":
            if self.r_over_q is None:
                raise ValueError(
                    "wake_model='resonant' requires wake_strength or r_over_q."
                )
            # `Cavity` defaults the frequency to 0, which would otherwise surface as a
            # `ZeroDivisionError` deep inside the grid sizer
            if not bool((self._mode_frequency != 0.0).all()):
                raise ValueError(
                    "wake_model='resonant' requires a non-zero mode frequency. "
                    "Pass frequency=... or mode_frequency=... ."
                )
        elif self.wake_model in ("auto", "periodic", "diffraction"):
            missing = [
                name
                for name in ("iris_radius", "cell_gap", "cell_period")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"wake_model={self.wake_model!r} requires "
                    + ", ".join(missing)
                    + ". Give the mechanical geometry, or use "
                    "wake_model='resonant' with wake_strength (or r_over_q)."
                )
        elif self.wake_model == "space_charge":
            if self.iris_radius is None:
                raise ValueError(
                    "wake_model='space_charge' requires iris_radius (pipe radius b)."
                )
        else:
            raise ValueError(f"Unknown wake_model {self.wake_model!r}.")

        if self.include_transverse and self.wake_model == "space_charge":
            raise ValueError(
                "The space-charge model here is longitudinal only; use Cheetah's "
                "SpaceChargeKick for the transverse part."
            )

    @property
    def _mode_frequency(self) -> torch.Tensor:
        """Wake mode frequency in Hz, defaulting to the RF frequency."""
        return (
            self.mode_frequency if self.mode_frequency is not None else self.frequency
        )

    @property
    def wake_strength(self) -> torch.Tensor:
        """Wake amplitude `w(0+) = 2 pi f_r (R/Q) = 2k` in V/C (resonant model only)."""
        if self.r_over_q is None:
            raise ValueError(
                "wake_strength is only defined for the resonant model (wake_strength= "
                f"or r_over_q=); this cavity uses {self.wake_model!r}. Use "
                "loss_factor_for(beam) and double it."
            )
        # Promote the frequency to the dtype of R/Q so the round trip does not lose
        # precision when `frequency` is float32
        amplitude = torch.as_tensor(self.r_over_q)
        omega_r = 2.0 * math.pi * torch.as_tensor(
            self._mode_frequency, dtype=amplitude.dtype, device=amplitude.device
        )
        return omega_r * amplitude

    @property
    def loss_factor(self) -> torch.Tensor:
        """Loss factor `k = w(0+) / 2` in V/C (Ferrario Eq. 18), resonant model only."""
        return 0.5 * self.wake_strength

    @property
    def geometry(self) -> CavityGeometry:
        """Wake description of this cavity, rebuilt only when its parameters change."""
        mode_frequency = self._mode_frequency
        signature = tuple(
            None
            if value is None
            else float(torch.as_tensor(value).detach().reshape(-1)[0])
            for value in (
                self.iris_radius,
                self.cell_gap,
                self.cell_period,
                self.length,
                self.cell_radius,
                self.r_over_q,
                mode_frequency,
                self.quality_factor,
                self.transverse_shunt_impedance,
            )
        )
        cached = getattr(self, "_geometry_cache", None)
        if cached is not None and cached[0] == signature:
            return cached[1]

        geometry = CavityGeometry(
            iris_radius=self.iris_radius,
            cell_gap=self.cell_gap,
            cell_period=self.cell_period,
            length=self.length,
            cell_radius=self.cell_radius,
            r_over_q=self.r_over_q,
            mode_frequency=mode_frequency,
            quality_factor=self.quality_factor,
            transverse_shunt_impedance=self.transverse_shunt_impedance,
        )
        self._geometry_cache = (signature, geometry)
        return geometry

    def longitudinal_kernel_for(
        self, beam: ParticleBeam
    ) -> LongitudinalWakeKernel | None:
        """Longitudinal wake kernel used for `beam`, or `None` for space charge."""
        if self.wake_model == "space_charge":
            return None
        return self.geometry.longitudinal_kernel(
            self.wake_model,
            sigma_tau=float(beam.tau.detach().std()),
            gamma=float(beam.relativistic_gamma.detach().reshape(-1)[0]),
        )

    def transverse_kernel_for(self, beam: ParticleBeam) -> TransverseWakeKernel:
        """Transverse dipole wake kernel used for `beam`."""
        regime = "periodic" if self.wake_model == "auto" else self.wake_model
        return self.geometry.transverse_kernel(
            regime,
            sigma_tau=float(beam.tau.detach().std()),
            gamma=float(beam.relativistic_gamma.detach().reshape(-1)[0]),
        )

    def loss_factor_for(self, beam: ParticleBeam) -> float:
        """Loss factor `k` in V/C (Ferrario Eqs. 17-18) for `beam`."""
        kernel = self.longitudinal_kernel_for(beam)
        if kernel is None:
            raise ValueError("The space-charge model has no loss factor.")
        return kernel.loss_factor

    def track_with_wake(self, incoming: Beam) -> Beam:
        """
        RF tracking plus the thin wake kick, placed according to `wake_position`.

        Call `track` from lattices, not this. `Segment` only ever calls `track`, and
        decides from `is_skippable` whether to visit the element at all.
        """
        if isinstance(incoming, ParameterBeam):
            raise AssertionError(
                "Wakefield tracking is currently only supported for `ParticleBeam`, "
                "because a wake needs the longitudinal charge distribution. Set "
                "wake=False to track a `ParameterBeam`."
            )

        if self.wake_position == "entrance":
            return self._track_rf(self._apply_wake(incoming))

        if self.wake_position == "exit":
            return self._apply_wake(self._track_rf(incoming))

        if self.wake_position == "split":
            outgoing = self._track_rf(self._apply_wake(incoming, scale=0.5))
            return self._apply_wake(outgoing, scale=0.5)

        raise ValueError(f"Unknown wake_position {self.wake_position!r}.")

    def _apply_wake(self, beam: ParticleBeam, *, scale: float = 1.0) -> ParticleBeam:
        """Add the wake's energy kick (times `scale`) to the momentum of `beam`."""
        if not self._beam_is_charged(beam):
            self.cached_wake_diagnostics = None
            return beam

        # NaN in, NaN out: a beam that is already invalid (e.g. below rest mass after
        # the RF) has no distribution to deposit, so pass it through like upstream does
        if not bool(torch.isfinite(beam.particles[..., 4]).all()):
            self.cached_wake_diagnostics = None
            return beam

        particles = beam.particles
        batch_shape = particles.shape[:-2]
        num_particles = particles.shape[-2]
        n_batch = math.prod(batch_shape) if batch_shape else 1

        flat = particles.reshape(n_batch, num_particles, 7)
        #We need to deal with survival_probabilities at some point, rn we just ignore them and assume all particles survive
        charges = torch.broadcast_to(
            beam.particle_charges, (*batch_shape, num_particles)
        ).reshape(n_batch, num_particles)

        # The reference energy can be vector-shaped after RF tracking
        energy = beam.energy
        if energy.shape == batch_shape and energy.numel() == n_batch:
            energy_flat = energy.reshape(n_batch)
        else:
            energy_flat = energy.detach().reshape(-1).mean().expand(n_batch)
        p0c = reference_momentum(energy_flat, beam.species.mass_eV)

        num_charges = float(beam.species.num_elementary_charges)
        out = flat.clone()
        diagnostics: list[dict] = []

        # We loop over the vector (batch) dimension, so large parameter scans are slower
        for index in range(n_batch):
            tau = flat[index, :, 4]
            charge = charges[index]

            if self.wake_model == "space_charge":
                delta_energy, diagnostic = self._space_charge_kick(
                    tau,
                    charge,
                    flat[index],
                    energy_flat[index],
                    beam.species.mass_eV,
                )
            else:
                kernel = self._kernel_from_scalars(
                    sigma_tau=float(tau.detach().std()),
                    gamma=float(energy_flat[index].detach() / beam.species.mass_eV),
                )
                delta_energy, diagnostic = longitudinal_energy_kick(
                    tau,
                    charge,
                    kernel,
                    num_elementary_charges=num_charges,
                    cells_per_feature=self.cells_per_feature,
                    max_grid=self.max_grid,
                )

            out[index, :, 5] = flat[index, :, 5] + scale * delta_energy / p0c[index]

            # Transverse wake has no benchmark against another code like bmad yet
            if self.include_transverse:
                transverse_kernel = self.geometry.transverse_kernel(
                    "periodic" if self.wake_model == "auto" else self.wake_model,
                    sigma_tau=float(tau.detach().std()),
                    gamma=float(energy_flat[index].detach() / beam.species.mass_eV),
                )
                for plane, (offset_index, momentum_index) in enumerate(
                    ((0, 1), (2, 3))
                ):
                    kick, transverse_diagnostic = transverse_dipole_wake_kick(
                        tau,
                        flat[index, :, offset_index],
                        charge,
                        transverse_kernel,
                        p0c[index],
                        num_elementary_charges=num_charges,
                        cells_per_feature=self.cells_per_feature,
                        max_grid=self.max_grid,
                    )
                    out[index, :, momentum_index] = (
                        flat[index, :, momentum_index] + scale * kick
                    )
                    diagnostic[f"transverse_{'xy'[plane]}"] = transverse_diagnostic

            diagnostics.append(diagnostic)

        if self.cache_diagnostics:
            self.cached_wake_diagnostics = {
                "per_batch": diagnostics,
                **diagnostics[0],
            }

        return ParticleBeam(
            particles=out.reshape(*batch_shape, num_particles, 7),
            energy=beam.energy,
            particle_charges=beam.particle_charges,
            survival_probabilities=beam.survival_probabilities,
            s=beam.s,
            species=beam.species,
        )

    def _kernel_from_scalars(
        self, *, sigma_tau: float, gamma: float
    ) -> LongitudinalWakeKernel:
        """Longitudinal kernel for a bunch length and Lorentz factor."""
        return self.geometry.longitudinal_kernel(
            self.wake_model, sigma_tau=sigma_tau, gamma=gamma
        )

    def _space_charge_kick(
        self,
        tau: torch.Tensor,
        charge: torch.Tensor,
        particles: torch.Tensor,
        energy: torch.Tensor,
        mass_eV: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict]:
        """Space-charge energy kick (Ferrario Eq. 7) from the beam's own size."""
        if self.beam_radius is not None:
            radius = self.beam_radius
        else:
            #Uniform-disc equivalent a = 2 sigma_r; might change convention later
            sigma_x = particles[:, 0].detach().std()
            sigma_y = particles[:, 2].detach().std()
            radius = 2.0 * torch.sqrt(torch.clamp(sigma_x * sigma_y, min=1e-30))

        return space_charge_wake_potential(
            tau,
            charge,
            length=self.length,
            pipe_radius=self.iris_radius,
            beam_radius=radius,
            gamma=energy
            / torch.as_tensor(float(mass_eV), device=energy.device, dtype=energy.dtype),
            cells_per_feature=self.cells_per_feature,
            max_grid=self.max_grid,
        )

    @staticmethod
    def _beam_is_charged(beam: ParticleBeam) -> bool:
        """Whether any particle in `beam` carries charge."""
        return not bool(torch.all(beam.particle_charges == 0))
