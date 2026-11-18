"""One-dimensional transport and two-well emergency scheduling."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import simpson
from scipy.special import exp1
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real


@dataclass
class Scenario:
    length: float = 1000.0
    duration: float = 10.0
    nx: int = 100
    nt: int = 100
    dispersion: float = 10.0
    velocity: float = 1.0
    area: float = 100.0
    porosity: float = 0.3
    source_position: float = 500.0
    peak_concentration: float = 100.0
    plume_width: float = 10.0
    pump_start_time: float = 1.0
    max_rate: float = 50.0
    construction_cost: float = 5000.0
    pumping_cost: float = 10.0
    initial_head: float = 50.0
    min_head: float = 45.0
    max_head: float = 52.0
    transmissivity: float = 100.0
    storativity: float = 0.0001
    observation_radius: float = 10.0
    min_spacing: float = 10.0

    def __post_init__(self):
        for name in ('length', 'duration', 'nx', 'nt', 'dispersion', 'velocity',
                     'area', 'porosity', 'plume_width', 'max_rate', 'transmissivity',
                     'storativity', 'observation_radius'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.nx < 4 or self.nt < 1 or int(self.nx) != self.nx or int(self.nt) != self.nt:
            raise ValueError('nx and nt must be integer grid counts')
        if not 0 < self.porosity <= 1 or not 0 <= self.source_position <= self.length:
            raise ValueError('Invalid porosity or source position')
        if not 0 <= self.pump_start_time <= self.duration:
            raise ValueError('Pumping start must lie within the simulation period')
        if not 0 <= self.min_spacing <= self.length or self.min_head > self.max_head:
            raise ValueError('Invalid spacing or head limits')
        if min(self.peak_concentration, self.construction_cost, self.pumping_cost) < 0:
            raise ValueError('Concentration and costs must be nonnegative')
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Scenario values must be finite')


class GroundwaterProblem(ElementwiseProblem):
    def __init__(self, scenario=None):
        self.scenario = scenario or Scenario()
        s = self.scenario
        self.dx, self.dt = s.length / s.nx, s.duration / s.nt
        self.grid = np.linspace(0, s.length, s.nx + 1)
        self.initial = s.peak_concentration * np.exp(-0.5 * ((self.grid - s.source_position) / s.plume_width) ** 2)
        self.initial[[0, -1]] = 0.0
        self.C_before_pump = self.solve_concentration(self.zero_schedule(), pre_pump=True)
        self.grad = np.gradient(self.C_before_pump, self.dx)
        variables = {}
        for i in range(1, 3):
            variables[f'well_positions_{i}'] = Integer(bounds=(0, s.nx))
            variables[f'Q_{i}'] = Real(bounds=(0.0, s.max_rate))
        super().__init__(vars=variables, n_obj=2, n_ieq_constr=5)

    @staticmethod
    def zero_schedule():
        return {'well_positions_1': 0, 'Q_1': 0.0, 'well_positions_2': 1, 'Q_2': 0.0}

    def decode(self, x):
        indices = np.array([x[f'well_positions_{i}'] for i in (1, 2)], dtype=float)
        rates = np.array([x[f'Q_{i}'] for i in (1, 2)], dtype=float)
        if (not np.isfinite(indices).all() or not np.isfinite(rates).all()
                or np.any(indices != np.round(indices)) or np.any(indices < 0)
                or np.any(indices > self.scenario.nx) or np.any(rates < 0)
                or np.any(rates > self.scenario.max_rate)):
            raise ValueError('Well indices or pumping rates are outside their bounds')
        return indices.astype(int), rates

    def pumping_windows(self, x):
        """Elapsed start times and operating durations in days."""
        self.decode(x)
        s = self.scenario
        # The initial plume is already present; advection is handled by the solver.
        starts = np.full(2, s.pump_start_time)
        return starts, np.maximum(0.0, s.duration - starts)

    def solve_concentration(self, x, pre_pump=False):
        s = self.scenario
        indices, rates = self.decode(x)
        starts, _ = self.pumping_windows(x)
        end = s.pump_start_time if pre_pump else s.duration
        # Upwind advection and explicit diffusion/removal with positivity-preserving substeps.
        coefficient = 2 * s.dispersion / self.dx**2 + s.velocity / self.dx + rates.sum() / (s.area * s.porosity)
        max_dt = min(self.dt, 0.9 / coefficient)
        events = sorted(set([0.0, float(end)] + [float(t) for t in starts if 0 < t < end]))
        concentration = self.initial.copy()
        for left, right in zip(events[:-1], events[1:]):
            steps = max(1, int(np.ceil((right - left) / max_dt)))
            dt = (right - left) / steps
            sink = np.zeros_like(concentration)
            if not pre_pump:
                np.add.at(sink, indices, rates * (left >= starts) / (s.area * s.porosity))
            for _ in range(steps):
                new = concentration.copy()
                diffusion = s.dispersion * (concentration[2:] - 2 * concentration[1:-1] + concentration[:-2]) / self.dx**2
                advection = s.velocity * (concentration[1:-1] - concentration[:-2]) / self.dx
                new[1:-1] += dt * (diffusion - advection - sink[1:-1] * concentration[1:-1])
                new[[0, -1]] = 0.0
                if not np.isfinite(new).all() or new.min() < -1e-9:
                    raise FloatingPointError('Transport integration produced invalid concentrations')
                concentration = np.maximum(new, 0.0)
        return concentration

    def raw_objectives_and_constraints(self, x):
        s = self.scenario
        indices, rates = self.decode(x)
        _, durations = self.pumping_windows(x)
        cost = 2 * s.construction_cost + s.pumping_cost * float(rates @ durations)
        remaining = float(simpson(self.solve_concentration(x) * s.area, x=self.grid))
        drawdown = np.zeros(2)
        active = durations > 0
        u = s.observation_radius**2 * s.storativity / (4 * s.transmissivity * durations[active])
        drawdown[active] = rates[active] * exp1(u) / (4 * np.pi * s.transmissivity)
        head = s.initial_head - drawdown
        g = np.r_[s.min_head - head, head - s.max_head,
                  s.min_spacing - abs(indices[0] - indices[1]) * self.dx]
        return np.array([cost, remaining]), g

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'], out['G'] = self.raw_objectives_and_constraints(x)
