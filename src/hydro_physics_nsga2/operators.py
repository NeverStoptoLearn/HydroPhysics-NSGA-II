"""Concentration-guided mixed-variable sampling and variation."""
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.mixed import MixedVariableMating
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Integer, Real


class IdentityMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        return X


class ConcentrationSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        s = problem.scenario
        concentration = problem.C_before_pump
        probabilities = concentration**2 + max(float(concentration.max())**2, 1.0) * 1e-6
        probabilities /= probabilities.sum()
        result = []
        for _ in range(n_samples):
            x = {}
            p = probabilities.copy()
            for i in (1, 2):
                index = int(np.random.choice(len(p), p=p))
                x[f'well_positions_{i}'] = index
                x[f'Q_{i}'] = float(np.random.uniform(0, s.max_rate))
                p[np.abs(problem.grid - problem.grid[index]) < s.min_spacing] = 0
                if p.sum() == 0:
                    p = np.ones_like(p)
                p /= p.sum()
            result.append(x)
        return result


class PhysicsGuidedMating(MixedVariableMating):
    def __init__(self, **kwargs):
        super().__init__(mutation={Integer: IdentityMutation(), Real: IdentityMutation()}, **kwargs)

    def _do(self, problem, pop, n_offsprings, parents=False, **kwargs):
        offspring = super()._do(problem, pop, n_offsprings, parents=parents, **kwargs)
        s = problem.scenario
        c, grad = problem.C_before_pump, problem.grad
        cmax = max(float(c.max()), 1e-12)
        gmax = max(float(np.abs(grad).max()), 1e-12)
        for individual in offspring:
            x = individual.X
            for i in (1, 2):
                position, rate = f'well_positions_{i}', f'Q_{i}'
                idx = int(x[position])
                direction = np.sign(grad[idx] / gmax + c[idx] / cmax)
                idx = int(np.clip(idx + direction + np.random.randint(-1, 2), 0, s.nx))
                if np.random.random() < 0.1:
                    idx = s.nx - idx
                x[position] = idx
                # Each rate is guided by its own well position after crossover.
                target = c[idx] / cmax + 0.5 * grad[idx] / gmax
                rate_value = x[rate] + 0.1 * (target - x[rate] / s.max_rate) * s.max_rate
                rate_value += np.random.normal(scale=0.05 * s.max_rate)
                if np.random.random() < 0.1:
                    rate_value = s.max_rate - rate_value
                x[rate] = float(np.clip(rate_value, 0, s.max_rate))
        return offspring
