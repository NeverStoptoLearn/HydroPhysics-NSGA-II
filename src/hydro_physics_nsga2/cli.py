from dataclasses import asdict
import argparse
import json
import numpy as np
from pymoo.core.mixed import MixedVariableGA, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.optimize import minimize
from .problem import GroundwaterProblem, Scenario
from .operators import ConcentrationSampling, PhysicsGuidedMating
from .experiment_io import start_run, save_json


def optimize(problem, population=50, generations=200):
    if population < 4 or generations < 1:
        raise ValueError('population >= 4 and generations >= 1 are required')
    algorithm = MixedVariableGA(pop_size=population, sampling=ConcentrationSampling(),
        mating=PhysicsGuidedMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        survival=RankAndCrowdingSurvival())
    return minimize(problem, algorithm, ('n_gen', generations), save_history=True, verbose=False)


def run(config, output=None):
    scenario = Scenario(**config.get('scenario', {}))
    problem = GroundwaterProblem(scenario)
    path = start_run({'scenario': asdict(scenario), 'optimizer': config.get('optimizer', {})}, output)
    result = optimize(problem, **config.get('optimizer', {}))
    solutions = [] if result.X is None else list(result.X)
    F, G, X = [], [], []
    for solution in solutions:
        f, g = problem.raw_objectives_and_constraints(solution)
        F.append(f); G.append(g)
        X.append([solution['well_positions_1'], solution['Q_1'], solution['well_positions_2'], solution['Q_2']])
    f = np.asarray(F).reshape(-1, 2)
    g = np.asarray(G).reshape(-1, 5)
    np.savez_compressed(path / 'results.npz', X=np.asarray(X).reshape(-1, 4), F=f, G=g)
    history = [{'generation': i + 1, 'evaluations': a.evaluator.n_eval,
                'F': a.pop.get('F').tolist(), 'G': a.pop.get('G').tolist()} for i, a in enumerate(result.history)]
    save_json(path / 'history.json', history)
    save_json(path / 'summary.json', {'status': 'ok' if len(f) else 'no_feasible_solution', 'solutions': len(f),
                                    'objectives': ['cost', 'residual_contamination_index']})
    from .visualization import plot_front
    plot_front(f, path / 'pareto.png')
    return path


def main():
    parser = argparse.ArgumentParser(description='Groundwater emergency scheduling')
    parser.add_argument('--config', default='configs/demo.json')
    parser.add_argument('--output')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = json.load(stream)
    print(run(config, args.output))
