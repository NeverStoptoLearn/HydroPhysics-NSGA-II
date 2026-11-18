import numpy as np
import pytest
from scipy.integrate import simpson
from hydro_physics_nsga2.problem import GroundwaterProblem, Scenario
from hydro_physics_nsga2.cli import optimize


def layout(a=50, b=51, q=20.0):
    return {'well_positions_1': a, 'Q_1': q, 'well_positions_2': b, 'Q_2': q}


def test_pumping_duration_and_cost_use_days():
    p = GroundwaterProblem()
    _, durations = p.pumping_windows(layout())
    np.testing.assert_allclose(durations, [9, 9])
    f, _ = p.raw_objectives_and_constraints(layout())
    assert f[0] == 10000 + 10 * (20 * 9 + 20 * 9)


def test_transport_nonnegative_and_pumping_removes_contaminant():
    p = GroundwaterProblem()
    baseline = p.solve_concentration(layout(q=0))
    pumped = p.solve_concentration(layout(q=50))
    assert np.isfinite(pumped).all() and pumped.min() >= 0
    assert np.all(pumped <= baseline + 1e-9)
    assert simpson(pumped, x=p.grid) < simpson(baseline, x=p.grid)


def test_well_index_maps_to_physical_plume_location():
    p = GroundwaterProblem()
    near = p.raw_objectives_and_constraints(layout())[0][1]
    far = p.raw_objectives_and_constraints(layout(10, 11))[0][1]
    assert near < far


def test_constraints_are_explicit_and_objectives_are_raw():
    p = GroundwaterProblem(Scenario(min_head=49.99))
    f, g = p.raw_objectives_and_constraints(layout(q=50))
    assert g.shape == (5,) and np.any(g > 0)
    assert f[1] == pytest.approx(simpson(p.solve_concentration(layout(q=50)) * p.scenario.area, x=p.grid))
    assert p.n_ieq_constr == 5


def test_invalid_parameters_and_schedules():
    with pytest.raises(ValueError):
        Scenario(pump_start_time=11)
    with pytest.raises(ValueError):
        GroundwaterProblem().decode(layout(a=101))


def test_small_optimization_exports_feasible_front():
    p = GroundwaterProblem()
    result = optimize(p, population=8, generations=2)
    assert result.X is not None
    assert np.isfinite(result.F).all()
    assert np.all(result.G <= 1e-8)
