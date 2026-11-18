# HydroPhysics-NSGA-II

Research code for **HydroPhysics-NSGA-II: A physics-informed multi-objective evolutionary algorithm for groundwater contaminant emergency scheduling**.

**Authors:** Zhengchen Zhou, Xuesong Yan, Dongcheng Li  
**Publication:** Swarm and Evolutionary Computation, 99 (2025), 102210  
**Paper:** [10.1016/j.swevo.2025.102210](https://doi.org/10.1016/j.swevo.2025.102210)

Optimize two pumping-well locations and pumping rates for a one-dimensional groundwater contamination emergency. Concentration-guided sampling and variation are combined with mixed-variable crossover and rank-and-crowding survival.

## Model and configuration

`scenario` controls domain length, duration, grid resolution, transport coefficients, plume parameters, pumping limits, costs, head limits, transmissivity, storativity, and minimum well spacing. Defaults are 1,000 m, 10 days, 100 spatial intervals, 100 requested time steps, dispersion 10 m²/d, velocity 1 m/d, and a Gaussian plume centered at 500 m.

Positions are integer grid indices; physical positions are index × grid spacing. Pumping begins at the configured elapsed start time and ends with the simulation. The initial plume is already present, and advection is handled by the transport solver. Finite-difference transport uses upwind advection, central diffusion, and adaptive substeps for nonnegative concentrations. Head drawdown uses the exponential-integral Theis expression with a configured observation radius and independent-well approximation.

Objectives are intervention cost and `integral(A * C(x, T) dx)`, a residual contamination index. This index is not converted to a physical mass. Head bounds and minimum well spacing are explicit constraints. Output `F` contains raw objectives; output `G` contains constraint values, with feasibility defined by `G <= 0`.

## Outputs

- `results.npz`: numeric decision matrix `X`, objective matrix `F`, and constraint matrix `G`.
- `history.json`: objective and constraint values for each generation.
- `pareto.png`: objective trade-off.
- `summary.json`: feasible-solution count and run status.

Decision columns are `[well_1_index, well_1_rate, well_2_index, well_2_rate]`. An empty feasible set is reported explicitly.

## Installation

Use Python 3.10–3.12 in a dedicated environment:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -e ".[dev]"
```

## Run

From the project root:

```bash
python -m hydro_physics_nsga2 --config configs/demo.json
python -m hydro_physics_nsga2 --config configs/experiment.json
```

`demo.json` provides a small execution example; `experiment.json` provides a larger configuration. An explicit output path can be supplied with `--output runs/example`. The destination must not already exist. Without that argument, a new timestamped directory is created under `runs/`.

Every run saves `config.json` and `environment.json` alongside results. Sampling follows the algorithm's natural randomness; no fixed random seed is configured. The demo validates execution, not publication-level accuracy or convergence.

## Project structure

```text
configs/             Small and larger experiment configurations
src/hydro_physics_nsga2/       Importable Python implementation and command-line entry point
notebooks/demo.ipynb  Lightweight example calling the package
tests/               Numerical and workflow checks
.github/workflows/   Automated tests and demo execution
```

## Tests

```bash
python -m pytest -q
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhou2025hydrophysics,
  title   = {{HydroPhysics-NSGA-II}: A physics-informed multi-objective evolutionary algorithm for groundwater contaminant emergency scheduling},
  author  = {Zhou, Zhengchen and Yan, Xuesong and Li, Dongcheng},
  journal = {Swarm and Evolutionary Computation},
  volume  = {99},
  pages   = {102210},
  year    = {2025},
  doi     = {10.1016/j.swevo.2025.102210}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Zhengchen Zhou (Zhouchen), China University of Geosciences (Wuhan)  
Email: [chenzz_14@outlook.com](mailto:chenzz_14@outlook.com)  
ORCID: [0009-0005-0160-1988](https://orcid.org/0009-0005-0160-1988)
