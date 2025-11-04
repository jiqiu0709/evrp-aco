# EVRP Ant Colony Solver

This repository contains a Python implementation of an ant colony optimisation
(ACO) solver tailored for the electric vehicle routing problem (EVRP). The
solver respects fixed fleet size, load capacity, and battery limits while using
recharging stations to restore energy.

After every ant constructs a solution, a SWAP* multi-level improver further
refines the routes via node swaps, N1 cross-route insertions, N2 swap-and-reinsert
operations, intra-route reordering, inter-route exchanges, and a
large-neighborhood destroy/repair phase. This layered local search complements
the existing 2-opt/relocate education stage and helps drive feasible
high-quality solutions.

Optional neural SWAP* sandwich passes mix energy distances with spatial
heuristics using threaded sub-route optimisation, while dedicated
intensification and diversification phases keep the search balanced. Min–Max ACO
constraints with half-life resets and elite injection guard against premature
convergence and reinforce high-quality tours.

## Running the Solver

```bash
python aco_evrp.py \
  --instance test/C50-S8-12001/C50-S8-12001 \
  --iterations 200 \
  --ants 20 \
  --seed 42
```

Batch mode across every instance in a directory (results saved to a report file):

```bash
python aco_evrp.py \
  --folder test \
  --iterations 200 \
  --ants 30 \
  --ant-workers 8 \
  --output results/test_runs.txt
```

Arguments:

- `--instance` (required): path prefix for the instance. For a file named
  `C50-S8-12001.txt`, supply `.../C50-S8-12001`.
- `--heuristic`: optional path to a pre-computed heuristic matrix. When omitted,
  the solver uses the reciprocal of the energy matrix.
- `--iterations`: number of optimisation iterations (default 200).
- `--ants`: number of ants per iteration (default scales with instance size).
- `--ant-workers`: worker threads used to build ants in parallel (defaults to CPU count).
- `--seed`: random seed for reproducibility.
- `--alpha`, `--beta`, `--rho`, `--q0`: standard ACO parameters for pheromone
  influence, heuristic influence, evaporation rate, and greedy selection weight.
- `--no-progress`: suppress the iteration progress bar (enabled by default when
  running interactively).
- `--time-relaxation`: relaxes the working-time limit during construction to
  encourage exploration (e.g. `0.15` allows up to 15% overtime before penalties
  kick in).
- `--penalty-update-interval`: frequency (in iterations) for adapting penalty
  coefficients based on the feasible/infeasible balance.
- `--enable-swap-star/--no-enable-swap-star`: toggle the SWAP* multi-neighborhood
  improver applied after solution construction (enabled by default).
- `--swap-star-iterations`: number of passes through SWAP* neighborhoods per
  solution (default 4).
- `--swap-star-sample-size`: maximum neighbors sampled per SWAP* operator
  before evaluating moves (default 60).
- `--swap-star-removal-rate`: fraction of customers removed during the SWAP*
  large-neighborhood destroy/repair step (default 0.2).
- `--swap-star-min-remove`: minimum customers removed during the SWAP* destroy
  phase (default 2).
- `--pheromone-halflife`: multiplier applied after pheromone evaporation to model
  half-life behaviour (default 0.5).
- `--pheromone-reset`: constant pheromone amount re-injected after evaporation
  (default 0.01).
- `--enable-neural-swap-star/--no-enable-neural-swap-star`: toggle the neural
  SWAP* sandwich stage that mixes spatial heuristics with threaded sub-route
  improvement (disabled by default).
- `--neural-swap-workers`: worker threads for the neural SWAP* sub-route
  optimiser (default 4).
- `--neural-swap-noise`: stochastic perturbation applied during neural SWAP*
  heuristic reconstruction (default 0.1).
- `--neural-swap-passes`: number of neural SWAP* sandwich passes per solution
  (default 1).
- `--enable-intensification/--no-enable-intensification`: toggle the
  global-best intensification phase that chains insertion rebuilds and deeper
  local search (enabled by default).
- `--intensification-multiplier`: scales education iterations during
  intensification (default 2.0).
- `--enable-diversification/--no-enable-diversification`: toggle pheromone and
  population diversification when stagnation is detected (enabled by default).
- `--diversification-interval`: number of stagnant iterations before triggering
  diversification (default 25).
- `--diversification-strength`: magnitude of the diversification pheromone
  noise (default 0.2).
- `--elitist-only/--no-elitist-only`: if enabled, only the best ant deposits
  pheromone; otherwise every ant reinforces its path (enabled by default).
- `--elite-injection-factor`: extra pheromone applied to elite-pool routes
  after each iteration (default 0.1).
- `--pheromone-halflife`: multiplier applied during pheromone evaporation to
  model half-life behaviour (default 0.5).
- `--pheromone-reset-constant`: baseline pheromone added after evaporation to
  prevent premature stagnation (default 0.01).
- `--folder`: solve every instance in a directory; use together with other
  tuning flags to reuse settings across the batch.
- `--output`: when running with `--folder`, write the formatted summary to this
  path (default `aco_batch_results.txt` inside the folder).

The batch report stores each instance in the format:

```
Instance tensor([[12371]], device='cuda:0'):
Cost: 440.3162
GT Cost: 409.8305
Gap: 7.44%
Best path found:
 Vehicle 1: 0 -> 10 -> ... -> 0
 ...
--------------------------------------------------
```

The solver prints the best set of routes (one per line, comma separated and
starting/ending at depot `0`) followed by the total energy cost of the solution.
If constraints remain violated, a warning and the list of unserved customers are
displayed. Additional diagnostics report any residual energy, time, or capacity
violations to help tune parameters.

## Constraints & Assumptions

- Travel times are loaded from the matching `-time.txt` matrix. Each vehicle is
  limited to 480 minutes (8 hours) per route.
- Upon reaching a customer, the solver adds a fixed 20-minute service duration;
  visiting a charging station adds 60 minutes.
- Energy consumption, capacity limits, and recharging behaviour follow the data
  defined in the instance's `-energy.txt` matrix and metadata file.

## Batch Mode Output

When targeting a folder, the solver enumerates every `*-energy.txt` file to find
matching instances, solves each one, and writes a report that matches the
expected format (including cost, ground-truth cost taken from
`*-evrp-time.txt`, percentage gap, and one line per vehicle route). Errors for
individual instances are recorded alongside the instance identifier so the rest
of the batch can continue.


