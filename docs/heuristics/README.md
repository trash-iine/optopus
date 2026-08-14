# Heuristics

Every heuristic implements `Heuristic<P>` (`clear` / `is_done` / `run_once` /
`run`). Heuristics are problem-agnostic: they only require traits on the
neighbor type, so any built-in or custom problem that satisfies those traits
plugs straight in.

## Base

| Algorithm | Required traits on neighbor | Notes |
|---|---|---|
| [LocalSearch](local_search.md) | `MoveToNeighbor`, `Rankable` | Greedy best-improving; halts at a local optimum. |
| [SimulatedAnnealing](simulated_annealing.md) | `MoveToNeighbor`, `Evaluate<f64>` | Boltzmann acceptance + multiplicative cooling. |
| [LateAcceptanceHillClimbing](late_acceptance.md) | `MoveToNeighbor`, `Evaluate<f64>` | Compares against the score `history_length` steps ago. |
| [TabuSearch](tabu_search.md) | `MoveToNeighbor`, `Rankable`, `EnabledTabu` | Best non-tabu neighbor + aspiration. |
| [RandomWalk](random_walk.md) | `MoveToNeighbor`, `Rankable` | Uniform random move; useful as perturbation. |
| [BeamSearch](beam_search.md) | `MoveToNeighbor`, `Rankable` | Maintains top-`k` candidates. |
| [RlSearch](rl_search.md) | `MoveToNeighbor`, `Evaluate<f64>`, `Clone` | Online REINFORCE over move features. |

## Meta

| Algorithm | Description |
|---|---|
| [Sequential / Iterated / Restart](meta.md) | Compose inner heuristics via the sub-run clone/merge pattern. |
| [GeneticAlgorithm](genetic_algorithm.md) | Tournament selection → `Crossover` → mutation → worst-replacement. |

## Crossover operators

Used by `GeneticAlgorithm`:

- `*UniformCrossover` — per-variable random parent (one per problem).
- `TspOrderCrossover` — Order Crossover (OX) for permutations.
- `JobShopPpxCrossover` — Precedence-Preserving Crossover for permutation-with-repetition.
- [`SubProblemBasedCrossover`](genetic_algorithm.md#subproblembasedcrossover)
  — generic crossover for any `P: SubProblemExtractable`.

## Problem-specific

| Algorithm | Problem | Notes |
|---|---|---|
| [BreakoutLocalSearchForMaxCut](breakout_local_search.md) | MaxCut | Greedy LS + adaptive perturbation. |
| [RlBreakoutLocalSearchForMaxCut](rl_breakout_local_search.md) | MaxCut | BLS machinery + learned (contextual-bandit) perturbation policy. |
| [LinKernighanHelsgaunForTsp](lkh.md) | TSP 2D | Variable-depth k-opt with candidate lists. |
