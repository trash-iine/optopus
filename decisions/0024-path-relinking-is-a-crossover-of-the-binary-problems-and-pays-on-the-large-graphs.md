# Path relinking is a crossover of the binary problems, and the greedy path pays on the large graphs

- Status: adopted
- Area: heuristic, max_cut, qubo
- Date: 2026-11-15
- Code: src/building_blocks/representation/binary/crossover.rs, src/problem/max_cut/crossover.rs

## Decision

Path relinking is `PathRelinkingCrossover`, a `Crossover` of MaxCut, QUBO,
MaxSAT and vertex cover, and not a heuristic of its own. The path flips the
variables the parents disagree on, greedily or at random, and the offspring
is the best solution at least a third of the path from both parents. The
operator, the two orders and the third were read in the path relinking paper
for QUBO that the genetic algorithm's page cites. That method's tabu search is
the mutation step of `GeneticAlgorithm`, so the method is a config. Its
reference set, which relinks every pair of members in both directions and is
rebuilt when no pair is left, is not reproduced. The parents are whatever
`parent_selection` picks.

There is one type and four `Crossover` impls, not a blanket impl over
`BinaryProblem`. MaxCut needs its own. A cut and its complement are the same
solution, so its impl relinks toward whichever labelling of the second parent
is nearer. A blanket impl would have walked the long way through assignments
that keep nothing of either cut.

## Do not retry

The random path is registered and is the worse of the two on every large
graph below, behind the uniform crossover as well. It stays because the paper
has it and it costs one constructor. Do not make it the default.

A QUBO measurement on the bqp instances of 500 and 1000 variables showed
nothing. All three crossovers reached the same value in every run within a
few seconds. So did the dense graph G1 at 30 s, and on G43 the three averages
were within 3 of each other. A claim about those sizes needs harder instances
than these.

## Appendix

`data/benchmarks/maxcut/path_relinking_gset.toml`, five runs, seed 42, 60 s,
the five runs in parallel on an Apple M4. `GeneticAlgorithm` with a population
of 10, tournament selection, and a tabu search of tenure 15 to 300 stopped
after 10000 moves without improvement, on each offspring and on each initial
member. Only `crossover_kind` differs. Average cut, higher is better.

| Graph | Vertices | PathRelinking | RandomPathRelinking | Uniform |
|---|---|---|---|---|
| G22 | 2000 | 13338.2 | 13323.8 | 13343.0 |
| G55 | 5000 | 10074.0 | 10018.0 | 10036.2 |
| G60 | 7000 | 13800.2 | 13628.6 | 13740.4 |
| G70 | 10000 | 9287.6 | 9016.4 | 9132.0 |

The standard deviations are 12 to 59, so G22 is a tie and the greedy path
leads from G55 up, by 38, 60 and 156. The greedy path made fewer moves than
the uniform crossover on G60, 8.3 against 11.4 million, which is the time its
O(d²) pricing takes.
