# Edge assembly crossover runs as its own genetic algorithm, and its first stage is enough on the instances here

- Status: adopted
- Area: heuristic, tsp
- Date: 2026-11-15
- Code: src/heuristic/specific/tsp/eax.rs

## Decision

`EdgeAssemblyCrossover` is a heuristic of its own under
`heuristic/specific/tsp`, and not a `Crossover` for `GeneticAlgorithm`. The
method is the crossover together with its replacement. A pair makes up to 30
offspring, each the first parent with one AB-cycle exchanged, and the chosen
one replaces that first parent only, so a member is replaced by a tour that
keeps most of its edges. `GeneticAlgorithm` makes one offspring per mating
and replaces its worst member, which keeps nothing of the member that leaves.

The paper is behind a paywall and was not read. The algorithm was read in the
implementation its first author publishes under the MIT license, the README
and the sources of the crossover and the main loop, and written again from
that reading in this crate's own structures. Taken from it are the pairing by
a random permutation, the single-cycle offspring, the join of sub-tours
smallest first over the 10 and then 50 nearest cities, the two selections,
the entropy loss computed edge by edge on the counts, and the stagnation rule.

Three things differ. The second stage, which exchanges a block of neighboring
AB-cycles found by a tabu search, is not implemented, so a run ends where the
published one changes stages. The first members come from
`AnchoredTourDescent`, Or-opt and 2-opt, where the published one uses 2-opt
alone. The edge counts are a short list per city and not a table of every
pair of cities, so memory grows with the edges in use.

## Do not retry

Registering the crossover for `GeneticAlgorithm` as a `crossover_kind` loses
the replacement, for the reason above, and was not measured.

The second stage is the next step, and only for instances well beyond a
thousand cities, where the first already ends at the optimum in every run.

## Appendix

`data/benchmarks/tsp/eax_tsplib.toml`, five runs, seed 42, a limit of 60 s,
the five runs in parallel on an Apple M4. Population 300, 30 offspring per
pair. Average tour length, with the average time the run took where it ended
by itself.

| Instance | EAX, entropy | EAX, greedy | Iterated LKH | ALNS |
|---|---|---|---|---|
| eil101 | 629.0 in 0.12 s | 629.0 in 0.08 s | 629.0 | 631.4 |
| ch150 | 6528.0 in 0.24 s | 6528.0 in 0.22 s | 6528.0 | 6558.6 |
| dsj1000 | 18660188.0 in 8.5 s | 18660188.0 in 3.4 s | 19039074.6 | 19027145.8 |

Every run of both selections reached the same length. For dsj1000 that length
is the optimal tour length the DIMACS challenge page lists. The values for
eil101 and ch150 were not checked against a source. Iterated LKH is
`Iterated` over `LinKernighanHelsgaun` and a kick of five random 2-opt moves.

The two selections cannot be told apart on these instances. The entropy
selection takes about twice the time on dsj1000 and reaches the same tour, so
what it buys, if anything, shows on larger instances than these.

One run each on 5000 cities drawn uniformly in a square, an instance that is
not in the repository, with a limit of 600 s for the search and 300 s for
ALNS. The entropy selection ended by itself at 50831436 after 257 s, the
greedy one at 50854730 after 136 s, and ALNS stood at 54432689 at its limit.
One run does not separate the two selections. The instance keeps only
nearest neighbours instead of a matrix, the case the symmetry check has to
let through.
