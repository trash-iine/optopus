# EdgeAssemblyCrossoverForTsp

**API:** [`EdgeAssemblyCrossoverForTsp`](../api/optopus/heuristic/struct.EdgeAssemblyCrossoverForTsp.html)

Problem-specific heuristic for the [TSP](../problems/tsp.md) with symmetric
distances. A genetic algorithm whose crossover, edge assembly crossover,
builds an offspring out of the edges of its two parents and almost nothing
else, and whose replacement keeps the population apart for a long time.

## Example

```rust
use optopus::prelude::*;

let tsp = Tsp::load_file("data/instances/tsp/eil101.tsp")?;
let mut state = SearchState::new(&tsp);

let mut eax = EdgeAssemblyCrossoverForTsp::new(
    StopCondition::duration(std::time::Duration::from_secs(30)),
    /* population_size    = */ 300,
    /* offspring_per_pair = */ 30,
);
eax.run(&mut state)?;
println!("tour length = {}", state.best_solution.objective);
```

Takes no `neighbor` type parameter, since it owns its move set. A run usually
ends before its time limit, see [When it stops](#when-it-stops).

## Algorithm sketch

The population starts as `population_size` tours. The first is the incumbent
and the rest are random, and each is taken to a local optimum of
[`AnchoredTourDescent`](../api/optopus/problem/tsp/struct.AnchoredTourDescent.html),
the Or-opt and 2-opt descent over nearest neighbours.

A generation draws a random order of the population and mates each member, as
the first parent, with the one after it, as the second. One `run_once` is one
mating, or the construction of one member while the population is being built.

### Crossover

1. Take the edges that only one of the two parents has. Every city has as
   many of them from one parent as from the other.
2. Split them into AB-cycles, closed walks that alternate between an edge of
   the first parent and an edge of the second. The walk starts at a random
   city and picks at random where it has a choice.
3. An offspring is the first parent with one AB-cycle exchanged. Its edges of
   the first parent are taken out and its edges of the second are put in.
   Every city still has two edges, so the result is one tour or several
   sub-tours.
4. Join the sub-tours, smallest first. For the smallest, remove one of its
   edges and one edge of another sub-tour, and add the two edges that connect
   the four ends, choosing the exchange that costs the least. The other edge
   is looked for among the 10 nearest cities of each city of the sub-tour,
   and among the 50 nearest if that finds none.

A mating makes one offspring from each of up to `offspring_per_pair`
AB-cycles, drawn without repetition.

An offspring differs from its first parent by one cycle and a few joins. A
mating reads both parents once, and builds each offspring in time that grows
with the cycle and the sub-tours it joins, which is less than the tour unless
the cycle splits it into large parts.

### Selection and replacement

The offspring the selection prefers replaces the first parent, if it is
shorter than that parent. Nothing else in the population changes. A member is
only ever replaced by a tour made mostly of its own edges, so two members do
not become the same tour just because one of them is good.

| `EaxSelection` | Prefers |
|---|---|
| `Greedy` | the shortest offspring |
| `Entropy` (default) | the offspring with the largest ratio of what it shortens to the edge entropy the population loses by taking it |

The edge entropy is `−Σ p ln p` over the share `p` of members that hold each
edge. An offspring that shortens the tour with edges the population has few
of scores above one that copies edges most members already have, and an
offspring that costs no entropy at all scores highest.

### When it stops { #when-it-stops }

Besides its stop condition, a run ends by itself in two cases.

- Every tour of the population has the same length.
- The best tour stagnates. Once it has stood for `1500 / offspring_per_pair`
  generations, a tenth of the generations run so far becomes the limit, and
  the run ends when the best tour has stood for that many.

### What is not here

This is the first stage of the published algorithm. That algorithm goes on
to a second stage when the first stagnates, which exchanges several
neighbouring AB-cycles at once and searches, by a tabu search over which
cycles to take, for a block that meets the rest of the tour at few cities.
The second stage is not implemented, so a run ends where the published one
would change stages.

## Constructor

```rust
EdgeAssemblyCrossoverForTsp::new(
    stop_condition: StopCondition,
    population_size: usize,      // at least 2
    offspring_per_pair: usize,   // at least 1
) -> Self
```

`stop_condition` decides when the run ends, see [Stop
conditions](../guide/stop_conditions.md), and counts one mating as one
iteration. The published values are `300` and `30`, the associated constants
`POPULATION_SIZE` and `OFFSPRING_PER_PAIR`.

| Builder | Sets | Default |
|---|---|---|
| `with_selection(EaxSelection)` | which offspring replaces the first parent | `EaxSelection::Entropy` |

Building the population costs one descent from a random tour per member, and
the memory is two neighbours per city per member.

`clear()` drops the population, so a second `run` builds a new one.

## Benchmark config

```toml
[[heuristics]]
kind = "EdgeAssemblyCrossover"
population_size = 300      # optional (default shown)
offspring_per_pair = 30    # optional (default shown)
selection = "Entropy"      # optional, Entropy (default) | Greedy
[heuristics.stop_condition]
max_duration_secs = 60.0
```

An instance with asymmetric distances fails each run with an error.

## References

- Nagata, Y. and Kobayashi, S. "A Powerful Genetic Algorithm Using Edge
  Assembly Crossover for the Traveling Salesman Problem." *INFORMS Journal on
  Computing*, 25(2), 346-363, 2013.
