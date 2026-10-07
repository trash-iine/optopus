# Bin Packing

**API:** [`BinPacking`](../api/optopus/problem/bin_packing/struct.BinPacking.html)

One-dimensional bin packing puts items of given sizes into bins of one
capacity, no bin over it, and uses as few bins as possible. The bin count alone
is a flat objective, since every packing with the same count ties, so the
search minimizes

```text
num_bins − (1 / num_bins) Σ_b (load_b / capacity)²
```

which is the count less Falkenauer's fill term. The term lies in `(0, 1]`, so
fewer bins always wins, and among packings with the same count it prefers the
ones whose bins are fuller, the ones a bin can be emptied from.
`BinPacking::lower_bound` gives `⌈Σ size / capacity⌉`.

## Example

```rust
use optopus::prelude::*;

let bp = BinPacking::load_file("data/instances/bin_packing/falkenauer/u250_00.txt")?;
let mut state = SearchState::new_with_seed(&bp, 42);
AdaptiveLargeNeighborhoodSearch::<BinPacking>::new(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* removal_fraction = */ 0.15,
    /* cooling_rate     = */ 0.9995,
)
.run(&mut state)?;
println!("{} bins, at least {}", state.best_solution.num_bins(), bp.lower_bound());
```

## Solution

[`BinPackingSolution`](../api/optopus/problem/bin_packing/struct.BinPackingSolution.html)
carries `bin_of`, the bin of every item, `loads`, the load of every bin, and
`objective`. The bins are always numbered `0..num_bins()` with none of them
empty, and a move that empties one renumbers the rest. Its `Distance` compares
which items share a bin, not the bin numbers.

## Neighbors

| Type | Config | Move |
|---|---|---|
| `BinPackingRelocateNeighbor` | `Relocate` | Move one item into another bin with room. Emptying a bin is how the count falls. |
| `BinPackingSwapNeighbor` | `Swap` | Exchange two items of different sizes in different bins when both still fit. |

Both are priced in O(1) from the loads, and neither ever overfills a bin.

## Ruin and recreate

`BinPacking` implements [`Ruinable`](../traits.md) with items as the elements
and bins as the containers, the case the trait was written around. Every open
bin is a bucket and one empty bin is always offered after them, so a repair
never runs out of room. An item that does not fit a bin costs infinity there,
and regret, the gap between an item's best bin and its next best, means
something. [ALNS](../heuristics/alns.md) runs on it without a local repair, and
so does [GreedyRandomizedConstruction](../heuristics/greedy_randomized_construction.md).

## Crossover

`BinPackingGroupCrossover` passes on whole bins rather than items, in the
spirit of Falkenauer's grouping crossover. The child takes bins from the two
parents alternately, fullest first, skipping any that holds an item already
placed, and packs what is left by First Fit Decreasing.

## File format

```text
n capacity
size_0
size_1
...
```

All values are whitespace separated integers. Falkenauer's 160 instances are
bundled as `data/instances/bin_packing/falkenauer/`, the uniform class
`u120` to `u1000` with capacity 150 and the triplets `t60` to `t501`, scaled by
ten to capacity 1000 so their sizes are integers. A triplet instance has an
optimum of exactly `n / 3` bins by construction.

## References

- Falkenauer, E. "A hybrid grouping genetic algorithm for bin packing."
  *Journal of Heuristics*, 2(1), 5-30, 1996.
- See [`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md)
  for instance sources and licensing.
