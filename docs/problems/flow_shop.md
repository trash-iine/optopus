# Permutation Flow Shop

**API:** [`FlowShop`](../api/optopus/problem/flow_shop/struct.FlowShop.html)

The permutation flow shop schedules `n_jobs` jobs on `n_machines` machines
that every job visits in the same order, machine 0 first. Every machine takes
the jobs in one common sequence, and that sequence is the solution. A machine
runs one job at a time and a job is on one machine at a time. With `p(i, j)`
the time job `j` spends on machine `i`, and `C(i, k)` the time the `k`-th job
of the sequence `π` leaves machine `i`, the makespan is minimized.

```text
C(i, k) = max(C(i-1, k), C(i, k-1)) + p(i, π_k)      C(·, 0) = C(0, ·) = 0
minimize  C(m, n)
```

It is strongly NP-hard from three machines on. Taillard's 120 instances are
the standard benchmark, and on them the hard block is the middle one, 50 jobs
on 20 machines and 100 on 20, rather than the largest.

## Example

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
println!("NEH = {}", fs.neh().objective);

let mut state = SearchState::new_with_seed(&fs, 42);
iterated_greedy_for_flow_shop(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* removal_count      = */ 4,
    /* temperature_factor = */ 0.4,
)
.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
println!("sequence = {:?}", state.best_solution.sequence);
```

[Iterated Greedy](../heuristics/iterated_greedy.md) is the search the problem
was built for. `fs.neh()` is the constructive heuristic of Nawaz, Enscore and
Ham, which is also exposed on its own.

## Solution

[`FlowShopSolution`](../api/optopus/problem/flow_shop/struct.FlowShopSolution.html)
carries `sequence`, a permutation of the jobs with the one run first first, and
`objective`, its makespan. `FlowShop::solution_from_sequence` wraps a sequence
of your own.

## Heads and tails

Every price on this problem comes from Taillard's acceleration. Split a
sequence anywhere into a prefix and a suffix. The head of the prefix is when
its last job leaves each machine, and the tail of the suffix is how long each
machine stays busy from the moment its first job starts there. The makespan is
the largest sum of head and tail over the machines, and neither side depends
on the other. Once both are tabulated for every split point, which is O(nm),
inserting a job at any place or taking one out is O(m), and trying a job at
all `n + 1` places is O(nm). That is what makes NEH O(n²m) and an Iterated
Greedy iteration cheap.

## Neighbors

| Type | Config | Move | Scan cost |
|---|---|---|---|
| `FlowShopInsertNeighbor` | `Relocate` | Take one job out and put it back at another place. | all `(n−1)²` moves in O(n²m) |
| `FlowShopSwapNeighbor` | `Swap` | Exchange the jobs at two positions. | each move O((j−i)·m) from the heads and tails |

Insertion is the neighborhood the flow shop literature relies on. Both moves
draw a random neighbor in O(nm).

## Ruin and recreate

`FlowShop` implements [`Ruinable`](../traits.md) with jobs as the elements and
the sequence as the one container, the same shape as the TSP's tour. The
partial solution keeps the heads and tails, so the trait's per-place
insertion cost is O(m) rather than a full schedule. That is what
[Iterated Greedy](../heuristics/iterated_greedy.md) and
[ALNS](../heuristics/alns.md) run on, through `iterated_greedy_for_flow_shop`
and `alns_for_flow_shop`. Both pair the search with `FlowShopInsertionDescent`,
the full insertion descent of Ruiz and Stützle, which takes every job out in a
random order and puts it back where the makespan is smallest until a pass
changes nothing.

## Crossover

`FlowShopOrderCrossover` is Order Crossover (OX), a stretch of the first
parent's sequence kept in place and the other jobs filled in the order the
second parent runs them.

## File format

```text
n_jobs n_machines
p(0,0) p(0,1) ... p(0,n-1)        one line of processing times per machine
...
```

This is Taillard's layout without his seed and bounds. The 120 Taillard
instances are bundled as `data/instances/flow_shop/taillard/ta001.txt` to
`ta120.txt`, regenerated from his published seeds by
`data/instances/scripts/fetch_pfsp.sh`.

## References

- Taillard, E. "Some efficient heuristic methods for the flow shop sequencing
  problem." *European Journal of Operational Research*, 47(1), 65-74, 1990.
  (The heads and tails.)
- Taillard, E. "Benchmarks for basic scheduling problems." *European Journal
  of Operational Research*, 64(2), 278-285, 1993.
- Nawaz, M., Enscore, E. E. and Ham, I. "A heuristic algorithm for the
  m-machine, n-job flow-shop sequencing problem." *Omega*, 11(1), 91-95, 1983.
- Ruiz, R. and Stützle, T. "A simple and effective iterated greedy algorithm
  for the permutation flowshop scheduling problem." *European Journal of
  Operational Research*, 177(3), 2033-2049, 2007.
- See [`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md)
  for instance sources and licensing.
