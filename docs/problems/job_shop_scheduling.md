# Job Shop Scheduling

**API:** [`JobShopScheduling`](../api/optopus/problem/job_shop_scheduling/struct.JobShopScheduling.html)

Job Shop Scheduling is one of the most-studied strongly NP-hard scheduling
problems.

Given `n_jobs` jobs and `n_machines` machines, each job `j` is a fixed
ordered sequence of operations `(machine, duration)` that must run on their
machines in that order: an operation cannot start before its predecessor in
the same job finishes, and a machine can process only one operation at a
time. Let `C_{j,k}` be the completion time of the `k`-th operation of job
`j`, with duration `p_{j,k}`; the *makespan* is the time the last operation
anywhere finishes. **Minimize** the makespan:

```text
minimize  max_j C_{j,last}
subject to  C_{j,k} ≥ C_{j,k-1} + p_{j,k}                 (precedence within a job)
            operations on the same machine do not overlap  (machine capacity)
```

## Example

Running a search and reading back the decoded schedule:

```rust
use optopus::prelude::*;

let inst = JobShopScheduling::new(
    "tiny".to_string(),
    /* n_machines = */ 2,
    vec![
        vec![(0, 2), (1, 3)],   // job 0: M0(2) → M1(3)
        vec![(1, 1), (0, 4)],   // job 1: M1(1) → M0(4)
    ],
);
let mut state = SearchState::new(&inst);
LocalSearch::<JobShopSwapNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("makespan = {}", sol.objective);
println!("operation order = {:?}", sol.operations); // decoded permutation-with-repetition
println!("completion times = {:?}", sol.completion_times); // finish time of each position above
```

## Solution

Solutions are encoded as a **permutation-with-repetition** of length
`n_jobs * n_machines` — the `k`-th occurrence of job `j` in the sequence
names the `k`-th operation of that job — and decoded by **left-shift
semi-active scheduling** into the completion times `C_{j,k}` from the
definition above.
[`JobShopSolution`](../api/optopus/problem/job_shop_scheduling/struct.JobShopSolution.html)
carries that encoding as `operations`, the per-position decoded completion
times as `completion_times` (so `completion_times[pos]` is the `C_{j,k}` of
the operation at position `pos`), and `objective`, which is `max_j C_{j,last}`,
the makespan being minimized. 

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `JobShopSwapNeighbor` | Swap `operations[i]` with `operations[i+1]`. | `iter + 1` |
| `JobShopRelocateNeighbor` | Remove `operations[i]` and reinsert it at another position. | `iter + 1` |

## Crossover

The crossover `JobShopPpxCrossover` is Precedence-Preserving Crossover (PPX): at each
child position, randomly choose a parent and append that parent's leftmost
unconsumed operation. Both parents are kept in sync, so the child remains a
precedence-feasible permutation-with-repetition.

## File format (Taillard / OR-Library)

```text
n_jobs n_machines
m d m d m d ...      (n_machines (machine, duration) pairs per job, one job per line)
m d m d m d ...
...
```

- Machine indices are **0-indexed**.
- Empty lines and `#`-prefixed comment lines are ignored.
- Whitespace within and between lines is flexible — the file is tokenized
  rather than read line-strictly.

```rust
use optopus::prelude::*;

let inst = JobShopScheduling::load_file("data/instances/jssp/ft06.txt")?;
# Ok::<(), optopus::error::OptError>(())
```

## References

- Fisher, H. and Thompson, G. L. "Probabilistic Learning Combinations of
  Local Job-Shop Scheduling Rules." In *Industrial Scheduling*, pp. 225-251.
  Prentice-Hall, 1963. (Source of the classic `ft06` instance.)
- Taillard, E. "Benchmarks for Basic Scheduling Problems." *European Journal
  of Operational Research*, 64(2), 278-285, 1993.
- See [`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md) for
  instance sources and licensing.
