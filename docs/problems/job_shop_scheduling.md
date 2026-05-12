# Job Shop Scheduling

Given `n_jobs` jobs and `n_machines` machines, where each job has a fixed
sequence of `(machine, duration)` operations, **minimize** the makespan
(time at which the last operation finishes).

## Encoding

A solution is a **permutation with repetition** of length `n_jobs * n_machines`:
each job index appears exactly `n_machines` times, and the *k*-th occurrence
(0-indexed) of job `j` represents operation `O(j, k)`. The operation sequence
is decoded via left-shift semi-active scheduling — every operation is started
as early as both its machine and its job's previous operation allow.

## Solution

```rust
pub struct JobShopSolution {
    pub operations: Vec<usize>,        // permutation-with-repetition; length n_jobs * n_machines
    pub objective: u32,                // makespan (Cmax)
    pub completion_times: Vec<u32>,    // finish time of each position in `operations`
}
```

`Rankable::is_better_than` returns `self.objective < other.objective`.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `JobShopSwapNeighbor` | Swap `operations[i]` with `operations[i+1]`; re-decode for the exact gain. | `iter + 1` |
| `JobShopRelocateNeighbor` | Remove `operations[i]` and reinsert it at another position; re-decode. | `iter + 1` |

Both implement `Rankable`, `Evaluate<f64>`, and `EnabledTabu`.

## Crossover

- `JobShopPpxCrossover` — Precedence-Preserving Crossover (PPX): at each
  child position, randomly choose a parent and append that parent's leftmost
  unconsumed operation. Both parents are kept in sync, so the child remains a
  precedence-feasible permutation-with-repetition.

## Construction

```rust
use optopus::prelude::*;

// Manual:
let inst = JobShopScheduling::new(
    "tiny".to_string(),
    /* n_machines = */ 2,
    vec![
        vec![(0, 2), (1, 3)],   // job 0: M0(2) → M1(3)
        vec![(1, 1), (0, 4)],   // job 1: M1(1) → M0(4)
    ],
);

// Load from file:
let inst = JobShopScheduling::load_file("data/jssp/ft06.txt")?;
# Ok::<(), optopus::error::OptError>(())
```

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

## Optional traits

- `Distance` — number of positions where the operation index differs.
