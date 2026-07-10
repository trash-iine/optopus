# QUBO

Quadratic Unconstrained Binary Optimization. Given a symmetric coefficient
matrix `Q`, **minimize**

```text
E(x) = Σ_{i ≤ j} Q[i][j] · x[i] · x[j]    (x ∈ {0,1}^n)
```

## Solution

```rust
pub struct QuboSolution {
    pub x: Vec<bool>,            // variable assignment
    pub gain: Vec<i32>,          // change in energy per flip (negative = improving)
    pub objective: i32,          // current energy
    // pub(crate) negative_gain_* — optional advanced index
}
```

Coefficients use `Coefficient = i32`. `Rankable::is_better_than` returns
`self.objective < other.objective` (lower is better).

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `QuboFlipNeighbor` | Flip one variable; gain refresh in O(degree). | `iter + 1` |
| `QuboSwapNeighbor` | Swap two variables with different values. | `iter + 2` |

Both implement `Rankable`, `Evaluate<f64>` *and* `Evaluate<i32>` (integer
gains avoid FP drift), and `EnabledTabu`.

## Crossover

- `QuboUniformCrossover` — per-variable random parent selection.
- `Qubo` implements `SubProblemExtractable`. Variables that agree in both
  parents are fixed; their contributions are folded into the linear terms of
  the sub-QUBO so the sub-problem stays self-contained.

## Construction

```rust
use optopus::prelude::*;

// Empty + manual:
let mut qubo = Qubo::new();
qubo.set_q(0, 1, 1);            // overwrite semantics
qubo.add_q(0, 0, -2);           // accumulate semantics

// From entries (last write wins on duplicates):
let qubo = Qubo::from_entries([
    (0, 1, 1),
    (0, 2, 2),
    (1, 2, 3),
    (0, 0, -1),                 // diagonal = linear term
]);

// Load from file:
let qubo = Qubo::load_file("data/instances/qubo/sample.qubo")?;
# Ok::<(), optopus::error::OptError>(())
```

## File format

```text
N M
i j v
i j v
...
```

- `N` — variable count, `M` — number of entries.
- Indices are **1-indexed**; converted to 0-indexed internally.
- `i == j` lines store the linear (diagonal) coefficient.
- Duplicate entries follow `set_q` semantics: the last write wins.

## Optional traits

- `Distance` — Hamming distance on `x`.
- `Evaluate<i32>` (in addition to `Evaluate<f64>`) for integer-precision SA /
  LAHC / RL Search.

## Notes

- An optional **`negative_gain` index** lets problem-specific algorithms
  enumerate only improving flips in O(|improving|). Not needed for standard
  heuristic use.

## References

- Kochenberger, G., Hao, J.-K., Glover, F., Lewis, M., Lü, Z., Wang, H., and
  Wang, Y. "The Unconstrained Binary Quadratic Programming Problem: A Survey."
  *Journal of Combinatorial Optimization*, 28(1), 58-81, 2014.
- Beasley, J. E. "Obtaining Test Problems via Internet." *Journal of Global
  Optimization*, 8(4), 429-433, 1996. (OR-Library, source of the bundled
  `bqp` instance set.)
- See [`data/instances/README.md`](../../data/instances/README.md) for
  instance sources and licensing.
