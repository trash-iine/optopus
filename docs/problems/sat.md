# MaxSAT

Given a propositional CNF formula, **maximize** the number of satisfied
clauses.

## Solution

```rust
pub struct SatSolution {
    pub x: Vec<bool>,        // assignment; x[i] is variable i+1 in DIMACS
    pub gain: Vec<i64>,      // change in satisfied count per flip
    pub n_satisfied: usize,  // number of currently satisfied clauses
}
```

`Rankable::is_better_than` returns `self.n_satisfied > other.n_satisfied`.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `SatFlipNeighbor` | Flip one variable; gain refresh in O(clauses_per_var). | `iter + 1` |
| `SatSwapNeighbor` | Swap two variables. | `iter + 2` |

Both implement `Rankable`, `Evaluate<f64>`, and `EnabledTabu`.

## Crossover

- `SatUniformCrossover` — per-variable random parent selection.
- `Sat` implements `SubProblemExtractable` for `SubProblemBasedCrossover`.

## Construction

```rust
use optopus::prelude::*;

// Build manually:
let mut sat = Sat::new(3);
sat.add_clause([1, -2, 3]);   // (x1 ∨ ¬x2 ∨ x3); literals are signed 1-indexed
sat.add_clause([-1, 2]);

// Load DIMACS CNF:
let sat = Sat::load_file("data/sat/example.cnf")?;
# Ok::<(), optopus::error::OptError>(())
```

Note the indexing convention: `add_clause` and the file format use **signed
1-indexed** literals (positive = positive literal, negative = negation).
Internally, `x[i]` is the truth value of variable `i + 1`.

## File format (DIMACS CNF)

```text
c optional comment lines
p cnf N M
1 -2 3 0
-1 2 0
...
```

- `N` — number of variables, `M` — number of clauses.
- Each clause line is a space-separated list of signed integers terminated by
  `0`; the sign carries the polarity, the magnitude is the variable index
  (1-indexed).

## Optional traits

- `Distance` — Hamming distance on `x`.
- `CdclEncodable` — natively solvable by the built-in CDCL engine via the
  identity encoding.
