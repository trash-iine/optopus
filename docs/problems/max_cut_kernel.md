# MaxCutKernel

**API:** [`MaxCutKernel`](../api/optopus/problem/max_cut/struct.MaxCutKernel.html)

Exact data reduction (*kernelization*) for [MaxCut](max_cut.md), based on
Ferizovic, Hespe, Lamm, Mnich, Schulz and Strash, *Engineering Kernelization
for Maximum Cut* ([arXiv:1905.10902](https://arxiv.org/abs/1905.10902)).

`MaxCutKernel::new` shrinks an instance by rules that preserve the optimum,
provably leaving a smaller instance, a constant offset, and a trace that
re-derives every removed vertex.

## Example

`MaxCutKernel` implements [`ProblemReduction`](../traits.md#problemreduction),
so a heuristic reaches the kernel through the two `SearchState` methods that
own the crossing.

```rust
let kernel = MaxCutKernel::new(&mc);          // run the rules once, not per cycle
let mut state = SearchState::new_with_seed(&mc, seed);

while !outer.is_done(&state) {
    let before = state.iteration;

    let mut sub = state.open_reduction(&kernel);   // project the warm start, seed from the parent RNG
    inner.run(&mut sub)?;
    state.close_reduction(&kernel, &sub);          // merge counters, install the lifted solution, update best

    if state.iteration == before {
        state.progress_iteration();                // see "budgets" below
    }
}
```

## The rules

| Rule | Precondition | Operation | Offset | Lifting |
|---|---|---|---|---|
| **Isolated** | `deg(v) = 0` | delete `v` | `0` | arbitrary |
| **Pendant** | `deg(v) = 1`, weight `w` | delete `v` | `max(0, w)` | opposite `a` iff `w > 0` |
| **Path** | `deg(v) = 2`, neighbors `a ≠ b`, weights `w₁, w₂` | delete `v`, add `Δ` to edge `(a,b)` | `max(0, w₁+w₂)` | whichever side pays |
| **Domination** | `\|w(v,a)\| ≥ Σ_{u≠a} \|w(v,u)\|` | merge `v` into `a` | see below | `x[v] = x[a] XOR opposite` |

**Path.** With `a` and `b` on the same side the best `v` can do is
`max(0, w₁+w₂)`; with them apart it is `max(w₁, w₂)`. The pair therefore
contributes a constant plus a term depending only on whether `(a,b)` is cut —
which *is* an edge, of weight `Δ = max(w₁,w₂) − max(0, w₁+w₂)`. For unit
weights that is `Δ = −1` with offset `+2`.

**Domination.** If one incident edge outweighs all the others, moving `v` to
the side that edge prefers gains at least as much as it can lose everywhere
else, so *some* optimum does exactly that. `v`'s side becomes a function of
`a`'s — a contraction, not a deletion, because `v`'s other edges are still
undecided and get carried over to `a` (with their sign flipped when `v` sits
opposite `a`). This is what lets the rules keep cascading on the weighted
graph that the path rule produces.

These rules run to a fixpoint in a work queue: a vertex is re-examined only when
its edges change. The re-queue set has to be read *before* the merge, not after:
carrying an edge over to the vertex `a` can cancel one of `a`'s existing edges to
exactly zero, which deletes it from both endpoints, so the vertex whose degree
just fell is no longer in `a`'s neighborhood to be found.  Reading it afterwards
left kernels with removable
vertices still in them (127 of 40000 random signed instances on 4-14
vertices), which in turn could leave a survivor with no edges at all — and
since `Graph` sizes itself from the largest vertex id it sees, that made the
kernel graph *shorter* than its own vertex list and every solution built for it
too short to lift. The compaction step now records an edgeless survivor as
isolated, so `graph.len() == original_of.len()` holds by construction rather
than by the queue having reached everything.

## What reduces and what does not

Reduction happens at *low degree*, so it is a question of the instance, not of
tuning.

| instance | n | avg degree | kernel n | removed |
|---|---|---|---|---|
| `ba_n10000_m001` (tree) | 10000 | 2.0 | **0** | **100%** |
| **G70** | 8646 | 2.3 | 2164 | **75%** |
| `er_n10000_d002` | 8589 | 2.3 | 2105 | 75% |
| `er_n20000_d003_pm10` | 19055 | 3.2 | 7780 | 59% |
| `ba_n10000_m002` | 10000 | 4.0 | 4955 | 50% |
| `er_n10000_d004` | 9807 | 4.1 | 7472 | 24% |
| `er_n05000_d005_w110` | 4969 | 5.1 | 3887 | 22% |
| G55 / G60 | 4969 / 6957 | 5.0 / 4.9 | 4351 / 6073 | 13% |
| G57 / G63 / G64 (regular) | 5000–7000 | 4–12 | unchanged | **0%** |
| G1, dense generated suite | 800–5000 | 48–1000 | unchanged | **0%** |

On a tree the kernel is empty. The reduction alone returns the exact
optimum (every edge cut).

On the other hand, on 4-regular graphs (G57 / G63 / G64) and a dense one (G1),
rules cannot fire.

## References

- Ferizovic, D., Hespe, D., Lamm, S., Mnich, M., Schulz, C. and Strash, D.
  "Engineering Kernelization for Maximum Cut." *ALENEX 2020*.
  [arXiv:1905.10902](https://arxiv.org/abs/1905.10902)
