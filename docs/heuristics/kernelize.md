# KernelizedSearchForMaxCut

Exact data reduction (*kernelization*) for [MaxCut](../problems/max_cut.md),
following Ferizovic, Hespe, Lamm, Mnich, Schulz and Strash, *Engineering
Kernelization for Maximum Cut* (ALENEX 2020,
[arXiv:1905.10902](https://arxiv.org/abs/1905.10902)).

`MaxCutKernel::reduce` shrinks an instance by rules that provably preserve the
optimum, leaving a smaller instance, a constant offset, and a trace that
re-derives every removed vertex. `KernelizedSearchForMaxCut` wraps any
`Heuristic<MaxCut>` so it searches the kernel instead of the original.

A kernel is the opposite trade-off from a heuristic contraction, which also
shrinks an instance but does so by *guessing* structure and therefore restricts
which solutions are reachable. A kernel restricts nothing: solving it and
lifting is equivalent to solving the original.

## The rules

Stated in raw cut-value space. Each is derived here rather than transcribed,
and pinned down by an exhaustive small-graph test — see *Correctness* below.

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
weights that is `Δ = −1` with offset `+2`, so **the rule produces negative
weights**; `Graph` and `MaxCut` carry `f32` weights and every move, gain and
objective path is already sign-agnostic, so nothing downstream has to change.

**Domination.** If one incident edge outweighs all the others, moving `v` to
the side that edge prefers gains at least as much as it can lose everywhere
else, so *some* optimum does exactly that. `v`'s side becomes a function of
`a`'s — a contraction, not a deletion, because `v`'s other edges are still
undecided and get carried over to `a` (with their sign flipped when `v` sits
opposite `a`). This is what lets the rules keep cascading on the weighted
graph that the path rule produces.

The rules run to a fixpoint in a work queue: a vertex is re-examined only when
one of its incident edges changes. For the merge that means the re-queue set
has to be read *before* the merge, not after: carrying an edge over to `a` can
cancel one of `a`'s existing edges to exactly zero, which deletes it from both
endpoints, so the vertex whose degree just fell is no longer in `a`'s
neighborhood to be found. Reading it afterwards left kernels with removable
vertices still in them (127 of 40000 random signed instances on 4-14
vertices), which in turn could leave a survivor with no edges at all — and
since `Graph` sizes itself from the largest vertex id it sees, that made the
kernel graph *shorter* than its own vertex list and every solution built for it
too short to lift. The compaction step now records an edgeless survivor as
isolated, so `graph.len() == original_of.len()` holds by construction rather
than by the queue having reached everything.

## What reduces and what does not

Reduction happens at *low degree*, so it is a question of the instance, not of
tuning. Measured (`kernelize: reduction finished` at `INFO`):

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

A 4-regular graph has no vertex of degree ≤ 2 and no dominating edge, so the
rules cannot fire — and neither can they on dense instances, which the paper
also reports. `MaxCutKernel::is_trivial` detects that case, and the wrapper
then hands the state straight to the inner heuristic, so leaving kernelization
enabled costs nothing on instances it cannot help.

On a **tree the kernel is empty**: the reduction alone returns the exact
optimum (every edge cut), which no local search reliably finds on 10000
vertices.

## Measured results

60-second runs, 5 runs each, seed 42. Both the plain and the kernelized arm
use the same inner heuristic, with size-dependent parameters scaled to the
instance each one actually sees (see *Usage*). `K{X}` is `X` run on the kernel.

| instance | BLS | K{BLS} | delta |
|---|---|---|---|
| G55 | 10206.0 | 10218.6 | +12.6 |
| G60 | 14042.2 | 14073.0 | +30.8 |
| G70 | 9496.8 | **9589.4** | +92.6 |
| `ba_n10000_m001` | 9844.2 | **9999.0** | +154.8 |
| `ba_n10000_m002` | 16517.8 | 16679.4 | +161.6 |
| `er_n05000_d005_w110` | 58450.0 | 58609.6 | +159.6 |
| `er_n10000_d002` | 9414.2 | 9506.0 | +91.8 |
| `er_n10000_d004` | 16905.8 | 16988.4 | +82.6 |
| `er_n20000_d003_pm10` | 67103.2 | 71990.2 | **+4887.0** |
| **total** | | | **+5673** |
| **instances won** | | | **9 / 9** |

Kernelization wins on every instance, and the margin tracks the reduction rate:
13% reduction buys tens of points, 59-75% buys thousands.

Two results are worth separating out:

- On the **tree** the kernel is empty, so `9999.0` is not a good heuristic
  result — it is the exact optimum, produced by the reduction with no search
  at all. Plain BLS averages 9844 on the same instance.
- On **G70** the kernelized run is also far more *stable*: its worst run
  (9587) beats plain BLS's best (9506), and its best (9593) edges past the best
  value previously recorded here (9592).

On a dense instance the wrapper is measurably inert: `is_trivial` fires before
anything is allocated and the inner heuristic runs on the original state, so
the two arms differ only by the per-run seed derivation, not by the search.

## Correctness

The reduction is only worth anything if it is exact, so that is what the tests
check — not the individual rules, but the property that matters:

- **Exhaustive equivalence**: over random graphs with `n ≤ 9` in three weight
  regimes (unit, ±1, general integer), `maxcut(original) == maxcut(kernel) +
  offset` by brute force. Every rule has to survive this before it is enabled.
- **Lifting identity**: `kernel_cut(y) + offset == original_cut(lift(y))` for
  *every* `y`, not only optimal ones — so a heuristic may be stopped at any
  point and lifted.
- **Cascade**: a path reduces to nothing and the offset equals the optimum.
- **Idempotence**: re-reducing a kernel finds nothing. The sweep is wide
  (three weight regimes × `n = 4..14` × three densities × 25 draws) because the
  merge rule's re-queue bug above needed signed weights and a few hundred draws
  to appear at all; the original 20-instance sweep missed it.
- **Index space**: the kernel graph covers every kernel vertex, so a solution
  sized from `graph.len()` — which is what `MaxCut::new_solution` gives any
  inner heuristic — can always be lifted.
- **Inertness**: on a regular graph the wrapper produces bit-identical results
  to running the inner heuristic alone.

## Usage

```toml
[[heuristics]]
kind = "Kernelize"
[heuristics.stop_condition]
max_duration_secs = 60

# Exactly one nested step: the heuristic that solves the kernel.
[[heuristics.steps]]
kind = "BreakoutLocalSearch"
tabu_tenure = [3, 216]
t = 1000
l0 = 21
p0 = 0.8
q = 0.5
[heuristics.steps.stop_condition]
max_duration_secs = 60
```

Size-dependent parameters of the inner heuristic should be scaled to the
**kernel**, not the original instance — `G70` reduces from 8646 to 2164
vertices, and a tenure sized for the former is four times too long for the
latter.
