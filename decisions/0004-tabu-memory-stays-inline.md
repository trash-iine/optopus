# Fat LTO removes the allocation-layout cliff, so the sparse map is not boxed

- Status: adopted
- Area: search_state
- Date: 2026-08
- Code: src/common/tabu.rs

## Decision

`TabuMemory` stores its `HashMap` for compound keys inline rather than behind a
`Box`. The map was briefly boxed to work around a large slowdown that turned out
to be an allocation arrangement, not the map.

A neighborhood scan walks three heap buffers in lockstep once per candidate, the
graph's vertex list, the solution's gains, and the dense tabu array. Where the
allocator puts those three decides whether they collide in cache, and anything
that perturbs the allocation sequence moves them.

## Measurement

Under `lto = "thin"`, storing the map inline drew an arrangement that cost
`TabuSearch` 47% on G32 and 35% on G1 at identical work, with bit-identical
objectives, move counts and per-run seeds. Leaving it inline and giving the
dense array 4096 slots of slack also fixed it, and boxing the map while giving
the dense array 512 slots put the slowdown back.

`lto = "fat"`, now set in `Cargo.toml`, removes it. The same inline arrangement
measures −2.0% and +0.2% under it.

## Do not retry

Ruled out by measurement, so none of these is worth revisiting: the struct's
size, the cache lines its fields occupy, inlining of `is_enabled`, the `TabuKey`
construction, the shape of the closure that reads the state, and code alignment.
Instruction counts in `run_once` were unchanged. The cost was stalls.
