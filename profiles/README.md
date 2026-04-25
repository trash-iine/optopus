# Profiling workflow

Directory for capturing performance traces by running `examples/prof_*.rs`
under `samply`.

Each `prof_*.rs` is wrapped with `StopCondition::duration` or `Restart` so
that it finishes in roughly 5 seconds, ensuring sufficient sample density.

## Capturing the baseline

For the initial / pre-optimization reference, capture all nine examples in a single batch:

```sh
cargo build --profile profiling --examples

mkdir -p profiles/baseline
for ex in prof_ls_maxcut_flip prof_ts_maxcut_flip prof_sa_maxcut \
         prof_beam_maxcut prof_ls_tsp prof_lkh_tsp \
         prof_ls_sat prof_ls_formula prof_ga_maxcut; do
  samply record --save-only \
    -o "profiles/baseline/${ex}.profile.json.gz" \
    "./target/profiling/examples/${ex}"
done
```

## Capturing a diff (post-optimization)

Write traces used for verifying optimizations to `profiles/current/`:

```sh
mkdir -p profiles/current
samply record --save-only \
  -o profiles/current/<example>.profile.json.gz \
  ./target/profiling/examples/<example>
```

## Viewing

```sh
samply load profiles/baseline/<example>.profile.json.gz
```

## Mapping

| Example                  | Target                                       |
|--------------------------|----------------------------------------------|
| `prof_ls_maxcut_flip`    | LocalSearch × MaxCutFlip  (G22)              |
| `prof_ts_maxcut_flip`    | TabuSearch  × MaxCutFlip  (G22)              |
| `prof_sa_maxcut`         | SA × MaxCut Flip + Swap   (G22)              |
| `prof_beam_maxcut`       | BeamSearch × MaxCutFlip   (G22)              |
| `prof_ls_tsp`            | LS × TSP TwoOpt + Relocate (berlin52)        |
| `prof_lkh_tsp`           | Lin-Kernighan (Helsgaun)  (berlin52)         |
| `prof_ls_sat`            | LS × SAT Flip + Swap      (n=500)            |
| `prof_ls_formula`        | LS × FormulaProblem Flip + Swap (n=60)       |
| `prof_ga_maxcut`         | GeneticAlgorithm × MaxCut (G22)              |

## Notes

- `profiles/baseline/` and `profiles/current/` are excluded via `.gitignore`.
  Trace files are local artifacts only.
- `cargo build --profile profiling` uses the `[profile.profiling]` setting in
  `Cargo.toml` (release + `debug = "line-tables-only"`), producing a binary
  with only the line information needed for symbol resolution.
