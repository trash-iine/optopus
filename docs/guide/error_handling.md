# Error Handling

**API:** [`OptError`](../api/optopus/error/enum.OptError.html)

Every fallible operation in Optopus returns `Result<_, OptError>`. The error
type is defined in [`optopus::error`](../api/optopus/error/index.html).

## Variants

| Variant | When raised | Common causes |
|---|---|---|
| [`Config(String)`](../api/optopus/error/enum.OptError.html#variant.Config) | A user-facing configuration error. | Invalid benchmark TOML field, missing required parameter, illegal range. |
| [`Io(std::io::Error)`](../api/optopus/error/enum.OptError.html#variant.Io) | Wrapped via `#[from]`; bubbles up from the `std::io` layer. | File not found, permission denied, EOF mid-read. |
| [`Parse(String)`](../api/optopus/error/enum.OptError.html#variant.Parse) | Generic format error not tied to a single line. | Malformed input that pre-loaders surface before line-by-line parsing. |
| [`TomlDe(toml::de::Error)`](../api/optopus/error/enum.OptError.html#variant.TomlDe) | Wrapped via `#[from]`; TOML deserialization failed. The inner error already carries line / column context. | Malformed benchmark config, an unknown heuristic `kind`, a field missing from a tagged `HeuristicConfig` variant. |
| [`TomlSer(toml::ser::Error)`](../api/optopus/error/enum.OptError.html#variant.TomlSer) | Wrapped via `#[from]`; serializing a value to TOML failed. | Writing a `BenchmarkReport` out with `write_to_dir`. |
| [`FileLoad { path, line, detail }`](../api/optopus/error/enum.OptError.html#variant.FileLoad) | Structured file-load error. `line == 0` indicates a file-level error not tied to one line. | TSPLIB / DIMACS / QUBO loaders hitting an unexpected token. |
| [`InvalidState(String)`](../api/optopus/error/enum.OptError.html#variant.InvalidState) | The search reached an inconsistent runtime state. | Empty neighborhood, attempting a move on an out-of-range index. |

## Matching on errors

```rust
use optopus::error::OptError;
use optopus::prelude::*;

match Qubo::load_file("instance.qubo") {
    Ok(prob) => { /* run heuristics */ }
    Err(OptError::FileLoad { path, line, detail }) => {
        eprintln!("Failed to parse {path} at line {line}: {detail}");
    }
    Err(OptError::Io(e)) => {
        eprintln!("I/O error: {e}");
    }
    Err(other) => {
        eprintln!("Unexpected error: {other}");
    }
}
```

**API:** [`Qubo::load_file`](../api/optopus/problem/qubo/struct.Qubo.html#method.load_file) · [`OptError::FileLoad`](../api/optopus/error/enum.OptError.html#variant.FileLoad) · [`OptError::Io`](../api/optopus/error/enum.OptError.html#variant.Io)

## In custom code

When implementing a custom problem or heuristic:

- Return [`OptError::InvalidState(...)`](../api/optopus/error/enum.OptError.html#variant.InvalidState) for runtime
  invariant violations (e.g.,
  [`MoveToNeighbor::apply_to_solution`](../api/optopus/trait_defs/trait.MoveToNeighbor.html#tymethod.apply_to_solution)
  with an invalid index).
- Return [`OptError::Parse(...)`](../api/optopus/error/enum.OptError.html#variant.Parse) or
  [`OptError::FileLoad { … }`](../api/optopus/error/enum.OptError.html#variant.FileLoad) from custom loaders.
- Let `?` propagate `std::io::Error` automatically —
  [`OptError: From<io::Error>`](../api/optopus/error/enum.OptError.html#trait-implementations) is derived.
