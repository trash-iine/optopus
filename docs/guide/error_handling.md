# Error Handling

Every fallible operation in Optopus returns `Result<_, OptError>`. The error
type is defined in [`optopus::error`](../../src/error.rs).

## Variants

| Variant | When raised | Common causes |
|---|---|---|
| `Config(String)` | A user-facing configuration error. | Invalid benchmark TOML field, missing required parameter, illegal range. |
| `Io(std::io::Error)` | Wrapped via `#[from]`; bubbles up from the `std::io` layer. | File not found, permission denied, EOF mid-read. |
| `Parse(String)` | Generic format error not tied to a single line. | Malformed input that pre-loaders surface before line-by-line parsing. |
| `FileLoad { path, line, detail }` | Structured file-load error. `line == 0` indicates a file-level error not tied to one line. | TSPLIB / DIMACS / QUBO loaders hitting an unexpected token. |
| `InvalidState(String)` | The search reached an inconsistent runtime state. | Empty neighborhood, attempting a move on an out-of-range index. |

## Display formatting

`OptError` implements `Display` via `thiserror`. `FileLoad` is rendered as
`<path>: line <line>: <detail>` (or `<path>: <detail>` when `line == 0`), which
makes loader errors point at the offending file and line directly.

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

## In custom code

When implementing a custom problem or heuristic:

- Return `OptError::InvalidState(...)` for runtime invariant violations
  (e.g., `MoveToNeighbor::apply_to_solution` with an invalid index).
- Return `OptError::Parse(...)` or `OptError::FileLoad { … }` from custom
  loaders.
- Let `?` propagate `std::io::Error` automatically — `OptError: From<io::Error>`
  is derived.
