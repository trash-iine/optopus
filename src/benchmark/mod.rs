//! Benchmark runner for comparing heuristics across problem instances.
//!
//! [`Benchmark::run_from_config`] runs every `(instance, heuristic)` pair from
//! a [`BenchmarkConfig`] (parsed from TOML) and returns a [`BenchmarkReport`]
//! that can be serialized back to TOML (see [`BenchmarkReport::write_to_dir`]).
//!
//! # Module layout
//!
//! - `config`, the TOML-facing types ([`BenchmarkConfig`], [`HeuristicConfig`], ...)
//!   and config validation
//! - `factory`, the generic heuristic factory (`build_heuristic` + visitors)
//! - `problems`, per-problem registration: to add a problem to the benchmark,
//!   add one line to its problem table and one impl block per trait there
//! - `runner`, the parallel run loop and per-run metrics
//! - `report`, the output report types and summary statistics
mod config;
mod factory;
mod problems;
mod report;
mod runner;

pub use config::{
    BenchmarkConfig, HeuristicConfig, InstanceConfig, NeighborKind, StopConditionConfig,
};
pub use problems::{BenchmarkProblem, BenchmarkSolution, ProblemKind};
pub use report::{BenchmarkReport, InstanceHeuristicResult, SingleRunResult, Summary};
pub use runner::Benchmark;
