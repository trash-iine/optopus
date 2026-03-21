//! optopus — combinatorial optimization library providing heuristic algorithms
//! for problems such as MaxCut, QUBO, SAT, TSP, and formula-based problems.
//!
//! # Overview
//!
//! - [`heuristic`] — heuristic algorithms (local search, simulated annealing, tabu search, beam search, etc.)
//! - [`problem`] — problem definitions and neighborhood structures (MaxCut, QUBO, SAT, TSP, Formula)
//! - [`search_state`] — search state management and core traits
//! - [`benchmark`] — utilities for running and recording benchmark experiments
//! - [`prelude`] — convenience re-exports of commonly used types and traits
//! - [`error`] — unified error type

pub mod benchmark;
pub mod error;
pub mod heuristic;
pub mod prelude;
pub mod problem;
pub mod search_state;
