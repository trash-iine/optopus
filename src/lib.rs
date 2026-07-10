//! optopus — combinatorial optimization library providing heuristic algorithms
//! for problems such as MaxCut, QUBO, SAT, TSP, and formula-based problems.
//!
//! # Quick start
//!
//! ```
//! use optopus::prelude::*;
//!
//! let mc = MaxCut::new(Graph::from_edges([
//!     (0, 1, 1.0),
//!     (0, 2, 1.0),
//!     (1, 2, 1.0),
//! ]));
//!
//! let mut state = SearchState::new(&mc);
//! let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::iterations(1_000_000),
//! );
//! ls.run(&mut state)?;
//!
//! println!("best cut = {}", state.best_solution.objective);
//! # Ok::<(), optopus::error::OptError>(())
//! ```
//!
//! # Overview
//!
//! - [`heuristic`] — heuristic algorithms (local search, simulated annealing, tabu search, beam search, etc.)
//! - [`problem`] — problem definitions and neighborhood structures (MaxCut, QUBO, SAT, TSP, Formula)
//! - [`search_state`] — search state management ([`SearchState`](search_state::SearchState))
//! - [`trait_defs`] — core library traits shared across the Problem, Heuristic, and SearchState layers
//! - [`common`] — shared data structures and helpers ([`Graph`](common::Graph), binary-solution utilities)
//! - [`benchmark`] — utilities for running and recording benchmark experiments
//! - [`prelude`] — convenience re-exports of commonly used types and traits
//! - [`error`] — unified error type

pub mod benchmark;
pub mod common;
pub mod error;
pub mod heuristic;
pub mod prelude;
pub mod problem;
pub mod search_state;
pub mod trait_defs;
