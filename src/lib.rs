//! optopus is a combinatorial optimization library providing heuristic
//! algorithms for problems such as MaxCut, QUBO, SAT, TSP, and formula-based
//! problems.
//!
//! Guides, per-problem and per-heuristic prose, and the benchmark viewer live on
//! the documentation site at <https://trash-iine.github.io/optopus/>.
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
//! - [`heuristic`] holds the search algorithms (local search, simulated
//!   annealing, tabu search, beam search and others).
//! - [`problem`] holds the problem definitions and their neighborhood
//!   structures (MaxCut, QUBO, SAT, TSP, Formula).
//! - [`search_state`] manages the state of a search, through
//!   [`SearchState`](search_state::SearchState).
//! - [`trait_defs`] holds the core traits shared across the problem, heuristic
//!   and search-state layers.
//! - [`common`] holds shared data structures and helpers such as
//!   [`Graph`](common::Graph) and the binary-solution utilities.
//! - [`benchmark`] runs and records benchmark experiments.
//! - [`prelude`] re-exports the commonly used types and traits.
//! - [`error`] defines the unified error type.

pub mod benchmark;
pub mod common;
pub mod error;
pub mod heuristic;
pub mod prelude;
pub mod problem;
pub mod search_state;
pub mod trait_defs;
