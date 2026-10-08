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
//! - [`building_blocks`] holds the pieces problems and heuristics are built
//!   from, in three groups, instance data such as
//!   [`Graph`](building_blocks::instance::Graph), helpers per solution representation,
//!   and problem-agnostic search machinery such as
//!   [`TabuMemory`](building_blocks::search::TabuMemory).
//! - [`benchmark`] runs and records benchmark experiments.
//! - [`prelude`] re-exports the commonly used types and traits.
//! - [`error`] defines the unified error type.

pub mod benchmark;
pub mod building_blocks;
pub mod error;
pub mod heuristic;
pub mod prelude;
pub mod problem;
pub mod search_state;
pub mod trait_defs;

/// The `rand` this crate is built against.
///
/// [`ProblemTrait::new_solution`](trait_defs::ProblemTrait::new_solution) takes
/// an `impl rand::Rng` and
/// [`EnabledTabu::add_to_tabu_map`](trait_defs::EnabledTabu::add_to_tabu_map) a
/// `rand::rngs::SmallRng`, so a problem defined outside the crate names `rand`
/// types. `use optopus::rand;` brings the same version into scope, with no
/// dependency of its own that could drift to an incompatible one.
pub use rand;
