//! Quadratic Unconstrained Binary Optimization (QUBO) problem definition and neighborhood structures.
//!
//! QUBO minimizes the energy `E(x) = Σ Q[i][j] * x[i] * x[j]` over binary variables `x ∈ {0,1}^n`.
//!
//! # Quick start
//!
//! ```
//! use optopus::prelude::*;
//!
//! // Build a QUBO instance
//! let qubo = Qubo::from_entries([
//!     (0, 1, 1),
//!     (0, 2, 1),
//!     (1, 2, 1),
//! ]);
//!
//! // Run LocalSearch with flip moves
//! let mut state = SearchState::new(&qubo);
//! let mut ls = LocalSearch::<QuboFlipNeighbor>::new(
//!     StopCondition::iterations(10_000),
//! );
//! ls.run(&mut state).unwrap();
//! println!("best = {}", state.best_solution.objective);
//! ```
//!
//! # Building a QUBO
//!
//! ```
//! use optopus::prelude::*;
//!
//! // Option 1: from_entries (set semantics — duplicate entries are overwritten)
//! let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 2)]);
//!
//! // Option 2: incremental construction
//! let mut qubo = Qubo::new();
//! qubo.set_q(0, 1, 1);   // set (overwrite)
//! qubo.add_q(0, 1, 1);   // accumulate → 2
//!
//! // Option 3: load from a sparse Q-matrix file
//! //   (format: "N M\n i j v\n ..." with 1-indexed entries; i == j gives the diagonal coefficient)
//! // let qubo = Qubo::load_file("data/qubo/sample.txt").unwrap();
//! ```
//!
//! # Reading QUBO structure
//!
//! ```
//! use optopus::prelude::*;
//!
//! let qubo = Qubo::from_entries([(0, 1, 3), (0, 2, 1), (1, 2, 2)]);
//!
//! // Coefficient via Index
//! assert_eq!(qubo[(0, 1)], 3);
//! assert_eq!(qubo[(0, 2)], 1);
//! assert_eq!(qubo[(3, 4)], 0);  // non-existent → 0
//!
//! // Stats
//! assert_eq!(qubo.num_of_variables(), 3);
//! assert_eq!(qubo.num_entries(), 3);
//! assert_eq!(qubo.degree(0), 2);
//!
//! // Iterate over entries (each entry appears once, with i <= j)
//! for (i, j, v) in qubo.entries() {
//!     println!("{i} -- {j} [Q={v}]");
//! }
//!
//! // Iterate over neighbors of a variable
//! for &(j, v) in qubo.neighbors(0) {
//!     println!("0 -- {j} [Q={v}]");
//! }
//! ```
//!
//! # Neighborhood moves
//!
//! | Move type | Description | Iteration cost |
//! |---|---|---|
//! | [`QuboFlipNeighbor`] | Flip one variable | 1 |
//! | [`QuboSwapNeighbor`] | Swap two variables with different values | 2 |
//!
//! # Applying heuristics
//!
//! ```
//! use optopus::prelude::*;
//! use std::time::Duration;
//!
//! let qubo = Qubo::from_entries([
//!     (0, 1, 1), (0, 2, 1), (0, 3, 1),
//!     (1, 2, 1), (1, 4, 1), (2, 5, 1),
//!     (3, 4, 1), (3, 5, 1), (4, 5, 1),
//! ]);
//!
//! // LocalSearch — greedy best-improving flip
//! let mut state = SearchState::new(&qubo);
//! let mut ls = LocalSearch::<QuboFlipNeighbor>::new(
//!     StopCondition::iterations(100_000),
//! );
//! ls.run(&mut state).unwrap();
//!
//! // TabuSearch — flip with tabu tenure [3, 7]
//! let mut state = SearchState::new(&qubo);
//! let mut ts = TabuSearch::<QuboFlipNeighbor>::new(
//!     StopCondition::iterations(100_000),
//!     (3, 7),
//!     None,
//! );
//! ts.run(&mut state).unwrap();
//!
//! // SimulatedAnnealing — flip with temperature schedule
//! let mut state = SearchState::new(&qubo);
//! let mut sa = SimulatedAnnealing::<QuboFlipNeighbor>::new(
//!     StopCondition::iterations(100_000),
//!     10.0,   // initial temperature
//!     0.9999, // cooling rate
//! );
//! sa.run(&mut state).unwrap();
//!
//! // Iterated Local Search — LocalSearch + RandomWalk perturbation
//! let mut state = SearchState::new(&qubo);
//! let mut ils = Iterated::new(
//!     StopCondition::iterations(1_000_000),
//!     Box::new(LocalSearch::<QuboFlipNeighbor>::new(
//!         StopCondition::failed_updates(1000),
//!     )),
//!     Box::new(RandomWalk::<QuboFlipNeighbor>::new(
//!         StopCondition::iterations(50),
//!     )),
//! );
//! ils.run(&mut state).unwrap();
//! ```
//!
//! # Working with solutions
//!
//! ```
//! use optopus::prelude::*;
//!
//! let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 2), (0, 2, 3)]);
//! let mut state = SearchState::new(&qubo);
//! let mut ls = LocalSearch::<QuboFlipNeighbor>::new(
//!     StopCondition::iterations(10_000),
//! );
//! ls.run(&mut state).unwrap();
//!
//! let sol = &state.best_solution;
//! println!("objective = {}", sol.objective);          // energy
//! println!("x = {:?}", sol.x);                        // variable assignment
//! println!("gain[0] = {}", sol.gain[0]);              // flip gain for variable 0
//! println!("found at iteration {}", state.best_iteration);
//! ```

mod crossover;
mod neighbor;
mod problem;

pub use crossover::QuboUniformCrossover;
pub use neighbor::{QuboFlipNeighbor, QuboSwapNeighbor};
pub use problem::{Qubo, QuboSolution};
