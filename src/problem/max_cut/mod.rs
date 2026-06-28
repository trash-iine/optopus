//! Maximum Cut (MaxCut) problem definition and neighborhood structures.
//!
//! Given an undirected weighted graph, MaxCut seeks a partition of the vertices into
//! two sets that maximizes the total weight of edges crossing the partition.
//!
//! # Quick start
//!
//! ```
//! use optopus::prelude::*;
//!
//! // Build a graph
//! let mc = MaxCut::from_edges([
//!     (0, 1, 1.0),
//!     (0, 2, 1.0),
//!     (1, 2, 1.0),
//! ]);
//!
//! // Run LocalSearch with flip moves
//! let mut state = SearchState::new(&mc);
//! let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::iterations(10_000),
//! );
//! ls.run(&mut state).unwrap();
//! println!("best = {}", state.best_solution.objective);
//! ```
//!
//! # Building a graph
//!
//! ```
//! use optopus::prelude::*;
//!
//! // Option 1: from_edges (set semantics — duplicate edges are overwritten)
//! let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0)]);
//!
//! // Option 2: from a Graph
//! let mut g = Graph::new();
//! g.set_weight(0, 1, 1.0);   // set (overwrite)
//! g.add_weight(0, 1, 0.5);   // accumulate → 1.5
//! let mc = MaxCut::new(g);
//!
//! // Option 3: load from file (format: "N M\n i j w\n ..." with 1-indexed vertices)
//! // let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G1").unwrap());
//! ```
//!
//! # Reading graph structure
//!
//! ```
//! use optopus::prelude::*;
//!
//! let mc = MaxCut::from_edges([(0, 1, 3.0), (0, 2, 1.0), (1, 2, 2.0)]);
//!
//! // Edge weight via Graph's Index
//! assert_eq!(mc.graph[(0, 1)], 3.0);
//! assert_eq!(mc.graph[(0, 2)], 1.0);
//! assert_eq!(mc.graph[(3, 4)], 0.0);  // non-existent → 0.0
//!
//! // Graph stats
//! assert_eq!(mc.graph.num_vertices(), 3);
//! assert_eq!(mc.graph.num_edges(), 3);
//! assert_eq!(mc.graph.degree(0), 2);
//!
//! // Iterate over edges (each edge appears once, with i < j)
//! for (i, j, w) in mc.graph.edges() {
//!     println!("{i} -- {j} [w={w}]");
//! }
//!
//! // Iterate over neighbors of a vertex
//! for &(j, w) in mc.graph.neighbors(0) {
//!     println!("0 -- {j} [w={w}]");
//! }
//! ```
//!
//! # Neighborhood moves
//!
//! | Move type | Description | Iteration cost |
//! |---|---|---|
//! | [`MaxCutFlipNeighbor`] | Flip one vertex to the opposite side | 1 |
//! | [`MaxCutSwapNeighbor`] | Swap two vertices on different sides | 2 |
//!
//! # Applying heuristics
//!
//! ```
//! use optopus::prelude::*;
//! use std::time::Duration;
//!
//! let mc = MaxCut::from_edges([
//!     (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),
//!     (1, 2, 1.0), (1, 4, 1.0), (2, 5, 1.0),
//!     (3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0),
//! ]);
//!
//! // LocalSearch — greedy best-improving flip
//! let mut state = SearchState::new(&mc);
//! let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::iterations(100_000),
//! );
//! ls.run(&mut state).unwrap();
//!
//! // TabuSearch — flip with tabu tenure [3, 7]
//! let mut state = SearchState::new(&mc);
//! let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::iterations(100_000),
//!     (3, 7),
//!     None,
//! );
//! ts.run(&mut state).unwrap();
//!
//! // SimulatedAnnealing — flip with temperature schedule
//! let mut state = SearchState::new(&mc);
//! let mut sa = SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
//!     StopCondition::iterations(100_000),
//!     10.0,   // initial temperature
//!     0.9999, // cooling rate
//! );
//! sa.run(&mut state).unwrap();
//!
//! // Iterated Local Search — LocalSearch + RandomWalk perturbation
//! let mut state = SearchState::new(&mc);
//! let mut ils = Iterated::new(
//!     StopCondition::iterations(1_000_000),
//!     Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
//!         StopCondition::failed_updates(1000),
//!     )),
//!     Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
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
//! let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]);
//! let mut state = SearchState::new(&mc);
//! let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::iterations(10_000),
//! );
//! ls.run(&mut state).unwrap();
//!
//! let sol = &state.best_solution;
//! println!("objective = {}", sol.objective);          // cut weight
//! println!("cut = {:?}", sol.cut);                    // partition assignment
//! println!("gain[0] = {}", sol.gain[0]);              // flip gain for vertex 0
//! println!("found at iteration {}", state.best_iteration);
//! ```

mod crossover;
mod neighbor;
mod problem;

pub use crossover::MaxCutUniformCrossover;
pub use neighbor::{MaxCutFlipNeighbor, MaxCutSwapNeighbor};
pub use problem::{MaxCut, MaxCutSolution};
