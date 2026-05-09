//! Job Shop Scheduling Problem (JSSP).
//!
//! Given `n_jobs` jobs and `n_machines` machines, each job has an ordered
//! sequence of operations specifying `(machine, duration)`. Operations within
//! a job must be processed in order, and a machine can process only one
//! operation at a time. The objective is to minimize the makespan (Cmax).
//!
//! Solutions are encoded as permutations-with-repetition: a sequence of length
//! `n_jobs * n_machines` where each job index appears exactly `n_machines`
//! times. The k-th occurrence of job `j` represents operation `O(j, k)`. A
//! left-shift semi-active schedule is reconstructed by walking the sequence.

mod crossover;
mod neighbor;
mod problem;

pub use crossover::JobShopPpxCrossover;
pub use neighbor::{JobShopRelocateNeighbor, JobShopSwapNeighbor};
pub use problem::{JobShopScheduling, JobShopSolution};
