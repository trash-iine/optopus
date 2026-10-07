//! Quadratic Assignment Problem (QAP), Koopmans and Beckmann form.
//!
//! `n` facilities go to `n` locations, one each, and the cost of an
//! assignment `p` is `Σ_i Σ_j a[i][j] · b[p(i)][p(j)]`, minimized. The one move
//! is the exchange of two facilities, priced in O(n), and with tabu search over
//! it this is Taillard's robust tabu search.

mod crossover;
mod neighbor;
mod problem;

pub use crossover::QapOrderCrossover;
pub use neighbor::QapSwapNeighbor;
pub use problem::{Qap, QapSolution};
