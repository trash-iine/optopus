use super::problem::{GraphColoring, GraphColoringSolution};
use crate::search_state::Crossover;
use rand::Rng;

/// Uniform crossover for Graph Coloring: each vertex's color is taken from one
/// parent or the other, chosen independently at random.
pub struct GraphColoringUniformCrossover;

impl Crossover<GraphColoring> for GraphColoringUniformCrossover {
    fn crossover(
        &mut self,
        prob: &GraphColoring,
        sol1: &GraphColoringSolution,
        sol2: &GraphColoringSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<GraphColoringSolution, crate::error::OptError> {
        let n = prob.graph.len();
        let colors = (0..n)
            .map(|v| {
                if rng.random_bool(0.5) {
                    sol1.colors[v]
                } else {
                    sol2.colors[v]
                }
            })
            .collect();
        Ok(prob.solution_from_colors(colors))
    }
}
