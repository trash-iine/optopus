use crate::search_state::Crossover;

use super::problem::{FlowShop, FlowShopSolution};

/// Order Crossover (OX) for the flow shop.
///
/// Copies a random contiguous stretch of `sol1`'s sequence and fills the
/// remaining positions with the other jobs in the order `sol2` runs them.
pub struct FlowShopOrderCrossover;

impl Crossover<FlowShop> for FlowShopOrderCrossover {
    fn crossover(
        &mut self,
        prob: &FlowShop,
        sol1: &FlowShopSolution,
        sol2: &FlowShopSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<FlowShopSolution, crate::error::OptError> {
        let child = crate::building_blocks::representation::order_crossover(
            &sol1.sequence,
            &sol2.sequence,
            rng,
        );
        Ok(prob.solution_from_sequence(child))
    }
}
