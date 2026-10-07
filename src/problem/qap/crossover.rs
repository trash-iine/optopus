use crate::search_state::Crossover;

use super::problem::{Qap, QapSolution};

/// Order Crossover (OX) over the assignment.
///
/// Copies the locations of a random contiguous run of facilities from `sol1`
/// and gives the other facilities the remaining locations in the order `sol2`
/// uses them.
pub struct QapOrderCrossover;

impl Crossover<Qap> for QapOrderCrossover {
    fn crossover(
        &mut self,
        prob: &Qap,
        sol1: &QapSolution,
        sol2: &QapSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<QapSolution, crate::error::OptError> {
        let child = crate::building_blocks::representation::order_crossover(
            &sol1.assignment,
            &sol2.assignment,
            rng,
        );
        Ok(prob.solution_from_assignment(child))
    }
}
