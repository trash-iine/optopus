//! Iterated Greedy over any [`Ruinable`] problem.

use crate::building_blocks::search::{greedy_insertion, random_removal};
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition, boltzmann_accept};
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluable, Evaluate, LocalRepair, Ruinable};

/// Iterated Greedy, the destruction and construction loop of Ruiz and Stützle.
///
/// Each iteration takes `removal_count` elements of the incumbent out at
/// random, puts each back at its cheapest place, runs the local search if one
/// was given, and accepts the result if it is no worse, or otherwise with
/// probability `exp(-Δ / temperature)`. The temperature is constant.
///
/// It is ruin-and-recreate with one fixed destroy operator, one fixed repair
/// operator and a fixed number of removals. That fixedness is the difference
/// from [`AdaptiveLargeNeighborhoodSearch`](crate::heuristic::AdaptiveLargeNeighborhoodSearch),
/// which chooses among several operators by a roulette, removes a fraction of
/// the elements and cools. On the permutation flow shop, where it was
/// proposed, the simple loop with a full insertion descent is the stronger of
/// the two.
///
/// # What the problem supplies
///
/// [`Ruinable`] and an [`Evaluate`] solution. The local search is a
/// [`LocalRepair`], optional in the type and part of the algorithm as
/// published. The temperature is in the units of the objective, so it is a
/// function of the instance, read once per run from the instance being
/// searched. The flow shop's is wired by
/// [`iterated_greedy_for_flow_shop`](crate::heuristic::iterated_greedy_for_flow_shop).
///
/// # References
///
/// - Ruiz, R. and Stützle, T. "A simple and effective iterated greedy
///   algorithm for the permutation flowshop scheduling problem." *European
///   Journal of Operational Research*, 177(3), 2033-2049, 2007.
///   [DOI](https://doi.org/10.1016/j.ejor.2005.12.009)
pub struct IteratedGreedy<P: Ruinable> {
    stop_condition: StopCondition,
    removal_count: usize,
    temperature_of: Box<dyn Fn(&P) -> f64>,
    /// The temperature for the instance of this run, resolved on its first
    /// iteration and dropped by [`clear`](Heuristic::clear).
    temperature: Option<f64>,
    local: Option<Box<dyn LocalRepair<P>>>,
    scratch: Vec<P::Element>,
}

impl<P: Ruinable> IteratedGreedy<P> {
    /// `temperature_of` gives the temperature for an instance, `|_| 1.0` for
    /// a fixed one.
    ///
    /// # Panics
    ///
    /// Panics if `removal_count` is zero, and on the first iteration of a run
    /// if `temperature_of` gives a temperature that is not positive.
    pub fn new(
        stop_condition: StopCondition,
        removal_count: usize,
        temperature_of: impl Fn(&P) -> f64 + 'static,
    ) -> Self {
        assert!(removal_count >= 1, "removal_count must be at least 1");
        Self {
            stop_condition,
            removal_count,
            temperature_of: Box::new(temperature_of),
            temperature: None,
            local: None,
            scratch: Vec::new(),
        }
    }

    /// Builder-style: the local search run after every reconstruction.
    pub fn with_local_repair(mut self, local: Box<dyn LocalRepair<P>>) -> Self {
        self.local = Some(local);
        self
    }
}

impl<P> Heuristic<P> for IteratedGreedy<P>
where
    P: Ruinable,
    P::Solution: Evaluate,
{
    fn clear(&mut self) {
        self.temperature = None;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let prob: &P = state.instance;
        let mut partial = prob.to_partial(&state.solution);
        let n = prob.num_elements(&partial);
        if n == 0 {
            state.progress_iteration();
            return Ok(());
        }

        let removed = random_removal(
            prob,
            &mut partial,
            self.removal_count.min(n),
            &mut state.rng,
            &mut self.scratch,
        );
        // The repair consumes the removal list, so the anchors are a copy,
        // made only when a local search will read them.
        let anchors = self.local.is_some().then(|| removed.clone());
        greedy_insertion(prob, &mut partial, removed, &mut state.rng);
        if let (Some(local), Some(anchors)) = (self.local.as_mut(), anchors.as_ref()) {
            local.repair_around(prob, &mut partial, anchors, &mut state.rng);
        }

        let temperature = *self.temperature.get_or_insert_with(|| {
            let t = (self.temperature_of)(prob);
            assert!(t > 0.0, "temperature must be positive, got {t}");
            t
        });
        let current = state.solution.evaluate().minimized();
        let candidate = prob.partial_energy(&partial);
        let worsening = Evaluable::Minimize(candidate - current);
        if candidate <= current || boltzmann_accept(worsening, temperature, &mut state.rng) {
            state.solution = prob.finish(&partial);
            state.iteration += 1;
            state.n_accepted += 1;
        } else {
            state.progress_iteration();
        }
        state.update_best();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::iterated_greedy_for_flow_shop;
    use crate::problem::FlowShop;
    use crate::problem::Tsp;
    use crate::problem::tsp::AnchoredTourDescent;

    fn ta001() -> FlowShop {
        FlowShop::load_file("data/instances/flow_shop/taillard/ta001.txt").unwrap()
    }

    #[test]
    fn improves_on_neh_and_keeps_a_valid_sequence() {
        let fs = ta001();
        let mut state = SearchState::new_with_seed(&fs, 3);
        iterated_greedy_for_flow_shop(StopCondition::iterations(300), 4, 0.4)
            .run(&mut state)
            .unwrap();
        let best = &state.best_solution;
        assert_eq!(best.objective, fs.makespan(&best.sequence));
        assert!(best.objective <= fs.neh().objective);
    }

    #[test]
    fn is_reproducible_under_seed() {
        let fs = ta001();
        let run = || {
            let mut state = SearchState::new_with_seed(&fs, 11);
            iterated_greedy_for_flow_shop(StopCondition::iterations(100), 4, 0.4)
                .run(&mut state)
                .unwrap();
            (state.best_solution.sequence.clone(), state.best_iteration)
        };
        assert_eq!(run(), run());
    }

    /// Nothing in the loop is flow shop specific. On a tour, with the
    /// anchored descent as its local search, it is a working ruin-and-recreate.
    #[test]
    fn runs_on_a_tour() {
        let coords: Vec<(f64, f64)> = (0..30)
            .map(|i| {
                let t = std::f64::consts::TAU * f64::from(i) / 30.0;
                (t.cos() * 10.0 + f64::from(i % 3), t.sin() * 10.0)
            })
            .collect();
        let tsp = Tsp::new("ring".to_string(), coords);
        let mut state = SearchState::new_with_seed(&tsp, 5);
        let start = state.solution.objective;
        IteratedGreedy::<Tsp>::new(StopCondition::iterations(500), 3, |_| 1.0)
            .with_local_repair(Box::new(AnchoredTourDescent::new()))
            .run(&mut state)
            .unwrap();
        assert!(state.best_solution.objective < start);
        assert_eq!(
            tsp.calculate_tour_length(&state.best_solution.tour)
                .unwrap(),
            state.best_solution.objective
        );
    }
}
