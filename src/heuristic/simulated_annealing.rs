use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluable, Evaluate, MoveToNeighbor, ProblemTrait};
use rand::Rng;

/// Returns `true` with Boltzmann probability `exp(-worsening / temperature)`.
///
/// Accepts an [`Evaluable<f64>`] value that encodes both the optimization direction and
/// the objective change. Improving moves are always accepted; worsening moves are
/// accepted with probability `exp(-worsening / T)`.
pub fn boltzmann_accept(
    delta: Evaluable<f64>,
    temperature: f64,
    rng: &mut rand::rngs::SmallRng,
) -> bool {
    let worsening = delta.worsening_amount();
    worsening < 0.0 || rng.random::<f64>() < (-worsening / temperature).exp()
}

/// Simulated annealing heuristic.
///
/// At each iteration a random neighbor is selected.
/// The move is accepted if it improves the current solution, or with probability
/// `exp(−evaluate(neighbor) / T)` otherwise, where `T` is the current temperature.
/// The temperature is multiplied by `cooling_rate` after each iteration.
///
/// Requires the neighbor type to implement [`Evaluable<f64>`], where the evaluation
/// value represents the **worsening** amount (positive = worse move).
///
/// # References
///
/// - Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. "Optimization by Simulated Annealing."
///   *Science*, 220(4598), 671-680, 1983.
///   [DOI](https://doi.org/10.1126/science.220.4598.671)
/// - Cerny, V. "Thermodynamical Approach to the Traveling Salesman Problem: An Efficient
///   Simulation Algorithm." *Journal of Optimization Theory and Applications*, 45(1), 41-51, 1985.
///   [DOI](https://doi.org/10.1007/BF00940812)
pub struct SimulatedAnnealing<N> {
    pub stop_condition: StopCondition,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    _neighbor: std::marker::PhantomData<N>,
    current_temperature: f64,
}

impl<N> SimulatedAnnealing<N> {
    /// Create a new [`SimulatedAnnealing`] heuristic with the given stopping condition, initial temperature, and cooling rate.
    pub fn new(stop_condition: StopCondition, initial_temperature: f64, cooling_rate: f64) -> Self {
        Self {
            stop_condition,
            initial_temperature,
            cooling_rate,
            current_temperature: initial_temperature,
            _neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for SimulatedAnnealing<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate,
{
    /// Reset the temperature to the initial value.
    fn clear(&mut self) {
        self.current_temperature = self.initial_temperature;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor: N = state.random_neighbor("SimulatedAnnealing")?;
        if boltzmann_accept(
            neighbor.evaluate(),
            self.current_temperature,
            &mut state.rng,
        ) {
            state.apply(&neighbor)?;
        } else {
            state.progress_iteration();
        }

        self.current_temperature *= self.cooling_rate;

        Ok(())
    }
}

/// Simulated annealing with a bang-bang (oscillating) temperature schedule.
///
/// The temperature alternates between cooling and reheating phases:
/// - **Cooling phase**: temperature is multiplied by `cooling_rate` each step.
///   When the temperature drops below `min_wave_threshold`, the phase switches to reheating.
/// - **Reheating phase**: temperature is divided by `cooling_rate` each step.
///   When the temperature exceeds `max_wave_threshold`, the phase switches back to cooling.
///
/// This creates a sawtooth temperature profile that helps escape local optima.
pub struct BangBangSimulatedAnnealing<N> {
    pub stop_condition: StopCondition,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub min_wave_threshold: f64,
    pub max_wave_threshold: f64,
    _neighbor: std::marker::PhantomData<N>,
    current_temperature: f64,
    is_going_down: bool,
}

impl<N> BangBangSimulatedAnnealing<N> {
    pub fn new(
        stop_condition: StopCondition,
        initial_temperature: f64,
        cooling_rate: f64,
        min_wave_threshold: f64,
        max_wave_threshold: f64,
    ) -> Self {
        Self {
            stop_condition,
            initial_temperature,
            cooling_rate,
            min_wave_threshold,
            max_wave_threshold,
            current_temperature: initial_temperature,
            is_going_down: true,
            _neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for BangBangSimulatedAnnealing<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate,
{
    /// Reset the temperature and phase to the initial state.
    fn clear(&mut self) {
        self.current_temperature = self.initial_temperature;
        self.is_going_down = true;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor: N = state.random_neighbor("SimulatedAnnealing (bang-bang)")?;

        if boltzmann_accept(
            neighbor.evaluate(),
            self.current_temperature,
            &mut state.rng,
        ) {
            state.apply(&neighbor)?;
        } else {
            state.progress_iteration();
        }

        if self.is_going_down {
            self.current_temperature *= self.cooling_rate;
            if self.current_temperature < self.min_wave_threshold {
                tracing::debug!("Wave detected, going up");
                self.is_going_down = false;
            }
        } else {
            self.current_temperature /= self.cooling_rate;
            if self.current_temperature > self.max_wave_threshold {
                tracing::debug!("Wave detected, going down");
                self.is_going_down = true;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::MaxCutFlipNeighbor;
    use crate::problem::max_cut::MaxCut;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn triangle() -> MaxCut {
        MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    }

    #[test]
    fn boltzmann_accept_always_accepts_improving_moves() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            assert!(boltzmann_accept(Evaluable::Maximize(1.0), 1e-12, &mut rng));
        }
    }

    #[test]
    fn boltzmann_accept_rejects_worsening_at_low_temperature() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            assert!(!boltzmann_accept(
                Evaluable::Maximize(-1.0),
                1e-12,
                &mut rng
            ));
        }
    }

    #[test]
    fn boltzmann_accept_mostly_accepts_worsening_at_high_temperature() {
        let mut rng = SmallRng::seed_from_u64(42);
        let accepted = (0..1000)
            .filter(|_| boltzmann_accept(Evaluable::Maximize(-1.0), 1e9, &mut rng))
            .count();
        assert!(accepted > 900, "accepted only {accepted} of 1000");
    }

    #[test]
    fn simulated_annealing_keeps_counter_invariant() {
        let mc = triangle();
        let mut state = SearchState::new(&mc);
        let mut sa = SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(100),
            1.0,
            0.95,
        );
        sa.run(&mut state).unwrap();
        assert_eq!(state.iteration, 100);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    #[test]
    fn bang_bang_keeps_counter_invariant_across_wave_switches() {
        let mc = triangle();
        let mut state = SearchState::new(&mc);
        // Fast cooling with a narrow wave band forces frequent phase switches.
        let mut sa = BangBangSimulatedAnnealing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(200),
            1.0,
            0.5,
            0.1,
            2.0,
        );
        sa.run(&mut state).unwrap();
        assert_eq!(state.iteration, 200);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }
}
