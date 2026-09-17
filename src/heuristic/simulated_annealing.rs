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
    let worsening = delta.minimized();
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
/// value represents the worsening amount (positive = worse move).
///
/// # References
///
/// - Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. "Optimization by Simulated Annealing."
///   Science, 220(4598), 671-680, 1983.
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
/// - Cooling phase: temperature is multiplied by `cooling_rate` each step.
///   When the temperature drops below `min_wave_threshold`, the phase switches to reheating.
/// - Reheating phase: temperature is divided by `cooling_rate` each step.
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

    /// The two deterministic ends of the Metropolis rule need one draw each,
    /// not a hundred: an improving move takes the short-circuit before the RNG
    /// is touched, and at 1e-12 the exponential underflows to exactly 0.0, so
    /// no draw can clear it. Only the warm end is statistical.
    #[test]
    fn boltzmann_accept_follows_the_metropolis_rule() {
        let mut rng = SmallRng::seed_from_u64(42);

        assert!(boltzmann_accept(Evaluable::Maximize(1.0), 1e-12, &mut rng));
        assert!(!boltzmann_accept(
            Evaluable::Maximize(-1.0),
            1e-12,
            &mut rng
        ));

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

    /// What makes this bang-bang rather than plain annealing is the schedule,
    /// so the schedule is what the test reads. Stepping `run_once` and
    /// recording the temperature shows both phases and both turning points;
    /// asserting on the counters instead would pass with `is_going_down`
    /// deleted and the heuristic silently reduced to a monotone cooling.
    #[test]
    fn bang_bang_oscillates_between_its_wave_thresholds() {
        let mc = triangle();
        let mut state = SearchState::new_with_seed(&mc, 42);
        // Halving per step over a band of one octave: the phase turns often
        // enough that 40 steps cover several full waves.
        let (min_wave, max_wave, cooling) = (0.1, 0.8, 0.5);
        let mut sa = BangBangSimulatedAnnealing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(40),
            1.0,
            cooling,
            min_wave,
            max_wave,
        );

        let mut temperatures = vec![sa.current_temperature];
        while !sa.is_done(&state) {
            sa.run_once(&mut state).unwrap();
            temperatures.push(sa.current_temperature);
        }

        let steps: Vec<f64> = temperatures.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(steps.iter().any(|&d| d < 0.0), "never cooled");
        assert!(steps.iter().any(|&d| d > 0.0), "never reheated");

        // A turn happens one step past the threshold, so the band the walk
        // stays inside is the thresholds widened by one step each way.
        let coldest = temperatures.iter().cloned().fold(f64::MAX, f64::min);
        let hottest = temperatures.iter().cloned().fold(f64::MIN, f64::max);
        assert!(coldest < min_wave, "never reached the cold turning point");
        assert!(coldest >= min_wave * cooling, "overshot below the band");
        assert!(hottest > max_wave, "never reached the hot turning point");
        assert!(hottest <= 1.0f64.max(max_wave / cooling), "overshot above");

        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }
}
