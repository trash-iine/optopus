use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{Evaluable, MoveToNeigbor, ProblemTrait, SearchState};
use rand::Rng;
use rand::seq::IteratorRandom;
use std::ops::{DivAssign, MulAssign};

/// Simulated annealing heuristic.
///
/// At each iteration a random neighbor is selected.
/// The move is accepted if it improves the current solution, or with probability
/// `exp(−evaluate(neighbor) / T)` otherwise, where `T` is the current temperature.
/// The temperature is multiplied by `cooling_rate` after each iteration.
///
/// Requires the neighbor type to implement [`Evaluable<f64>`], where the evaluation
/// value represents the **worsening** amount (positive = worse move).
pub struct SimulatedAnnealing<N> {
    pub stop_condition: StopCondition,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    phantom_neighbor: std::marker::PhantomData<N>,
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
            phantom_neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for SimulatedAnnealing<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Evaluable<f64>,
{
    /// Reset the temperature to the initial value.
    fn clear(&mut self) {
        self.current_temperature = self.initial_temperature;
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor = N::iter(&state.instance, &state.solution)
            .choose(&mut rand::rng())
            .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;
        if state.is_neighbor_better_than_current(&neighbor)
            || rand::rng().random::<f64>() < (-neighbor.evaluate() / self.current_temperature).exp()
        {
            state.apply(&neighbor)?;
        }

        self.current_temperature *= self.cooling_rate;

        return Ok(());
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
    pub min_wave_threashold: f64,
    pub max_wave_threashold: f64,
    phantom_neighbor: std::marker::PhantomData<N>,
    current_temperature: f64,
    is_going_down: bool,
}

impl<N> BangBangSimulatedAnnealing<N> {
    pub fn new(
        stop_condition: StopCondition,
        initial_temperature: f64,
        cooling_rate: f64,
        min_wave_threashold: f64,
        max_wave_threashold: f64,
    ) -> Self {
        Self {
            stop_condition,
            initial_temperature,
            cooling_rate,
            min_wave_threashold,
            max_wave_threashold,
            current_temperature: initial_temperature,
            is_going_down: true,
            phantom_neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for BangBangSimulatedAnnealing<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Evaluable<f64>,
{
    /// Reset the temperature and phase to the initial state.
    fn clear(&mut self) {
        self.current_temperature = self.initial_temperature;
        self.is_going_down = true;
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor = N::iter(&state.instance, &state.solution)
            .choose(&mut rand::rng())
            .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;

        if state.is_neighbor_better_than_current(&neighbor)
            || rand::rng().random::<f64>() < (-neighbor.evaluate() / self.current_temperature).exp()
        {
            state.apply(&neighbor)?;
        }

        if self.is_going_down {
            self.current_temperature.mul_assign(self.cooling_rate);
            if self.current_temperature < self.min_wave_threashold {
                tracing::debug!("Wave detected, going up");
                self.is_going_down = false;
            }
        } else {
            self.current_temperature.div_assign(self.cooling_rate);
            if self.current_temperature > self.max_wave_threashold {
                tracing::debug!("Wave detected, going down");
                self.is_going_down = true;
            }
        }

        return Ok(());
    }
}
