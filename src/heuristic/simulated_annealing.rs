use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{Evaluable, MoveToNeigbor, ProblemTrait, SearchState};
use rand::seq::IteratorRandom;
use rand::Rng;
use std::ops::{DivAssign, MulAssign};

pub struct SimulatedAnnealing<N> {
    pub stop_condition: StopCondition,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    phantom_neighbor: std::marker::PhantomData<N>,
    current_temperature: f64,
}

impl<N> SimulatedAnnealing<N> {
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
    fn clear(&mut self) {
        self.current_temperature = self.initial_temperature;
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &mut self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), OptError> {
        let neighbor = N::iter(&state.instance, &state.solution)
            .choose(&mut rand::rng())
            .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;
        if state.is_neighbor_better_than_current(&neighbor)
            || rand::rng().random::<f64>()
                < (-neighbor.evaluate() / self.current_temperature).exp()
        {
            state.apply(&neighbor)?;
        }

        self.current_temperature.mul_assign(self.cooling_rate);

        return Ok(());
    }
}

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
    fn clear(&mut self) {
        self.current_temperature = self.initial_temperature;
        self.is_going_down = true;
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &mut self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), OptError> {
        let neighbor = N::iter(&state.instance, &state.solution)
            .choose(&mut rand::rng())
            .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;

        if state.is_neighbor_better_than_current(&neighbor)
            || rand::rng().random::<f64>()
                < (-neighbor.evaluate() / self.current_temperature).exp()
        {
            state.apply(&neighbor)?;
        }

        if self.is_going_down {
            self.current_temperature.mul_assign(self.cooling_rate);
            if self.current_temperature < self.min_wave_threashold {
                tracing::info!("Wave detected, going up");
                self.is_going_down = false;
            }
        } else {
            self.current_temperature.div_assign(self.cooling_rate);
            if self.current_temperature > self.max_wave_threashold {
                tracing::info!("Wave detected, going down");
                self.is_going_down = true;
            }
        }

        return Ok(());
    }
}
