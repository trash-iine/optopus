use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{Evaluable, MoveToNeigbor, ProblemTrait, SearchState};
use rand::seq::IteratorRandom;
use rand::Rng;
use std::cell::RefCell;
use std::ops::{DivAssign, MulAssign};

pub struct SimulatedAnnealing<N> {
    pub stop_condition: StopCondition,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    phantom_neighbor: std::marker::PhantomData<N>,
    current_temperature: RefCell<f64>,
}

impl<N> SimulatedAnnealing<N> {
    pub fn new(stop_condition: StopCondition, initial_temperature: f64, cooling_rate: f64) -> Self {
        Self {
            stop_condition,
            initial_temperature,
            cooling_rate,
            current_temperature: RefCell::new(initial_temperature),
            phantom_neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for SimulatedAnnealing<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Evaluable<f64>,
{
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), OptError> {
        let neighbor = N::iter(&state.instance, &state.solution)
            .choose(&mut rand::rng())
            .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;
        if state.is_neighbor_better_than_current(&neighbor)
            || rand::rng().random::<f64>()
                < (-neighbor.evaluate() / *self.current_temperature.borrow()).exp()
        {
            state.apply(&neighbor)?;
        }

        self.current_temperature
            .borrow_mut()
            .mul_assign(self.cooling_rate);

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
    current_temperature: RefCell<f64>,
    is_going_down: RefCell<bool>,
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
            current_temperature: RefCell::new(initial_temperature),
            is_going_down: RefCell::new(true),
            phantom_neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for BangBangSimulatedAnnealing<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Evaluable<f64>,
{
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), OptError> {
        let neighbor = N::iter(&state.instance, &state.solution)
            .choose(&mut rand::rng())
            .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;

        if state.is_neighbor_better_than_current(&neighbor)
            || rand::rng().random::<f64>()
                < (-neighbor.evaluate() / *self.current_temperature.borrow()).exp()
        {
            state.apply(&neighbor)?;
        }

        if *self.is_going_down.borrow() {
            self.current_temperature
                .borrow_mut()
                .mul_assign(self.cooling_rate);
            if *self.current_temperature.borrow() < self.min_wave_threashold {
                tracing::info!("Wave detected, going up");
                *self.is_going_down.borrow_mut() = false;
            }
        } else {
            self.current_temperature
                .borrow_mut()
                .div_assign(self.cooling_rate);
            if *self.current_temperature.borrow() > self.max_wave_threashold {
                tracing::info!("Wave detected, going down");
                *self.is_going_down.borrow_mut() = true;
            }
        }

        return Ok(());
    }
}
