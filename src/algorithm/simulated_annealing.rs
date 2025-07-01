use super::{Heuristic, StopCondition};
use crate::search_state::{EnumerateMoveToNeighbor, Evaluable, ProblemTrait, SearchState};
use rand::seq::IteratorRandom;
use rand::Rng;
use std::cell::RefCell;
use std::ops::{DivAssign, MulAssign};

pub struct SimulatedAnnealing<MoveToNeighbor> {
    pub stop_condition: StopCondition,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    phantom_neighbor: std::marker::PhantomData<MoveToNeighbor>,
    current_temperature: RefCell<f64>,
}

impl<MoveToNeighbor> SimulatedAnnealing<MoveToNeighbor> {
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

impl<Problem, MoveToNeighbor> Heuristic<Problem> for SimulatedAnnealing<MoveToNeighbor>
where
    Problem: ProblemTrait,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
    MoveToNeighbor: Evaluable<f64>,
{
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let neighbor = state
            .iter_on_move_to_neighbor()
            .choose(&mut rand::rng())
            .ok_or("No neighbor found")?;
        if state.is_move_to_be_better_than_currernt(&neighbor)
            || rand::rng().random::<f64>()
                < (-neighbor.evaluate() / *self.current_temperature.borrow()).exp()
        {
            state.apply(&neighbor);
        }

        self.current_temperature
            .borrow_mut()
            .mul_assign(self.cooling_rate);

        return Ok(());
    }
}

pub struct BangBangSimulatedAnnealing<MoveToNeighbor> {
    pub stop_condition: StopCondition,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub min_wave_threashold: f64,
    pub max_wave_threashold: f64,
    phantom_neighbor: std::marker::PhantomData<MoveToNeighbor>,
    current_temperature: RefCell<f64>,
    is_going_down: RefCell<bool>,
}

impl<MoveToNeighbor> BangBangSimulatedAnnealing<MoveToNeighbor> {
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

impl<Problem, MoveToNeighbor> Heuristic<Problem> for BangBangSimulatedAnnealing<MoveToNeighbor>
where
    Problem: ProblemTrait,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
    MoveToNeighbor: Evaluable<f64>,
{
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let neighbor = state
            .iter_on_move_to_neighbor()
            .choose(&mut rand::rng())
            .ok_or("No neighbor found")?;

        if state.is_move_to_be_better_than_currernt(&neighbor)
            || rand::rng().random::<f64>()
                < (-neighbor.evaluate() / *self.current_temperature.borrow()).exp()
        {
            state.apply(&neighbor);
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
