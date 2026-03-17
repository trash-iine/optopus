use std::collections::HashMap;

use rand::seq::IteratorRandom;

use super::super::{Heuristic, StopCondition, TabuSearch};
use crate::error::OptError;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::problem::{MaxCut, MaxCutSwapNeighbor};
use crate::search_state::{
    EnabledTabu, MoveToNeigbor, ProblemTrait, Rankable, SearchState, filter_best,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PerturbationType {
    Strong,
    WeakFlip,
    WeakSwap,
}

fn is_same_solution(
    prev_solution: &<MaxCut as ProblemTrait>::Solution,
    current_solution: &<MaxCut as ProblemTrait>::Solution,
) -> bool {
    if prev_solution.cut.len() != current_solution.cut.len() {
        return false;
    }

    prev_solution.cut.iter().all(|(i, &value)| {
        current_solution
            .cut
            .get(i)
            .map_or(false, |&current_value| value == current_value)
    })
}

pub struct BreakoutLocalSearch {
    tabu_tenure: (u64, u64),
    stop_condition: StopCondition,
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
    omega: u64,
    l: u64,
    prev_best_objective: Option<f32>,
    prev_solution: Option<<MaxCut as ProblemTrait>::Solution>,
    tabu_map: HashMap<usize, u64>,
}

impl BreakoutLocalSearch {
    pub fn new(
        tabu_tenure: (u64, u64),
        stop_condition: StopCondition,
        t: u64,
        l0: u64,
        p0: f64,
        q: f64,
    ) -> Self {
        Self {
            tabu_tenure,
            stop_condition,
            t,
            l0,
            p0,
            q,
            omega: 0,
            l: l0,
            prev_best_objective: None,
            prev_solution: None,
            tabu_map: HashMap::new(),
        }
    }

    fn local_search_with_updating_tabu(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        loop {
            let mut best_move_option: Option<MaxCutFlipNeighbor> = None;
            for neighbor in MaxCutFlipNeighbor::iter(&state.instance, &state.solution) {
                if !state.is_neighbor_better_than_current(&neighbor) {
                    continue;
                }

                if let Some(best_move) = best_move_option
                    && best_move.is_better_than(&neighbor)
                {
                    best_move_option = Some(best_move);
                } else {
                    best_move_option = Some(neighbor);
                }
            }

            if let Some(best_move) = best_move_option {
                best_move.add_to_tabu_map(
                    &mut self.tabu_map,
                    state.iteration,
                    self.tabu_tenure,
                );
                state.apply(&best_move)?;
            } else {
                return Ok(());
            }
        }
    }

    fn update_omega(&mut self, state: &SearchState<'_, MaxCut>) {
        if let Some(prev_best_objective) = self.prev_best_objective
            && prev_best_objective >= state.solution.objective
        {
            self.omega += 1;
        } else {
            self.omega = 0;
        }

        if self.omega > self.t {
            self.omega = 0;
        }

        self.prev_best_objective = Some(state.best_solution.objective);
    }

    fn update_l(&mut self, state: &SearchState<'_, MaxCut>) {
        if let Some(ref prev_solution) = self.prev_solution
            && is_same_solution(prev_solution, &state.solution)
        {
            self.l += 1;
        } else {
            self.l = self.l0;
        }

        self.prev_solution = Some(state.solution.clone());
    }

    fn get_perturbation_type(&self) -> PerturbationType {
        if self.omega == 0 {
            PerturbationType::Strong
        } else {
            let p = (std::f64::consts::E.powf(-(self.omega as f64 / self.t as f64))).max(self.p0);

            let prob: f64 = rand::random_range(0.0..=1.0);
            if prob <= p * self.q {
                PerturbationType::WeakFlip
            } else if prob <= p {
                PerturbationType::WeakSwap
            } else {
                PerturbationType::Strong
            }
        }
    }

    fn apply_strong_perturbation(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        for _ in 0..self.l {
            let neighbor = MaxCutFlipNeighbor::iter(&state.instance, &state.solution)
                .choose(&mut rand::rng())
                .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;

            neighbor.add_to_tabu_map(
                &mut self.tabu_map,
                state.iteration,
                self.tabu_tenure,
            );
            state.apply(&neighbor)?;
        }
        Ok(())
    }

    fn apply_weak_flip_perturbation(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        let sc = StopCondition::new(Some(self.l + state.iteration), None, None);
        let tabu_map = std::mem::take(&mut self.tabu_map);
        let mut perturb =
            TabuSearch::<MaxCutFlipNeighbor>::new(sc, self.tabu_tenure, Some(tabu_map));
        perturb.run(state)?;
        self.tabu_map = perturb.take_tabu_map();
        Ok(())
    }

    fn apply_weak_swap_perturbation(
        &mut self,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        for _ in 0..self.l {
            let mut v0_best = Vec::new();
            let mut v1_best = Vec::new();
            let mut v0_tabu = Vec::new();
            let mut v1_tabu = Vec::new();
            for neighbor in MaxCutFlipNeighbor::iter(&state.instance, &state.solution) {
                if state.solution.cut[&neighbor.i] {
                    if let Some(sample) = v0_best.first() {
                        if neighbor.is_better_than(sample) {
                            v0_best = vec![neighbor];
                        } else if !sample.is_better_than(&neighbor) {
                            v0_best.push(neighbor);
                        }
                    } else {
                        v0_best.push(neighbor);
                    }
                } else {
                    if let Some(sample) = v1_best.first() {
                        if neighbor.is_better_than(sample) {
                            v1_best = vec![neighbor];
                        } else if !sample.is_better_than(&neighbor) {
                            v1_best.push(neighbor);
                        }
                    } else {
                        v1_best.push(neighbor);
                    }
                }

                if !neighbor.is_move_enabled(&self.tabu_map, state.iteration) {
                    continue;
                }

                if state.solution.cut[&neighbor.i] {
                    if let Some(sample) = v0_tabu.first() {
                        if neighbor.is_better_than(sample) {
                            v0_tabu = vec![neighbor];
                        } else if !sample.is_better_than(&neighbor) {
                            v0_tabu.push(neighbor);
                        }
                    } else {
                        v0_tabu.push(neighbor);
                    }
                } else {
                    if let Some(sample) = v1_tabu.first() {
                        if neighbor.is_better_than(sample) {
                            v1_tabu = vec![neighbor];
                        } else if !sample.is_better_than(&neighbor) {
                            v1_tabu.push(neighbor);
                        }
                    } else {
                        v1_tabu.push(neighbor);
                    }
                }
            }

            let mut best_list = filter_best(v0_best.iter().flat_map(|v0| {
                v1_best.iter().map(|v1| MaxCutSwapNeighbor {
                    i: v0.i,
                    j: v1.i,
                    gain: state.solution.gain[&v0.i]
                        + state.solution.gain[&v1.i]
                        + if state.instance.has_edge(v0.i, v1.i) {
                            2.0 * state.instance.get_weight(v0.i, v1.i)
                        } else {
                            0.0
                        },
                })
            }));
            if let Some(best_move) = best_list.pop()
                && best_move.move_to_be_better_than(
                    &state.instance,
                    &state.solution,
                    &state.best_solution,
                )
            {
                best_move.add_to_tabu_map(
                    &mut self.tabu_map,
                    state.iteration,
                    self.tabu_tenure,
                );
                state.apply(&best_move)?;
            } else {
                let i = v0_tabu
                    .iter()
                    .min_by(|a, b| {
                        self.tabu_map
                            .get(&a.i)
                            .unwrap_or(&0)
                            .cmp(&self.tabu_map.get(&b.i).unwrap_or(&0))
                    })
                    .ok_or_else(|| OptError::InvalidState("No tabu v0".to_string()))?
                    .i;
                let j = v1_tabu
                    .iter()
                    .min_by(|a, b| {
                        self.tabu_map
                            .get(&a.i)
                            .unwrap_or(&0)
                            .cmp(&self.tabu_map.get(&b.i).unwrap_or(&0))
                    })
                    .ok_or_else(|| OptError::InvalidState("No tabu v1".to_string()))?
                    .i;
                let neighbor = MaxCutSwapNeighbor {
                    i: i,
                    j: j,
                    gain: state.solution.gain[&i]
                        + state.solution.gain[&j]
                        + if state.instance.has_edge(i, j) {
                            2.0 * state.instance.get_weight(i, j)
                        } else {
                            0.0
                        },
                };
                neighbor.add_to_tabu_map(
                    &mut self.tabu_map,
                    state.iteration,
                    self.tabu_tenure,
                );
                state.apply(&neighbor)?;
                /*
                               let mut best_list = filter_best(v0_tabu.iter().flat_map(|v0| {
                                   v1_tabu.iter().map(|v1| MaxCutSwapNeighbor {
                                       i: v0.i,
                                       j: v1.i,
                                       gain: state.solution.gain[&v0.i]
                                           + state.solution.gain[&v1.i]
                                           + if state.instance.has_edge(v0.i, v1.i) {
                                               2.0 * state.instance.get_weight(v0.i, v1.i)
                                           } else {
                                               0.0
                                           },
                                   })
                               }));

                               if let Some(best_move) = best_list.pop() {
                                   best_move.add_to_tabu_map(
                                       &mut self.tabu_map.borrow_mut(),
                                       state.iteration,
                                       self.tabu_tenure,
                                   );
                                   state.apply(&best_move);
                               } else {
                                   tracing::warn!("No valid swap neighbor found for perturbation");
                                   state.progress_iteration();
                               }
                */
            }
        }
        Ok(())
    }
}

impl Heuristic<MaxCut> for BreakoutLocalSearch {
    fn clear(&mut self) {
        self.omega = 0;
        self.l = self.l0;
        self.prev_best_objective = None;
        self.prev_solution = None;
        self.tabu_map = HashMap::new();
    }

    fn run_once<'a>(
        &mut self,
        state: &mut SearchState<'a, MaxCut>,
    ) -> Result<(), OptError> {
        self.local_search_with_updating_tabu(state)?;

        self.update_omega(state);
        self.update_l(state);

        match self.get_perturbation_type() {
            PerturbationType::Strong => {
                self.apply_strong_perturbation(state)?;
            }
            PerturbationType::WeakFlip => {
                self.apply_weak_flip_perturbation(state)?;
            }
            PerturbationType::WeakSwap => {
                self.apply_weak_swap_perturbation(state)?;
            }
        }

        Ok(())
    }

    fn is_done<'a>(&self, state: &SearchState<'a, MaxCut>) -> bool {
        self.stop_condition.is_done(state)
    }
}
