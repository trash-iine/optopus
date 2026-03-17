use std::cell::RefCell;
use std::cmp::Ordering;

use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{MoveToNeigbor, ProblemTrait, Rankable, SearchState};

/// ビーム探索アルゴリズム。
///
/// `beam_width` 個の候補解を並行して管理し、各ステップで全候補の近傍を展開して
/// 上位 `beam_width` 個を次のビームとして保持します。
/// 各ステップで最良解が `SearchState::best_solution` に自動的に記録されます。
///
/// # Example
///
/// ```
/// use optopus::heuristic::{BeamSearch, StopCondition, Heuristic};
/// use optopus::search_state::SearchState;
/// use optopus::problem::{MaxCut, MaxCutFlipNeighbor};
///
/// let mut mc = MaxCut::new();
/// mc.add_weight(0, 1, 1.0);
/// mc.add_weight(0, 2, 1.0);
/// mc.add_weight(1, 2, 1.0);
///
/// let mut state = SearchState::new(&mc);
/// let bs = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(
///     StopCondition::iterations(1000),
///     5,
/// );
/// bs.run(&mut state).unwrap();
/// ```
pub struct BeamSearch<P: ProblemTrait, N> {
    pub stop_condition: StopCondition,
    pub beam_width: usize,
    beam: RefCell<Vec<P::Solution>>,
    _phantom: std::marker::PhantomData<N>,
}

impl<P: ProblemTrait, N> BeamSearch<P, N> {
    pub fn new(stop_condition: StopCondition, beam_width: usize) -> Self {
        if beam_width == 0 {
            panic!("beam_width must be greater than 0");
        }
        Self {
            stop_condition,
            beam_width,
            beam: RefCell::new(Vec::new()),
            _phantom: std::marker::PhantomData,
        }
    }

    fn clear_beam(&self) {
        self.beam.borrow_mut().clear();
    }
}

impl<P, N> Heuristic<P> for BeamSearch<P, N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Rankable,
{
    fn clear(&mut self) {
        self.clear_beam();
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let mut beam = self.beam.borrow_mut();

        // 初回: state.solution からビームを初期化
        if beam.is_empty() {
            beam.push(state.solution.clone());
        }

        // 全ビーム候補の近傍を展開して candidates に収集
        let mut candidates: Vec<P::Solution> = Vec::new();
        for beam_sol in beam.iter() {
            for neighbor in N::iter(&state.instance, beam_sol) {
                let mut candidate = beam_sol.clone();
                neighbor.apply_to_solution(&state.instance, &mut candidate)?;
                candidates.push(candidate);
            }
        }

        // 近傍が空なら iteration だけ進めて終了
        if candidates.is_empty() {
            drop(beam);
            state.progress_iteration();
            return Ok(());
        }

        // Rankable による降順ソート（良い順）→ 上位 beam_width を保持
        candidates.sort_unstable_by(|a, b| {
            if a.is_better_than(b) {
                Ordering::Less
            } else if b.is_better_than(a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        candidates.truncate(self.beam_width);

        // state をビームの最良解で更新
        state.solution = candidates[0].clone();
        drop(beam);

        state.update_best();
        state.progress_iteration();

        *self.beam.borrow_mut() = candidates;

        Ok(())
    }

    fn run<'a>(&self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        self.clear_beam();
        while !self.is_done(state) {
            self.run_once(state)?;
        }
        Ok(())
    }
}
