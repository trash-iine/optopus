use optopus::algorithm::{Heuristic, LocalSearch, RandomWalk, StopCondition, TabuSearch};
use optopus::problem::max_cut::MaxCutFlipNeighbor;
use optopus::problem::MaxCut;
use optopus::search_state::{EnumerateMoveToNeighbor, SearchState};

struct BreakoutLocalSearch {
    max_failed_update: usize,
}

impl Heuristic<MaxCut> for BreakoutLocalSearch {
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, MaxCut>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut best_move_option = None;
        for neighbor in state.iter_on_move_to_neighbor() {
            if !state.is_move_to_be_better_than_currernt(&neighbor) {
                continue;
            }

            if let Some(best_move) = best_move_option {
                if state.is_first_move_better_than_second(&neighbor, &best_move) {
                    best_move_option = Some(neighbor);
                }
            } else {
                best_move_option = Some(neighbor);
            }
        }

        if let Some(best_move) = best_move_option {
            state.update(&best_move);
            self.max_failed_update = 0;
        } else {
            self.max_failed_update += 1;
        }

        Ok(())
    }

    fn is_done<'a>(&self, state: &SearchState<'a, MaxCut>) -> bool {
        todo!()
    }
}

fn main() {
    let mut mc = MaxCut::new();
    mc.add_weight(0, 1, 1.0);
    mc.add_weight(0, 2, 1.0);
    mc.add_weight(1, 2, 1.0);

    let mut state = SearchState::new(&mc, rand::rng());
    let sc = StopCondition::new(Some(1000), None, None);
    let ls = LocalSearch::<MaxCutFlipNeighbor>::new(sc);
    ls.run(&mut state);
}
