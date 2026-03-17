use std::collections::HashSet;
use std::sync::Arc;

use super::definition::{Sat, SatSolution};
use crate::{error::OptError, search_state::{EnabledTabu, Evaluable, MoveToNeigbor, Rankable}};

// ---------------------------------------------------------------------------
// Flip 近傍 (1-opt)
// ---------------------------------------------------------------------------

/// 変数を1つフリップする近傍
#[derive(Debug, Clone)]
pub struct SatFlipNeighbor {
    pub i: usize,
    /// 充足節数の変化量 (正 = 改善)
    pub gain: i64,
}

impl Rankable for SatFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain // 充足節数を最大化
    }
}

impl EnabledTabu for SatFlipNeighbor {
    type TabuMap = std::collections::HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&self.i)
            .map_or(true, |&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + d);
    }
}

// SAT は最大化問題: SA の受理確率には「悪化量 = -gain」を渡す
impl Evaluable<f64> for SatFlipNeighbor {
    fn evaluate(&self) -> f64 {
        -(self.gain as f64)
    }
}

impl MoveToNeigbor<Sat> for SatFlipNeighbor {
    fn apply_to_solution(&self, prob: &Sat, sol: &mut SatSolution) -> Result<(), OptError> {
        // フリップにより gain が変化する可能性がある変数を収集 (適用前に実施)
        let mut affected: HashSet<usize> = HashSet::new();
        for clause in prob.clauses_of_var(self.i) {
            for &lit in clause {
                let j = lit.unsigned_abs() as usize - 1;
                if j != self.i {
                    affected.insert(j);
                }
            }
        }

        // x[i] をフリップ
        sol.x[self.i] = !sol.x[self.i];

        // n_satisfied を更新
        sol.n_satisfied = (sol.n_satisfied as i64 + self.gain) as usize;

        // gain[i] を更新: 再フリップすると符号が反転
        sol.gain[self.i] = -self.gain;

        // 共有節を持つ変数の gain を再計算
        for j in affected {
            sol.gain[j] = prob.calc_gain(&sol.x, j);
        }
        Ok(())
    }

    fn iter(prob: &Sat, sol: &SatSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.n_vars();
        let gain = sol.gain.clone();
        (0..n).map(move |i| SatFlipNeighbor { i, gain: gain[i] })
    }

    fn move_to_be_better_than(&self, _: &Sat, src: &SatSolution, other: &SatSolution) -> bool {
        (src.n_satisfied as i64 + self.gain) > other.n_satisfied as i64
    }
}

// ---------------------------------------------------------------------------
// Swap 近傍 (2-opt: 同じ節に含まれる変数ペアを同時にフリップ)
// ---------------------------------------------------------------------------

/// 変数 i と j を同時にフリップする近傍
/// 節を共有するペアのみ列挙することで探索空間を絞る
#[derive(Debug, Clone)]
pub struct SatSwapNeighbor {
    pub i: usize,
    pub j: usize,
    /// 充足節数の変化量 (正 = 改善)
    pub gain: i64,
}

impl Rankable for SatSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl EnabledTabu for SatSwapNeighbor {
    type TabuMap = std::collections::HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let ok_i = tabu_map.get(&self.i).map_or(true, |&t| iteration > t);
        let ok_j = tabu_map.get(&self.j).map_or(true, |&t| iteration > t);
        ok_i && ok_j
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + d);
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.j, iteration + d);
    }
}

impl Evaluable<f64> for SatSwapNeighbor {
    fn evaluate(&self) -> f64 {
        -(self.gain as f64)
    }
}

impl MoveToNeigbor<Sat> for SatSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(&self, prob: &Sat, sol: &mut SatSolution) -> Result<(), OptError> {
        // i をフリップ → j をフリップ (gain[j] は i フリップ後の値を使う)
        let flip_i = SatFlipNeighbor {
            i: self.i,
            gain: sol.gain[self.i],
        };
        flip_i.apply_to_solution(prob, sol)?;

        let flip_j = SatFlipNeighbor {
            i: self.j,
            gain: sol.gain[self.j],
        };
        flip_j.apply_to_solution(prob, sol)?;
        Ok(())
    }

    fn iter(prob: &Sat, sol: &SatSolution) -> impl Iterator<Item = Self> + Send {
        let x = Arc::new(sol.x.clone());
        let gain = Arc::new(sol.gain.clone());

        // 節を共有するペア (i, j) を重複なく列挙する
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut items: Vec<SatSwapNeighbor> = Vec::new();

        for clause in prob.all_clauses() {
            for (a, &lit_a) in clause.iter().enumerate() {
                for &lit_b in &clause[a + 1..] {
                    let i = lit_a.unsigned_abs() as usize - 1;
                    let j = lit_b.unsigned_abs() as usize - 1;
                    let pair = (i.min(j), i.max(j));
                    if !seen.insert(pair) {
                        continue;
                    }
                    let (i, j) = pair;

                    // gain_swap = gain_i + gain_j_after_flip_i
                    let gain_j_after_flip_i =
                        prob.calc_gain_with_virtual_flip(&x, i, j);
                    let swap_gain = gain[i] + gain_j_after_flip_i;

                    items.push(SatSwapNeighbor { i, j, gain: swap_gain });
                }
            }
        }

        items.into_iter()
    }

    fn move_to_be_better_than(&self, _: &Sat, src: &SatSolution, other: &SatSolution) -> bool {
        (src.n_satisfied as i64 + self.gain) > other.n_satisfied as i64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::SearchState;

    /// (x1 ∨ x2), (¬x1 ∨ x3), (¬x2 ∨ ¬x3)
    fn make_sat() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 2]);
        sat.add_clause([-1, 3]);
        sat.add_clause([-2, -3]);
        sat
    }

    fn make_solution(sat: &Sat, x: Vec<bool>) -> SatSolution {
        let gain: Vec<i64> = (0..sat.n_vars()).map(|i| sat.calc_gain(&x, i)).collect();
        let n_satisfied = sat.calc_satisfied(&x);
        SatSolution { x, gain, n_satisfied }
    }

    #[test]
    fn test_search_state_new() {
        let sat = make_sat();
        let _state = SearchState::new(&sat);
    }

    #[test]
    fn test_flip_gain_matches_energy_delta() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatFlipNeighbor::iter(&sat, &sol) {
            let mut x2 = sol.x.clone();
            x2[neighbor.i] = !x2[neighbor.i];
            let expected_delta = sat.calc_satisfied(&x2) as i64 - sol.n_satisfied as i64;
            assert_eq!(
                neighbor.gain, expected_delta,
                "flip {}: gain={} expected={}",
                neighbor.i, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_flip_apply_consistency() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatFlipNeighbor::iter(&sat, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&sat, &mut s).unwrap();

            let expected_n_sat = sat.calc_satisfied(&s.x);
            assert_eq!(
                s.n_satisfied, expected_n_sat,
                "flip {}: n_satisfied={} expected={}",
                neighbor.i, s.n_satisfied, expected_n_sat
            );

            for i in 0..sat.n_vars() {
                let expected_gain = sat.calc_gain(&s.x, i);
                assert_eq!(
                    s.gain[i], expected_gain,
                    "flip {}: gain[{}]={} expected={}",
                    neighbor.i, i, s.gain[i], expected_gain
                );
            }
        }
    }

    #[test]
    fn test_swap_gain_matches_energy_delta() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatSwapNeighbor::iter(&sat, &sol) {
            let mut x2 = sol.x.clone();
            x2[neighbor.i] = !x2[neighbor.i];
            x2[neighbor.j] = !x2[neighbor.j];
            let expected_delta = sat.calc_satisfied(&x2) as i64 - sol.n_satisfied as i64;
            assert_eq!(
                neighbor.gain, expected_delta,
                "swap ({},{}): gain={} expected={}",
                neighbor.i, neighbor.j, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_swap_apply_consistency() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatSwapNeighbor::iter(&sat, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&sat, &mut s).unwrap();

            let expected_n_sat = sat.calc_satisfied(&s.x);
            assert_eq!(
                s.n_satisfied, expected_n_sat,
                "swap ({},{}): n_satisfied={} expected={}",
                neighbor.i, neighbor.j, s.n_satisfied, expected_n_sat
            );
        }
    }
}
