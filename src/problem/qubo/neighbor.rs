use super::problem::{Coefficient, Qubo};
use crate::{
    problem::qubo::problem::QuboSolution,
    search_state::{EnabledTabu, Evaluable, MoveToNeigbor, Rankable},
};

// ---------------------------------------------------------------------------
// Flip 近傍 (1-opt)
// ---------------------------------------------------------------------------

/// 変数を1つフリップする近傍
#[derive(Debug, Clone)]
pub struct QuboFlipNeighbour {
    pub i: usize,
    /// 目的関数値の変化量 (負 = 改善)
    pub gain: Coefficient,
}

impl Rankable for QuboFlipNeighbour {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl EnabledTabu for QuboFlipNeighbour {
    type TabuMap = std::collections::HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&self.i)
            .map_or(true, |&tabu_tenure| iteration > tabu_tenure)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let tabu_duration = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + tabu_duration);
    }
}

impl Evaluable<Coefficient> for QuboFlipNeighbour {
    fn evaluate(&self) -> Coefficient {
        self.gain
    }
}

impl MoveToNeigbor<Qubo> for QuboFlipNeighbour {
    fn apply_to_solution(&self, prob: &Qubo, sol: &mut QuboSolution) {
        let bi = *sol
            .x
            .get(&self.i)
            .expect(format!("{} is not found in solution", self.i).as_str());

        // 1. 変数をフリップ
        sol.x.insert(self.i, !bi);

        // 2. gain[i] を更新: 再度フリップすると元に戻る = 符号反転
        sol.gain.insert(self.i, -self.gain);

        // 3. i の近傍 j の gain を更新
        //    x_i が bi → !bi に変化したとき、gain[j] の変化量は:
        //      bi == x_j のとき: +Q[i][j]
        //      bi != x_j のとき: -Q[i][j]
        for (&j, &q) in prob.iter_on_adjacency(self.i) {
            if j == self.i {
                continue; // 対角項は他変数の gain に影響しない
            }
            let bj = *sol
                .x
                .get(&j)
                .expect(format!("{} is not found in solution", j).as_str());
            let delta = if bi == bj { q } else { -q };
            *sol.gain.entry(j).or_insert(0) += delta;
        }

        // 4. 目的関数値を更新
        sol.objective += self.gain;
    }

    fn iter(prob: &Qubo, sol: &QuboSolution) -> impl Iterator<Item = Self> + Send {
        // 変数インデックスが連続 0..n である保証はないため keys() から収集する
        let vars: Vec<usize> = prob.iter_on_variables().copied().collect();
        let gain = sol.gain.clone();
        vars.into_iter().map(move |i| QuboFlipNeighbour {
            i,
            gain: gain[&i],
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &Qubo,
        src: &QuboSolution,
        other: &QuboSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}

// ---------------------------------------------------------------------------
// Swap 近傍 (2-opt: 変数を2つ同時にフリップ)
// ---------------------------------------------------------------------------

/// 変数 i と j を同時にフリップする近傍
#[derive(Debug, Clone)]
pub struct QuboSwapNeighbour {
    pub i: usize,
    pub j: usize,
    /// 目的関数値の変化量 (負 = 改善)
    pub gain: Coefficient,
}

impl Rankable for QuboSwapNeighbour {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl EnabledTabu for QuboSwapNeighbour {
    type TabuMap = std::collections::HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let enabled_i = tabu_map
            .get(&self.i)
            .map_or(true, |&t| iteration > t);
        let enabled_j = tabu_map
            .get(&self.j)
            .map_or(true, |&t| iteration > t);
        enabled_i && enabled_j
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

impl MoveToNeigbor<Qubo> for QuboSwapNeighbour {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(&self, prob: &Qubo, sol: &mut QuboSolution) {
        // i をフリップ → j をフリップ (gain[j] は i フリップ後の値を使う)
        let flip_i = QuboFlipNeighbour {
            i: self.i,
            gain: sol.gain[&self.i],
        };
        flip_i.apply_to_solution(prob, sol);

        let flip_j = QuboFlipNeighbour {
            i: self.j,
            gain: sol.gain[&self.j],
        };
        flip_j.apply_to_solution(prob, sol);
    }

    fn iter(prob: &Qubo, sol: &QuboSolution) -> impl Iterator<Item = Self> + Send {
        let vars: Vec<usize> = prob.iter_on_variables().copied().collect();
        let n = vars.len();
        let x = sol.x.clone();
        let gain = sol.gain.clone();

        // 全ペア (i, j) に対して gain を事前計算
        // gain(swap i,j) = gain_i + gain_j + delta_ij
        // delta_ij = Q[i][j] * (x_i == x_j ? 1 : -1)
        let mut items: Vec<QuboSwapNeighbour> = Vec::with_capacity(n * (n - 1) / 2);
        for (idx_i, &i) in vars.iter().enumerate() {
            for &j in &vars[idx_i + 1..] {
                let q_ij = prob.get_q(i, j).unwrap_or(0);
                let bi = x[&i];
                let bj = x[&j];
                let delta_ij = if bi == bj { q_ij } else { -q_ij };
                let swap_gain = gain[&i] + gain[&j] + delta_ij;
                items.push(QuboSwapNeighbour { i, j, gain: swap_gain });
            }
        }
        items.into_iter()
    }

    fn move_to_be_better_than(
        &self,
        _: &Qubo,
        src: &QuboSolution,
        other: &QuboSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::SearchState;
    use std::collections::HashMap;

    fn make_qubo() -> Qubo {
        let mut q = Qubo::new();
        // 3変数のQUBO: Q[0][1]=1, Q[1][2]=2, Q[0][2]=3, Q[2][2]=1
        q.set_q(0, 1, 1);
        q.set_q(1, 2, 2);
        q.set_q(0, 2, 3);
        q.set_q(2, 2, 1);
        q
    }

    fn make_solution(qubo: &Qubo, x: HashMap<usize, bool>) -> QuboSolution {
        let gain: HashMap<_, _> = x
            .keys()
            .map(|&i| (i, qubo.calculate_gain(&x, i)))
            .collect();
        let objective = qubo.calculate_energy(&x);
        QuboSolution { x, gain, objective }
    }

    #[test]
    fn test_flip_apply_consistency() {
        let qubo = make_qubo();
        let sol = make_solution(
            &qubo,
            HashMap::from([(0, true), (1, false), (2, true)]),
        );

        for neighbor in QuboFlipNeighbour::iter(&qubo, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&qubo, &mut s);

            // objective が正しく更新されているか
            let expected_obj = qubo.calculate_energy(&s.x);
            assert_eq!(
                s.objective, expected_obj,
                "flip {}: objective={} expected={}",
                neighbor.i, s.objective, expected_obj
            );

            // gain が正しく更新されているか
            for &i in qubo.iter_on_variables() {
                let expected_gain = qubo.calculate_gain(&s.x, i);
                assert_eq!(
                    s.gain[&i], expected_gain,
                    "flip {}: gain[{}]={} expected={}",
                    neighbor.i, i, s.gain[&i], expected_gain
                );
            }
        }
    }

    #[test]
    fn test_flip_gain_matches_energy_delta() {
        let qubo = make_qubo();
        let sol = make_solution(
            &qubo,
            HashMap::from([(0, true), (1, false), (2, true)]),
        );

        for neighbor in QuboFlipNeighbour::iter(&qubo, &sol) {
            let mut flipped_x = sol.x.clone();
            flipped_x.insert(neighbor.i, !flipped_x[&neighbor.i]);
            let expected_delta = qubo.calculate_energy(&flipped_x) - sol.objective;
            assert_eq!(
                neighbor.gain, expected_delta,
                "flip {}: gain={} expected delta={}",
                neighbor.i, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_swap_apply_consistency() {
        let qubo = make_qubo();
        let sol = make_solution(
            &qubo,
            HashMap::from([(0, true), (1, false), (2, true)]),
        );

        for neighbor in QuboSwapNeighbour::iter(&qubo, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&qubo, &mut s);

            let expected_obj = qubo.calculate_energy(&s.x);
            assert_eq!(
                s.objective, expected_obj,
                "swap ({},{}): objective={} expected={}",
                neighbor.i, neighbor.j, s.objective, expected_obj
            );
        }
    }

    #[test]
    fn test_swap_gain_matches_energy_delta() {
        let qubo = make_qubo();
        let sol = make_solution(
            &qubo,
            HashMap::from([(0, true), (1, false), (2, true)]),
        );

        for neighbor in QuboSwapNeighbour::iter(&qubo, &sol) {
            let mut flipped_x = sol.x.clone();
            flipped_x.insert(neighbor.i, !flipped_x[&neighbor.i]);
            flipped_x.insert(neighbor.j, !flipped_x[&neighbor.j]);
            let expected_delta = qubo.calculate_energy(&flipped_x) - sol.objective;
            assert_eq!(
                neighbor.gain, expected_delta,
                "swap ({},{}): gain={} expected delta={}",
                neighbor.i, neighbor.j, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_search_state_new() {
        let qubo = make_qubo();
        let _state = SearchState::new(&qubo, rand::rng());
    }
}
