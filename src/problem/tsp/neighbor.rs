use std::sync::Arc;

use super::definition::{TspSolution, TspWithCoordinates};
use crate::search_state::{EnabledTabu, Evaluable, MoveToNeigbor, Rankable};

// ---------------------------------------------------------------------------
// 2-opt 近傍
// ---------------------------------------------------------------------------

/// 2-opt 移動: エッジ (tour[i], tour[i+1]) と (tour[j], tour[(j+1)%n]) を除去し、
/// (tour[i], tour[j]) と (tour[i+1], tour[(j+1)%n]) で繋ぎ直す。
/// 実装上は tour[i+1..=j] を反転させる。
#[derive(Debug, Clone)]
pub struct TspTwoOptNeighbor {
    pub i: usize,
    pub j: usize,
    /// 目的関数値の変化量 (負 = 改善)
    pub gain: f64,
}

impl Rankable for TspTwoOptNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

// TSP は最小化問題: gain = ツアー長の変化量 (正 = 悪化) → SA に直接渡せる
impl Evaluable<f64> for TspTwoOptNeighbor {
    fn evaluate(&self) -> f64 {
        self.gain
    }
}

impl EnabledTabu for TspTwoOptNeighbor {
    // (i, j) のペアをキーにして禁断リストを管理する
    type TabuMap = std::collections::HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let key = (self.i.min(self.j), self.i.max(self.j));
        tabu_map.get(&key).map_or(true, |&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        let key = (self.i.min(self.j), self.i.max(self.j));
        tabu_map.insert(key, iteration + d);
    }
}

impl MoveToNeigbor<TspWithCoordinates> for TspTwoOptNeighbor {
    fn apply_to_solution(&self, _: &TspWithCoordinates, sol: &mut TspSolution) {
        sol.tour[self.i + 1..=self.j].reverse();
        sol.objective += self.gain;
    }

    fn iter(prob: &TspWithCoordinates, sol: &TspSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.get_n();
        let tour = Arc::new(sol.tour.clone());
        let coords = Arc::new(prob.coordinates.clone());

        (0..n - 1).flat_map(move |i| {
            let tour = Arc::clone(&tour);
            let coords = Arc::clone(&coords);
            // i=0 のとき j=n-1 は全体の逆順（無向グラフでは gain=0 の自明移動）なので除外
            let max_j = if i == 0 { n - 1 } else { n };
            (i + 2..max_j).map(move |j| {
                let dist = |a: usize, b: usize| -> f64 {
                    let (x1, y1) = coords[a];
                    let (x2, y2) = coords[b];
                    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
                };
                let gain = dist(tour[i], tour[j]) + dist(tour[i + 1], tour[(j + 1) % n])
                    - dist(tour[i], tour[i + 1])
                    - dist(tour[j], tour[(j + 1) % n]);
                TspTwoOptNeighbor { i, j, gain }
            })
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &TspWithCoordinates,
        src: &TspSolution,
        other: &TspSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }
}

// ---------------------------------------------------------------------------
// Relocate 近傍 (Or-opt 1-city)
// ---------------------------------------------------------------------------

/// Relocate 移動: ツアー上の都市 tour[pos] を取り出し、tour[ins] の直後に挿入する。
/// ins は元のツアー上のインデックスで指定する。
#[derive(Debug, Clone)]
pub struct TspRelocateNeighbor {
    pub pos: usize,
    pub ins: usize,
    /// 目的関数値の変化量 (負 = 改善)
    pub gain: f64,
}

impl Rankable for TspRelocateNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluable<f64> for TspRelocateNeighbor {
    fn evaluate(&self) -> f64 {
        self.gain
    }
}

impl EnabledTabu for TspRelocateNeighbor {
    // (pos, ins) のペアをキーにして禁断リストを管理する
    type TabuMap = std::collections::HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&(self.pos, self.ins))
            .map_or(true, |&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert((self.pos, self.ins), iteration + d);
    }
}

impl MoveToNeigbor<TspWithCoordinates> for TspRelocateNeighbor {
    fn apply_to_solution(&self, _: &TspWithCoordinates, sol: &mut TspSolution) {
        let city = sol.tour.remove(self.pos);
        // pos を除去後のインデックス変化を補正して挿入位置を決める
        let insert_at = if self.ins < self.pos {
            self.ins + 1 // ins は pos より前なのでインデックスは不変 → 直後に挿入
        } else {
            self.ins // ins > pos だったので除去後は ins-1 に移動 → ins に挿入して直後に
        };
        sol.tour.insert(insert_at, city);
        sol.objective += self.gain;
    }

    fn iter(prob: &TspWithCoordinates, sol: &TspSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.get_n();
        let tour = Arc::new(sol.tour.clone());
        let coords = Arc::new(prob.coordinates.clone());

        (0..n).flat_map(move |pos| {
            let tour = Arc::clone(&tour);
            let coords = Arc::clone(&coords);
            let prev = (pos + n - 1) % n;
            let next = (pos + 1) % n;

            (0..n).filter_map(move |ins| {
                // pos 自身または直前 (prev) への挿入は元の位置と同じなので除外
                if ins == pos || ins == prev {
                    return None;
                }

                let dist = |a: usize, b: usize| -> f64 {
                    let (x1, y1) = coords[a];
                    let (x2, y2) = coords[b];
                    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
                };

                let ins_next = (ins + 1) % n;

                // pos を prev-pos-next から取り除くコスト削減
                let removal_gain = dist(tour[prev], tour[pos]) + dist(tour[pos], tour[next])
                    - dist(tour[prev], tour[next]);

                // pos を ins-ins_next の間へ挿入する追加コスト
                let insertion_cost = dist(tour[ins], tour[pos]) + dist(tour[pos], tour[ins_next])
                    - dist(tour[ins], tour[ins_next]);

                let gain = insertion_cost - removal_gain;
                Some(TspRelocateNeighbor { pos, ins, gain })
            })
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &TspWithCoordinates,
        src: &TspSolution,
        other: &TspSolution,
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
    use crate::problem::tsp::definition::calculate_tour_length;

    fn make_square_tsp() -> TspWithCoordinates {
        // 正方形の4都市: (0,0), (1,0), (1,1), (0,1)
        // 辺長 = 1, 対角線 = sqrt(2)
        TspWithCoordinates::new(
            "square".to_string(),
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
    }

    #[test]
    fn test_two_opt_gain_calculation() {
        let tsp = make_square_tsp();
        // 初期ツアー [0,1,3,2] は交差あり (長さ = 1 + sqrt(2) + 1 + sqrt(2))
        let tour = vec![0, 1, 3, 2];
        let objective = calculate_tour_length(&tsp, &tour);
        let sol = TspSolution { tour, objective };

        // 2-opt (i=1, j=2): エッジ (1,3) と (2,0) を (1,2) と (3,0) に置換
        // → ツアー [0,1,2,3] (長さ = 4.0)
        let neighbors: Vec<_> = TspTwoOptNeighbor::iter(&tsp, &sol).collect();
        let best = neighbors
            .iter()
            .min_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap())
            .unwrap();
        assert!(best.gain < 0.0, "最良の2-opt移動は改善のはず");

        let mut sol2 = sol.clone();
        best.apply_to_solution(&tsp, &mut sol2);
        let expected = calculate_tour_length(&tsp, &sol2.tour);
        assert!((sol2.objective - expected).abs() < 1e-9);
    }

    #[test]
    fn test_two_opt_apply_consistency() {
        let tsp = make_square_tsp();
        let sol = TspSolution {
            tour: vec![0, 2, 1, 3],
            objective: calculate_tour_length(&tsp, &vec![0, 2, 1, 3]),
        };

        for neighbor in TspTwoOptNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s);
            let expected = calculate_tour_length(&tsp, &s.tour);
            assert!(
                (s.objective - expected).abs() < 1e-9,
                "2-opt ({},{}) after: objective={} expected={}",
                neighbor.i,
                neighbor.j,
                s.objective,
                expected
            );
        }
    }

    #[test]
    fn test_relocate_apply_consistency() {
        let tsp = make_square_tsp();
        let sol = TspSolution {
            tour: vec![0, 1, 2, 3],
            objective: calculate_tour_length(&tsp, &vec![0, 1, 2, 3]),
        };

        for neighbor in TspRelocateNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s);
            let expected = calculate_tour_length(&tsp, &s.tour);
            assert!(
                (s.objective - expected).abs() < 1e-9,
                "relocate (pos={}, ins={}) after: objective={} expected={}",
                neighbor.pos,
                neighbor.ins,
                s.objective,
                expected
            );
        }
    }

    #[test]
    fn test_relocate_tour_is_valid_permutation() {
        let tsp = make_square_tsp();
        let sol = TspSolution {
            tour: vec![0, 1, 2, 3],
            objective: calculate_tour_length(&tsp, &vec![0, 1, 2, 3]),
        };

        for neighbor in TspRelocateNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s);
            let mut sorted = s.tour.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2, 3], "ツアーは有効な順列のはず");
        }
    }
}
