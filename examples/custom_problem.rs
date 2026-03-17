//! 独自の最適化問題を定義するサンプル。
//!
//! ProblemTrait / MoveToNeigbor / Rankable を実装して、
//! 組み込みの LocalSearch を使って解を求めます。
//!
//! ここでは「整数ベクトルの要素和を最大化する」単純な問題を例に取ります。
//!
//! 実行方法:
//! ```
//! cargo run --example custom_problem
//! ```

use optopus::prelude::*;

// ─── 問題定義 ───────────────────────────────────────────────
/// N個の整数値 (0 or 1) の和を最大化する問題（カバーの例）
struct OneMaxProblem {
    n: usize,
}

// ─── 解の定義 ────────────────────────────────────────────────
#[derive(Clone)]
struct OneMaxSolution {
    bits: Vec<bool>,
}

impl OneMaxSolution {
    fn objective(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }
}

impl Rankable for OneMaxSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective() > other.objective()
    }
}

impl ProblemTrait for OneMaxProblem {
    type Solution = OneMaxSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> OneMaxSolution {
        OneMaxSolution {
            bits: (0..self.n).map(|_| rng.random_bool(0.5)).collect(),
        }
    }
}

// ─── 近傍定義（1ビットフリップ）─────────────────────────────
struct FlipMove {
    index: usize,
}

impl MoveToNeigbor<OneMaxProblem> for FlipMove {
    fn apply_to_solution(
        &self,
        _prob: &OneMaxProblem,
        sol: &mut OneMaxSolution,
    ) -> Result<(), optopus::error::OptError> {
        sol.bits[self.index] = !sol.bits[self.index];
        Ok(())
    }

    fn iter(prob: &OneMaxProblem, _sol: &OneMaxSolution) -> impl Iterator<Item = Self> + Send {
        (0..prob.n).map(|i| FlipMove { index: i })
    }

    fn move_to_be_better_than(
        &self,
        prob: &OneMaxProblem,
        src: &OneMaxSolution,
        other: &OneMaxSolution,
    ) -> bool {
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned)
            .expect("apply_to_solution should not fail");
        cloned.is_better_than(other)
    }
}

impl Rankable for FlipMove {
    fn is_better_than(&self, _other: &Self) -> bool {
        false // 移動同士の比較は不要（LocalSearch は解の優劣で選ぶ）
    }
}

// ─── メイン ─────────────────────────────────────────────────
fn main() {
    let prob = OneMaxProblem { n: 20 };
    let mut state = SearchState::new(&prob);

    let ls = LocalSearch::<FlipMove>::new(StopCondition::iterations(10_000));
    ls.run(&mut state).unwrap();

    println!(
        "best = {:?}  (objective = {}/{})",
        state.best_solution.bits,
        state.best_solution.objective(),
        prob.n
    );
}
