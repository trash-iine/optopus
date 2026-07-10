use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;

use super::problem::{JobShopScheduling, JobShopSolution};
use crate::{
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};

/// Minimum operation-sequence length for parallel neighborhood evaluation.
///
/// Each candidate costs a full O(n) schedule decode, so the neighborhood scan
/// is O(n²) (swap) / O(n³) (relocate) — heavy enough for rayon to pay off on
/// larger instances. Candidates are collected in index order, so results are
/// identical to the serial path regardless of thread count.
const PARALLEL_ITER_MIN_OPS: usize = 400;

/// Adjacent-pair swap on the operation sequence.
///
/// Swaps `operations[i]` with `operations[i+1]`. `gain` is the change in
/// makespan (negative = improvement) relative to the solution the move was
/// enumerated from; the schedule is re-decoded after the swap to obtain the
/// exact value.
#[derive(Debug, Clone)]
pub struct JobShopSwapNeighbor {
    pub i: usize,
    pub gain: f64,
}

impl Rankable for JobShopSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for JobShopSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for JobShopSwapNeighbor {
    /// Keyed by the swap position `i`.
    type TabuMap = HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map.get(&self.i).is_none_or(|&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + d);
    }
}

impl MoveToNeighbor<JobShopScheduling> for JobShopSwapNeighbor {
    fn apply_to_solution(
        &self,
        prob: &JobShopScheduling,
        sol: &mut JobShopSolution,
    ) -> Result<(), OptError> {
        sol.operations.swap(self.i, self.i + 1);
        let (makespan, completions) = prob.decode(&sol.operations)?;
        sol.objective = makespan;
        sol.completion_times = completions;
        Ok(())
    }

    fn iter(prob: &JobShopScheduling, sol: &JobShopSolution) -> impl Iterator<Item = Self> + Send {
        let n = sol.operations.len();
        let base = sol.objective as f64;
        let items: Vec<Self> = if n >= PARALLEL_ITER_MIN_OPS {
            (0..n.saturating_sub(1))
                .into_par_iter()
                .filter(|&i| sol.operations[i] != sol.operations[i + 1])
                .map_init(
                    || sol.operations.clone(),
                    |tentative, i| {
                        tentative.swap(i, i + 1);
                        let makespan = prob
                            .compute_makespan(tentative)
                            .expect("swap of valid sequence stays valid");
                        tentative.swap(i, i + 1);
                        JobShopSwapNeighbor {
                            i,
                            gain: makespan as f64 - base,
                        }
                    },
                )
                .collect()
        } else {
            let mut items = Vec::with_capacity(n.saturating_sub(1));
            let mut tentative = sol.operations.clone();
            for i in 0..n.saturating_sub(1) {
                if sol.operations[i] == sol.operations[i + 1] {
                    continue; // identity swap (same job)
                }
                tentative.swap(i, i + 1);
                let makespan = prob
                    .compute_makespan(&tentative)
                    .expect("swap of valid sequence stays valid");
                tentative.swap(i, i + 1);
                let gain = makespan as f64 - base;
                items.push(JobShopSwapNeighbor { i, gain });
            }
            items
        };
        items.into_iter()
    }

    /// Compares via the cached `gain` (exact: makespans are integers), which
    /// is relative to the solution the move was enumerated from (`src`).
    fn move_to_be_better_than(
        &self,
        _prob: &JobShopScheduling,
        src: &JobShopSolution,
        other: &JobShopSolution,
    ) -> bool {
        src.objective as f64 + self.gain < other.objective as f64
    }

    /// O(n) + one decode: collects the non-identity swap positions once,
    /// picks one uniformly, and evaluates only that candidate (instead of
    /// decoding all n of them).
    fn random_neighbor(
        prob: &JobShopScheduling,
        sol: &JobShopSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let n = sol.operations.len();
        let positions: Vec<usize> = (0..n.saturating_sub(1))
            .filter(|&i| sol.operations[i] != sol.operations[i + 1])
            .collect();
        if positions.is_empty() {
            return None;
        }
        let i = positions[rng.random_range(0..positions.len())];
        let mut tentative = sol.operations.clone();
        tentative.swap(i, i + 1);
        let makespan = prob
            .compute_makespan(&tentative)
            .expect("swap of valid sequence stays valid");
        Some(Self {
            i,
            gain: makespan as f64 - sol.objective as f64,
        })
    }
}

/// Removes `operations[from]` and re-inserts it at position `to` (in the
/// post-removal indexing — i.e. `to ∈ 0..n-1`).
///
/// `gain` is the change in makespan relative to the solution the move was
/// enumerated from (negative = improvement).
#[derive(Debug, Clone)]
pub struct JobShopRelocateNeighbor {
    pub from: usize,
    pub to: usize,
    pub gain: f64,
}

impl Rankable for JobShopRelocateNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for JobShopRelocateNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for JobShopRelocateNeighbor {
    /// Keyed by the `(from, to)` pair.
    type TabuMap = HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&(self.from, self.to))
            .is_none_or(|&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert((self.from, self.to), iteration + d);
    }
}

fn relocate_in_place(operations: &mut Vec<usize>, from: usize, to: usize) {
    let job = operations.remove(from);
    operations.insert(to, job);
}

impl MoveToNeighbor<JobShopScheduling> for JobShopRelocateNeighbor {
    fn apply_to_solution(
        &self,
        prob: &JobShopScheduling,
        sol: &mut JobShopSolution,
    ) -> Result<(), OptError> {
        relocate_in_place(&mut sol.operations, self.from, self.to);
        let (makespan, completions) = prob.decode(&sol.operations)?;
        sol.objective = makespan;
        sol.completion_times = completions;
        Ok(())
    }

    fn iter(prob: &JobShopScheduling, sol: &JobShopSolution) -> impl Iterator<Item = Self> + Send {
        let n = sol.operations.len();
        let base = sol.objective as f64;
        let relocate_row = |tentative: &mut Vec<usize>, from: usize| {
            let mut row = Vec::with_capacity(n.saturating_sub(1));
            for to in 0..n {
                if to == from {
                    continue; // identity move
                }
                relocate_in_place(tentative, from, to);
                let makespan = prob
                    .compute_makespan(tentative)
                    .expect("relocate of valid sequence stays valid");
                relocate_in_place(tentative, to, from);
                let gain = makespan as f64 - base;
                row.push(JobShopRelocateNeighbor { from, to, gain });
            }
            row
        };
        let items: Vec<Self> = if n >= PARALLEL_ITER_MIN_OPS {
            (0..n)
                .into_par_iter()
                .map_init(
                    || sol.operations.clone(),
                    |buf, from| relocate_row(buf, from),
                )
                .collect::<Vec<Vec<Self>>>()
                .into_iter()
                .flatten()
                .collect()
        } else {
            let mut items = Vec::with_capacity(n * n.saturating_sub(1));
            let mut tentative = sol.operations.clone();
            for from in 0..n {
                items.extend(relocate_row(&mut tentative, from));
            }
            items
        };
        items.into_iter()
    }

    /// Compares via the cached `gain` (exact: makespans are integers), which
    /// is relative to the solution the move was enumerated from (`src`).
    fn move_to_be_better_than(
        &self,
        _prob: &JobShopScheduling,
        src: &JobShopSolution,
        other: &JobShopSolution,
    ) -> bool {
        src.objective as f64 + self.gain < other.objective as f64
    }

    /// O(1) + one decode: samples a uniformly random `(from, to)` pair with
    /// `to != from` and evaluates only that candidate (instead of decoding
    /// all n·(n−1) of them).
    fn random_neighbor(
        prob: &JobShopScheduling,
        sol: &JobShopSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let n = sol.operations.len();
        if n < 2 {
            return None;
        }
        let from = rng.random_range(0..n);
        let to = {
            let t = rng.random_range(0..n - 1);
            if t >= from { t + 1 } else { t }
        };
        let mut tentative = sol.operations.clone();
        relocate_in_place(&mut tentative, from, to);
        let makespan = prob
            .compute_makespan(&tentative)
            .expect("relocate of valid sequence stays valid");
        Some(Self {
            from,
            to,
            gain: makespan as f64 - sol.objective as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inst() -> JobShopScheduling {
        // 2 jobs × 2 machines
        // job 0: M0(2) → M1(3)
        // job 1: M1(1) → M0(4)
        JobShopScheduling::new(
            "tiny".to_string(),
            2,
            vec![vec![(0, 2), (1, 3)], vec![(1, 1), (0, 4)]],
        )
    }

    fn make_sol(inst: &JobShopScheduling, ops: Vec<usize>) -> JobShopSolution {
        let (objective, completion_times) = inst.decode(&ops).unwrap();
        JobShopSolution {
            operations: ops,
            objective,
            completion_times,
        }
    }

    #[test]
    fn test_swap_apply_consistency() {
        let inst = make_inst();
        let sol = make_sol(&inst, vec![0, 1, 1, 0]);
        for n in JobShopSwapNeighbor::iter(&inst, &sol) {
            let mut s = sol.clone();
            let before = s.objective;
            n.apply_to_solution(&inst, &mut s).unwrap();
            let recomputed = inst.decode(&s.operations).unwrap().0;
            assert_eq!(s.objective, recomputed);
            assert!(
                (n.gain - (s.objective as f64 - before as f64)).abs() < 1e-9,
                "swap gain inconsistent at i={}",
                n.i
            );
            // job-count invariant
            let mut counts = vec![0usize; inst.n_jobs];
            for &j in &s.operations {
                counts[j] += 1;
            }
            assert_eq!(counts, vec![inst.n_machines, inst.n_machines]);
        }
    }

    #[test]
    fn test_swap_skips_same_job_pairs() {
        let inst = make_inst();
        let sol = make_sol(&inst, vec![0, 0, 1, 1]);
        let moves: Vec<_> = JobShopSwapNeighbor::iter(&inst, &sol).collect();
        // Only i=1 (between job0 and job1) is a non-identity swap.
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].i, 1);
    }

    #[test]
    fn test_relocate_apply_consistency() {
        let inst = make_inst();
        let sol = make_sol(&inst, vec![0, 1, 0, 1]);
        for n in JobShopRelocateNeighbor::iter(&inst, &sol) {
            let mut s = sol.clone();
            let before = s.objective;
            n.apply_to_solution(&inst, &mut s).unwrap();
            let recomputed = inst.decode(&s.operations).unwrap().0;
            assert_eq!(s.objective, recomputed);
            assert!((n.gain - (s.objective as f64 - before as f64)).abs() < 1e-9);
            let mut counts = vec![0usize; inst.n_jobs];
            for &j in &s.operations {
                counts[j] += 1;
            }
            assert_eq!(counts, vec![inst.n_machines, inst.n_machines]);
        }
    }

    #[test]
    fn test_swap_finds_improvement_when_available() {
        let inst = make_inst();
        // [0, 0, 1, 1] schedules everything serially:
        //   pos 0: job0 op0 on M0 finish 2
        //   pos 1: job0 op1 on M1 finish 5
        //   pos 2: job1 op0 on M1 finish 6
        //   pos 3: job1 op1 on M0 finish 10
        // makespan = 10
        let sol = make_sol(&inst, vec![0, 0, 1, 1]);
        assert_eq!(sol.objective, 10);
        let best = JobShopSwapNeighbor::iter(&inst, &sol)
            .map(|n| n.gain)
            .fold(f64::INFINITY, f64::min);
        assert!(best < 0.0);
    }

    /// Reference implementation of the default `move_to_be_better_than`
    /// (clone + apply). Used to assert that the override agrees with it.
    fn reference_move_to_be_better_than<M: MoveToNeighbor<JobShopScheduling>>(
        m: &M,
        prob: &JobShopScheduling,
        src: &JobShopSolution,
        other: &JobShopSolution,
    ) -> bool {
        let mut cloned = src.clone();
        m.apply_to_solution(prob, &mut cloned).unwrap();
        cloned.is_better_than(other)
    }

    #[test]
    fn test_swap_move_to_be_better_than_matches_default() {
        let inst = make_inst();
        // Two starting solutions covering both serial and interleaved layouts.
        for ops in [vec![0, 1, 1, 0], vec![0, 0, 1, 1], vec![1, 0, 0, 1]] {
            let src = make_sol(&inst, ops);
            // Compare against `src` itself ("is the neighbor strictly better
            // than current?") and against an alternative `other` to exercise
            // both improving and non-improving outcomes.
            let other_alt = make_sol(&inst, vec![1, 1, 0, 0]);
            for other in [src.clone(), other_alt.clone()] {
                for n in JobShopSwapNeighbor::iter(&inst, &src) {
                    let expected = reference_move_to_be_better_than(&n, &inst, &src, &other);
                    let got = n.move_to_be_better_than(&inst, &src, &other);
                    assert_eq!(
                        got, expected,
                        "swap i={} disagrees: src={:?} other.obj={} got={} expected={}",
                        n.i, src.operations, other.objective, got, expected,
                    );
                }
            }
        }
    }

    #[test]
    fn test_random_neighbor_samples_member_of_iter() {
        use rand::SeedableRng;
        let inst = make_inst();
        let sol = make_sol(&inst, vec![0, 1, 0, 1]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);

        let swaps: Vec<_> = JobShopSwapNeighbor::iter(&inst, &sol).collect();
        for _ in 0..20 {
            let m = <JobShopSwapNeighbor as MoveToNeighbor<JobShopScheduling>>::random_neighbor(
                &inst, &sol, &mut rng,
            )
            .unwrap();
            assert!(swaps.iter().any(|s| s.i == m.i && s.gain == m.gain));
        }

        let relocs: Vec<_> = JobShopRelocateNeighbor::iter(&inst, &sol).collect();
        for _ in 0..20 {
            let m =
                <JobShopRelocateNeighbor as MoveToNeighbor<JobShopScheduling>>::random_neighbor(
                    &inst, &sol, &mut rng,
                )
                .unwrap();
            assert!(
                relocs
                    .iter()
                    .any(|r| r.from == m.from && r.to == m.to && r.gain == m.gain)
            );
        }
    }

    #[test]
    fn test_random_swap_neighbor_none_for_single_job() {
        use rand::SeedableRng;
        // One job only: every adjacent pair is the same job → no swap moves.
        let inst = JobShopScheduling::new("single".to_string(), 2, vec![vec![(0, 2), (1, 3)]]);
        let sol = make_sol(&inst, vec![0, 0]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        assert!(JobShopSwapNeighbor::iter(&inst, &sol).next().is_none());
        assert!(
            <JobShopSwapNeighbor as MoveToNeighbor<JobShopScheduling>>::random_neighbor(
                &inst, &sol, &mut rng
            )
            .is_none()
        );
    }

    #[test]
    fn test_relocate_move_to_be_better_than_matches_default() {
        let inst = make_inst();
        for ops in [vec![0, 1, 0, 1], vec![0, 0, 1, 1], vec![1, 0, 1, 0]] {
            let src = make_sol(&inst, ops);
            let other_alt = make_sol(&inst, vec![1, 1, 0, 0]);
            for other in [src.clone(), other_alt.clone()] {
                for n in JobShopRelocateNeighbor::iter(&inst, &src) {
                    let expected = reference_move_to_be_better_than(&n, &inst, &src, &other);
                    let got = n.move_to_be_better_than(&inst, &src, &other);
                    assert_eq!(
                        got, expected,
                        "relocate from={} to={} disagrees: src={:?} other.obj={} got={} expected={}",
                        n.from, n.to, src.operations, other.objective, got, expected,
                    );
                }
            }
        }
    }
}
