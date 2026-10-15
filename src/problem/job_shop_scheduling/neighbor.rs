use rand::Rng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use super::problem::{JobShopScheduling, JobShopSolution, PartialSchedule};
use crate::{
    building_blocks::search::{TabuKey, TabuMemory},
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor},
};

/// Minimum operation-sequence length for parallel neighborhood evaluation.
///
/// Each candidate costs a schedule of the sequence after the prefix it
/// shares with the solution, so the neighborhood scan is O(n²) (swap) /
/// O(n³) (relocate), heavy enough for rayon to pay off on larger instances.
/// Candidates are collected in index order, so results are identical to the
/// serial path regardless of thread count.
const PARALLEL_ITER_MIN_OPS: usize = 400;

/// Swap positions a parallel scan hands each task, which schedules the
/// prefix before its first position once.
const PARALLEL_SWAP_CHUNK: usize = 32;

/// When each operation of `operations` finishes, and the latest finish from
/// each position on, one longer so that past the end reads zero.
fn trace(prob: &JobShopScheduling, operations: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let mut partial = PartialSchedule::new(prob);
    let finish: Vec<u32> = operations.iter().map(|&j| partial.push(prob, j)).collect();
    let mut latest = vec![0; finish.len() + 1];
    for p in (0..finish.len()).rev() {
        latest[p] = latest[p + 1].max(finish[p]);
    }
    (finish, latest)
}

/// The swaps of positions `range` with their gains over `base`.
///
/// Swapping `i` and `i + 1` leaves the schedule of `operations[..i]` as it
/// is, so the prefix is scheduled once and advanced by one operation per
/// position, and each candidate schedules only from `i` on.
fn swap_scan(
    prob: &JobShopScheduling,
    operations: &[usize],
    base: i64,
    (finish, latest): (&[u32], &[u32]),
    range: std::ops::Range<usize>,
) -> Vec<JobShopSwapNeighbor> {
    let mut prefix = PartialSchedule::new(prob);
    for &j in &operations[..range.start] {
        prefix.push(prob, j);
    }
    let (mut work, mut shadow) = (prefix.clone(), prefix.clone());
    let mut items = Vec::with_capacity(range.len());
    for i in range {
        if operations[i] != operations[i + 1] {
            work.clone_from(&prefix);
            work.push(prob, operations[i + 1]);
            work.push(prob, operations[i]);
            shadow.clone_from(&prefix);
            shadow.push(prob, operations[i]);
            shadow.push(prob, operations[i + 1]);
            let makespan =
                work.finish_against(prob, &mut shadow, operations, i + 2, finish, latest);
            items.push(JobShopSwapNeighbor {
                i,
                gain: i64::from(makespan) - base,
            });
        }
        prefix.push(prob, operations[i]);
    }
    items
}

/// The relocations of `operations[from]` to every other place, with their
/// gains over `base`, `prefix` and `work` being scratch.
///
/// With `from` taken out the rest reads `rest`, and the move to `to` puts the
/// operation back before `rest[to]`, so the schedule of `rest[..to]` is
/// shared by every later `to` and advanced by one operation per place.
fn relocate_row(
    prob: &JobShopScheduling,
    operations: &[usize],
    base: i64,
    from: usize,
    (prefix, work): &mut (PartialSchedule, PartialSchedule),
) -> Vec<JobShopRelocateNeighbor> {
    let n = operations.len();
    let moved = operations[from];
    let rest = |k: usize| {
        if k < from {
            operations[k]
        } else {
            operations[k + 1]
        }
    };
    *prefix = PartialSchedule::new(prob);
    let mut row = Vec::with_capacity(n.saturating_sub(1));
    for to in 0..n {
        if to != from {
            work.clone_from(prefix);
            work.push(prob, moved);
            for k in to..n - 1 {
                work.push(prob, rest(k));
            }
            row.push(JobShopRelocateNeighbor {
                from,
                to,
                gain: i64::from(work.makespan) - base,
            });
        }
        if to < n - 1 {
            prefix.push(prob, rest(to));
        }
    }
    row
}

/// Adjacent-pair swap on the operation sequence.
///
/// Swaps `operations[i]` with `operations[i+1]`. `gain` is the change in
/// makespan (negative = improvement) relative to the solution the move was
/// enumerated from; the schedule is re-decoded after the swap to obtain the
/// exact value.
#[derive(Debug, Clone)]
pub struct JobShopSwapNeighbor {
    /// Left index of the swapped pair (swaps `operations[i]` and `operations[i+1]`).
    pub i: usize,
    /// Change in makespan (negative = improvement).
    pub gain: i64,
}

impl JobShopSwapNeighbor {
    /// Builds the swap of positions `i` and `i + 1`, decoding the swapped
    /// sequence once to obtain the exact makespan delta.
    ///
    /// This is the single-move path, used by
    /// [`random_neighbor`](MoveToNeighbor::random_neighbor) and by callers
    /// building one specific move. [`iter`](MoveToNeighbor::iter) deliberately
    /// does not go through here: it reuses one scratch buffer across the
    /// whole neighborhood instead of cloning the sequence per candidate.
    ///
    /// Swapping two positions that hold the same job is an identity move; the
    /// neighborhood scan skips those, but one built here is still evaluated
    /// correctly (`gain == 0`) rather than rejected.
    ///
    /// # Panics
    ///
    /// Panics if `i + 1` is out of range for `sol.operations`, or if
    /// `sol.operations` is not a valid operation sequence for `prob`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// // job 0: M0(2) → M1(3), job 1: M1(1) → M0(4)
    /// let inst = JobShopScheduling::new(
    ///     "tiny".to_string(),
    ///     2,
    ///     vec![vec![(0, 2), (1, 3)], vec![(1, 1), (0, 4)]],
    /// );
    /// let operations = vec![0, 1, 0, 1];
    /// let (objective, completion_times) = inst.decode(&operations).unwrap();
    /// let sol = JobShopSolution { operations, objective, completion_times };
    ///
    /// let m = JobShopSwapNeighbor::new(&inst, &sol, 1);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&inst, &mut after).unwrap();
    /// assert_eq!(i64::from(after.objective), i64::from(sol.objective) + m.gain);
    /// ```
    pub fn new(prob: &JobShopScheduling, sol: &JobShopSolution, i: usize) -> Self {
        let mut tentative = sol.operations.clone();
        tentative.swap(i, i + 1);
        let makespan = prob.compute_makespan(&tentative);
        Self {
            i,
            gain: i64::from(makespan) - i64::from(sol.objective),
        }
    }
}

impl Evaluate for JobShopSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for JobShopSwapNeighbor {
    /// The move is tabu while position `i` is still blocked at the current iteration.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(TabuKey::DenseVar(self.i), iteration)
    }

    /// Applying the move forbids position `i` for a tenure the memory draws.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::DenseVar(self.i), iteration, rng);
    }
}

impl MoveToNeighbor<JobShopScheduling> for JobShopSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

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
        let ops = &sol.operations[..];
        let positions = ops.len().saturating_sub(1);
        let base = i64::from(sol.objective);
        let (finish, latest) = trace(prob, ops);
        let traced = (&finish[..], &latest[..]);
        let items: Vec<Self> = if ops.len() >= PARALLEL_ITER_MIN_OPS {
            (0..positions)
                .into_par_iter()
                .step_by(PARALLEL_SWAP_CHUNK)
                .map(|start| {
                    swap_scan(
                        prob,
                        ops,
                        base,
                        traced,
                        start..(start + PARALLEL_SWAP_CHUNK).min(positions),
                    )
                })
                .collect::<Vec<Vec<Self>>>()
                .into_iter()
                .flatten()
                .collect()
        } else {
            swap_scan(prob, ops, base, traced, 0..positions)
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
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
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
        Some(Self::new(prob, sol, i))
    }
}

/// Removes `operations[from]` and re-inserts it at position `to` (in the
/// post-removal indexing, i.e. `to ∈ 0..n-1`).
///
/// `gain` is the change in makespan (negative = improvement) relative to the
/// solution the move was enumerated from; the schedule is re-decoded after
/// the relocation to obtain the exact value.
#[derive(Debug, Clone)]
pub struct JobShopRelocateNeighbor {
    /// Index of the operation to be relocated.
    pub from: usize,
    /// Destination index, in post-removal indexing (`to ∈ 0..n-1`).
    pub to: usize,
    /// Change in makespan (negative = improvement).
    pub gain: i64,
}

impl JobShopRelocateNeighbor {
    /// Builds the relocation of `operations[from]` to position `to`, decoding
    /// the resulting sequence once to obtain the exact makespan delta.
    ///
    /// This is the single-move path, used by
    /// [`random_neighbor`](MoveToNeighbor::random_neighbor) and by callers
    /// building one specific move. [`iter`](MoveToNeighbor::iter) deliberately
    /// does not go through here: it reuses one scratch buffer across the
    /// whole neighborhood instead of cloning the sequence per candidate.
    ///
    /// `to` is in post-removal indexing, so `to == from` is the identity move;
    /// the neighborhood scan skips it, but one built here is still evaluated
    /// correctly (`gain == 0`) rather than rejected.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is out of range for `sol.operations`, or if
    /// `sol.operations` is not a valid operation sequence for `prob`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// // job 0: M0(2) → M1(3), job 1: M1(1) → M0(4)
    /// let inst = JobShopScheduling::new(
    ///     "tiny".to_string(),
    ///     2,
    ///     vec![vec![(0, 2), (1, 3)], vec![(1, 1), (0, 4)]],
    /// );
    /// let operations = vec![0, 1, 0, 1];
    /// let (objective, completion_times) = inst.decode(&operations).unwrap();
    /// let sol = JobShopSolution { operations, objective, completion_times };
    ///
    /// let m = JobShopRelocateNeighbor::new(&inst, &sol, 3, 0);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&inst, &mut after).unwrap();
    /// assert_eq!(after.operations, vec![1, 0, 1, 0]);
    /// assert_eq!(i64::from(after.objective), i64::from(sol.objective) + m.gain);
    /// ```
    pub fn new(prob: &JobShopScheduling, sol: &JobShopSolution, from: usize, to: usize) -> Self {
        let mut tentative = sol.operations.clone();
        relocate_in_place(&mut tentative, from, to);
        let makespan = prob.compute_makespan(&tentative);
        Self {
            from,
            to,
            gain: i64::from(makespan) - i64::from(sol.objective),
        }
    }
}

impl Evaluate for JobShopRelocateNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for JobShopRelocateNeighbor {
    /// Keyed by the `(from, to)` move, so re-making the same relocation is what is blocked.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.from, self.to), iteration)
    }

    /// Applying it forbids that `(from, to)` pair.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.from, self.to), iteration, rng);
    }
}

fn relocate_in_place(operations: &mut Vec<usize>, from: usize, to: usize) {
    let job = operations.remove(from);
    operations.insert(to, job);
}

impl MoveToNeighbor<JobShopScheduling> for JobShopRelocateNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

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
        let ops = &sol.operations[..];
        let n = ops.len();
        let base = i64::from(sol.objective);
        let scratch = || (PartialSchedule::new(prob), PartialSchedule::new(prob));
        let items: Vec<Self> = if n >= PARALLEL_ITER_MIN_OPS {
            (0..n)
                .into_par_iter()
                .map_init(scratch, |s, from| relocate_row(prob, ops, base, from, s))
                .collect::<Vec<Vec<Self>>>()
                .into_iter()
                .flatten()
                .collect()
        } else {
            let mut items = Vec::with_capacity(n * n.saturating_sub(1));
            let mut s = scratch();
            for from in 0..n {
                items.extend(relocate_row(prob, ops, base, from, &mut s));
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
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// O(1) + one decode: samples a uniformly random `(from, to)` pair with
    /// `to != from` and evaluates only that candidate (instead of decoding
    /// all n·(n−1) of them).
    fn random_neighbor(
        prob: &JobShopScheduling,
        sol: &JobShopSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let (from, to) =
            crate::building_blocks::search::random_distinct_pair(sol.operations.len(), rng)?;
        Some(Self::new(prob, sol, from, to))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trait_defs::Rankable;

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
                n.gain == i64::from(s.objective) - i64::from(before),
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
    fn test_new_matches_iter() {
        let inst = make_inst();
        let sol = make_sol(&inst, vec![0, 1, 0, 1]);

        for m in JobShopSwapNeighbor::iter(&inst, &sol) {
            let rebuilt = JobShopSwapNeighbor::new(&inst, &sol, m.i);
            assert_eq!(rebuilt.gain, m.gain, "swap i={}", m.i);
        }
        for m in JobShopRelocateNeighbor::iter(&inst, &sol) {
            let rebuilt = JobShopRelocateNeighbor::new(&inst, &sol, m.from, m.to);
            assert_eq!(
                rebuilt.gain, m.gain,
                "relocate (from={}, to={})",
                m.from, m.to
            );
        }
    }

    #[test]
    fn test_new_evaluates_identity_moves_as_zero_gain() {
        let inst = make_inst();
        let sol = make_sol(&inst, vec![0, 0, 1, 1]);
        // i=0 swaps two operations of the same job; iter skips it.
        assert_eq!(JobShopSwapNeighbor::new(&inst, &sol, 0).gain, 0);
        assert_eq!(JobShopRelocateNeighbor::new(&inst, &sol, 2, 2).gain, 0);
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
            assert_eq!(n.gain, i64::from(s.objective) - i64::from(before));
            let mut counts = vec![0usize; inst.n_jobs];
            for &j in &s.operations {
                counts[j] += 1;
            }
            assert_eq!(counts, vec![inst.n_machines, inst.n_machines]);
        }
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

    /// A random instance with `n_jobs` jobs on `n_machines` machines and a
    /// shuffled sequence for it.
    fn random_case(
        n_jobs: usize,
        n_machines: usize,
        seed: u64,
    ) -> (JobShopScheduling, JobShopSolution) {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        let mut rng = SmallRng::seed_from_u64(seed);
        let jobs = (0..n_jobs)
            .map(|_| {
                let mut machines: Vec<usize> = (0..n_machines).collect();
                machines.shuffle(&mut rng);
                machines
                    .into_iter()
                    .map(|m| (m, rng.random_range(1..20)))
                    .collect()
            })
            .collect();
        let inst = JobShopScheduling::new("random".to_string(), n_machines, jobs);
        let sol = crate::search_state::ProblemTrait::new_solution(&inst, &mut rng);
        (inst, sol)
    }

    /// The scans that schedule only past the shared prefix list the same
    /// moves, in the same order, at the gains a full decode gives, on the
    /// serial path and on the parallel one.
    #[test]
    fn scans_price_every_move_as_a_full_decode() {
        for (n_jobs, n_machines) in [(6, 4), (20, 20)] {
            let (inst, sol) = random_case(n_jobs, n_machines, 11);
            let n = sol.operations.len();
            let decoded =
                |ops: &[usize]| i64::from(inst.compute_makespan(ops)) - i64::from(sol.objective);
            let swaps: Vec<_> = JobShopSwapNeighbor::iter(&inst, &sol)
                .map(|m| (m.i, m.gain))
                .collect();
            let expected: Vec<_> = (0..n - 1)
                .filter(|&i| sol.operations[i] != sol.operations[i + 1])
                .map(|i| {
                    let mut ops = sol.operations.clone();
                    ops.swap(i, i + 1);
                    (i, decoded(&ops))
                })
                .collect();
            assert_eq!(swaps, expected);
            if n < PARALLEL_ITER_MIN_OPS {
                let relocates: Vec<_> = JobShopRelocateNeighbor::iter(&inst, &sol)
                    .map(|m| (m.from, m.to, m.gain))
                    .collect();
                let expected: Vec<_> = (0..n)
                    .flat_map(|from| {
                        (0..n)
                            .filter(move |&to| to != from)
                            .map(move |to| (from, to))
                    })
                    .map(|(from, to)| {
                        let mut ops = sol.operations.clone();
                        relocate_in_place(&mut ops, from, to);
                        (from, to, decoded(&ops))
                    })
                    .collect();
                assert_eq!(relocates, expected);
            }
        }
    }
}
