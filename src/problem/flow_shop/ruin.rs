//! The flow shop's answer to [`Ruinable`], with jobs as the elements and the one
//! sequence as the only container.
//!
//! This is the single-container case of the trait, the same as a tour. What
//! makes it cheap is that the partial keeps Taillard's heads and tails of its
//! sequence. The trait prices one place at a time, and recomputing the
//! schedule per place would make a scan over all places O(n²m). With the
//! table each place is O(m) and a whole scan O(nm), the cost Taillard's
//! acceleration gives NEH and Iterated Greedy. The table is rebuilt in O(nm)
//! after every edit, once per inserted job, which is what the scan costs
//! anyway.

use rand::rngs::SmallRng;

use super::problem::{FlowShop, FlowShopSolution};
use super::taillard::HeadsTails;
use crate::building_blocks::search::AnchoredSweep;
use crate::trait_defs::{LocalRepair, Ruinable};

/// Marks a job that is not currently in [`FlowShopPartial::sequence`].
const NOT_PLACED: usize = usize::MAX;

/// A sequence mid-ruin, with the jobs taken out of it.
///
/// The sequence, the position of every job in it, and its heads and tails are
/// kept in step by one private refresh, the only place any of them is
/// derived, and every edit calls it.
pub struct FlowShopPartial {
    /// The jobs currently placed, in order.
    sequence: Vec<usize>,
    /// Job to index in `sequence`, or [`NOT_PLACED`].
    pos: Vec<usize>,
    /// Heads and tails of `sequence`.
    table: HeadsTails,
}

impl FlowShopPartial {
    /// Re-derives the positions and the table from the sequence, in O(nm).
    fn refresh(&mut self, prob: &FlowShop) {
        self.pos.fill(NOT_PLACED);
        for (k, &job) in self.sequence.iter().enumerate() {
            self.pos[job] = k;
        }
        self.table.rebuild(prob, &self.sequence);
    }

    /// The makespan of the placed jobs.
    fn makespan(&self) -> u32 {
        self.table.makespan()
    }

    /// The place where `job` costs least, the earliest one on a tie, and the
    /// makespan it gives.
    fn best_place(&self, prob: &FlowShop, job: usize) -> (usize, u32) {
        (0..=self.sequence.len())
            .map(|place| (place, self.table.makespan_with(prob, place, job)))
            .min_by_key(|&(_, makespan)| makespan)
            .expect("there is always at least one place")
    }
}

impl Ruinable for FlowShop {
    type Element = usize;
    type Partial = FlowShopPartial;

    fn to_partial(&self, sol: &FlowShopSolution) -> FlowShopPartial {
        let mut partial = FlowShopPartial {
            sequence: sol.sequence.clone(),
            pos: vec![NOT_PLACED; self.n_jobs],
            table: HeadsTails::default(),
        };
        partial.refresh(self);
        partial
    }

    fn finish(&self, partial: &FlowShopPartial) -> FlowShopSolution {
        debug_assert_eq!(partial.sequence.len(), self.n_jobs, "every job is placed");
        FlowShopSolution {
            sequence: partial.sequence.clone(),
            objective: partial.makespan(),
        }
    }

    fn elements(&self, partial: &FlowShopPartial, out: &mut Vec<usize>) {
        out.clear();
        out.extend_from_slice(&partial.sequence);
    }

    fn num_elements(&self, partial: &FlowShopPartial) -> usize {
        partial.sequence.len()
    }

    fn remove_all(&self, partial: &mut FlowShopPartial, set: &[usize]) {
        for &job in set {
            partial.pos[job] = NOT_PLACED;
        }
        let pos = &partial.pos;
        partial.sequence.retain(|&job| pos[job] != NOT_PLACED);
        partial.refresh(self);
    }

    fn removal_gain(&self, partial: &FlowShopPartial, element: usize) -> f64 {
        let without = partial.table.makespan_without(partial.pos[element]);
        f64::from(partial.makespan()) - f64::from(without)
    }

    /// Two jobs are related when they load the machines alike, measured as
    /// the L1 distance between their processing times. Smaller is closer.
    fn relatedness(&self, a: usize, b: usize) -> f64 {
        (0..self.n_machines)
            .map(|i| {
                f64::from(
                    self.processing_time(i, a)
                        .abs_diff(self.processing_time(i, b)),
                )
            })
            .sum()
    }

    fn num_buckets(&self, _partial: &FlowShopPartial) -> usize {
        1
    }

    /// Before the first job, between any two, and after the last.
    fn num_places(&self, partial: &FlowShopPartial, _bucket: usize) -> usize {
        partial.sequence.len() + 1
    }

    fn insertion_cost(
        &self,
        partial: &FlowShopPartial,
        _bucket: usize,
        place: usize,
        element: usize,
    ) -> f64 {
        f64::from(partial.table.makespan_with(self, place, element)) - f64::from(partial.makespan())
    }

    fn insert(&self, partial: &mut FlowShopPartial, _bucket: usize, place: usize, element: usize) {
        partial.sequence.insert(place, element);
        partial.refresh(self);
    }

    fn partial_energy(&self, partial: &FlowShopPartial) -> f64 {
        f64::from(partial.makespan())
    }
}

/// The insertion descent of Ruiz and Stützle's Iterated Greedy.
///
/// Every job, in a random order, is taken out and put back where the makespan
/// is smallest, and the move is kept only if it shortens the schedule. Passes
/// repeat until one changes nothing, which is a local optimum of the
/// insertion neighborhood. A pass is O(n²m) with the heads and tails.
///
/// It sweeps the whole sequence and ignores the anchors it is handed. The
/// paper's local search is the full descent, and on a flow shop a job moved
/// anywhere changes the schedule of every job after it, so there is no
/// neighborhood of the re-inserted jobs to stay in.
#[derive(Debug, Default)]
pub struct FlowShopInsertionDescent {
    sweep: AnchoredSweep,
    max_passes: Option<usize>,
}

impl FlowShopInsertionDescent {
    /// A descent that runs to a local optimum.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style: stops after `max_passes` passes even if the last one
    /// still improved. Unbounded unless set.
    ///
    /// # Panics
    ///
    /// Panics if `max_passes` is zero.
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        assert!(max_passes >= 1, "max_passes must be at least 1");
        self.max_passes = Some(max_passes);
        self
    }
}

impl LocalRepair<FlowShop> for FlowShopInsertionDescent {
    fn repair_around(
        &mut self,
        prob: &FlowShop,
        partial: &mut FlowShopPartial,
        _anchors: &[usize],
        rng: &mut SmallRng,
    ) {
        if partial.sequence.len() < 2 {
            return;
        }
        self.sweep.set_order(partial.sequence.iter().copied());
        self.sweep
            .sweep(rng, self.max_passes.unwrap_or(usize::MAX), |job| {
                let before = partial.makespan();
                let from = partial.pos[job];
                partial.sequence.remove(from);
                partial.refresh(prob);
                let (place, makespan) = partial.best_place(prob, job);
                let improved = makespan < before;
                partial
                    .sequence
                    .insert(if improved { place } else { from }, job);
                partial.refresh(prob);
                improved
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    fn instance(n: usize, m: usize, seed: u64) -> FlowShop {
        let mut rng = SmallRng::seed_from_u64(seed);
        let rows = (0..m)
            .map(|_| (0..n).map(|_| rng.random_range(1..100)).collect())
            .collect();
        FlowShop::new("random", rows)
    }

    /// The trait's prices agree with recomputing the edited sequence.
    #[test]
    fn ruin_prices_match_a_full_recompute() {
        let fs = instance(8, 3, 3);
        let mut rng = SmallRng::seed_from_u64(4);
        let mut seq: Vec<usize> = (0..8).collect();
        seq.shuffle(&mut rng);
        let sol = fs.solution_from_sequence(seq.clone());
        let mut partial = fs.to_partial(&sol);
        assert_eq!(fs.partial_energy(&partial), f64::from(sol.objective));

        for &job in &seq {
            let k = seq.iter().position(|&j| j == job).unwrap();
            let mut without = seq.clone();
            without.remove(k);
            let expected = f64::from(sol.objective) - f64::from(fs.makespan(&without));
            assert_eq!(fs.removal_gain(&partial, job), expected);
        }

        fs.remove_all(&mut partial, &[seq[2], seq[5]]);
        let rest: Vec<usize> = seq
            .iter()
            .copied()
            .filter(|&j| j != seq[2] && j != seq[5])
            .collect();
        assert_eq!(fs.partial_energy(&partial), f64::from(fs.makespan(&rest)));
        for place in 0..fs.num_places(&partial, 0) {
            let mut edited = rest.clone();
            edited.insert(place, seq[2]);
            let expected = f64::from(fs.makespan(&edited)) - f64::from(fs.makespan(&rest));
            assert_eq!(fs.insertion_cost(&partial, 0, place, seq[2]), expected);
        }

        fs.insert(&mut partial, 0, 1, seq[2]);
        fs.insert(&mut partial, 0, 4, seq[5]);
        let done = fs.finish(&partial);
        assert_eq!(done.objective, fs.makespan(&done.sequence));
    }

    /// The descent never worsens a sequence, and what it returns is a local
    /// optimum, so no single insertion improves it.
    #[test]
    fn descent_reaches_an_insertion_local_optimum() {
        use crate::search_state::MoveToNeighbor;

        let fs = instance(12, 5, 7);
        let mut rng = SmallRng::seed_from_u64(8);
        let start = crate::search_state::ProblemTrait::new_solution(&fs, &mut rng);
        let mut partial = fs.to_partial(&start);
        FlowShopInsertionDescent::new().repair_around(&fs, &mut partial, &[], &mut rng);
        let done = fs.finish(&partial);
        assert!(done.objective <= start.objective);
        assert_eq!(done.objective, fs.makespan(&done.sequence));
        assert!(
            super::super::FlowShopInsertNeighbor::iter(&fs, &done).all(|m| m.gain >= 0.0),
            "some insertion still improves the descent's result"
        );
    }
}
