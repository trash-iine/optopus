//! Heads and tails of a job sequence, the bookkeeping behind Taillard's
//! acceleration.
//!
//! A permutation flow shop sequence splits anywhere into a prefix `A` and a
//! suffix `B`, and its makespan is `max_i (head(A)_i + tail(B)_i)`. The head of
//! a prefix is when its last job leaves each machine, and the tail of a suffix
//! is how long each machine stays busy from the moment its first job starts
//! there. Neither depends on the other side of the split, so once both are
//! tabulated for every split point, inserting a job at any place or taking one
//! out is priced in O(m), and all `n + 1` insertion places in O(nm). That is
//! the speed-up of Taillard (1990) that NEH and Iterated Greedy rely on.

use super::problem::FlowShop;

/// Heads and tails of one sequence, flat and row-major by position.
///
/// `heads[k * m + i]` is when the job at position `k` leaves machine `i`.
/// `tails[k * m + i]` is the time from the start of the job at position `k`
/// on machine `i` to the end of the schedule, its own processing included.
#[derive(Debug, Clone, Default)]
pub(crate) struct HeadsTails {
    heads: Vec<u32>,
    tails: Vec<u32>,
    m: usize,
}

impl HeadsTails {
    /// Tabulates `sequence` from scratch in O(nm), reusing the buffers.
    pub(crate) fn rebuild(&mut self, prob: &FlowShop, sequence: &[usize]) {
        let m = prob.n_machines;
        self.m = m;
        self.heads.resize(sequence.len() * m, 0);
        self.tails.resize(sequence.len() * m, 0);
        for (k, &job) in sequence.iter().enumerate() {
            if k == 0 {
                self.heads[..m].fill(0);
            } else {
                self.heads.copy_within((k - 1) * m..k * m, k * m);
            }
            prob.advance(&mut self.heads[k * m..(k + 1) * m], job);
        }
        for (k, &job) in sequence.iter().enumerate().rev() {
            let mut next_machine = 0;
            for i in (0..m).rev() {
                let busy =
                    self.tail_from(k + 1, i).max(next_machine) + prob.processing_time(i, job);
                self.tails[k * m + i] = busy;
                next_machine = busy;
            }
        }
    }

    /// How many jobs the table covers.
    fn len(&self) -> usize {
        self.heads.len().checked_div(self.m).unwrap_or(0)
    }

    /// The makespan of the tabulated sequence.
    pub(crate) fn makespan(&self) -> u32 {
        self.heads.last().copied().unwrap_or(0)
    }

    /// The head of the prefix that ends just before `place`, machine `i`.
    fn head_before(&self, place: usize, i: usize) -> u32 {
        if place == 0 {
            0
        } else {
            self.heads[(place - 1) * self.m + i]
        }
    }

    /// The tail of the suffix that starts at `place`, machine `i`.
    fn tail_from(&self, place: usize, i: usize) -> u32 {
        self.tails.get(place * self.m + i).copied().unwrap_or(0)
    }

    /// The makespan once `job` is inserted at `place`, between the jobs at
    /// `place - 1` and `place`, in O(m).
    pub(crate) fn makespan_with(&self, prob: &FlowShop, place: usize, job: usize) -> u32 {
        let mut done = 0;
        let mut makespan = 0;
        for i in 0..self.m {
            done = done.max(self.head_before(place, i)) + prob.processing_time(i, job);
            makespan = makespan.max(done + self.tail_from(place, i));
        }
        makespan
    }

    /// The place where inserting `job` gives the smallest makespan, the
    /// earliest one on a tie, and that makespan, in O(nm).
    pub(crate) fn best_insertion(&self, prob: &FlowShop, job: usize) -> (usize, u32) {
        (0..=self.len())
            .map(|place| (place, self.makespan_with(prob, place, job)))
            .min_by_key(|&(_, makespan)| makespan)
            .expect("there is always at least one place")
    }

    /// The makespan once the job at position `k` is taken out, in O(m).
    pub(crate) fn makespan_without(&self, k: usize) -> u32 {
        (0..self.m)
            .map(|i| self.head_before(k, i) + self.tail_from(k + 1, i))
            .max()
            .unwrap_or(0)
    }

    /// The makespan once the jobs at positions `lo..=hi` are replaced by
    /// `middle`, in O((hi - lo + 1) m). `row` is scratch, so a caller pricing
    /// many replacements allocates it once.
    pub(crate) fn makespan_replacing(
        &self,
        prob: &FlowShop,
        lo: usize,
        hi: usize,
        middle: impl Iterator<Item = usize>,
        row: &mut Vec<u32>,
    ) -> u32 {
        row.clear();
        row.extend((0..self.m).map(|i| self.head_before(lo, i)));
        for job in middle {
            prob.advance(row, job);
        }
        row.iter()
            .enumerate()
            .map(|(i, &head)| head + self.tail_from(hi + 1, i))
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;

    fn random_instance(n: usize, m: usize, seed: u64) -> FlowShop {
        use rand::Rng;
        let mut rng = SmallRng::seed_from_u64(seed);
        let rows = (0..m)
            .map(|_| (0..n).map(|_| rng.random_range(1..100)).collect())
            .collect();
        FlowShop::new("random", rows)
    }

    /// Every O(m) price agrees with recomputing the edited sequence from
    /// scratch, at every place, on random sequences.
    #[test]
    fn prices_match_a_full_recompute() {
        let prob = random_instance(9, 4, 1);
        let mut rng = SmallRng::seed_from_u64(2);
        let mut ht = HeadsTails::default();
        for _ in 0..20 {
            let mut seq: Vec<usize> = (0..prob.n_jobs).collect();
            seq.shuffle(&mut rng);
            let job = seq.pop().unwrap();
            ht.rebuild(&prob, &seq);
            assert_eq!(ht.makespan(), prob.makespan(&seq));
            for place in 0..=seq.len() {
                let mut edited = seq.clone();
                edited.insert(place, job);
                assert_eq!(ht.makespan_with(&prob, place, job), prob.makespan(&edited));
            }
            let (place, best) = ht.best_insertion(&prob, job);
            assert_eq!(best, ht.makespan_with(&prob, place, job));
            assert!((0..place).all(|p| ht.makespan_with(&prob, p, job) > best));
            for k in 0..seq.len() {
                let mut edited = seq.clone();
                edited.remove(k);
                assert_eq!(ht.makespan_without(k), prob.makespan(&edited));
            }
            for lo in 0..seq.len() {
                for hi in lo..seq.len() {
                    let mut edited = seq.clone();
                    edited.swap(lo, hi);
                    let middle = edited[lo..=hi].iter().copied();
                    assert_eq!(
                        ht.makespan_replacing(&prob, lo, hi, middle, &mut Vec::new()),
                        prob.makespan(&edited)
                    );
                }
            }
        }
    }
}
