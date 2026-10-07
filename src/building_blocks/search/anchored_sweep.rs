//! The sweep an anchored descent runs, shared by every problem that has one.
//!
//! A descent that repairs around the places a ruin just touched does two
//! things that have nothing to do with its moves. It widens the anchors by a
//! ring of their nearest partners, so a vertex displaced by the edit is
//! reconsidered too, and it sweeps that list first-improvement until a pass
//! finds nothing. Both [`vrp::ops::Descent`](crate::problem::vrp) and
//! [`AnchoredTourDescent`](crate::problem::tsp::AnchoredTourDescent) run
//! exactly this, with only the move set differing, so the list and the loop
//! live here and the move set comes in as a closure.

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// The sweep list and the marks that keep it free of duplicates, kept between
/// calls so a descent allocates nothing per iteration.
#[derive(Debug)]
pub struct AnchoredSweep {
    /// The vertices a pass visits, shuffled in place each time.
    order: Vec<usize>,
    /// Membership marks that keep `order` free of duplicates without sorting it.
    seen: Vec<bool>,
    /// How many partners of each anchor join the list.
    ring: usize,
}

impl Default for AnchoredSweep {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            seen: Vec::new(),
            ring: Self::DEFAULT_RING,
        }
    }
}

impl AnchoredSweep {
    /// Partners of an anchor that a sweep visits along with it, unless
    /// [`with_ring`](Self::with_ring) says otherwise.
    ///
    /// Deliberately far below the granularity of the candidate lists: the
    /// anchors of a large ruin, widened by a full candidate list of 20,
    /// already cover most of a mid-sized instance, and a sweep that touches
    /// everything is the full descent the caller was trying not to pay for.
    /// The nearest handful is where a displaced vertex actually lands.
    pub const DEFAULT_RING: usize = 5;

    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style: how many nearest partners of each anchor the sweep
    /// visits along with it. Defaults to [`DEFAULT_RING`](Self::DEFAULT_RING).
    /// Zero sweeps the anchors alone.
    pub fn with_ring(mut self, ring: usize) -> Self {
        self.ring = ring;
        self
    }

    /// The ring width in force.
    pub fn ring(&self) -> usize {
        self.ring
    }

    /// Sizes the marks for vertex ids below `n`. Cheap to call every time,
    /// since it only reallocates when `n` changes.
    pub fn ensure(&mut self, n: usize) {
        if self.seen.len() != n {
            self.seen = vec![false; n];
        }
    }

    /// Makes the sweep list exactly `all`, for a full descent.
    pub fn set_order(&mut self, all: impl IntoIterator<Item = usize>) {
        self.order.clear();
        self.order.extend(all);
    }

    /// Makes the sweep list the anchors plus the first `ring` partners of
    /// each, in that order and without repeats, dropping any vertex `keep`
    /// rejects (one not currently placed, say).
    ///
    /// # Panics
    ///
    /// Panics if a vertex id is at or beyond the `n` given to
    /// [`ensure`](Self::ensure).
    pub fn collect_around(
        &mut self,
        anchors: &[usize],
        neighbors: &[Vec<usize>],
        keep: impl Fn(usize) -> bool,
    ) {
        self.order.clear();
        for &u in anchors {
            for &v in std::iter::once(&u).chain(neighbors[u].iter().take(self.ring)) {
                if !self.seen[v] && keep(v) {
                    self.seen[v] = true;
                    self.order.push(v);
                }
            }
        }
        for &u in &self.order {
            self.seen[u] = false;
        }
    }

    /// Sweeps the list until a pass finds nothing, or `max_passes` are
    /// spent. Each pass shuffles the list and calls `improve` on every entry,
    /// which applies one improving move anchored there and says whether it
    /// found one.
    pub fn sweep(
        &mut self,
        rng: &mut SmallRng,
        max_passes: usize,
        mut improve: impl FnMut(usize) -> bool,
    ) {
        for _ in 0..max_passes {
            self.order.shuffle(rng);
            let mut improved = false;
            for &u in &self.order {
                if improve(u) {
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Anchors come first in their own order, each followed by its ring, and
    /// a vertex reached twice is listed once.
    #[test]
    fn collect_around_widens_without_repeats() {
        let neighbors = vec![vec![1, 2], vec![0, 2], vec![0, 1], vec![0]];
        let mut sweep = AnchoredSweep::new();
        sweep.ensure(4);
        sweep.collect_around(&[0, 3], &neighbors, |_| true);
        assert_eq!(sweep.order, vec![0, 1, 2, 3]);
        sweep.collect_around(&[3], &neighbors, |v| v != 0);
        assert_eq!(sweep.order, vec![3]);
        assert!(sweep.seen.iter().all(|&s| !s), "marks are reset after use");
    }

    /// The ring width bounds how far each anchor is widened. Zero is the
    /// anchors alone, one adds each anchor's nearest partner.
    #[test]
    fn with_ring_bounds_the_widening() {
        let neighbors = vec![vec![1, 2], vec![0, 2], vec![0, 1], vec![0]];
        let mut alone = AnchoredSweep::new().with_ring(0);
        alone.ensure(4);
        alone.collect_around(&[0, 3], &neighbors, |_| true);
        assert_eq!(alone.order, vec![0, 3]);

        let mut one = AnchoredSweep::new().with_ring(1);
        one.ensure(4);
        one.collect_around(&[3], &neighbors, |_| true);
        assert_eq!(one.order, vec![3, 0]);
    }

    /// A pass that improves nothing ends the sweep, whatever `max_passes`
    /// says, and every entry is visited once per pass.
    #[test]
    fn sweep_stops_at_the_first_quiet_pass() {
        let mut sweep = AnchoredSweep::new();
        sweep.set_order(0..3);
        let mut rng = SmallRng::seed_from_u64(1);
        let mut visits = 0;
        sweep.sweep(&mut rng, 10, |_| {
            visits += 1;
            visits <= 3
        });
        assert_eq!(visits, 6, "one improving pass, then one quiet pass");
    }
}
