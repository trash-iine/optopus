//! Iterative beam search over any [`Ruinable`] problem.

use rand::seq::SliceRandom;

use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluate, Ruinable};

/// One way to place the next element: in `parent`, at `(bucket, place)`,
/// leaving a partial of energy `energy`.
#[derive(Clone, Copy)]
struct Child {
    energy: f64,
    parent: usize,
    bucket: usize,
    place: usize,
}

impl Child {
    /// The lower energy first, and a fixed order among equals so a pass
    /// replays under a seed.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.energy
            .total_cmp(&other.energy)
            .then(self.parent.cmp(&other.parent))
            .then(self.bucket.cmp(&other.bucket))
            .then(self.place.cmp(&other.place))
    }
}

/// Iterative beam search.
///
/// A pass builds a solution from nothing, one element at a time. It keeps
/// the `width` best partial solutions, the beam. For the next element, every
/// placement in every partial of the beam is priced, and the `width`
/// cheapest results are the next beam. When every element is placed, the best
/// of the beam is the solution of the pass.
///
/// Each `run_once` is one pass. The first runs at `initial_width`, and each
/// later one at `growth` times the width before it, so a run starts with a
/// greedy construction and spends its later time on wider and wider searches.
/// The elements are placed in an order drawn anew for each pass.
///
/// What is taken from the flow shop search of the second reference is the
/// restart at a width that grows geometrically from one. Its branching is
/// not. There a child appends any job not yet scheduled to an end of the
/// sequence and is ranked by a bound with the idle time, which only a flow
/// shop has. Here the next element is fixed by the drawn order, a child
/// inserts it anywhere, and the rank is the energy of the partial solution.
///
/// A pass is not interrupted by an iteration count, and it is by a time
/// limit, which ends it between two elements with its beam discarded.
///
/// # What the problem supplies
///
/// [`Ruinable`], whose [`insertion_cost`](Ruinable::insertion_cost) prices a
/// placement and whose [`partial_energy`](Ruinable::partial_energy) ranks the
/// partial solutions, and a [`Partial`](Ruinable::Partial) that is `Clone`,
/// since two placements in the same partial each need their own copy.
///
/// A pass prices `width · n` placements per element, so its cost grows with
/// the square of the number of elements, and it copies up to `width`
/// partials per element.
///
/// # Example
///
/// ```
/// use optopus::prelude::*;
///
/// let coords = (0..12)
///     .map(|i| {
///         let theta = std::f64::consts::TAU * i as f64 / 12.0;
///         (theta.cos(), theta.sin())
///     })
///     .collect();
/// let tsp = Tsp::new("ring".into(), coords);
/// let mut state = SearchState::new_with_seed(&tsp, 1);
/// IterativeBeamSearch::<Tsp>::new(StopCondition::iterations(4))
///     .run(&mut state)?;
/// assert!(state.best_solution.objective < 6.3);
/// # Ok::<(), optopus::error::OptError>(())
/// ```
///
/// # References
///
/// - Ow, P. S. and Morton, T. E. "Filtered Beam Search in Scheduling."
///   *International Journal of Production Research*, 26(1), 35-62, 1988.
/// - Libralesso, L., Focke, P. A., Secardin, A. and Jost, V. "Iterative beam
///   search algorithms for the permutation flowshop." *European Journal of
///   Operational Research*, 301(1), 217-234, 2022.
pub struct IterativeBeamSearch<P: Ruinable> {
    stop_condition: StopCondition,
    initial_width: usize,
    growth: f64,
    /// The width of the next pass.
    width: usize,
    /// The elements, in the order the current pass places them.
    order: Vec<P::Element>,
    /// The cheapest placements of the current element found so far, at most
    /// twice the width, then the ones kept.
    children: Vec<Child>,
}

impl<P: Ruinable> IterativeBeamSearch<P> {
    /// The width of the first pass, unless
    /// [`with_initial_width`](Self::with_initial_width) says otherwise.
    pub const INITIAL_WIDTH: usize = 1;
    /// The factor the width grows by from one pass to the next, unless
    /// [`with_growth`](Self::with_growth) says otherwise.
    pub const GROWTH: f64 = 2.0;

    /// A search that starts greedy and doubles its width each pass.
    pub fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            initial_width: Self::INITIAL_WIDTH,
            growth: Self::GROWTH,
            width: Self::INITIAL_WIDTH,
            order: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Builder-style: the width of the first pass.
    ///
    /// # Panics
    ///
    /// Panics if `width` is zero.
    pub fn with_initial_width(mut self, width: usize) -> Self {
        assert!(width >= 1, "initial_width must be at least 1");
        self.initial_width = width;
        self.width = width;
        self
    }

    /// Builder-style: the factor the width grows by from one pass to the
    /// next. One keeps every pass at the initial width, each with a new
    /// order of the elements.
    ///
    /// # Panics
    ///
    /// Panics if `growth` is below one or not finite.
    pub fn with_growth(mut self, growth: f64) -> Self {
        assert!(
            growth >= 1.0 && growth.is_finite(),
            "growth must be at least 1"
        );
        self.growth = growth;
        self
    }
}

/// Keeps the `width` cheapest of `children` and returns the dearest of them.
fn keep_cheapest(children: &mut Vec<Child>, width: usize) -> Child {
    children.select_nth_unstable_by(width - 1, Child::cmp);
    children.truncate(width);
    children[width - 1]
}

impl<P> Heuristic<P> for IterativeBeamSearch<P>
where
    P: Ruinable,
    P::Partial: Clone,
    P::Solution: Evaluate,
{
    fn clear(&mut self) {
        self.width = self.initial_width;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let prob: &P = state.instance;
        let mut root = prob.to_partial(&state.solution);
        prob.elements(&root, &mut self.order);
        prob.remove_all(&mut root, &self.order);
        self.order.shuffle(&mut state.rng);

        let mut beam = vec![root];
        // Grown by what is pushed, not sized by the width, which outgrows
        // any allocation once the passes are fast enough.
        let mut next: Vec<P::Partial> = Vec::new();
        for &element in &self.order {
            if self.stop_condition.is_done(state) {
                return Ok(());
            }
            self.children.clear();
            // Once the buffer has been cut to the width, a placement dearer
            // than the dearest kept can never be kept.
            let mut cutoff: Option<Child> = None;
            for (parent, partial) in beam.iter().enumerate() {
                let energy = prob.partial_energy(partial);
                for bucket in 0..prob.num_buckets(partial) {
                    for place in 0..prob.num_places(partial, bucket) {
                        let cost = prob.insertion_cost(partial, bucket, place, element);
                        if !cost.is_finite() {
                            continue;
                        }
                        let child = Child {
                            energy: energy + cost,
                            parent,
                            bucket,
                            place,
                        };
                        if cutoff.is_some_and(|c| child.cmp(&c).is_gt()) {
                            continue;
                        }
                        self.children.push(child);
                        if self.children.len() >= self.width.saturating_mul(2) {
                            cutoff = Some(keep_cheapest(&mut self.children, self.width));
                        }
                    }
                }
            }
            if self.children.is_empty() {
                return Err(OptError::InvalidState(
                    "IterativeBeamSearch: an element has no finite placement, \
                     see Ruinable::insertion_cost"
                        .into(),
                ));
            }
            if self.children.len() > self.width {
                keep_cheapest(&mut self.children, self.width);
            }
            // Grouped by parent, so the last child of a parent takes its
            // partial and only the others copy it.
            self.children
                .sort_unstable_by(|a, b| a.parent.cmp(&b.parent).then_with(|| a.cmp(b)));
            let mut parents = beam.drain(..).enumerate();
            for group in self.children.chunk_by(|a, b| a.parent == b.parent) {
                let (_, parent) = parents
                    .find(|&(i, _)| i == group[0].parent)
                    .expect("the children are sorted by parent");
                for (mut partial, child) in std::iter::repeat_n(parent, group.len()).zip(group) {
                    prob.insert(&mut partial, child.bucket, child.place, element);
                    next.push(partial);
                }
            }
            drop(parents);
            std::mem::swap(&mut beam, &mut next);
        }

        let (_, best) = beam
            .iter()
            .map(|partial| (prob.partial_energy(partial), partial))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("the beam holds at least the root");
        state.solution = prob.finish(best);
        state.iteration += 1;
        state.n_accepted += 1;
        state.update_best();
        self.width = (self.width as f64 * self.growth).ceil() as usize;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{BinPacking, FlowShop, Tsp, Vrp};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn scattered_tsp(n: usize, seed: u64) -> Tsp {
        let mut rng = SmallRng::seed_from_u64(seed);
        let coords = (0..n)
            .map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)))
            .collect();
        Tsp::new("scattered".into(), coords)
    }

    /// The width doubles pass by pass, and a run replays under a seed.
    #[test]
    fn the_width_grows_and_a_run_replays() {
        let tsp = scattered_tsp(40, 1);
        let run = || {
            let mut state = SearchState::new_with_seed(&tsp, 5);
            let mut search = IterativeBeamSearch::<Tsp>::new(StopCondition::iterations(5));
            search.run(&mut state).unwrap();
            assert_eq!(search.width, 32);
            let best = &state.best_solution;
            assert_eq!(
                tsp.calculate_tour_length(&best.tour).unwrap(),
                best.objective
            );
            best.objective
        };
        assert_eq!(run(), run());
    }

    /// With the order of the elements held fixed, a wider beam contains the
    /// greedy path only by chance, but over several orders the wider search
    /// is the better one on average.
    #[test]
    fn a_wide_beam_beats_the_greedy_construction_on_average() {
        let tsp = scattered_tsp(60, 2);
        let average = |width: usize| {
            let mut total = 0.0;
            for seed in 0..10 {
                let mut state = SearchState::new_with_seed(&tsp, seed);
                IterativeBeamSearch::<Tsp>::new(StopCondition::iterations(1))
                    .with_initial_width(width)
                    .run(&mut state)
                    .unwrap();
                total += state.best_solution.objective;
            }
            total / 10.0
        };
        assert!(average(16) < average(1));
    }

    #[test]
    fn it_builds_valid_solutions_of_the_other_ruinable_problems() {
        let mut rng = SmallRng::seed_from_u64(1);
        let sizes: Vec<_> = (0..40).map(|_| rng.random_range(20..100)).collect();
        let bp = BinPacking::new("random", 150, sizes);
        let mut state = SearchState::new_with_seed(&bp, 2);
        IterativeBeamSearch::<BinPacking>::new(StopCondition::iterations(4))
            .run(&mut state)
            .unwrap();
        assert!(state.best_solution.loads.iter().all(|&l| l <= bp.capacity));

        let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta001.txt").unwrap();
        let mut state = SearchState::new_with_seed(&fs, 3);
        IterativeBeamSearch::<FlowShop>::new(StopCondition::iterations(4))
            .run(&mut state)
            .unwrap();
        let best = &state.best_solution;
        assert_eq!(best.objective, fs.makespan(&best.sequence));

        let mut coords = vec![(0.0, 0.0)];
        let mut demands = vec![0];
        for i in 0..12 {
            let theta = std::f64::consts::TAU * i as f64 / 12.0;
            coords.push((theta.cos(), theta.sin()));
            demands.push(1);
        }
        let vrp = Vrp::new("ring", coords, demands, 3, 4);
        let mut state = SearchState::new_with_seed(&vrp, 4);
        IterativeBeamSearch::<Vrp>::new(StopCondition::iterations(4))
            .run(&mut state)
            .unwrap();
        assert_eq!(state.best_solution.overload, 0);
        vrp.validate_routes(&state.best_solution.routes).unwrap();
    }

    /// `clear` starts the widths over, which is what a restart or a second
    /// `run` on the same value relies on.
    #[test]
    fn clear_starts_the_widths_over() {
        let tsp = scattered_tsp(20, 3);
        let mut search = IterativeBeamSearch::<Tsp>::new(StopCondition::iterations(3))
            .with_initial_width(2)
            .with_growth(3.0);
        let mut state = SearchState::new_with_seed(&tsp, 1);
        search.run(&mut state).unwrap();
        assert_eq!(search.width, 54);
        search.clear();
        assert_eq!(search.width, 2);
    }

    /// On a tour of a few cities every pass is quick, so the width doubles
    /// past anything that could be allocated, and the passes still run.
    #[test]
    fn the_width_may_outgrow_memory() {
        let tsp = scattered_tsp(5, 4);
        let mut state = SearchState::new_with_seed(&tsp, 1);
        let mut search = IterativeBeamSearch::<Tsp>::new(StopCondition::iterations(80));
        search.run(&mut state).unwrap();
        assert_eq!(search.width, usize::MAX);
    }

    #[test]
    #[should_panic(expected = "initial_width must be at least 1")]
    fn a_zero_width_is_rejected() {
        let _ = IterativeBeamSearch::<Tsp>::new(StopCondition::iterations(1)).with_initial_width(0);
    }

    #[test]
    #[should_panic(expected = "growth must be at least 1")]
    fn a_shrinking_growth_is_rejected() {
        let _ = IterativeBeamSearch::<Tsp>::new(StopCondition::iterations(1)).with_growth(0.5);
    }
}
