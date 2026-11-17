//! Adaptive Iterated Local Search for vehicle routing, the variant for large
//! instances.

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::Vrp;
use crate::problem::vrp::ops::build_neighbor_lists;
use crate::problem::vrp::{AnchoredRouteDescent, RouteAdjacency, VrpPartial};
use crate::search_state::SearchState;
use crate::trait_defs::{LocalRepair, Rankable, Ruinable};

/// An average that is the plain mean of its first `window` values and an
/// exponential one of weight `1 / window` after that.
#[derive(Debug, Clone)]
struct MovingMean {
    window: u64,
    seen: u64,
    value: f64,
}

impl MovingMean {
    fn new(window: u64) -> Self {
        Self {
            window,
            seen: 0,
            value: 0.0,
        }
    }

    fn push(&mut self, x: f64) {
        self.seen += 1;
        if self.seen < self.window {
            self.value = (self.value * (self.seen - 1) as f64 + x) / self.seen as f64;
        } else {
            let weight = 1.0 / self.window as f64;
            self.value = self.value * (1.0 - weight) + x * weight;
        }
    }
}

/// Which customers a perturbation takes out.
#[derive(Debug, Clone, Copy)]
enum Removal {
    /// The customers nearest one drawn customer.
    Concentric,
    /// Runs of consecutive customers, each following one drawn customer on
    /// its route.
    Sequential,
}

/// Where a removed customer goes back.
#[derive(Debug, Clone, Copy)]
enum Insertion {
    /// Beside its nearest placed customer.
    Distance,
    /// Beside whichever of its nearest placed customers costs the least.
    Cost,
}

/// A removal with the number of customers it takes, which follows how far
/// its perturbations have been ending from the reference solution.
#[derive(Debug, Clone)]
struct Operator {
    removal: Removal,
    /// The perturbation degree, the number of customers to remove.
    degree: f64,
    /// How far from the reference the local search has been ending after
    /// this removal.
    reached: MovingMean,
    uses: u64,
}

/// Adaptive Iterated Local Search, the variant for large instances.
///
/// Every iteration perturbs a reference solution, descends, and decides
/// whether the result becomes the new reference. Two things adapt while it
/// runs, how many customers a perturbation removes and how good a result has
/// to be to replace the reference, and both tighten toward the end of the
/// run.
///
/// # The perturbation
///
/// A removal and an insertion are drawn. The concentric removal takes the
/// customers nearest one drawn customer, and the sequential removal takes
/// runs of consecutive customers that follow a drawn customer on its route.
/// The removed customers go back in a random order. Insertion by distance
/// puts each beside its nearest placed customer, and insertion by cost beside
/// whichever of its `neighbors` nearest placed customers makes the smallest
/// detour. Neither looks at capacity.
///
/// The descent that follows is the granular route descent anchored at the
/// customers that were moved, run to a local optimum. It prices an overload
/// at the penalty [`Vrp`] charges, which outweighs any distance, so it
/// restores feasibility first and shortens the routes after.
///
/// # The perturbation degree
///
/// Each removal keeps its own degree, the number of customers it takes.
/// The distance between a result and the reference is the number of the
/// result's edges the reference does not have. Every `gamma` uses of a
/// removal, its degree is multiplied by the ideal distance over the distance
/// its results have been ending at, so a removal whose results fall back
/// onto the reference removes more, and one whose results land far away
/// removes less. The ideal distance itself falls geometrically from
/// `distance_max` to `distance_min` over the run.
///
/// # The acceptance
///
/// A feasible result of cost `f` becomes the reference when
/// `f ≤ low + eta · (mean − low)`, where `mean` is a moving average of the
/// costs the descent has been ending at and `low` the best of them over the
/// last `gamma` to `2 · gamma` iterations. `eta` falls geometrically from one
/// to `eta_min` over the run, so the search accepts an average result at the
/// start and nearly only a recent best at the end.
///
/// # The length of the run
///
/// Both schedules need the number of iterations the run will make. It is, in
/// this order, the one set by
/// [`with_expected_iterations`](Self::with_expected_iterations), the stop
/// condition's iteration limit, or an estimate from the share of the stop
/// condition's time limit that has passed, renewed every `gamma` iterations.
/// A run with none of the three is an error.
///
/// # Example
///
/// ```
/// use optopus::prelude::*;
///
/// let mut coords = vec![(0.0, 0.0)];
/// let mut demands = vec![0];
/// for i in 0..24 {
///     let theta = std::f64::consts::TAU * i as f64 / 24.0;
///     coords.push((10.0 * theta.cos(), 10.0 * theta.sin()));
///     demands.push(1);
/// }
/// let vrp = Vrp::new("ring", coords, demands, 6, 5);
/// let mut state = SearchState::new_with_seed(&vrp, 1);
/// AdaptiveIteratedLocalSearchForVrp::new(StopCondition::iterations(300))
///     .run(&mut state)?;
/// assert!(state.best_solution.is_feasible());
/// # Ok::<(), optopus::error::OptError>(())
/// ```
///
/// # References
///
/// - Máximo, V. R., Cordeau, J.-F. and Nascimento, M. C. V. "AILS-II: An
///   Adaptive Iterated Local Search Heuristic for the Large-Scale Capacitated
///   Vehicle Routing Problem." *INFORMS Journal on Computing*, 36(4),
///   974-986, 2024.
pub struct AdaptiveIteratedLocalSearch {
    stop_condition: StopCondition,
    gamma: u64,
    distance_min: f64,
    distance_max: f64,
    eta_min: f64,
    neighbors: usize,
    expected_iterations: Option<u64>,

    operators: [Operator; 2],
    ideal_distance: f64,
    eta: f64,
    /// The factors the ideal distance and `eta` are multiplied by each
    /// iteration, from the current estimate of the run's length.
    distance_decay: f64,
    eta_decay: f64,
    costs: MovingMean,
    /// The best cost since the window before last opened, and since the last
    /// one did.
    low: f64,
    low_next: f64,
    iterations: u64,
    started: bool,

    descent: AnchoredRouteDescent,
    /// For each customer, its nearest customers and how many of them are
    /// nearer than the depot.
    nearest: Vec<Vec<usize>>,
    depot_rank: Vec<usize>,
    reference: Option<RouteAdjacency>,
    removed: Vec<usize>,
    absent: Vec<bool>,
    everyone: Vec<usize>,
}

impl AdaptiveIteratedLocalSearch {
    /// Iterations between two adjustments of a removal's degree, and the
    /// window of the two averages, unless [`with_gamma`](Self::with_gamma)
    /// says otherwise.
    pub const GAMMA: u64 = 30;
    /// The distance from the reference a result should end at, at the end
    /// and at the start of the run, unless
    /// [`with_distances`](Self::with_distances) says otherwise.
    pub const DISTANCE_MIN: f64 = 15.0;
    pub const DISTANCE_MAX: f64 = 30.0;
    /// What `eta` falls to, unless [`with_eta_min`](Self::with_eta_min) says
    /// otherwise.
    pub const ETA_MIN: f64 = 0.01;
    /// The nearest customers an insertion by cost and a move of the descent
    /// consider, unless [`with_neighbors`](Self::with_neighbors) says
    /// otherwise.
    pub const NEIGHBORS: usize = 40;
    /// The nearest customers kept per customer, which bounds a concentric
    /// removal.
    const KEPT: usize = 100;

    /// A search with the published parameters.
    pub fn new(stop_condition: StopCondition) -> Self {
        let operator = |removal| Operator {
            removal,
            degree: Self::DISTANCE_MAX,
            reached: MovingMean::new(Self::GAMMA),
            uses: 0,
        };
        Self {
            stop_condition,
            gamma: Self::GAMMA,
            distance_min: Self::DISTANCE_MIN,
            distance_max: Self::DISTANCE_MAX,
            eta_min: Self::ETA_MIN,
            neighbors: Self::NEIGHBORS,
            expected_iterations: None,
            operators: [operator(Removal::Sequential), operator(Removal::Concentric)],
            ideal_distance: Self::DISTANCE_MAX,
            eta: 1.0,
            distance_decay: 1.0,
            eta_decay: 1.0,
            costs: MovingMean::new(Self::GAMMA),
            low: f64::INFINITY,
            low_next: f64::INFINITY,
            iterations: 0,
            started: false,
            descent: Self::descent(Self::NEIGHBORS),
            nearest: Vec::new(),
            depot_rank: Vec::new(),
            reference: None,
            removed: Vec::new(),
            absent: Vec::new(),
            everyone: Vec::new(),
        }
    }

    /// The descent of an iteration, run until a pass changes nothing.
    fn descent(neighbors: usize) -> AnchoredRouteDescent {
        AnchoredRouteDescent::new()
            .with_granularity(neighbors)
            .with_max_passes(usize::MAX)
    }

    /// Builder-style: the iterations between two adjustments of a removal's
    /// degree, which is also the window of the two averages.
    ///
    /// # Panics
    ///
    /// Panics if `gamma` is zero.
    pub fn with_gamma(mut self, gamma: u64) -> Self {
        assert!(gamma >= 1, "gamma must be at least 1");
        self.gamma = gamma;
        self
    }

    /// Builder-style: the distance from the reference a result should end
    /// at, at the end and at the start of the run.
    ///
    /// # Panics
    ///
    /// Panics unless `0 < min <= max`.
    pub fn with_distances(mut self, min: f64, max: f64) -> Self {
        assert!(
            min > 0.0 && min <= max,
            "distances must satisfy 0 < min <= max, got ({min}, {max})"
        );
        self.distance_min = min;
        self.distance_max = max;
        self
    }

    /// Builder-style: what `eta` falls to by the end of the run.
    ///
    /// # Panics
    ///
    /// Panics if `eta_min` is outside `(0, 1]`.
    pub fn with_eta_min(mut self, eta_min: f64) -> Self {
        assert!(eta_min > 0.0 && eta_min <= 1.0, "eta_min must be in (0, 1]");
        self.eta_min = eta_min;
        self
    }

    /// Builder-style: how many nearest customers an insertion by cost and a
    /// move of the descent consider.
    ///
    /// # Panics
    ///
    /// Panics if `neighbors` is zero.
    pub fn with_neighbors(mut self, neighbors: usize) -> Self {
        assert!(neighbors >= 1, "neighbors must be at least 1");
        self.neighbors = neighbors;
        self.descent = Self::descent(neighbors);
        self
    }

    /// Builder-style: the number of iterations the two schedules are laid
    /// out over, for a run whose stop condition does not say.
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is zero.
    pub fn with_expected_iterations(mut self, iterations: u64) -> Self {
        assert!(iterations >= 1, "expected_iterations must be at least 1");
        self.expected_iterations = Some(iterations);
        self
    }

    /// Whether anything says how long the run is.
    fn has_a_length(&self) -> bool {
        self.expected_iterations.is_some()
            || self.stop_condition.max_iteration.is_some()
            || self.stop_condition.max_duration.is_some()
    }

    /// Puts every schedule back to the start of a run.
    fn rewind(&mut self) {
        for op in &mut self.operators {
            op.degree = self.distance_max;
            op.reached = MovingMean::new(self.gamma);
            op.uses = 0;
        }
        self.ideal_distance = self.distance_max;
        self.eta = 1.0;
        self.distance_decay = 1.0;
        self.eta_decay = 1.0;
        self.costs = MovingMean::new(self.gamma);
        self.low = f64::INFINITY;
        self.low_next = f64::INFINITY;
        self.iterations = 0;
        self.started = false;
        self.reference = None;
    }

    /// Renews the two decay factors from the number of iterations the run is
    /// now expected to make.
    fn renew_decays(&mut self, state: &SearchState<'_, Vrp>) {
        let total = match (
            self.expected_iterations,
            self.stop_condition.max_iteration,
            self.stop_condition.max_duration,
        ) {
            (Some(total), _, _) | (None, Some(total), _) => total as f64,
            (None, None, Some(limit)) => {
                let share = state.duration().as_secs_f64() / limit.as_secs_f64();
                if share <= 0.0 {
                    return;
                }
                self.iterations as f64 / share
            }
            (None, None, None) => return,
        };
        if total >= 1.0 {
            self.distance_decay = (self.distance_min / self.distance_max).powf(1.0 / total);
            self.eta_decay = self.eta_min.powf(1.0 / total);
        }
    }

    /// Reads the instance and takes the incumbent to a local optimum, which
    /// becomes the first reference.
    fn start(&mut self, state: &mut SearchState<'_, Vrp>) {
        let prob: &Vrp = state.instance;
        let n = prob.get_n();
        self.nearest = build_neighbor_lists(prob, Self::KEPT);
        self.depot_rank = (0..=n)
            .map(|c| {
                let to_depot = prob.distance(c, 0);
                self.nearest[c]
                    .iter()
                    .position(|&o| prob.distance(c, o) >= to_depot)
                    .unwrap_or(usize::MAX)
            })
            .collect();
        self.absent = vec![false; n + 1];
        self.everyone = (1..=n).collect();
        if self.stop_condition.max_iteration.is_some() || self.expected_iterations.is_some() {
            self.renew_decays(state);
        }

        let mut partial = prob.to_partial(&state.solution);
        self.descent
            .repair_around(prob, &mut partial, &self.everyone, &mut state.rng);
        state.solution = prob.finish(&partial);
        self.reference = Some(RouteAdjacency::from_routes(n, &state.solution.routes));
        self.started = true;
        state.iteration += 1;
        state.n_accepted += 1;
        state.update_best();
    }

    /// Takes customers out of `partial` by `removal`, about `degree` of
    /// them, leaving them in `self.removed`.
    fn remove(
        &mut self,
        prob: &Vrp,
        partial: &mut VrpPartial,
        removal: Removal,
        degree: f64,
        rng: &mut SmallRng,
    ) {
        let n = prob.get_n();
        self.removed.clear();
        match removal {
            Removal::Concentric => {
                let center = rng.random_range(1..=n);
                let count = (degree.ceil() as usize).min(self.nearest[center].len());
                self.removed
                    .extend_from_slice(&self.nearest[center][..count]);
                prob.remove_all(partial, &self.removed);
            }
            Removal::Sequential => {
                let wanted = (degree as usize).clamp(1, n);
                while self.removed.len() < wanted {
                    let anchor = loop {
                        let c = rng.random_range(1..=n);
                        if !self.absent[c] {
                            break c;
                        }
                    };
                    // The customers after the anchor, around the route and
                    // back to the anchor itself.
                    let (r, pos) = partial.locate(anchor);
                    let route = &partial.routes()[r];
                    let take = (wanted - self.removed.len()).min(route.len());
                    let first = self.removed.len();
                    self.removed
                        .extend((1..=take).map(|k| route[(pos + k) % route.len()]));
                    for &c in &self.removed[first..] {
                        self.absent[c] = true;
                    }
                    prob.remove_all(partial, &self.removed[first..]);
                }
            }
        }
        for &c in &self.removed {
            self.absent[c] = true;
        }
    }

    /// Where `c` goes back, as a route and an index in it.
    ///
    /// The customers of `c`'s nearest list that are placed are walked nearest
    /// first, up to `reach` of them, and the one after which `c` makes the
    /// smallest detour is kept. The depot takes its turn where it falls in
    /// that order, standing for the start of every route. `c` then goes
    /// before or after the one kept, whichever is cheaper.
    fn place(&self, prob: &Vrp, partial: &VrpPartial, c: usize, reach: usize) -> (usize, usize) {
        let routes = partial.routes();
        let detour =
            |a: usize, b: usize| prob.distance(a, c) + prob.distance(c, b) - prob.distance(a, b);
        let after = |r: usize, pos: usize| {
            let route = &routes[r];
            let next = route.get(pos + 1).copied().unwrap_or(0);
            detour(route[pos], next)
        };

        // (detour, route, index of the neighbor in it, or `None` for the
        // depot at the start of the route)
        let mut best: Option<(f64, usize, Option<usize>)> = None;
        let starts = |best: &mut Option<(f64, usize, Option<usize>)>| {
            let mut any = false;
            for (r, route) in routes.iter().enumerate() {
                // An idle vehicle stands for every idle one of its type.
                if prob.num_places(partial, r) == 0 {
                    continue;
                }
                any = true;
                let cost = detour(0, route.first().copied().unwrap_or(0));
                if best.is_none_or(|(b, _, _)| cost < b) {
                    *best = Some((cost, r, None));
                }
            }
            any
        };
        let mut seen = 0;
        for (rank, &o) in self.nearest[c].iter().enumerate() {
            if seen >= reach {
                break;
            }
            if rank == self.depot_rank[c] && starts(&mut best) {
                seen += 1;
                if seen >= reach {
                    break;
                }
            }
            if self.absent[o] {
                continue;
            }
            seen += 1;
            let (r, pos) = partial.locate(o);
            let cost = after(r, pos);
            if best.is_none_or(|(b, _, _)| cost < b) {
                best = Some((cost, r, Some(pos)));
            }
        }
        if best.is_none() {
            // Nothing near `c` is placed, so every place is priced.
            starts(&mut best);
            for (r, route) in routes.iter().enumerate() {
                for pos in 0..route.len() {
                    let cost = after(r, pos);
                    if best.is_none_or(|(b, _, _)| cost < b) {
                        best = Some((cost, r, Some(pos)));
                    }
                }
            }
        }
        let (cost, r, at) = best.expect("a solution has at least one route to start");
        let route = &routes[r];
        match at {
            // After the depot at the start, or before it at the end.
            None => {
                let at_end = detour(route.last().copied().unwrap_or(0), 0);
                if cost < at_end {
                    (r, 0)
                } else {
                    (r, route.len())
                }
            }
            Some(pos) => {
                let prev = if pos == 0 { 0 } else { route[pos - 1] };
                let before = detour(prev, route[pos]);
                if cost < before {
                    (r, pos + 1)
                } else {
                    (r, pos)
                }
            }
        }
    }

    /// One iteration: perturb the reference, descend, adapt and decide.
    fn step(&mut self, state: &mut SearchState<'_, Vrp>) {
        let prob: &Vrp = state.instance;
        let n = prob.get_n();
        let which = state.rng.random_range(0..self.operators.len());
        let removal = self.operators[which].removal;
        let degree = self.operators[which]
            .degree
            .clamp(1.0, n.saturating_sub(2).max(1) as f64);
        let insertion = if state.rng.random_bool(0.5) {
            Insertion::Distance
        } else {
            Insertion::Cost
        };

        let mut partial = prob.to_partial(&state.solution);
        self.remove(prob, &mut partial, removal, degree, &mut state.rng);
        self.removed.shuffle(&mut state.rng);
        let reach = match insertion {
            Insertion::Distance => 1,
            Insertion::Cost => self.neighbors,
        };
        for k in 0..self.removed.len() {
            let c = self.removed[k];
            let (r, place) = self.place(prob, &partial, c, reach);
            prob.insert(&mut partial, r, place, c);
            self.absent[c] = false;
        }
        self.descent
            .repair_around(prob, &mut partial, &self.removed, &mut state.rng);
        let result = prob.finish(&partial);

        let reference = self.reference.as_ref().expect("the search has started");
        let adjacency = RouteAdjacency::from_routes(n, &result.routes);
        let distance = adjacency.broken_pairs_from(reference) as f64;

        self.iterations += 1;
        if self.iterations.is_multiple_of(self.gamma) {
            self.renew_decays(state);
        }
        self.ideal_distance =
            (self.ideal_distance * self.distance_decay).clamp(self.distance_min, self.distance_max);

        let op = &mut self.operators[which];
        op.uses += 1;
        op.reached.push(distance);
        if op.uses.is_multiple_of(self.gamma) {
            // A removal whose results all fell back onto the reference
            // divides by zero here, and the bound below is its degree.
            op.degree = (op.degree / op.reached.value * self.ideal_distance)
                .clamp(1.0, n.saturating_sub(2).max(1) as f64);
        }

        let accepted = if result.is_feasible() {
            let cost = result.objective;
            self.costs.push(cost);
            if self.iterations.is_multiple_of(self.gamma) {
                self.low = self.low_next;
                self.low_next = f64::INFINITY;
            }
            self.low_next = self.low_next.min(cost);
            self.low = self.low.min(cost);
            self.eta = (self.eta * self.eta_decay).max(self.eta_min);
            cost <= self.low + self.eta * (self.costs.value - self.low)
        } else {
            // A result that breaks a constraint stays out of the averages,
            // and replaces only a reference that breaks one worse.
            self.eta = (self.eta * self.eta_decay).max(self.eta_min);
            result.is_better_than(&state.solution)
        };

        if accepted {
            state.solution = result;
            self.reference = Some(adjacency);
            state.iteration += 1;
            state.n_accepted += 1;
            state.update_best();
        } else {
            state.progress_iteration();
        }
    }
}

impl Heuristic<Vrp> for AdaptiveIteratedLocalSearch {
    fn clear(&mut self) {
        self.rewind();
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Vrp>) -> Result<(), OptError> {
        if state.instance.get_n() == 0 {
            state.progress_iteration();
            return Ok(());
        }
        if !self.started {
            if !self.has_a_length() {
                return Err(OptError::InvalidState(
                    "AdaptiveIteratedLocalSearch needs the length of the run: an iteration or \
                     time limit in its stop condition, or with_expected_iterations"
                        .into(),
                ));
            }
            self.start(state);
            return Ok(());
        }
        self.step(state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Customers on three rings around the depot.
    fn rings(n: usize, capacity: i64, vehicles: usize) -> Vrp {
        let mut coords = vec![(0.0, 0.0)];
        let mut demands = vec![0];
        for i in 0..n {
            let theta = std::f64::consts::TAU * i as f64 / n as f64;
            let radius = 10.0 * (1 + i % 3) as f64;
            coords.push((radius * theta.cos(), radius * theta.sin()));
            demands.push(1 + (i % 4) as i64);
        }
        Vrp::new("rings", coords, demands, capacity, vehicles)
    }

    fn started(vrp: &Vrp, seed: u64) -> (AdaptiveIteratedLocalSearch, SearchState<'_, Vrp>) {
        let mut state = SearchState::new_with_seed(vrp, seed);
        let mut search = AdaptiveIteratedLocalSearch::new(StopCondition::iterations(1_000));
        search.run_once(&mut state).unwrap();
        (search, state)
    }

    #[test]
    fn it_improves_stays_feasible_and_replays_under_a_seed() {
        let vrp = rings(30, 15, 6);
        let run = || {
            let mut state = SearchState::new_with_seed(&vrp, 3);
            let initial = state.best_solution.objective;
            AdaptiveIteratedLocalSearch::new(StopCondition::iterations(60))
                .run(&mut state)
                .unwrap();
            let best = &state.best_solution;
            assert!(best.objective < initial);
            assert!(best.is_feasible());
            vrp.validate_routes(&best.routes).unwrap();
            assert_eq!(
                vrp.objective_under(best, vrp.penalty_weight()),
                best.objective
            );
            best.routes.clone()
        };
        assert_eq!(run(), run());
    }

    /// Both removals take about the asked number of customers, each once,
    /// and what they take is what is missing afterward.
    #[test]
    fn a_removal_takes_the_asked_number_of_customers() {
        let vrp = rings(60, 25, 10);
        let (mut search, state) = started(&vrp, 5);
        let mut rng = SmallRng::seed_from_u64(1);
        for removal in [Removal::Concentric, Removal::Sequential] {
            for degree in [1.0, 7.4, 30.0] {
                let mut partial = vrp.to_partial(&state.solution);
                search.remove(&vrp, &mut partial, removal, degree, &mut rng);
                let expected = match removal {
                    Removal::Concentric => degree.ceil() as usize,
                    Removal::Sequential => degree as usize,
                };
                assert_eq!(search.removed.len(), expected);
                let mut sorted = search.removed.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), expected, "no customer is taken twice");
                let left: usize = partial.routes().iter().map(Vec::len).sum();
                assert_eq!(left + expected, 60);
                for &c in &search.removed {
                    assert!(search.absent[c]);
                    search.absent[c] = false;
                }
            }
        }
    }

    /// A sequential removal takes customers that followed one another on a
    /// route of the reference.
    #[test]
    fn a_sequential_removal_takes_a_run_of_a_route() {
        let vrp = rings(60, 25, 10);
        let (mut search, state) = started(&vrp, 7);
        let mut rng = SmallRng::seed_from_u64(2);
        let mut partial = vrp.to_partial(&state.solution);
        search.remove(&vrp, &mut partial, Removal::Sequential, 3.0, &mut rng);
        let route = state
            .solution
            .routes
            .iter()
            .find(|r| r.contains(&search.removed[0]))
            .unwrap();
        if route.len() >= 3 {
            let at = route.iter().position(|&c| c == search.removed[0]).unwrap();
            for (k, &c) in search.removed.iter().enumerate() {
                assert_eq!(c, route[(at + k) % route.len()]);
            }
        }
    }

    /// Insertion by distance puts a customer next to its nearest placed
    /// customer, on the cheaper side.
    #[test]
    fn insertion_by_distance_goes_beside_the_nearest_customer() {
        let vrp = rings(60, 25, 10);
        let (mut search, state) = started(&vrp, 9);
        let mut partial = vrp.to_partial(&state.solution);
        let c = 17;
        vrp.remove_all(&mut partial, &[c]);
        search.absent[c] = true;
        let (r, place) = search.place(&vrp, &partial, c, 1);
        let route = &partial.routes()[r];
        let nearest = search.nearest[c][0];
        if search.depot_rank[c] > 0 {
            let beside = [place.checked_sub(1), Some(place)]
                .into_iter()
                .flatten()
                .filter_map(|p| route.get(p))
                .any(|&o| o == nearest);
            assert!(
                beside,
                "customer {c} went to {place} of {route:?}, not by {nearest}"
            );
        }
    }

    /// The degree of a removal moves toward what makes its results end at
    /// the ideal distance, and the two schedules reach their ends.
    #[test]
    fn the_schedules_reach_their_ends() {
        let vrp = rings(30, 15, 6);
        let mut state = SearchState::new_with_seed(&vrp, 11);
        let mut search = AdaptiveIteratedLocalSearch::new(StopCondition::iterations(90));
        search.run(&mut state).unwrap();
        assert!((search.eta - AdaptiveIteratedLocalSearch::ETA_MIN).abs() < 2e-3);
        assert!(search.ideal_distance < AdaptiveIteratedLocalSearch::DISTANCE_MIN + 0.2);
        for op in &search.operators {
            assert!(op.uses > 0);
            assert!(op.degree >= 1.0 && op.degree <= 28.0);
        }
    }

    #[test]
    fn the_moving_mean_is_plain_then_exponential() {
        let mut mean = MovingMean::new(3);
        mean.push(3.0);
        mean.push(5.0);
        assert_eq!(mean.value, 4.0);
        mean.push(10.0);
        assert!((mean.value - (4.0 * 2.0 / 3.0 + 10.0 / 3.0)).abs() < 1e-12);
    }

    /// With nothing to say how long the run is, the schedules have no
    /// length, and the run says so.
    #[test]
    fn a_run_without_a_length_is_an_error() {
        let vrp = rings(20, 25, 4);
        let mut state = SearchState::new_with_seed(&vrp, 1);
        let err = AdaptiveIteratedLocalSearch::new(StopCondition::failed_updates(10))
            .run(&mut state)
            .unwrap_err();
        assert!(err.to_string().contains("length of the run"), "{err}");

        let mut state = SearchState::new_with_seed(&vrp, 1);
        AdaptiveIteratedLocalSearch::new(StopCondition::failed_updates(10))
            .with_expected_iterations(100)
            .run(&mut state)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "distances must satisfy")]
    fn a_minimum_above_the_maximum_is_rejected() {
        let _ =
            AdaptiveIteratedLocalSearch::new(StopCondition::iterations(1)).with_distances(5.0, 2.0);
    }
}
