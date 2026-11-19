//! Slack Induction by String Removals, a ruin-and-recreate search for vehicle
//! routing with one ruin and one recreate.

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::building_blocks::search::best_two_insertions;
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::Vrp;
use crate::problem::vrp::VrpPartial;
use crate::search_state::SearchState;
use crate::trait_defs::Ruinable;

/// Slack Induction by String Removals.
///
/// Each iteration takes strings of consecutive customers out of a few routes
/// that pass near one seed customer, puts them back one at a time at the
/// cheapest place found, and keeps the result by a simulated annealing test.
/// There is no local search and no bank of operators to choose from.
///
/// The ruin walks the customers outward from a drawn seed and takes one
/// string, whole or split around a run that stays, from each route it meets
/// until it has ruined a drawn number of routes. The recreate puts the
/// customers back in a drawn order, each at the cheapest place priced with the
/// problem's penalty, skipping each place with probability `blink_rate`. A
/// candidate is accepted when `c' < c − T · ln(u)`, with the temperature `T`
/// falling geometrically to its final value over `cooling_iterations`
/// iterations. The temperatures are in the units of the objective.
///
/// # Example
///
/// ```
/// use optopus::prelude::*;
///
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0)];
/// let vrp = Vrp::new("cross", coords, vec![0, 1, 1, 1, 1], 2, 2);
/// let mut state = SearchState::new_with_seed(&vrp, 1);
/// SlackInductionByStringRemovalForVrp::new(StopCondition::iterations(500), 500)
///     .with_temperatures(1.0, 0.01)
///     .run(&mut state)?;
/// assert_eq!(state.best_solution.overload, 0);
/// # Ok::<(), optopus::error::OptError>(())
/// ```
///
/// # References
///
/// - Christiaens, J. and Vanden Berghe, G. "Slack Induction by String Removals
///   for Vehicle Routing Problems." *Transportation Science*, 54(2), 417-433,
///   2020.
pub struct SlackInductionByStringRemoval {
    stop_condition: StopCondition,
    cooling_iterations: u64,
    initial_temperature: f64,
    final_temperature: f64,
    average_removal: f64,
    max_string_length: usize,
    split_stop_rate: f64,
    blink_rate: f64,
    /// The temperature of the next iteration, unset until the first.
    temperature: Option<f64>,
    /// The customers by distance from the seed, sorted as far as the walk got.
    by_distance: Vec<(f64, usize)>,
    /// Which routes this iteration has already ruined.
    ruined: Vec<bool>,
    /// The customers this iteration took out.
    removed: Vec<usize>,
}

impl SlackInductionByStringRemoval {
    /// The temperature the search starts at, unless
    /// [`with_temperatures`](Self::with_temperatures) says otherwise.
    pub const INITIAL_TEMPERATURE: f64 = 100.0;
    /// The temperature the search ends at.
    pub const FINAL_TEMPERATURE: f64 = 1.0;
    /// The average number of customers an iteration removes, unless
    /// [`with_removal`](Self::with_removal) says otherwise.
    pub const AVERAGE_REMOVAL: f64 = 10.0;
    /// The longest string an iteration removes from one route.
    pub const MAX_STRING_LENGTH: usize = 10;
    /// The probability that the run a split string keeps stops growing,
    /// unless [`with_split_stop_rate`](Self::with_split_stop_rate) says
    /// otherwise.
    pub const SPLIT_STOP_RATE: f64 = 0.01;
    /// The probability that a place is skipped, unless
    /// [`with_blink_rate`](Self::with_blink_rate) says otherwise.
    pub const BLINK_RATE: f64 = 0.01;

    /// A search whose temperature reaches its final value after
    /// `cooling_iterations` iterations.
    ///
    /// # Panics
    ///
    /// Panics if `cooling_iterations` is zero.
    pub fn new(stop_condition: StopCondition, cooling_iterations: u64) -> Self {
        assert!(
            cooling_iterations >= 1,
            "cooling_iterations must be at least 1"
        );
        Self {
            stop_condition,
            cooling_iterations,
            initial_temperature: Self::INITIAL_TEMPERATURE,
            final_temperature: Self::FINAL_TEMPERATURE,
            average_removal: Self::AVERAGE_REMOVAL,
            max_string_length: Self::MAX_STRING_LENGTH,
            split_stop_rate: Self::SPLIT_STOP_RATE,
            blink_rate: Self::BLINK_RATE,
            temperature: None,
            by_distance: Vec::new(),
            ruined: Vec::new(),
            removed: Vec::new(),
        }
    }

    /// Builder-style: the temperatures the search starts and ends at, in the
    /// units of the objective.
    ///
    /// # Panics
    ///
    /// Panics unless `initial >= last > 0` and `initial` is finite.
    pub fn with_temperatures(mut self, initial: f64, last: f64) -> Self {
        assert!(
            last > 0.0 && initial >= last && initial.is_finite(),
            "temperatures must satisfy initial >= final > 0 and be finite, got ({initial}, {last})"
        );
        self.initial_temperature = initial;
        self.final_temperature = last;
        self
    }

    /// Builder-style: the average number of customers an iteration removes
    /// and the longest string it takes from one route.
    ///
    /// # Panics
    ///
    /// Panics if `average_removal` is not positive and finite or
    /// `max_string_length` is zero.
    pub fn with_removal(mut self, average_removal: f64, max_string_length: usize) -> Self {
        assert!(
            average_removal > 0.0 && average_removal.is_finite(),
            "average_removal must be positive and finite"
        );
        assert!(
            max_string_length >= 1,
            "max_string_length must be at least 1"
        );
        self.average_removal = average_removal;
        self.max_string_length = max_string_length;
        self
    }

    /// Builder-style: the probability that the run of customers a split
    /// string keeps stops growing at each step. A small rate keeps a long
    /// run, so what is removed sits at the two ends of the route.
    ///
    /// # Panics
    ///
    /// Panics if `rate` is outside `[0, 1]`.
    pub fn with_split_stop_rate(mut self, rate: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&rate),
            "split_stop_rate must be in [0, 1]"
        );
        self.split_stop_rate = rate;
        self
    }

    /// Builder-style: the probability that the recreate skips a place. Zero
    /// puts every customer at its cheapest place.
    ///
    /// # Panics
    ///
    /// Panics if `rate` is outside `[0, 1)`.
    pub fn with_blink_rate(mut self, rate: f64) -> Self {
        assert!((0.0..1.0).contains(&rate), "blink_rate must be in [0, 1)");
        self.blink_rate = rate;
        self
    }

    /// Takes strings out of the routes nearest one seed customer, leaving
    /// the customers taken in `self.removed`.
    fn ruin(&mut self, prob: &Vrp, partial: &mut VrpPartial, rng: &mut SmallRng) {
        self.removed.clear();
        let routes = partial.routes();
        let (served, in_use) = routes
            .iter()
            .filter(|r| !r.is_empty())
            .fold((0usize, 0usize), |(s, t), r| (s + r.len(), t + 1));
        if served == 0 {
            return;
        }
        let string_cap = (self.max_string_length as f64).min(served as f64 / in_use as f64);
        // No more strings than customers, which also keeps a huge removal
        // from overflowing the range.
        let most_strings =
            (4.0 * self.average_removal / (1.0 + string_cap) - 1.0).min(prob.get_n() as f64);
        let strings = if most_strings > 0.0 {
            rng.random_range(1.0..most_strings + 1.0) as usize
        } else {
            1
        };

        let n = prob.get_n();
        let seed = rng.random_range(1..=n);
        self.ruined.clear();
        self.ruined.resize(routes.len(), false);
        let split_stop_rate = self.split_stop_rate;
        // Takes a string from the route `c` is on, unless that route is
        // already ruined, and says whether it did.
        let ruin_route_of =
            |c: usize, ruined: &mut [bool], removed: &mut Vec<usize>, rng: &mut SmallRng| {
                let (r, pos) = partial.locate(c);
                if ruined[r] {
                    return false;
                }
                ruined[r] = true;
                let route = &routes[r];
                let longest = string_cap.min(route.len() as f64);
                let len = (rng.random_range(1.0..longest + 1.0) as usize).min(route.len());
                Self::take_string(removed, split_stop_rate, route, pos, len, rng);
                true
            };

        // The seed leads the walk, so its string is taken before anything is
        // sorted, and an iteration that wants one string sorts nothing.
        ruin_route_of(seed, &mut self.ruined, &mut self.removed, rng);
        let mut done = 1;
        if done < strings {
            self.by_distance.clear();
            self.by_distance
                .extend((1..=n).map(|c| (prob.distance(seed, c), c)));
            // The seed leads even when another customer sits on it.
            self.by_distance[seed - 1].0 = f64::NEG_INFINITY;
            let nearer =
                |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
            // The walk rarely goes far, so the list is sorted a block at a
            // time.
            let (mut start, mut block) = (0, 16);
            while start < n && done < strings {
                let end = (start + block).min(n);
                if end < n {
                    self.by_distance[start..].select_nth_unstable_by(end - start, nearer);
                }
                self.by_distance[start..end].sort_unstable_by(nearer);
                for &(_, c) in &self.by_distance[start..end] {
                    if done == strings {
                        break;
                    }
                    if ruin_route_of(c, &mut self.ruined, &mut self.removed, rng) {
                        done += 1;
                    }
                }
                start = end;
                block *= 2;
            }
        }
        prob.remove_all(partial, &self.removed);
    }

    /// Adds to `removed` `len` customers of `route`, in a string that holds
    /// position `pos`, whole or split around a run that stays.
    fn take_string(
        removed: &mut Vec<usize>,
        split_stop_rate: f64,
        route: &[usize],
        pos: usize,
        len: usize,
        rng: &mut SmallRng,
    ) {
        let spare = route.len() - len;
        let mut kept = 0;
        if spare > 0 && rng.random_bool(0.5) {
            kept = 1;
            while kept < spare && rng.random::<f64>() >= split_stop_rate {
                kept += 1;
            }
        }
        let span = len + kept;
        let first = rng.random_range((pos + 1).saturating_sub(span)..=pos.min(route.len() - span));
        // The run that stays lies inside the string when the string has two
        // ends to put it between.
        let before = if kept == 0 {
            len
        } else if len >= 2 {
            rng.random_range(1..len)
        } else {
            rng.random_range(0..=1)
        };
        removed.extend_from_slice(&route[first..first + before]);
        removed.extend_from_slice(&route[first + before + kept..first + span]);
    }

    /// Puts the removed customers back, each at the cheapest place the blinks
    /// let it see.
    fn recreate(&mut self, prob: &Vrp, partial: &mut VrpPartial, rng: &mut SmallRng) {
        self.removed.shuffle(rng);
        // The order, drawn with the weights four, four, two and one.
        match rng.random_range(0..11) {
            // As the shuffle left them.
            0..4 => {}
            // The largest demand first.
            4..8 => self
                .removed
                .sort_by_key(|&c| std::cmp::Reverse(prob.demands[c])),
            // The farthest from the depot first.
            8..10 => self
                .removed
                .sort_by(|&a, &b| prob.distance(0, b).total_cmp(&prob.distance(0, a))),
            // The closest to the depot first.
            _ => self
                .removed
                .sort_by(|&a, &b| prob.distance(0, a).total_cmp(&prob.distance(0, b))),
        }
        for &c in &self.removed {
            let mut best = (f64::INFINITY, usize::MAX, 0);
            for bucket in 0..prob.num_buckets(partial) {
                for place in 0..prob.num_places(partial, bucket) {
                    if rng.random::<f64>() < self.blink_rate {
                        continue;
                    }
                    let cost = prob.insertion_cost(partial, bucket, place, c);
                    if cost < best.0 {
                        best = (cost, bucket, place);
                    }
                }
            }
            if best.1 == usize::MAX {
                // Every place was skipped, which a short solution can do.
                best = best_two_insertions(prob, partial, c).0;
            }
            prob.insert(partial, best.1, best.2, c);
        }
    }
}

impl Heuristic<Vrp> for SlackInductionByStringRemoval {
    fn clear(&mut self) {
        self.temperature = None;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Vrp>) -> Result<(), OptError> {
        let prob: &Vrp = state.instance;
        let mut partial = prob.to_partial(&state.solution);
        let current = prob.partial_energy(&partial);
        let temperature = *self.temperature.get_or_insert(self.initial_temperature);

        self.ruin(prob, &mut partial, &mut state.rng);
        self.recreate(prob, &mut partial, &mut state.rng);
        let candidate = prob.partial_energy(&partial);

        // One minus a draw from [0, 1) is in (0, 1], so the logarithm is
        // finite.
        let slack = -temperature * (1.0 - state.rng.random::<f64>()).ln();
        if candidate < current + slack {
            state.solution = prob.finish(&partial);
            state.iteration += 1;
            state.n_accepted += 1;
            state.update_best();
        } else {
            state.progress_iteration();
        }

        // The factor that takes the initial temperature to the final one in
        // `cooling_iterations` steps.
        let cooling = (self.final_temperature / self.initial_temperature)
            .powf(1.0 / self.cooling_iterations as f64);
        self.temperature = Some((temperature * cooling).max(self.final_temperature));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Customers on three rings around the depot, so routes have neighbors.
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

    fn sisr(iterations: u64) -> SlackInductionByStringRemoval {
        SlackInductionByStringRemoval::new(StopCondition::iterations(iterations), iterations)
            .with_temperatures(5.0, 0.05)
    }

    #[test]
    fn it_improves_a_random_start_and_ends_feasible() {
        let vrp = rings(40, 20, 8);
        let mut state = SearchState::new_with_seed(&vrp, 3);
        let initial = state.best_solution.objective;
        sisr(4_000).run(&mut state).unwrap();
        let best = &state.best_solution;
        assert!(best.objective < initial);
        assert_eq!(best.overload, 0);
        vrp.validate_routes(&best.routes).unwrap();
        assert_eq!(
            vrp.objective_under(best, vrp.penalty_weight()),
            best.objective
        );
    }

    #[test]
    fn it_replays_under_a_seed() {
        let vrp = rings(30, 15, 6);
        let run = || {
            let mut state = SearchState::new_with_seed(&vrp, 11);
            sisr(1_500).run(&mut state).unwrap();
            state.best_solution.routes.clone()
        };
        assert_eq!(run(), run());
    }

    /// A ruin takes each string from a different route, and what it takes is
    /// what is missing afterward.
    #[test]
    fn a_ruin_takes_at_most_one_string_per_route() {
        let vrp = rings(40, 20, 8);
        let mut rng = rand::SeedableRng::seed_from_u64(5);
        let sol = crate::search_state::ProblemTrait::new_solution(&vrp, &mut rng);
        let mut search = sisr(1);
        for _ in 0..200 {
            let mut partial = vrp.to_partial(&sol);
            search.ruin(&vrp, &mut partial, &mut rng);
            assert!(!search.removed.is_empty());
            let mut left = 0;
            let mut ruined = 0;
            for (before, after) in sol.routes.iter().zip(partial.routes()) {
                left += after.len();
                let taken = before.len() - after.len();
                ruined += usize::from(taken > 0);
                // Eight routes of five customers cap a string at five.
                assert!(taken <= 5);
                // What stays keeps its order.
                let kept: Vec<usize> = before
                    .iter()
                    .copied()
                    .filter(|c| !search.removed.contains(c))
                    .collect();
                assert_eq!(&kept, after);
            }
            // 4 · 10 / (1 + 5) − 1 is 5.67, so at most six strings.
            assert!(ruined <= 6, "{ruined} routes were ruined");
            assert_eq!(left + search.removed.len(), 40);
        }
    }

    /// A split string leaves customers between the two parts it takes.
    #[test]
    fn a_split_string_keeps_a_run_inside_it() {
        let route: Vec<usize> = (1..=12).collect();
        let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(8);
        let mut search = sisr(1).with_split_stop_rate(1.0);
        let mut split = 0;
        for _ in 0..200 {
            search.removed.clear();
            SlackInductionByStringRemoval::take_string(
                &mut search.removed,
                search.split_stop_rate,
                &route,
                5,
                4,
                &mut rng,
            );
            assert_eq!(search.removed.len(), 4);
            let (lo, hi) = (search.removed[0], search.removed[3]);
            assert!(lo <= 6 && 6 <= hi, "the string spans position 5");
            if hi - lo == 4 {
                split += 1;
            } else {
                assert_eq!(hi - lo, 3, "a whole string is consecutive");
            }
        }
        assert!(
            split > 50 && split < 150,
            "half the strings are split, got {split}"
        );
    }

    /// With no blinks the recreate is the greedy one, so a customer taken
    /// from an optimal place goes back to a place that costs no more.
    #[test]
    fn without_blinks_every_customer_goes_to_its_cheapest_place() {
        let vrp = rings(24, 12, 6);
        let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(2);
        let sol = crate::search_state::ProblemTrait::new_solution(&vrp, &mut rng);
        let mut search = sisr(1).with_blink_rate(0.0);
        let mut partial = vrp.to_partial(&sol);
        let before = vrp.partial_energy(&partial);
        search.removed = vec![7];
        vrp.remove_all(&mut partial, &[7]);
        search.recreate(&vrp, &mut partial, &mut rng);
        assert!(vrp.partial_energy(&partial) <= before + 1e-9);
    }

    #[test]
    #[should_panic(expected = "cooling_iterations must be at least 1")]
    fn zero_cooling_iterations_are_rejected() {
        let _ = SlackInductionByStringRemoval::new(StopCondition::iterations(1), 0);
    }

    /// A removal whose four times overflows still draws a count of strings,
    /// at most one per customer.
    #[test]
    fn a_huge_removal_ruins_every_route() {
        let vrp = rings(40, 20, 8);
        let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(5);
        let sol = crate::search_state::ProblemTrait::new_solution(&vrp, &mut rng);
        let mut partial = vrp.to_partial(&sol);
        let mut search = sisr(1).with_removal(f64::MAX, 10);
        search.ruin(&vrp, &mut partial, &mut rng);
        assert!(!search.removed.is_empty());
    }

    #[test]
    #[should_panic(expected = "average_removal must be positive and finite")]
    fn an_infinite_removal_is_rejected() {
        let _ = sisr(1).with_removal(f64::INFINITY, 10);
    }

    #[test]
    #[should_panic(expected = "temperatures must satisfy")]
    fn a_rising_temperature_is_rejected() {
        let _ = sisr(1).with_temperatures(1.0, 2.0);
    }
}
