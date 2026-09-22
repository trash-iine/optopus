use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use super::adjacency::RouteAdjacency;
use super::ops;
use crate::common::{DistanceStore, EdgeWeightType};
use crate::error::OptError;
use crate::trait_defs::{Distance, Evaluable, Evaluate, ProblemTrait};

/// Returns the capacity overflow of a route load: `max(0, load - capacity)`.
#[inline]
pub(crate) fn overload_of(load: i64, capacity: i64) -> i64 {
    (load - capacity).max(0)
}

/// Checks that `routes` visits every customer `1..=n` exactly once, the
/// partition check behind [`Vrp::validate_routes`].
fn validate_partition(n: usize, routes: &[Vec<usize>]) -> Result<(), OptError> {
    let mut seen = vec![false; n + 1];
    let mut count = 0;
    for route in routes {
        for &c in route {
            if c == 0 || c > n {
                return Err(OptError::InvalidState(format!(
                    "route contains invalid customer index {c} (valid: 1..={n})"
                )));
            }
            if seen[c] {
                return Err(OptError::InvalidState(format!(
                    "customer {c} appears more than once"
                )));
            }
            seen[c] = true;
            count += 1;
        }
    }
    if count != n {
        return Err(OptError::InvalidState(format!(
            "routes visit {count} customers, expected {n}"
        )));
    }
    Ok(())
}

/// Checks that an instance has a depot and that every coordinate is finite,
/// the two things a [`DistanceStore`] cannot be built without.
fn check_finite(coordinates: &[(f64, f64)]) -> Result<(), String> {
    if coordinates.is_empty() {
        return Err("Vrp requires at least the depot node".to_string());
    }
    for (i, &(x, y)) in coordinates.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(format!("node {i}: coordinates must be finite"));
        }
    }
    Ok(())
}

/// Fleet size to use when the caller does not specify one.
///
/// `ceil(total demand / capacity)` is only a lower bound, the bin-packing
/// relaxation, and a fleet that small frequently cannot serve the customers at
/// all: three demand-2 customers never fit into two capacity-3 vehicles, and
/// CVRPLIB's `X-n101-k25` needs 26 vehicles despite the `k25` in its name.
///
/// So the fleet is sized by first-fit-decreasing, which always yields a feasible
/// packing, plus a 10% margin: the distance-optimal solution routinely uses a
/// few more vehicles than the minimum, because splitting a remote customer onto
/// its own route can be cheaper than detouring to it. Idle vehicles are free
/// (an empty route has distance `0`), so an overly generous fleet costs only a
/// little search time, whereas too small a fleet makes the optimum unreachable.
fn default_fleet_size(demands: &[i64], capacity: i64) -> usize {
    let mut sorted: Vec<i64> = demands.iter().copied().filter(|&d| d > 0).collect();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let mut bins: Vec<i64> = Vec::new();
    for demand in sorted {
        match bins.iter_mut().find(|load| **load + demand <= capacity) {
            Some(load) => *load += demand,
            // A customer larger than one vehicle still gets its own route.
            None => bins.push(demand),
        }
    }
    let packed = bins.len().max(1);
    packed + (packed / 10).max(1)
}

/// Returns the amount by which `time` exceeds `limit` (`0.0` when it does not).
///
/// An unconstrained vehicle type carries `limit == f64::INFINITY`, for which
/// `time - limit` is `-inf` and the excess is `0.0`.
#[inline]
pub(crate) fn time_excess_of(time: f64, limit: f64) -> f64 {
    (time - limit).max(0.0)
}

/// Returns how many vehicles of a type are still missing from its `min_count`.
#[inline]
pub(crate) fn shortfall_of(min_count: usize, used: i64) -> i64 {
    (min_count as i64 - used).max(0)
}

/// One vehicle type of a heterogeneous fleet.
///
/// A type owns a contiguous block of `max_count` *slots* in every solution (see
/// [`Vrp`]); every route driven by one of those slots is priced, timed and
/// capacity-checked with this type's parameters.
///
/// It deserializes straight from a `[[vehicle_types]]` entry of the instance
/// file ([`Vrp::load_file`]); the optional keys default as documented
/// there, and the loader validates what it reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VehicleType {
    /// Human-readable label (reporting only; not required to be unique).
    pub name: String,
    /// Load a vehicle of this type can carry before incurring an overload penalty.
    pub capacity: i64,
    /// Travel speed in distance units per time unit (`route_time = distance / speed + service`).
    pub speed: f64,
    /// Cost charged once for each *used* (non-empty) vehicle of this type.
    #[serde(default)]
    pub fixed_cost: f64,
    /// Cost charged per distance unit travelled by a vehicle of this type.
    #[serde(default)]
    pub variable_cost_per_distance: f64,
    /// Minimum number of vehicles of this type that must be used; a shortfall is
    /// penalized exactly like an overload.
    #[serde(default)]
    pub min_count: usize,
    /// Number of slots of this type — the maximum number of vehicles available.
    pub max_count: usize,
    /// Maximum duration of a single route of this type; `f64::INFINITY` = unconstrained.
    #[serde(default = "default_max_route_time")]
    pub max_route_time: f64,
}

impl VehicleType {
    /// A vehicle type with no costs, no minimum count and no route-time limit.
    ///
    /// # Panics
    /// Panics if `capacity <= 0`, `speed <= 0` (or non-finite), or `max_count == 0`.
    pub fn new(name: impl Into<String>, capacity: i64, speed: f64, max_count: usize) -> Self {
        let vt = Self {
            name: name.into(),
            capacity,
            speed,
            fixed_cost: 0.0,
            variable_cost_per_distance: 0.0,
            min_count: 0,
            max_count,
            max_route_time: f64::INFINITY,
        };
        if let Err(msg) = vt.validate() {
            panic!("{msg}");
        }
        vt
    }

    /// Sets the fixed (per used vehicle) and variable (per distance unit) costs.
    #[must_use]
    pub fn with_costs(mut self, fixed_cost: f64, variable_cost_per_distance: f64) -> Self {
        self.fixed_cost = fixed_cost;
        self.variable_cost_per_distance = variable_cost_per_distance;
        self
    }

    /// Sets the minimum number of vehicles of this type that must be used.
    #[must_use]
    pub fn with_min_count(mut self, min_count: usize) -> Self {
        self.min_count = min_count;
        self
    }

    /// Sets the maximum duration of one route of this type.
    #[must_use]
    pub fn with_max_route_time(mut self, max_route_time: f64) -> Self {
        self.max_route_time = max_route_time;
        self
    }

    /// Checks the invariants every constructor and loader enforces.
    fn validate(&self) -> Result<(), String> {
        let who = &self.name;
        if self.capacity <= 0 {
            return Err(format!("vehicle type '{who}': capacity must be positive"));
        }
        if self.speed <= 0.0 || !self.speed.is_finite() {
            return Err(format!(
                "vehicle type '{who}': speed must be positive and finite"
            ));
        }
        if self.max_count == 0 {
            return Err(format!(
                "vehicle type '{who}': max_count must be at least 1"
            ));
        }
        if self.min_count > self.max_count {
            return Err(format!(
                "vehicle type '{who}': min_count ({}) must not exceed max_count ({})",
                self.min_count, self.max_count
            ));
        }
        // Costs bound the penalty weight from above, so a negative cost would
        // break the "infeasible always ranks worse" guarantee.
        if !self.fixed_cost.is_finite() || self.fixed_cost < 0.0 {
            return Err(format!(
                "vehicle type '{who}': fixed_cost must be finite and non-negative"
            ));
        }
        if !self.variable_cost_per_distance.is_finite() || self.variable_cost_per_distance < 0.0 {
            return Err(format!(
                "vehicle type '{who}': variable_cost_per_distance must be finite and non-negative"
            ));
        }
        if self.max_route_time <= 0.0 || self.max_route_time.is_nan() {
            return Err(format!(
                "vehicle type '{who}': max_route_time must be positive (use infinity for none)"
            ));
        }
        Ok(())
    }
}

/// Which aggregation of the per-route times the objective charges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveMode {
    /// Sum of every route's duration — additive, the usual routing objective.
    TotalTime,
    /// Duration of the longest route — a min-max ("when is the last vehicle
    /// back?") objective.
    Makespan,
}

/// A solution to the VRP.
///
/// `routes[s]` is the ordered customer list of *slot* `s` (the depot is implicit
/// at both ends); `routes.len()` is always [`Vrp::num_slots`], and an idle
/// vehicle is an empty route. Which vehicle type a slot belongs to is fixed by
/// the instance ([`Vrp::type_of_slot`]), so a move between two slots of
/// different types is also a change of vehicle type.
///
/// Every other field is a cache that the neighborhood moves maintain
/// incrementally; [`Vrp::solution_from_routes`] recomputes all of them from
/// scratch and is the reference the incremental updates are tested against.
///
/// All three violations (capacity, route time, minimum fleet usage) are soft,
/// weighted by [`Vrp::penalty_weight`] exactly as [`crate::problem::Vrp`]
/// handles capacity.
#[derive(Debug, Clone)]
pub struct VrpSolution {
    /// Ordered customers per slot; length is always `num_slots`.
    pub routes: Vec<Vec<usize>>,
    /// Total demand of each slot's route.
    pub route_loads: Vec<i64>,
    /// Raw travelled distance of each slot's route (`0.0` when empty).
    pub route_distance: Vec<f64>,
    /// Duration of each slot's route: `route_distance / speed + Σ service_time`.
    pub route_time: Vec<f64>,
    /// Per vehicle type, the number of non-empty slots of that type.
    pub used_count: Vec<usize>,
    /// Sum of `route_time` over all slots.
    pub total_time: f64,
    /// Largest `route_time` over the non-empty slots (`0.0` if all are empty).
    pub makespan: f64,
    /// `Σ max(0, load_s − capacity(type of s))`.
    pub overload: i64,
    /// `Σ max(0, route_time_s − max_route_time(type of s))`.
    pub time_excess: f64,
    /// `Σ_types max(0, min_count − used_count)`.
    pub min_count_shortfall: i64,
    /// `Σ fixed_cost(type)` over the used slots plus
    /// `Σ variable_cost_per_distance(type) × route_distance` over all slots.
    pub total_cost: f64,
    /// Penalty-augmented objective (see [`Vrp`]); lower is better.
    pub objective: f64,
}

impl Evaluate for VrpSolution {
    /// Vrp minimizes its penalty-augmented objective.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.objective)
    }
}

impl Distance for VrpSolution {
    /// Broken-pairs dissimilarity: how many of the two solutions' customer
    /// adjacencies the other one does not have.
    ///
    /// It counts trips, not slot labels, so permuting the routes or driving
    /// one of them backwards is a distance of `0`, which the obvious
    /// alternative, comparing each customer's route index, gets wrong on exactly
    /// the solutions a search meets most often. It is also blind to the vehicle
    /// *type* a trip is assigned to: two solutions driving the same trips with
    /// different vehicles are at distance `0`, since diversity here means "a
    /// different set of trips", which is what a recombination operator can act
    /// on.
    ///
    /// The underlying count is directional; this takes the larger of the two
    /// directions, so `a.distance(b) == b.distance(a)`. Hybrid Genetic Search
    /// ranks diversity on the directional count instead, the form Vidal's biased
    /// fitness is defined on.
    fn distance(&self, other: &Self) -> usize {
        let n: usize = self.routes.iter().map(|r| r.len()).sum();
        let a = RouteAdjacency::from_routes(n, &self.routes);
        let b = RouteAdjacency::from_routes(n, &other.routes);
        a.broken_pairs_from(&b).max(b.broken_pairs_from(&a))
    }
}

impl VrpSolution {
    /// Total travelled distance over all slots, the sum of `route_distance`.
    pub fn total_distance(&self) -> f64 {
        self.route_distance.iter().sum()
    }

    /// The time the objective charges under `mode`: the total or the longest
    /// route time.
    pub fn time_component(&self, mode: ObjectiveMode) -> f64 {
        match mode {
            ObjectiveMode::TotalTime => self.total_time,
            ObjectiveMode::Makespan => self.makespan,
        }
    }

    /// The three violations as the objective sums them,
    /// `overload + time_excess + min_count_shortfall`; `0.0` exactly when the
    /// solution is feasible.
    pub fn violation(&self) -> f64 {
        self.overload as f64 + self.time_excess + self.min_count_shortfall as f64
    }
}

/// The Vehicle Routing Problem: a depot (node `0`) and `n` customers
/// (`1..=n`) with integer demands and service times, served by a fleet of
/// [`VehicleType`]s. Each type contributes `max_count` *slots*, and every
/// solution has exactly one route per slot. The homogeneous CVRP is the
/// one-type instance every plain constructor builds ([`Vrp::new`],
/// [`Vrp::with_rounding`], [`Vrp::with_nearest_neighbors`],
/// [`Vrp::from_distance_matrix`], [`Vrp::load_file`] on a CVRPLIB file):
/// speed `1`, no service times, no costs, no route-time limit, so the
/// objective is plainly the total distance plus the capacity penalty.
/// [`Vrp::with_fleet`] and a TOML instance file build the general case.
///
/// # Slots are bound to types in fixed contiguous blocks
///
/// Type `0` owns slots `0..max_count[0]`, type `1` the next block, and so on;
/// the mapping is computed once at construction and never changes, so a
/// slot's type is part of the instance, never of the solution. Relocating or
/// swapping a customer *between slots of different types* is therefore
/// exactly a change of vehicle type, and the same three moves that serve the
/// homogeneous CVRP cover fleet composition too. Idle slots of one type are
/// interchangeable, see [`Vrp::idle_representatives`].
///
/// # Objective
///
/// ```text
/// time_component = TotalTime => total_time | Makespan => makespan
/// objective = time_component
///           + cost_weight * (fixed costs of used vehicles + variable costs)
///           + penalty_weight() * (overload + time_excess + min_count_shortfall)
/// ```
///
/// Minimized. The distances live in a [`DistanceStore`]. An instance is built
/// from 2D coordinates, either as the full distance matrix or as the `k`
/// nearest neighbours of every node with the remaining pairs computed from
/// the coordinates on demand ([`Vrp::with_nearest_neighbors`]), or from a
/// distance matrix given directly ([`Vrp::from_distance_matrix`]). Distances
/// from coordinates mirror TSPLIB semantics: rounding selects nearest-integer
/// `EUC_2D` (the CVRPLIB standard) versus plain Euclidean (the default for
/// programmatically constructed instances).
#[derive(Debug, Clone)]
pub struct Vrp {
    pub name: String,
    /// Distances between the nodes; index `0` is the depot, `1..=n` are the
    /// customers.
    distances: DistanceStore,
    /// Node demands; `demands[0] == 0` (the depot).
    pub demands: Vec<i64>,
    /// Node service times; `service_times[0] == 0.0` (the depot).
    pub service_times: Vec<f64>,
    /// The fleet, in slot-block order. Private because `slot_terms` copies
    /// what pricing reads of it: a write here would leave the two disagreeing
    /// about what a route costs.
    vehicle_types: Vec<VehicleType>,
    /// Which aggregation of route times the objective charges.
    objective_mode: ObjectiveMode,
    /// Weight of the monetary cost term relative to the time term.
    cost_weight: f64,
    /// What slot `s`'s vehicle type is, and what pricing a route edit reads
    /// of it, laid out per slot so the hot loop follows one index instead of
    /// two.
    slot_terms: Vec<SlotTerms>,
    /// Lazily computed penalty weight (see [`Vrp::penalty_weight`]).
    penalty_weight: OnceLock<f64>,
}

/// The numbers of a slot's vehicle type that price a route edit, copied per
/// slot from [`VehicleType`] at construction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotTerms {
    pub vehicle_type: usize,
    pub capacity: i64,
    pub speed: f64,
    pub max_route_time: f64,
    pub fixed_cost: f64,
    pub variable_cost_per_distance: f64,
}

impl Vrp {
    /// Largest number of nodes (depot + customers) for which the coordinate
    /// constructors keep the full distance matrix (memory `nodes² × 8` bytes,
    /// 32 MB at this cap). Larger instances keep
    /// [`Vrp::NEAREST_NEIGHBORS_ABOVE_CAP`] neighbours per node.
    pub const DIST_MATRIX_MAX_N: usize = 2000;
    /// Neighbours per node kept above the cap, the granularity the descents
    /// use by default.
    pub const NEAREST_NEIGHBORS_ABOVE_CAP: usize = 20;

    /// Creates a CVRP instance with plain (non-rounded) Euclidean distances.
    ///
    /// `coordinates[0]` / `demands[0]` are the depot (demand ignored). One
    /// vehicle type of `capacity`, speed `1`, no costs and no route-time
    /// limit, `num_vehicles` slots. If `num_vehicles == 0` it defaults to
    /// `default_fleet_size`: the demands packed first-fit-decreasing, plus a
    /// 10% margin. Not `ceil(total_demand / capacity)`, which is only a lower
    /// bound on the bin count and often admits no feasible assignment.
    ///
    /// # Panics
    /// Panics if `coordinates` and `demands` differ in length, if `coordinates`
    /// is empty, or if `capacity <= 0`.
    pub fn new(
        name: impl Into<String>,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
    ) -> Self {
        Self::plain(
            name.into(),
            coordinates,
            demands,
            capacity,
            num_vehicles,
            false,
        )
    }

    /// Like [`Vrp::new`] but with nearest-integer `EUC_2D` distances.
    pub fn with_rounding(
        name: impl Into<String>,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
    ) -> Self {
        Self::plain(
            name.into(),
            coordinates,
            demands,
            capacity,
            num_vehicles,
            true,
        )
    }

    /// Creates an instance that keeps only the `k` nearest neighbours of every
    /// node, `nodes × k` distances instead of `nodes²`. Any other pair is
    /// computed from the coordinates when asked for, so the instance answers
    /// exactly like the full matrix would, only slower on pairs outside the
    /// lists. `k` larger than `nodes - 1` keeps every other node. The other
    /// arguments are those of [`Vrp::new`], with `rounded` selecting
    /// nearest-integer `EUC_2D` distances as [`Vrp::with_rounding`] does.
    ///
    /// # Panics
    /// As [`Vrp::new`].
    pub fn with_nearest_neighbors(
        name: impl Into<String>,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
        rounded: bool,
        k: usize,
    ) -> Self {
        check_finite(&coordinates).unwrap_or_else(|msg| panic!("{msg}"));
        let distances = DistanceStore::nearest_from_coordinates(
            coordinates,
            Self::edge_weight_type(rounded),
            k,
        );
        Self::plain_from_store(name.into(), distances, demands, capacity, num_vehicles)
    }

    /// Creates an instance from a distance matrix given directly, so it has
    /// no coordinates. `matrix[i][j]` is the distance between nodes `i` and
    /// `j`, node `0` being the depot, and is expected to be symmetric, since
    /// the 2-opt gains price a segment reversal from the four exchanged edges
    /// alone. Returns an error unless the matrix is square. The other
    /// arguments are those of [`Vrp::new`].
    ///
    /// # Panics
    /// As [`Vrp::new`].
    pub fn from_distance_matrix(
        name: impl Into<String>,
        matrix: Vec<Vec<f64>>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
    ) -> Result<Self, OptError> {
        Ok(Self::plain_from_store(
            name.into(),
            DistanceStore::from_matrix(matrix)?,
            demands,
            capacity,
            num_vehicles,
        ))
    }

    /// Creates a heterogeneous-fleet instance.
    ///
    /// `coordinates[0]` / `demands[0]` / `service_times[0]` describe the depot
    /// (its demand and service time are forced to zero). Slots are laid out in
    /// `vehicle_types` order, `max_count` of them per type. Distances follow
    /// [`Vrp::new`] when `rounded` is false and [`Vrp::with_rounding`] when it
    /// is true.
    ///
    /// # Panics
    ///
    /// Panics on any input a well-formed instance cannot have, this being the
    /// programmatic constructor ([`Vrp::load_file`] reports the same conditions
    /// as [`OptError::Config`]). Specifically: empty `coordinates`;
    /// `coordinates`, `demands` and `service_times` of differing lengths; a
    /// non-finite coordinate; a negative demand or service time; an empty
    /// `vehicle_types`; a vehicle type with a non-positive `capacity` /
    /// `speed`, a zero `max_count`, `min_count > max_count`, a negative cost or
    /// a non-positive `max_route_time`; a negative or non-finite `cost_weight`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_fleet(
        name: impl Into<String>,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        service_times: Vec<f64>,
        vehicle_types: Vec<VehicleType>,
        objective_mode: ObjectiveMode,
        cost_weight: f64,
        rounded: bool,
    ) -> Self {
        Self::build(
            name.into(),
            coordinates,
            demands,
            service_times,
            vehicle_types,
            objective_mode,
            cost_weight,
            rounded,
        )
        .unwrap_or_else(|msg| panic!("{msg}"))
    }

    /// The edge-weight type a `rounded` flag stands for.
    fn edge_weight_type(rounded: bool) -> EdgeWeightType {
        if rounded {
            EdgeWeightType::Euc2d
        } else {
            EdgeWeightType::Continuous
        }
    }

    /// The single-type instance every CVRP constructor builds.
    fn plain(
        name: String,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
        rounded: bool,
    ) -> Self {
        let distances = Self::store_from_coordinates(coordinates, rounded);
        Self::plain_from_store(name, distances, demands, capacity, num_vehicles)
    }

    fn plain_from_store(
        name: String,
        distances: DistanceStore,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
    ) -> Self {
        assert!(capacity > 0, "capacity must be positive");
        let num_vehicles = if num_vehicles == 0 {
            default_fleet_size(&demands, capacity)
        } else {
            num_vehicles
        };
        let service_times = vec![0.0; demands.len()];
        let fleet = vec![VehicleType::new("vehicle", capacity, 1.0, num_vehicles)];
        Self::from_store(
            name,
            distances,
            demands,
            service_times,
            fleet,
            ObjectiveMode::TotalTime,
            1.0,
        )
        .unwrap_or_else(|msg| panic!("{msg}"))
    }

    /// Full matrix up to [`Vrp::DIST_MATRIX_MAX_N`] nodes, nearest neighbours
    /// above it.
    ///
    /// # Panics
    /// Panics on an empty coordinate list or a non-finite coordinate; the
    /// loader calls [`check_finite`] itself to report them instead.
    fn store_from_coordinates(coordinates: Vec<(f64, f64)>, rounded: bool) -> DistanceStore {
        check_finite(&coordinates).unwrap_or_else(|msg| panic!("{msg}"));
        let ewt = Self::edge_weight_type(rounded);
        if coordinates.len() <= Self::DIST_MATRIX_MAX_N {
            DistanceStore::from_coordinates(coordinates, ewt)
        } else {
            DistanceStore::nearest_from_coordinates(
                coordinates,
                ewt,
                Self::NEAREST_NEIGHBORS_ABOVE_CAP,
            )
        }
    }

    /// The fleet constructor behind [`Vrp::with_fleet`] and the TOML loader.
    /// `Err` carries the message the former panics with and the latter wraps.
    #[allow(clippy::too_many_arguments)]
    fn build(
        name: String,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        service_times: Vec<f64>,
        vehicle_types: Vec<VehicleType>,
        objective_mode: ObjectiveMode,
        cost_weight: f64,
        rounded: bool,
    ) -> Result<Self, String> {
        check_finite(&coordinates)?;
        let distances = Self::store_from_coordinates(coordinates, rounded);
        Self::from_store(
            name,
            distances,
            demands,
            service_times,
            vehicle_types,
            objective_mode,
            cost_weight,
        )
    }

    /// The one assembly point: validates everything but the coordinates
    /// (already inside the store), forces the depot's demand and service time
    /// to zero, and lays the slots out.
    fn from_store(
        name: String,
        distances: DistanceStore,
        mut demands: Vec<i64>,
        mut service_times: Vec<f64>,
        vehicle_types: Vec<VehicleType>,
        objective_mode: ObjectiveMode,
        cost_weight: f64,
    ) -> Result<Self, String> {
        if distances.is_empty() {
            return Err("Vrp requires at least the depot node".to_string());
        }
        if distances.len() != demands.len() || distances.len() != service_times.len() {
            return Err(
                "distances, demands and service_times must cover the same nodes".to_string(),
            );
        }
        for (i, &d) in demands.iter().enumerate() {
            if d < 0 {
                return Err(format!("node {i}: demand must be non-negative"));
            }
        }
        for (i, &s) in service_times.iter().enumerate() {
            if !s.is_finite() || s < 0.0 {
                return Err(format!(
                    "node {i}: service_time must be finite and non-negative"
                ));
            }
        }
        if vehicle_types.is_empty() {
            return Err("Vrp requires at least one vehicle type".to_string());
        }
        for vt in &vehicle_types {
            vt.validate()?;
        }
        if !cost_weight.is_finite() || cost_weight < 0.0 {
            return Err("cost_weight must be finite and non-negative".to_string());
        }

        // The depot is never served.
        demands[0] = 0;
        service_times[0] = 0.0;

        let mut slot_terms = Vec::new();
        for (t, vt) in vehicle_types.iter().enumerate() {
            slot_terms.extend(std::iter::repeat_n(
                SlotTerms {
                    vehicle_type: t,
                    capacity: vt.capacity,
                    speed: vt.speed,
                    max_route_time: vt.max_route_time,
                    fixed_cost: vt.fixed_cost,
                    variable_cost_per_distance: vt.variable_cost_per_distance,
                },
                vt.max_count,
            ));
        }

        Ok(Self {
            name,
            distances,
            demands,
            service_times,
            vehicle_types,
            objective_mode,
            cost_weight,
            slot_terms,
            penalty_weight: OnceLock::new(),
        })
    }

    /// Number of customers (excludes the depot).
    pub fn get_n(&self) -> usize {
        self.demands.len() - 1
    }

    /// The node coordinates the instance was built from, index `0` the
    /// depot, `None` for an instance built from a distance matrix.
    pub fn coordinates(&self) -> Option<&[(f64, f64)]> {
        self.distances.coordinates()
    }

    /// Whether distances from the coordinates are rounded to the nearest
    /// integer (`EUC_2D`), `None` for an instance built from a distance
    /// matrix.
    pub fn rounded(&self) -> Option<bool> {
        self.distances
            .edge_weight_type()
            .map(|ewt| ewt == EdgeWeightType::Euc2d)
    }

    /// The fleet, in slot-block order.
    #[inline]
    pub fn vehicle_types(&self) -> &[VehicleType] {
        &self.vehicle_types
    }

    /// Which aggregation of route times the objective charges.
    #[inline]
    pub fn objective_mode(&self) -> ObjectiveMode {
        self.objective_mode
    }

    /// Weight of the monetary cost term relative to the time term.
    #[inline]
    pub fn cost_weight(&self) -> f64 {
        self.cost_weight
    }

    /// The same instance charging the other aggregation of route times.
    ///
    /// Nothing cached depends on the mode, so this is the one fleet
    /// parameter that can be changed after construction. Solutions built
    /// under the old mode carry an objective the new one does not agree
    /// with; rebuild them with [`Vrp::solution_from_routes`].
    #[must_use]
    pub fn with_objective_mode(mut self, objective_mode: ObjectiveMode) -> Self {
        self.objective_mode = objective_mode;
        self.penalty_weight = OnceLock::new();
        self
    }

    /// Number of vehicle slots, `Σ max_count`, and the length of every
    /// solution's `routes`.
    pub fn num_slots(&self) -> usize {
        self.slot_terms.len()
    }

    /// The vehicle type index of slot `s`.
    ///
    /// # Panics
    /// Panics if `s >= num_slots()`.
    #[inline]
    pub fn type_of_slot(&self, s: usize) -> usize {
        self.slot_terms[s].vehicle_type
    }

    /// The vehicle type driving slot `s`.
    ///
    /// # Panics
    /// Panics if `s >= num_slots()`.
    #[inline]
    pub fn vehicle_type_of_slot(&self, s: usize) -> &VehicleType {
        &self.vehicle_types[self.type_of_slot(s)]
    }

    /// The pricing terms of slot `s`'s type.
    #[inline]
    pub(crate) fn slot_terms(&self, s: usize) -> &SlotTerms {
        &self.slot_terms[s]
    }

    /// The first empty slot of every vehicle type that has one, in slot order.
    ///
    /// Empty slots of one type are label-equivalent, so a move that opens a
    /// new route need only consider one of them; enumerating every empty slot
    /// would offer the same move once per idle vehicle.
    pub fn idle_representatives<'a>(
        &'a self,
        routes: &'a [Vec<usize>],
    ) -> impl Iterator<Item = usize> + 'a {
        self.destination_slots(routes)
            .filter(|&s| routes[s].is_empty())
    }

    /// Whether `slot` may receive a customer: it is in use, or it is the
    /// first idle slot of its vehicle type. The point form of
    /// [`Vrp::destination_slots`], for a sampler that draws a slot rather
    /// than enumerating them.
    pub fn is_destination_slot(&self, routes: &[Vec<usize>], slot: usize) -> bool {
        if !routes[slot].is_empty() {
            return true;
        }
        let t = self.type_of_slot(slot);
        // The type owns a contiguous block, so its earlier slots are the only
        // ones that could hold the representative.
        !(0..slot)
            .rev()
            .take_while(|&r| self.type_of_slot(r) == t)
            .any(|r| routes[r].is_empty())
    }

    /// The slots a move may put a customer into: every slot in use, plus the
    /// idle representatives, in slot order.
    ///
    /// One pass, so a neighborhood enumerating destinations asks the
    /// question once per slot rather than once per slot per candidate.
    pub fn destination_slots<'a>(
        &'a self,
        routes: &'a [Vec<usize>],
    ) -> impl Iterator<Item = usize> + 'a {
        let mut idle_type = usize::MAX;
        routes.iter().enumerate().filter_map(move |(s, route)| {
            if !route.is_empty() {
                return Some(s);
            }
            let t = self.type_of_slot(s);
            if t == idle_type {
                return None;
            }
            idle_type = t;
            Some(s)
        })
    }

    /// Distance between nodes `i` and `j` (either may be the depot, index
    /// `0`), see [`DistanceStore::distance`].
    #[inline]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.distances.distance(i, j)
    }

    /// The objective of `sol`'s caches under an arbitrary penalty weight, the
    /// formula [`Vrp`] states with `penalty` in place of
    /// [`Vrp::penalty_weight`]. The searches that adapt their own penalty
    /// price their solutions with it; `sol.objective` itself is not read.
    pub fn objective_under(&self, sol: &VrpSolution, penalty: f64) -> f64 {
        sol.time_component(self.objective_mode)
            + self.cost_weight * sol.total_cost
            + penalty * sol.violation()
    }

    /// Penalty weight applied per unit of overload, of route-time excess and of
    /// missing minimum-count vehicle.
    ///
    /// It must dominate everything the objective can otherwise trade away, so it
    /// is built as an upper bound on the whole non-penalty part of the objective,
    /// plus one:
    ///
    /// - **Time.** Every route visits its customers once and returns to the
    ///   depot, so all routes together traverse at most `n + num_slots` edges,
    ///   none longer than the largest pairwise distance — bounding total
    ///   distance. Dividing by the *slowest* type's speed and adding every
    ///   customer's service time bounds `total_time`, and therefore also
    ///   `makespan`, which never exceeds it.
    /// - **Cost.** At most `max_count` vehicles of each type are charged their
    ///   fixed cost, and the variable cost is at most the highest per-distance
    ///   rate times the same total-distance bound. The sum is scaled by
    ///   `cost_weight`, as in the objective.
    ///
    /// The bound is deliberately loose — only strict dominance matters, and the
    /// cost of looseness is a flatter penalty landscape, not a wrong optimum.
    ///
    /// Dominance is *per unit* of violation: overload and shortfall are integers,
    /// so any violation of them costs at least the full weight and thus more than
    /// any feasible alternative can save. Route-time excess is continuous, so a
    /// violation smaller than one time unit is scaled down accordingly — a
    /// deliberate consequence of modelling a continuous constraint with a finite
    /// weight.
    pub fn penalty_weight(&self) -> f64 {
        *self.penalty_weight.get_or_init(|| {
            let max_total_distance =
                (self.get_n() + self.num_slots()) as f64 * self.distances.max_distance();

            let min_speed = self
                .vehicle_types
                .iter()
                .map(|vt| vt.speed)
                .fold(f64::INFINITY, f64::min);
            let total_service: f64 = self.service_times.iter().sum();
            let time_bound = max_total_distance / min_speed + total_service;

            let fixed_bound: f64 = self
                .vehicle_types
                .iter()
                .map(|vt| vt.fixed_cost * vt.max_count as f64)
                .sum();
            let max_rate = self
                .vehicle_types
                .iter()
                .map(|vt| vt.variable_cost_per_distance)
                .fold(0.0_f64, f64::max);
            let cost_bound = fixed_bound + max_rate * max_total_distance;

            time_bound + self.cost_weight * cost_bound + 1.0
        })
    }

    /// Distance of a single route: `depot → route[0] → … → route[last] → depot`.
    /// An empty route has distance `0`.
    pub fn route_distance(&self, route: &[usize]) -> f64 {
        ops::route_distance(&self.distances, route)
    }

    /// Total service time of a route.
    pub fn route_service_time(&self, route: &[usize]) -> f64 {
        route.iter().map(|&c| self.service_times[c]).sum()
    }

    /// Builds a [`VrpSolution`] from a slot assignment, computing every
    /// cached field from scratch.
    ///
    /// This is the reference implementation the incremental neighborhood updates
    /// are tested against. It assumes `routes` is a valid partition of the
    /// customers; use [`Vrp::validate_routes`] to check.
    ///
    /// # Panics
    /// Panics if `routes.len() != num_slots()`.
    pub fn solution_from_routes(&self, routes: Vec<Vec<usize>>) -> VrpSolution {
        assert_eq!(
            routes.len(),
            self.num_slots(),
            "a Vrp solution has exactly one route per vehicle slot"
        );

        let slots = self.num_slots();
        let mut route_loads = vec![0i64; slots];
        let mut route_distance = vec![0.0; slots];
        let mut route_time = vec![0.0; slots];
        let mut used_count = vec![0usize; self.vehicle_types.len()];
        let mut total_time = 0.0;
        let mut makespan = 0.0_f64;
        let mut overload = 0i64;
        let mut time_excess = 0.0;
        let mut total_cost = 0.0;

        for (s, route) in routes.iter().enumerate() {
            let t = self.type_of_slot(s);
            let vt = &self.vehicle_types[t];

            // One pass: the route's demand, its service time and the legs
            // between its customers, with the depot legs added after.
            let mut load = 0i64;
            let mut service = 0.0;
            let mut dist = 0.0;
            let mut prev = 0;
            for &c in route {
                load += self.demands[c];
                service += self.service_times[c];
                dist += self.distance(prev, c);
                prev = c;
            }
            let dist = if route.is_empty() {
                0.0
            } else {
                dist + self.distance(prev, 0)
            };
            let time = if route.is_empty() {
                0.0
            } else {
                dist / vt.speed + service
            };

            route_loads[s] = load;
            route_distance[s] = dist;
            route_time[s] = time;

            overload += overload_of(load, vt.capacity);
            time_excess += time_excess_of(time, vt.max_route_time);
            total_time += time;
            total_cost += vt.variable_cost_per_distance * dist;
            if !route.is_empty() {
                used_count[t] += 1;
                total_cost += vt.fixed_cost;
                if time > makespan {
                    makespan = time;
                }
            }
        }

        let min_count_shortfall: i64 = self
            .vehicle_types
            .iter()
            .zip(&used_count)
            .map(|(vt, &used)| shortfall_of(vt.min_count, used as i64))
            .sum();
        let mut sol = VrpSolution {
            routes,
            route_loads,
            route_distance,
            route_time,
            used_count,
            total_time,
            makespan,
            overload,
            time_excess,
            min_count_shortfall,
            total_cost,
            objective: 0.0,
        };
        sol.objective = self.objective_under(&sol, self.penalty_weight());
        sol
    }

    /// Validates that `routes` has one entry per slot and visits every customer
    /// `1..=n` exactly once.
    pub fn validate_routes(&self, routes: &[Vec<usize>]) -> Result<(), OptError> {
        if routes.len() != self.num_slots() {
            return Err(OptError::InvalidState(format!(
                "routes has {} entries, expected one per slot ({})",
                routes.len(),
                self.num_slots()
            )));
        }
        validate_partition(self.get_n(), routes)
    }
    /// Loads an instance file: the TOML fleet format below when the path ends
    /// in `.toml`, CVRPLIB otherwise.
    ///
    /// The CVRPLIB reader accepts the standard TSPLIB-style header (`NAME`,
    /// `DIMENSION`, `EDGE_WEIGHT_TYPE: EUC_2D`, `CAPACITY`, and an optional
    /// `COMMENT` carrying `No of trucks: K`), followed by the
    /// `NODE_COORD_SECTION`, `DEMAND_SECTION`, and `DEPOT_SECTION`. Node `1`
    /// in the file (the depot per `DEPOT_SECTION`) is re-indexed to `0`
    /// internally, and the result is the single-type instance [`Vrp::new`]
    /// builds. Files with at most [`Vrp::DIST_MATRIX_MAX_N`] nodes get the
    /// full distance matrix, larger ones keep
    /// [`Vrp::NEAREST_NEIGHBORS_ABOVE_CAP`] neighbours per node.
    ///
    /// The TOML format:
    ///
    /// ```toml
    /// name = "demo-fleet"
    /// objective_mode = "TotalTime"   # or "Makespan"
    /// cost_weight = 1.0              # optional, default 1.0
    /// rounded = false                # optional, default false
    ///
    /// [depot]
    /// x = 0.0
    /// y = 0.0
    ///
    /// [[vehicle_types]]
    /// name = "truck"
    /// capacity = 100
    /// speed = 1.0
    /// fixed_cost = 50.0                 # optional, default 0.0
    /// variable_cost_per_distance = 0.1  # optional, default 0.0
    /// min_count = 3                     # optional, default 0
    /// max_count = 5
    /// max_route_time = 480.0            # optional, default: unconstrained
    ///
    /// [[customers]]
    /// id = 1
    /// x = 3.0
    /// y = 4.0
    /// demand = 10
    /// service_time = 15.0   # optional, default 0.0
    /// ```
    ///
    /// Customer `id`s must be exactly `1..=n` — no gaps, no duplicates — and
    /// index the customer within the instance. `name` may be omitted, in which
    /// case the file stem is used. Unknown keys are rejected, so a misspelled
    /// optional field is an error rather than a silently ignored setting.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, OptError> {
        let path = path.as_ref();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
        {
            Self::load_toml(path)
        } else {
            Self::load_cvrplib(path)
        }
    }

    /// Reads the TOML fleet format regardless of the file's name, the reader
    /// [`Vrp::load_file`] picks by extension.
    pub fn load_toml(path: &std::path::Path) -> Result<Self, OptError> {
        let text = std::fs::read_to_string(path)?;
        let raw: RawFile = toml::from_str(&text)?;

        if raw.customers.is_empty() {
            return Err(OptError::Config(
                "instance defines no customers".to_string(),
            ));
        }

        // Customer ids must form exactly 1..=n.
        let n = raw.customers.len();
        let mut seen = vec![false; n + 1];
        for c in &raw.customers {
            if c.id == 0 || c.id > n {
                return Err(OptError::Config(format!(
                    "customer id {} is out of range 1..={n}",
                    c.id
                )));
            }
            if seen[c.id] {
                return Err(OptError::Config(format!("duplicate customer id {}", c.id)));
            }
            seen[c.id] = true;
        }

        let mut coordinates = vec![(raw.depot.x, raw.depot.y); n + 1];
        let mut demands = vec![0i64; n + 1];
        let mut service_times = vec![0.0; n + 1];
        for c in &raw.customers {
            coordinates[c.id] = (c.x, c.y);
            demands[c.id] = c.demand;
            service_times[c.id] = c.service_time;
        }

        let name = raw.name.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("vrp")
                .to_string()
        });

        Self::build(
            name,
            coordinates,
            demands,
            service_times,
            raw.vehicle_types,
            raw.objective_mode,
            raw.cost_weight,
            raw.rounded,
        )
        .map_err(OptError::Config)
    }

    /// Reads the CVRPLIB format regardless of the file's name, the reader
    /// [`Vrp::load_file`] picks by extension.
    pub fn load_cvrplib(path: &std::path::Path) -> Result<Self, OptError> {
        use crate::common::InstanceLines;

        let mut lines = InstanceLines::open(path)?;

        let mut name: Option<String> = None;
        let mut dimension: Option<usize> = None;
        let mut capacity: Option<i64> = None;
        let mut num_vehicles: usize = 0;

        // Header: parse until the first *_SECTION keyword.
        let first_section =
            loop {
                let line = lines
                    .next_line()?
                    .ok_or_else(|| lines.err("unexpected end of file in header"))?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let key_upper = trimmed
                    .split(|c: char| c == ':' || c.is_whitespace())
                    .next()
                    .unwrap_or("")
                    .to_ascii_uppercase();
                if key_upper.ends_with("_SECTION") {
                    break key_upper;
                }
                let value = trimmed
                    .split_once(':')
                    .map(|(_, v)| v.trim().to_string())
                    .unwrap_or_default();
                match key_upper.as_str() {
                    "NAME" => name = Some(value),
                    "DIMENSION" => {
                        dimension = Some(value.parse::<usize>().map_err(|e| {
                            lines.err(format!("failed to parse DIMENSION value: {e}"))
                        })?);
                    }
                    "CAPACITY" => {
                        capacity = Some(value.parse::<i64>().map_err(|e| {
                            lines.err(format!("failed to parse CAPACITY value: {e}"))
                        })?);
                    }
                    "EDGE_WEIGHT_TYPE" => {
                        let ewt = value.to_ascii_uppercase();
                        if ewt != "EUC_2D" {
                            return Err(lines.err(format!(
                                "unsupported EDGE_WEIGHT_TYPE '{ewt}' (only EUC_2D is supported)"
                            )));
                        }
                    }
                    "COMMENT" => {
                        // Best-effort extraction of "No of trucks: K".
                        if let Some(k) = parse_trucks_from_comment(&value) {
                            num_vehicles = k;
                        }
                    }
                    _ => {}
                }
            };

        let dim = dimension.ok_or_else(|| lines.err("'DIMENSION: N' not found in header"))?;
        let capacity = capacity.ok_or_else(|| lines.err("'CAPACITY: Q' not found in header"))?;

        let mut coordinates: Vec<(f64, f64)> = vec![(0.0, 0.0); dim];
        let mut demands: Vec<i64> = vec![0; dim];

        // Sections can appear in any order; drive off the section keyword.
        let mut section = first_section;
        let mut depot_index: Option<usize> = None;
        loop {
            match section.as_str() {
                "NODE_COORD_SECTION" => {
                    for _ in 0..dim {
                        let line = lines
                            .next_data_line()?
                            .ok_or_else(|| lines.err("unexpected EOF in NODE_COORD_SECTION"))?;
                        let mut t = line.split_whitespace();
                        let idx: usize = lines.parse_next(&mut t, "node index")?;
                        let x: f64 = lines.parse_next(&mut t, "x coordinate")?;
                        let y: f64 = lines.parse_next(&mut t, "y coordinate")?;
                        if idx < 1 || idx > dim {
                            return Err(
                                lines.err(format!("node index {idx} out of range 1..={dim}"))
                            );
                        }
                        coordinates[idx - 1] = (x, y);
                    }
                }
                "DEMAND_SECTION" => {
                    for _ in 0..dim {
                        let line = lines
                            .next_data_line()?
                            .ok_or_else(|| lines.err("unexpected EOF in DEMAND_SECTION"))?;
                        let mut t = line.split_whitespace();
                        let idx: usize = lines.parse_next(&mut t, "node index")?;
                        let d: i64 = lines.parse_next(&mut t, "demand")?;
                        if idx < 1 || idx > dim {
                            return Err(
                                lines.err(format!("node index {idx} out of range 1..={dim}"))
                            );
                        }
                        demands[idx - 1] = d;
                    }
                }
                "DEPOT_SECTION" => {
                    // First entry is the depot id (1-indexed); list ends with -1.
                    let line = lines
                        .next_data_line()?
                        .ok_or_else(|| lines.err("unexpected EOF in DEPOT_SECTION"))?;
                    let mut t = line.split_whitespace();
                    let depot: i64 = lines.parse_next(&mut t, "depot id")?;
                    depot_index = Some(depot as usize);
                    // Consume until the terminating -1 (and any extra depots, unsupported).
                    // Peek subsequent data lines for -1.
                    loop {
                        let l = lines.next_data_line()?;
                        match l {
                            None => break,
                            Some(l) => {
                                let v: i64 = l
                                    .split_whitespace()
                                    .next()
                                    .unwrap_or("-1")
                                    .parse()
                                    .unwrap_or(-1);
                                if v == -1 {
                                    break;
                                }
                            }
                        }
                    }
                }
                "EOF" => break,
                other => {
                    return Err(lines.err(format!("unexpected section '{other}'")));
                }
            }

            // Advance to the next section keyword (or EOF).
            match lines.next_data_line()? {
                None => break,
                Some(line) => {
                    section = line
                        .trim()
                        .split(|c: char| c == ':' || c.is_whitespace())
                        .next()
                        .unwrap_or("EOF")
                        .to_ascii_uppercase();
                }
            }
        }

        // Re-index so the depot is node 0. CVRPLIB depots are almost always node 1;
        // if it is elsewhere, swap it into slot 0.
        let depot = depot_index.unwrap_or(1);
        if depot >= 1 && depot <= dim && depot != 1 {
            coordinates.swap(0, depot - 1);
            demands.swap(0, depot - 1);
        }
        demands[0] = 0; // depot carries no demand

        let name = name.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("vrp")
                .to_string()
        });

        Ok(Self::plain(
            name,
            coordinates,
            demands,
            capacity,
            num_vehicles,
            true,
        ))
    }
}

/// Extracts `K` from a `No of trucks: K` style comment fragment.
fn parse_trucks_from_comment(comment: &str) -> Option<usize> {
    let lower = comment.to_ascii_lowercase();
    let idx = lower.find("trucks")?;
    let rest = &comment[idx + "trucks".len()..];
    // Skip separators like ':' and spaces, then read the first integer.
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<usize>().ok()
}

// ---------------------------------------------------------------------------
// TOML deserialization scaffold (private to this module)
// ---------------------------------------------------------------------------

fn default_cost_weight() -> f64 {
    1.0
}

fn default_max_route_time() -> f64 {
    f64::INFINITY
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    #[serde(default)]
    name: Option<String>,
    objective_mode: ObjectiveMode,
    #[serde(default = "default_cost_weight")]
    cost_weight: f64,
    #[serde(default)]
    rounded: bool,
    depot: RawDepot,
    #[serde(default)]
    vehicle_types: Vec<VehicleType>,
    #[serde(default)]
    customers: Vec<RawCustomer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDepot {
    x: f64,
    y: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCustomer {
    id: usize,
    x: f64,
    y: f64,
    demand: i64,
    #[serde(default)]
    service_time: f64,
}

impl ProblemTrait for Vrp {
    type Solution = VrpSolution;

    /// Randomized capacity-aware greedy assignment: the customers are shuffled
    /// with `rng`, then each is placed on the slot with the most *remaining*
    /// capacity among those that can still take it feasibly (falling back to the
    /// slot with the most remaining capacity overall). Remaining capacity is
    /// per-slot, so the larger vehicle types fill first — which is also what
    /// keeps a heterogeneous fleet feasible when the small types alone cannot
    /// carry the demand.
    ///
    /// Route-time limits and minimum counts are not considered here; the search
    /// resolves them through the penalty.
    fn new_solution(&self, rng: &mut impl rand::Rng) -> VrpSolution {
        let n = self.get_n();
        let slots = self.num_slots();
        let mut routes: Vec<Vec<usize>> = vec![Vec::new(); slots];
        let mut remaining: Vec<i64> = (0..slots)
            .map(|s| self.vehicle_type_of_slot(s).capacity)
            .collect();

        let mut customers: Vec<usize> = (1..=n).collect();
        customers.shuffle(rng);

        for c in customers {
            let d = self.demands[c];
            // A slot that still fits the customer beats one that does not,
            // then the most remaining capacity, then the lowest slot index.
            let s = (0..slots)
                .max_by_key(|&s| (remaining[s] >= d, remaining[s], std::cmp::Reverse(s)))
                .expect("a Vrp instance always has at least one slot");
            routes[s].push(c);
            remaining[s] -= d;
        }

        self.solution_from_routes(routes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Two truck slots (capacity 10, speed 1, fixed 5, variable 0.1, min 1,
    /// route-time limit 12) and one van slot (capacity 5, speed 2, fixed 1,
    /// variable 0.5, min 1, no limit).
    ///
    /// Depot at the origin; c1 = (3, 0), c2 = (0, 4), c3 = (-1, 0), so every
    /// distance used below is exact in binary floating point except d(c2, c3).
    fn fixture(mode: ObjectiveMode) -> Vrp {
        Vrp::with_fleet(
            "fix",
            vec![(0.0, 0.0), (3.0, 0.0), (0.0, 4.0), (-1.0, 0.0)],
            vec![0, 6, 6, 2],
            vec![0.0, 1.0, 2.0, 0.5],
            vec![
                VehicleType::new("truck", 10, 1.0, 2)
                    .with_costs(5.0, 0.1)
                    .with_min_count(1)
                    .with_max_route_time(12.0),
                VehicleType::new("van", 5, 2.0, 1)
                    .with_costs(1.0, 0.5)
                    .with_min_count(1),
            ],
            mode,
            1.0,
            false,
        )
    }

    #[test]
    fn slots_are_contiguous_blocks_in_declaration_order() {
        let prob = fixture(ObjectiveMode::TotalTime);
        assert_eq!(prob.num_slots(), 3);
        assert_eq!(prob.type_of_slot(0), 0);
        assert_eq!(prob.type_of_slot(1), 0);
        assert_eq!(prob.type_of_slot(2), 1);
        assert_eq!(prob.vehicle_type_of_slot(2).name, "van");
        assert_eq!(prob.get_n(), 3);
    }

    #[test]
    fn solution_from_routes_computes_every_cached_field() {
        let prob = fixture(ObjectiveMode::TotalTime);
        // Slot 0 (truck): depot -> c1 -> c2 -> depot = 3 + 5 + 4 = 12 distance,
        // load 12 (overload 2), time 12/1 + (1 + 2) = 15 (excess 3 over 12).
        // Slot 2 (van): depot -> c3 -> depot = 2 distance, load 2, time 2/2 + 0.5 = 1.5.
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![], vec![3]]);

        assert_eq!(sol.route_loads, vec![12, 0, 2]);
        assert!((sol.route_distance[0] - 12.0).abs() < 1e-9);
        assert_eq!(sol.route_distance[1], 0.0);
        assert!((sol.route_distance[2] - 2.0).abs() < 1e-9);
        assert!((sol.route_time[0] - 15.0).abs() < 1e-9);
        assert_eq!(sol.route_time[1], 0.0);
        assert!((sol.route_time[2] - 1.5).abs() < 1e-9);
        assert_eq!(sol.used_count, vec![1, 1]);
        assert!((sol.total_time - 16.5).abs() < 1e-9);
        assert!((sol.makespan - 15.0).abs() < 1e-9);
        assert_eq!(sol.overload, 2);
        assert!((sol.time_excess - 3.0).abs() < 1e-9);
        assert_eq!(sol.min_count_shortfall, 0);
        // Fixed 6.0 (two used trucks and a van) plus variable 2.2.
        assert!((sol.total_cost - 8.2).abs() < 1e-9);

        let expected = 16.5 + 8.2 + prob.penalty_weight() * 5.0;
        assert!((sol.objective - expected).abs() < 1e-6);
    }

    #[test]
    fn empty_fleet_type_counts_as_a_shortfall() {
        let prob = fixture(ObjectiveMode::TotalTime);
        // Everything on the trucks: the van's min_count of 1 is unmet.
        let sol = prob.solution_from_routes(vec![vec![1], vec![2, 3], vec![]]);
        assert_eq!(sol.used_count, vec![2, 0]);
        assert_eq!(sol.min_count_shortfall, 1);
    }

    #[test]
    fn objective_mode_changes_the_objective_of_the_same_routes() {
        let routes = vec![vec![1, 2], vec![], vec![3]];
        let total = fixture(ObjectiveMode::TotalTime).solution_from_routes(routes.clone());
        let makespan = fixture(ObjectiveMode::Makespan).solution_from_routes(routes);
        // Same penalty weight (the instances differ only in the aggregation),
        // so the objectives differ by exactly total_time - makespan.
        assert!((total.objective - makespan.objective - (16.5 - 15.0)).abs() < 1e-6);
        assert!(makespan.objective < total.objective);
    }

    #[test]
    fn penalty_weight_dominates_time_and_cost() {
        let prob = fixture(ObjectiveMode::TotalTime);
        // Feasible: the trucks carry one big customer each, the van the small one.
        let feasible = prob.solution_from_routes(vec![vec![1], vec![2], vec![3]]);
        assert_eq!(feasible.overload, 0);
        assert_eq!(feasible.min_count_shortfall, 0);
        assert!(feasible.time_excess < 1e-9);

        // Overload by 2 while being much cheaper in time and cost.
        let overloaded = prob.solution_from_routes(vec![vec![1, 2], vec![], vec![3]]);
        assert_eq!(overloaded.overload, 2);
        assert!(feasible.objective < overloaded.objective);

        // Missing the van's minimum count, again on an otherwise cheap routing.
        let short = prob.solution_from_routes(vec![vec![1], vec![2, 3], vec![]]);
        assert_eq!(short.overload, 0);
        assert_eq!(short.min_count_shortfall, 1);
        assert!(feasible.objective < short.objective);

        // The route-time limit: one truck doing everything exceeds 12.
        let slow = prob.solution_from_routes(vec![vec![1, 2, 3], vec![], vec![]]);
        assert!(slow.time_excess > 1.0);
        assert!(feasible.objective < slow.objective);

        // The weight also dominates the cost term: a fleet-cost saving of a
        // whole vehicle cannot buy a unit of infeasibility.
        assert!(
            prob.penalty_weight() > prob.cost_weight * (2.0 * 5.0 + 1.0),
            "penalty must exceed the total fixed cost of the fleet"
        );
    }

    #[test]
    fn new_solution_is_a_valid_partition() {
        let prob = fixture(ObjectiveMode::Makespan);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(4);
        for _ in 0..20 {
            let sol = prob.new_solution(&mut rng);
            assert_eq!(sol.routes.len(), prob.num_slots());
            prob.validate_routes(&sol.routes).unwrap();
        }
    }

    #[test]
    fn new_solution_is_reproducible_under_a_fixed_seed() {
        let prob = fixture(ObjectiveMode::TotalTime);
        let mut a = rand::rngs::SmallRng::seed_from_u64(9);
        let mut b = rand::rngs::SmallRng::seed_from_u64(9);
        assert_eq!(
            prob.new_solution(&mut a).routes,
            prob.new_solution(&mut b).routes
        );
    }

    #[test]
    fn validate_routes_detects_bad_partitions() {
        let prob = fixture(ObjectiveMode::TotalTime);
        assert!(prob.validate_routes(&[vec![1, 2], vec![], vec![3]]).is_ok());
        assert!(
            prob.validate_routes(&[vec![1, 1], vec![], vec![3]])
                .is_err()
        );
        assert!(prob.validate_routes(&[vec![1, 2], vec![], vec![]]).is_err());
        assert!(
            prob.validate_routes(&[vec![1, 2, 4], vec![], vec![]])
                .is_err()
        );
        // One route per slot, not per used vehicle.
        assert!(prob.validate_routes(&[vec![1, 2], vec![3]]).is_err());
    }

    #[test]
    #[should_panic(expected = "capacity must be positive")]
    fn new_rejects_a_non_positive_capacity() {
        let _ = VehicleType::new("bad", 0, 1.0, 1);
    }

    #[test]
    #[should_panic(expected = "speed must be positive")]
    fn new_rejects_a_non_positive_speed() {
        let _ = VehicleType::new("bad", 5, 0.0, 1);
    }

    #[test]
    #[should_panic(expected = "max_count must be at least 1")]
    fn new_rejects_a_zero_max_count() {
        let _ = VehicleType::new("bad", 5, 1.0, 0);
    }

    #[test]
    #[should_panic(expected = "min_count")]
    fn new_rejects_min_count_above_max_count() {
        let _ = Vrp::with_fleet(
            "bad",
            vec![(0.0, 0.0), (1.0, 0.0)],
            vec![0, 1],
            vec![0.0, 0.0],
            vec![VehicleType::new("t", 5, 1.0, 1).with_min_count(2)],
            ObjectiveMode::TotalTime,
            1.0,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "at least one vehicle type")]
    fn new_rejects_an_empty_fleet() {
        let _ = Vrp::with_fleet(
            "bad",
            vec![(0.0, 0.0), (1.0, 0.0)],
            vec![0, 1],
            vec![0.0, 0.0],
            Vec::new(),
            ObjectiveMode::TotalTime,
            1.0,
            false,
        );
    }

    // -----------------------------------------------------------------------
    // TOML loading
    // -----------------------------------------------------------------------

    fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        path.push(format!(
            "optopus_fleet_vrp_{}_{}_{name}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    const GOOD_TOML: &str = r#"
name = "demo-fleet"
objective_mode = "Makespan"
cost_weight = 2.0
rounded = true

[depot]
x = 0.0
y = 0.0

[[vehicle_types]]
name = "truck"
capacity = 100
speed = 1.0
fixed_cost = 50.0
variable_cost_per_distance = 0.1
min_count = 1
max_count = 2
max_route_time = 480.0

[[vehicle_types]]
name = "van"
capacity = 40
speed = 1.5
max_count = 1

[[customers]]
id = 1
x = 3.0
y = 4.0
demand = 10
service_time = 15.0

[[customers]]
id = 2
x = -3.0
y = 4.0
demand = 20
"#;

    #[test]
    fn load_toml_roundtrip() {
        let path = write_temp("ok", GOOD_TOML);
        let prob = Vrp::load_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(prob.name, "demo-fleet");
        assert_eq!(prob.objective_mode, ObjectiveMode::Makespan);
        assert_eq!(prob.cost_weight, 2.0);
        assert_eq!(prob.rounded(), Some(true));
        assert_eq!(prob.get_n(), 2);
        assert_eq!(prob.demands, vec![0, 10, 20]);
        assert_eq!(prob.service_times, vec![0.0, 15.0, 0.0]);
        let coordinates = prob.coordinates().unwrap();
        assert_eq!(coordinates[0], (0.0, 0.0));
        assert_eq!(coordinates[1], (3.0, 4.0));
        assert_eq!(prob.num_slots(), 3);
        assert_eq!(prob.type_of_slot(1), 0);
        assert_eq!(prob.type_of_slot(2), 1);

        let truck = &prob.vehicle_types[0];
        assert_eq!(truck.capacity, 100);
        assert_eq!(truck.min_count, 1);
        assert_eq!(truck.max_route_time, 480.0);
        // Defaults for the omitted optional fields.
        let van = &prob.vehicle_types[1];
        assert_eq!(van.fixed_cost, 0.0);
        assert_eq!(van.variable_cost_per_distance, 0.0);
        assert_eq!(van.min_count, 0);
        assert!(van.max_route_time.is_infinite());

        // rounded = true: depot -> (3, 4) is exactly 5.
        assert_eq!(prob.distance(0, 1), 5.0);
    }

    #[test]
    fn load_file_defaults_name_to_the_file_stem() {
        let body = GOOD_TOML.replace("name = \"demo-fleet\"\n", "");
        let path = write_temp("stem", &body);
        let prob = Vrp::load_file(&path).unwrap();
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);
        assert_eq!(prob.name, stem);
    }

    fn load_err(name: &str, body: &str) -> OptError {
        let path = write_temp(name, body);
        let err = Vrp::load_file(&path).expect_err("must fail");
        let _ = std::fs::remove_file(&path);
        err
    }

    #[test]
    fn load_file_rejects_invalid_vehicle_types() {
        let err = load_err("cap", &GOOD_TOML.replace("capacity = 100", "capacity = 0"));
        assert!(matches!(err, OptError::Config(_)), "{err}");
        assert!(err.to_string().contains("capacity"), "{err}");

        let err = load_err("speed", &GOOD_TOML.replace("speed = 1.0", "speed = 0.0"));
        assert!(err.to_string().contains("speed"), "{err}");

        let err = load_err("maxc", &GOOD_TOML.replace("max_count = 2", "max_count = 0"));
        assert!(err.to_string().contains("max_count"), "{err}");

        let err = load_err("minc", &GOOD_TOML.replace("min_count = 1", "min_count = 5"));
        assert!(err.to_string().contains("min_count"), "{err}");

        let err = load_err(
            "fleet",
            &GOOD_TOML.replace("[[vehicle_types]]", "[[unused_types]]"),
        );
        // Unknown keys are rejected before the emptiness check.
        assert!(matches!(err, OptError::TomlDe(_)), "{err}");
    }

    #[test]
    fn load_file_rejects_inconsistent_customer_ids() {
        let err = load_err("gap", &GOOD_TOML.replace("id = 2", "id = 3"));
        assert!(err.to_string().contains("out of range"), "{err}");

        let err = load_err("dup", &GOOD_TOML.replace("id = 2", "id = 1"));
        assert!(err.to_string().contains("duplicate"), "{err}");

        let body = GOOD_TOML.split("[[customers]]").next().unwrap().to_string();
        let err = load_err("none", &body);
        assert!(err.to_string().contains("no customers"), "{err}");
    }

    #[test]
    fn load_file_rejects_an_unknown_field() {
        let err = load_err(
            "typo",
            &GOOD_TOML.replace("max_route_time = 480.0", "max_rout_time = 480.0"),
        );
        assert!(matches!(err, OptError::TomlDe(_)), "{err}");
    }

    #[test]
    fn load_file_reports_a_missing_file_as_io() {
        let err = Vrp::load_file("/nonexistent/optopus/fleet.toml").unwrap_err();
        assert!(matches!(err, OptError::Io(_)), "{err}");
    }

    // -----------------------------------------------------------------------
    // The homogeneous CVRP
    // -----------------------------------------------------------------------

    /// Depot at origin + 4 customers on the axes, each demand 1, capacity 2,
    /// 2 vehicles.
    fn square_vrp() -> Vrp {
        Vrp::new(
            "sq",
            vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
            vec![0, 1, 1, 1, 1],
            2,
            2,
        )
    }

    #[test]
    fn route_distance_includes_depot_legs() {
        let vrp = square_vrp();
        // depot(0,0) -> c1(1,0) -> c2(0,1) -> depot : 1 + sqrt(2) + 1
        let d = vrp.route_distance(&[1, 2]);
        assert!((d - (2.0 + 2.0_f64.sqrt())).abs() < 1e-9);
        assert_eq!(vrp.route_distance(&[]), 0.0);
    }

    #[test]
    fn overload_is_penalized() {
        let vrp = square_vrp();
        // All four customers on one route: load 4 > capacity 2 → overload 2.
        let sol = vrp.solution_from_routes(vec![vec![1, 2, 3, 4], vec![]]);
        assert_eq!(sol.overload, 2);
        assert!(sol.objective > sol.total_distance());
        assert!((sol.objective - (sol.total_distance() + vrp.penalty_weight() * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn num_vehicles_defaults_to_a_feasible_fleet() {
        let vrp = Vrp::new(
            "d",
            vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            vec![0, 2, 2, 2],
            3,
            0,
        );
        // The bin-packing bound `ceil(6 / 3) = 2` is not achievable — only one
        // demand-2 customer fits per capacity-3 vehicle — so the default is the
        // 3 vehicles that actually pack, plus the margin.
        assert_eq!(vrp.num_slots(), 4);
        let routes = vec![vec![1], vec![2], vec![3], vec![]];
        assert_eq!(vrp.solution_from_routes(routes).overload, 0);
    }

    #[test]
    fn default_fleet_size_packs_every_customer() {
        // 10 unit demands, capacity 3: four vehicles pack them, plus the margin.
        assert_eq!(default_fleet_size(&[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3), 5);
        // A customer larger than a vehicle still gets a route of its own.
        assert_eq!(default_fleet_size(&[0, 7], 3), 2);
        // No customers at all still leaves one vehicle.
        assert_eq!(default_fleet_size(&[0], 5), 2);
    }

    #[test]
    fn parse_trucks_from_comment_variants() {
        assert_eq!(
            parse_trucks_from_comment("Min no of trucks: 5, Optimal value: 784"),
            Some(5)
        );
        assert_eq!(parse_trucks_from_comment("no trucks here"), None);
    }

    #[test]
    fn load_cvrplib_roundtrip() {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        path.push(format!("optopus_vrp_{}.vrp", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            "NAME : test\n\
             COMMENT : (test, Min no of trucks: 2, Optimal value: 6)\n\
             TYPE : CVRP\n\
             DIMENSION : 5\n\
             EDGE_WEIGHT_TYPE : EUC_2D\n\
             CAPACITY : 2\n\
             NODE_COORD_SECTION\n\
             1 0 0\n\
             2 1 0\n\
             3 0 1\n\
             4 -1 0\n\
             5 0 -1\n\
             DEMAND_SECTION\n\
             1 0\n\
             2 1\n\
             3 1\n\
             4 1\n\
             5 1\n\
             DEPOT_SECTION\n\
             1\n\
             -1\n\
             EOF\n"
        )
        .unwrap();

        let vrp = Vrp::load_file(&path).unwrap();
        assert_eq!(vrp.name, "test");
        assert_eq!(vrp.get_n(), 4);
        assert_eq!(vrp.vehicle_types.len(), 1);
        assert_eq!(vrp.vehicle_type_of_slot(0).capacity, 2);
        assert_eq!(vrp.num_slots(), 2);
        assert_eq!(vrp.rounded(), Some(true));
        assert_eq!(vrp.demands, vec![0, 1, 1, 1, 1]);
        // depot at origin, customer 1 at (1,0): distance 1
        assert_eq!(vrp.distance(0, 1), 1.0);
        let _ = std::fs::remove_file(&path);
    }
    /// The depot and twenty customers on a lattice, so repeated distances are
    /// in play.
    fn lattice() -> (Vec<(f64, f64)>, Vec<i64>) {
        let coords: Vec<(f64, f64)> = (0..21).map(|i| ((i % 5) as f64, (i / 5) as f64)).collect();
        let mut demands = vec![1; 21];
        demands[0] = 0;
        (coords, demands)
    }

    #[test]
    fn from_distance_matrix_reads_the_matrix_as_given() {
        let matrix = vec![
            vec![0.0, 3.0, 7.0],
            vec![3.0, 0.0, 5.0],
            vec![7.0, 5.0, 0.0],
        ];
        let vrp = Vrp::from_distance_matrix("m", matrix, vec![0, 1, 1], 5, 1).unwrap();
        assert_eq!(vrp.get_n(), 2);
        assert_eq!(vrp.distance(0, 2), 7.0);
        assert_eq!(vrp.route_distance(&[1, 2]), 3.0 + 5.0 + 7.0);
        assert!(vrp.coordinates().is_none());
        assert!(vrp.rounded().is_none());
        assert!(vrp.penalty_weight() > 3.0 * 7.0);
    }

    #[test]
    fn nearest_store_answers_like_the_full_matrix() {
        let (coords, demands) = lattice();
        let full = Vrp::with_rounding("full", coords.clone(), demands.clone(), 5, 0);
        let sparse = Vrp::with_nearest_neighbors("sparse", coords, demands, 5, 0, true, 4);
        assert_eq!(sparse.rounded(), Some(true));
        assert_eq!(sparse.coordinates(), full.coordinates());
        assert_eq!(sparse.num_slots(), full.num_slots());
        for i in 0..21 {
            for j in 0..21 {
                assert_eq!(
                    sparse.distance(i, j),
                    full.distance(i, j),
                    "pair ({i}, {j})"
                );
            }
        }
        assert_eq!(sparse.penalty_weight(), full.penalty_weight());
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
        let sol = full.new_solution(&mut rng);
        let again = sparse.solution_from_routes(sol.routes.clone());
        assert_eq!(again.objective, sol.objective);
    }

    #[test]
    fn load_file_above_the_cap_keeps_the_coordinates() {
        use std::io::Write;
        let dim = Vrp::DIST_MATRIX_MAX_N + 1;
        let mut contents = format!(
            "NAME : big\nTYPE : CVRP\nDIMENSION : {dim}\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : 50\nNODE_COORD_SECTION\n"
        );
        for i in 0..dim {
            contents.push_str(&format!("{} {} {}\n", i + 1, (i * 7) % 101, (i * 13) % 89));
        }
        contents.push_str("DEMAND_SECTION\n");
        for i in 0..dim {
            contents.push_str(&format!("{} {}\n", i + 1, if i == 0 { 0 } else { 1 }));
        }
        contents.push_str("DEPOT_SECTION\n1\n-1\nEOF\n");
        let mut path = std::env::temp_dir();
        path.push(format!("optopus_vrp_big_{}.vrp", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(contents.as_bytes())
            .unwrap();
        let vrp = Vrp::load_file(&path).unwrap();
        assert_eq!(vrp.get_n(), dim - 1);
        assert_eq!(vrp.coordinates().map(|c| c.len()), Some(dim));
        assert_eq!(vrp.distance(0, 1), 7f64.hypot(13.0).round());
        assert_eq!(vrp.distance(0, 1), vrp.distance(1, 0));
        let _ = std::fs::remove_file(&path);
    }
}
