use rand::seq::SliceRandom;
use std::sync::OnceLock;

use super::adjacency::RouteAdjacency;
use crate::error::OptError;
use crate::search_state::{Distance, ProblemTrait};

/// Returns the capacity overflow of a route load: `max(0, load - capacity)`.
#[inline]
pub(crate) fn overload_of(load: i64, capacity: i64) -> i64 {
    (load - capacity).max(0)
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

/// A solution to the Capacitated Vehicle Routing Problem.
///
/// The fleet is fixed at [`Vrp::num_vehicles`] routes; `routes.len()` therefore
/// always equals `num_vehicles` (idle vehicles are represented by empty routes).
/// Each route is the ordered list of customer indices (`1..=n`) it visits; the
/// depot (index `0`) is implicit at the start and end of every route.
///
/// Capacity is a soft constraint handled with a penalty, exactly like
/// [`crate::problem::VertexCover`]: `objective = distance + penalty_weight * overload`,
/// with [`Vrp::penalty_weight`] large enough that any optimum is feasible.
/// Move gains are computed on the fly from [`Vrp::distance`] (backed by the lazily
/// built distance matrix).
#[derive(Debug, Clone)]
pub struct VrpSolution {
    /// `routes[r]` is the ordered list of customers served by vehicle `r`
    /// (depot implicit at both ends). Length is always `num_vehicles`.
    pub routes: Vec<Vec<usize>>,
    /// Cached total demand of each route.
    pub route_loads: Vec<i64>,
    /// True total travel distance (depot → customers → depot, summed over routes).
    pub distance: f64,
    /// Total capacity overflow `Σ max(0, load_r − Q)`.
    pub overload: i64,
    /// Penalty-augmented objective: `distance + penalty_weight * overload`.
    pub objective: f64,
}

impl crate::trait_defs::Evaluate for VrpSolution {
    /// VRP minimizes its penalty-augmented objective.
    fn evaluate(&self) -> crate::trait_defs::Evaluable<f64> {
        crate::trait_defs::Evaluable::Minimize(self.objective)
    }
}

impl Distance for VrpSolution {
    /// Broken-pairs dissimilarity: how many of the two solutions' customer
    /// adjacencies the other one does not have.
    ///
    /// It counts trips, not vehicle labels, so permuting the routes or driving
    /// one of them backwards is a distance of `0`, which the obvious
    /// alternative, comparing each customer's route index, gets wrong on exactly
    /// the solutions a search meets most often. Two solutions are at distance
    /// `0` precisely when they describe the same set of trips.
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

/// The coordinates an instance was built from and whether a distance is
/// rounded to the nearest integer. Only instances built from coordinates have
/// one, so [`Vrp::from_distance_matrix`] instances have no geometry at all.
#[derive(Debug, Clone)]
struct Geometry {
    /// Node coordinates; index `0` is the depot, `1..=n` are the customers.
    coordinates: Vec<(f64, f64)>,
    rounded: bool,
}

impl Geometry {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let (x1, y1) = self.coordinates[i];
        let (x2, y2) = self.coordinates[j];
        let dx = x1 - x2;
        let dy = y1 - y2;
        let d = (dx * dx + dy * dy).sqrt();
        if self.rounded { d.round() } else { d }
    }

    /// The `k` nearest other nodes of every node with their distances, in
    /// ascending distance, ties by node index.
    fn nearest_rows(&self, k: usize) -> Vec<Vec<(usize, f64)>> {
        let n = self.coordinates.len();
        (0..n)
            .map(|i| {
                let mut row: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, self.distance(i, j)))
                    .collect();
                row.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                row.truncate(k);
                row
            })
            .collect()
    }
}

/// Where an instance keeps its distances.
#[derive(Debug, Clone)]
enum DistanceStore {
    /// The full `nodes × nodes` matrix, row-major. The geometry is present
    /// only when the matrix was built from coordinates.
    Full {
        matrix: Vec<f64>,
        geometry: Option<Geometry>,
    },
    /// For each node, its `k` nearest other nodes with their distances, in
    /// ascending distance. Any other pair is computed from the geometry, so
    /// the geometry is not optional here.
    Nearest {
        rows: Vec<Vec<(usize, f64)>>,
        geometry: Geometry,
    },
}

/// The Capacitated Vehicle Routing Problem (CVRP).
///
/// A depot (node `0`) and `n` customers (`1..=n`) with integer demands are
/// served by a homogeneous fleet of `num_vehicles` vehicles, each of capacity
/// `capacity`. The objective is to minimize total travel distance such that
/// every customer is visited exactly once and no route's demand exceeds
/// capacity (the latter enforced via a penalty, see [`VrpSolution`]).
///
/// An instance is built from 2D coordinates, either as the full distance
/// matrix ([`Vrp::new`], [`Vrp::with_rounding`]) or as the `k` nearest
/// neighbours of every node with the remaining pairs computed from the
/// coordinates on demand ([`Vrp::with_nearest_neighbors`]), or from a
/// distance matrix given directly ([`Vrp::from_distance_matrix`]). Distances
/// from coordinates mirror TSPLIB semantics: rounding selects nearest-integer
/// `EUC_2D` (the CVRPLIB standard, used by [`Vrp::load_file`]) versus plain
/// Euclidean (the default for programmatically constructed instances).
#[derive(Debug, Clone)]
pub struct Vrp {
    pub name: String,
    /// Node demands; `demands[0] == 0` (the depot).
    pub demands: Vec<i64>,
    /// Vehicle capacity `Q`.
    pub capacity: i64,
    /// Fixed fleet size (number of routes in every solution).
    pub num_vehicles: usize,
    distances: DistanceStore,
    /// Lazily computed penalty weight (see [`Vrp::penalty_weight`]).
    penalty_weight: OnceLock<f64>,
}

impl Vrp {
    /// Largest number of nodes (depot + customers) for which [`Vrp::load_file`]
    /// keeps the full distance matrix (memory `nodes² × 8` bytes, 32 MB at
    /// this cap). Larger files keep [`Vrp::NEAREST_NEIGHBORS_ABOVE_CAP`]
    /// neighbours per node.
    pub const DIST_MATRIX_MAX_N: usize = 2000;
    /// Neighbours per node that [`Vrp::load_file`] keeps above the cap, the
    /// granularity the descents use by default.
    pub const NEAREST_NEIGHBORS_ABOVE_CAP: usize = 20;

    /// Creates a CVRP instance with plain (non-rounded) Euclidean distances.
    ///
    /// `coordinates[0]` / `demands[0]` are the depot (demand ignored). If
    /// `num_vehicles == 0` it defaults to `default_fleet_size`: the demands
    /// packed first-fit-decreasing, plus a 10% margin. Not
    /// `ceil(total_demand / capacity)`, which is only a lower bound on the bin
    /// count and often admits no feasible assignment.
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
        Self::build(
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
        Self::build(
            name.into(),
            coordinates,
            demands,
            capacity,
            num_vehicles,
            true,
        )
    }

    fn build(
        name: String,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
        rounded: bool,
    ) -> Self {
        let geometry = Geometry {
            coordinates,
            rounded,
        };
        let nodes = geometry.coordinates.len();
        let mut matrix = Vec::with_capacity(nodes * nodes);
        for a in 0..nodes {
            for b in 0..nodes {
                matrix.push(geometry.distance(a, b));
            }
        }
        Self::from_store(
            name,
            nodes,
            DistanceStore::Full {
                matrix,
                geometry: Some(geometry),
            },
            demands,
            capacity,
            num_vehicles,
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
        let geometry = Geometry {
            coordinates,
            rounded,
        };
        let nodes = geometry.coordinates.len();
        let rows = geometry.nearest_rows(k.min(nodes.saturating_sub(1)));
        Self::from_store(
            name.into(),
            nodes,
            DistanceStore::Nearest { rows, geometry },
            demands,
            capacity,
            num_vehicles,
        )
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
        let nodes = matrix.len();
        let mut flat = Vec::with_capacity(nodes * nodes);
        for (i, row) in matrix.into_iter().enumerate() {
            if row.len() != nodes {
                return Err(OptError::InvalidState(format!(
                    "distance matrix row {} has {} entries, expected {}",
                    i,
                    row.len(),
                    nodes
                )));
            }
            flat.extend(row);
        }
        Ok(Self::from_store(
            name.into(),
            nodes,
            DistanceStore::Full {
                matrix: flat,
                geometry: None,
            },
            demands,
            capacity,
            num_vehicles,
        ))
    }

    fn from_store(
        name: String,
        nodes: usize,
        distances: DistanceStore,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
    ) -> Self {
        assert!(nodes > 0, "VRP requires at least the depot node");
        assert_eq!(
            nodes,
            demands.len(),
            "distances and demands must cover the same nodes"
        );
        assert!(capacity > 0, "capacity must be positive");

        let num_vehicles = if num_vehicles == 0 {
            default_fleet_size(&demands, capacity)
        } else {
            num_vehicles
        };

        Self {
            name,
            demands,
            capacity,
            num_vehicles,
            distances,
            penalty_weight: OnceLock::new(),
        }
    }

    /// Number of customers (excludes the depot).
    pub fn get_n(&self) -> usize {
        self.demands.len() - 1
    }

    /// Number of nodes (depot + customers).
    fn num_nodes(&self) -> usize {
        self.demands.len()
    }

    /// The node coordinates the instance was built from, index `0` the
    /// depot, `None` for an instance built from a distance matrix.
    pub fn coordinates(&self) -> Option<&[(f64, f64)]> {
        self.geometry().map(|g| g.coordinates.as_slice())
    }

    /// Whether distances from the coordinates are rounded to the nearest
    /// integer (`EUC_2D`), `None` for an instance built from a distance
    /// matrix.
    pub fn rounded(&self) -> Option<bool> {
        self.geometry().map(|g| g.rounded)
    }

    fn geometry(&self) -> Option<&Geometry> {
        match &self.distances {
            DistanceStore::Full { geometry, .. } => geometry.as_ref(),
            DistanceStore::Nearest { geometry, .. } => Some(geometry),
        }
    }

    /// Distance between nodes `i` and `j` (either may be the depot, index `0`).
    ///
    /// A full matrix answers with one lookup. A nearest-neighbour store scans
    /// the short list of `i` and falls back to the coordinates for any pair
    /// outside it, so the value is the same either way.
    #[inline]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        match &self.distances {
            DistanceStore::Full { matrix, .. } => matrix[i * self.num_nodes() + j],
            DistanceStore::Nearest { rows, geometry } => rows[i]
                .iter()
                .find(|(c, _)| *c == j)
                .map_or_else(|| geometry.distance(i, j), |(_, d)| *d),
        }
    }

    /// Penalty weight applied per unit of capacity overflow.
    ///
    /// Chosen strictly larger than the maximum possible total distance
    /// (`nodes × max_edge`), so any optimum of `distance + weight * overload` is
    /// feasible (overload `0`) whenever a feasible solution exists.
    pub fn penalty_weight(&self) -> f64 {
        *self.penalty_weight.get_or_init(|| {
            let n = self.num_nodes();
            let mut max_edge = 0.0_f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = self.distance(i, j);
                    if d > max_edge {
                        max_edge = d;
                    }
                }
            }
            (self.get_n() + self.num_vehicles) as f64 * max_edge + 1.0
        })
    }

    /// Distance of a single route: `depot → route[0] → … → route[last] → depot`.
    /// An empty route has distance `0`.
    pub fn route_distance(&self, route: &[usize]) -> f64 {
        if route.is_empty() {
            return 0.0;
        }
        let mut d = self.distance(0, route[0]);
        for w in route.windows(2) {
            d += self.distance(w[0], w[1]);
        }
        d + self.distance(route[route.len() - 1], 0)
    }

    /// Builds a [`VrpSolution`] from a route partition, computing all cached
    /// fields (loads, distance, overload, objective). Assumes `routes` is a valid
    /// partition of the customers; use [`Vrp::validate_routes`] to check.
    pub fn solution_from_routes(&self, routes: Vec<Vec<usize>>) -> VrpSolution {
        let route_loads: Vec<i64> = routes
            .iter()
            .map(|r| r.iter().map(|&c| self.demands[c]).sum())
            .collect();
        let distance: f64 = routes.iter().map(|r| self.route_distance(r)).sum();
        let overload: i64 = route_loads
            .iter()
            .map(|&l| overload_of(l, self.capacity))
            .sum();
        let objective = distance + self.penalty_weight() * overload as f64;
        VrpSolution {
            routes,
            route_loads,
            distance,
            overload,
            objective,
        }
    }

    /// Validates that `routes` visits every customer `1..=n` exactly once.
    pub fn validate_routes(&self, routes: &[Vec<usize>]) -> Result<(), OptError> {
        let n = self.get_n();
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

    /// Loads a CVRP instance from a CVRPLIB-format file.
    ///
    /// Accepts the standard TSPLIB-style header (`NAME`, `DIMENSION`,
    /// `EDGE_WEIGHT_TYPE: EUC_2D`, `CAPACITY`, and an optional
    /// `COMMENT` carrying `No of trucks: K`), followed by the
    /// `NODE_COORD_SECTION`, `DEMAND_SECTION`, and `DEPOT_SECTION`. Node `1` in
    /// the file (the depot per `DEPOT_SECTION`) is re-indexed to `0` internally.
    ///
    /// Files with at most [`Vrp::DIST_MATRIX_MAX_N`] nodes get the full
    /// distance matrix, larger ones keep
    /// [`Vrp::NEAREST_NEIGHBORS_ABOVE_CAP`] neighbours per node.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, OptError> {
        use crate::common::InstanceLines;

        let path = path.as_ref();
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

        Ok(if dim <= Self::DIST_MATRIX_MAX_N {
            Self::build(name, coordinates, demands, capacity, num_vehicles, true)
        } else {
            Self::with_nearest_neighbors(
                name,
                coordinates,
                demands,
                capacity,
                num_vehicles,
                true,
                Self::NEAREST_NEIGHBORS_ABOVE_CAP,
            )
        })
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

impl ProblemTrait for Vrp {
    type Solution = VrpSolution;

    /// Randomized least-loaded assignment: customers are shuffled and each is
    /// placed on the least-loaded route that keeps it feasible (or the globally
    /// least-loaded route if none can). Always produces exactly `num_vehicles`
    /// routes.
    fn new_solution(&self, rng: &mut impl rand::Rng) -> VrpSolution {
        let n = self.get_n();
        let v = self.num_vehicles;
        let mut routes: Vec<Vec<usize>> = vec![Vec::new(); v];
        let mut loads: Vec<i64> = vec![0; v];

        let mut customers: Vec<usize> = (1..=n).collect();
        customers.shuffle(rng);

        for c in customers {
            let d = self.demands[c];
            let feasible = (0..v)
                .filter(|&r| loads[r] + d <= self.capacity)
                .min_by_key(|&r| loads[r]);
            let r = feasible
                .unwrap_or_else(|| (0..v).min_by_key(|&r| loads[r]).expect("num_vehicles >= 1"));
            routes[r].push(c);
            loads[r] += d;
        }

        self.solution_from_routes(routes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

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
        assert!(sol.objective > sol.distance);
        assert!((sol.objective - (sol.distance + vrp.penalty_weight() * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn penalty_weight_dominates_distance() {
        let vrp = square_vrp();
        let feasible = vrp.solution_from_routes(vec![vec![1, 2], vec![3, 4]]);
        let infeasible = vrp.solution_from_routes(vec![vec![1, 2, 3], vec![4]]);
        assert_eq!(infeasible.overload, 1);
        assert!(
            feasible.objective < infeasible.objective,
            "any feasible solution must beat any infeasible one"
        );
    }

    #[test]
    fn new_solution_is_valid_partition() {
        let vrp = square_vrp();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        let sol = vrp.new_solution(&mut rng);
        assert_eq!(sol.routes.len(), 2);
        vrp.validate_routes(&sol.routes).unwrap();
    }

    #[test]
    fn validate_routes_detects_duplicates_and_missing() {
        let vrp = square_vrp();
        assert!(vrp.validate_routes(&[vec![1, 2, 3, 4], vec![]]).is_ok());
        assert!(vrp.validate_routes(&[vec![1, 1, 3, 4], vec![]]).is_err());
        assert!(vrp.validate_routes(&[vec![1, 2, 3], vec![]]).is_err());
        assert!(vrp.validate_routes(&[vec![1, 2, 3, 5], vec![]]).is_err());
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
        assert_eq!(vrp.num_vehicles, 4);
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
    fn load_file_roundtrip() {
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
        assert_eq!(vrp.capacity, 2);
        assert_eq!(vrp.num_vehicles, 2);
        assert_eq!(vrp.rounded(), Some(true));
        assert_eq!(vrp.demands, vec![0, 1, 1, 1, 1]);
        // depot at origin, customer 1 at (1,0): distance 1
        assert_eq!(vrp.distance(0, 1), 1.0);
        let _ = std::fs::remove_file(&path);
    }
    /// The depot and twenty customers on a lattice, so repeated distances
    /// exercise the tie breaking of the nearest-neighbour rows.
    fn lattice() -> (Vec<(f64, f64)>, Vec<i64>) {
        let coords: Vec<(f64, f64)> = (0..21).map(|i| ((i % 5) as f64, (i / 5) as f64)).collect();
        let mut demands = vec![1; 21];
        demands[0] = 0;
        (coords, demands)
    }

    #[test]
    fn from_distance_matrix_rejects_a_ragged_matrix() {
        let ragged = vec![vec![0.0, 1.0], vec![1.0, 0.0, 2.0]];
        assert!(Vrp::from_distance_matrix("ragged", ragged, vec![0, 1], 5, 1).is_err());
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
        assert_eq!(sparse.num_vehicles, full.num_vehicles);
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
    fn nearest_store_clamps_k_to_the_other_nodes() {
        let vrp = Vrp::with_nearest_neighbors(
            "tiny",
            vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)],
            vec![0, 1, 1],
            5,
            1,
            false,
            10,
        );
        assert_eq!(vrp.distance(1, 2), 5f64.sqrt());
        assert_eq!(vrp.distance(2, 0), 2.0);
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
