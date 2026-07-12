use rand::seq::SliceRandom;
use std::sync::OnceLock;

use crate::error::OptError;
use crate::search_state::{Distance, ProblemTrait, Rankable};

/// Returns the capacity overflow of a route load: `max(0, load - capacity)`.
#[inline]
pub(crate) fn overload_of(load: i64, capacity: i64) -> i64 {
    (load - capacity).max(0)
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

impl Rankable for VrpSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

impl Distance for VrpSolution {
    /// Position-based dissimilarity: number of customers assigned to a different
    /// route index in the two solutions. A rough diversity proxy for GA parent
    /// selection (route indices are not canonicalized).
    fn distance(&self, other: &Self) -> usize {
        let a = self.customer_route_map();
        let b = other.customer_route_map();
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }
}

impl VrpSolution {
    /// Builds `map[c] = route index serving customer c` (`map[0]` unused).
    fn customer_route_map(&self) -> Vec<usize> {
        let n: usize = self.routes.iter().map(|r| r.len()).sum();
        let mut map = vec![usize::MAX; n + 1];
        for (r, route) in self.routes.iter().enumerate() {
            for &c in route {
                if c < map.len() {
                    map[c] = r;
                }
            }
        }
        map
    }
}

/// Maximum number of nodes (depot + customers) for which the full distance
/// matrix is precomputed (memory `nodes² × 8` bytes). Larger instances fall back
/// to computing each distance on the fly.
pub const VRP_DIST_MATRIX_MAX_N: usize = 2000;

/// The Capacitated Vehicle Routing Problem (CVRP).
///
/// A depot (node `0`) and `n` customers (`1..=n`) with 2D coordinates and
/// integer demands are served by a homogeneous fleet of `num_vehicles` vehicles,
/// each of capacity `capacity`. The objective is to minimize total travel
/// distance such that every customer is visited exactly once and no route's
/// demand exceeds capacity (the latter enforced via a penalty, see [`VrpSolution`]).
///
/// Distances mirror TSPLIB semantics: `rounded` selects nearest-integer `EUC_2D`
/// (the CVRPLIB standard, used by [`Vrp::load_file`]) versus plain Euclidean
/// (the default for programmatically constructed instances).
#[derive(Debug, Clone)]
pub struct Vrp {
    pub name: String,
    /// Node coordinates; index `0` is the depot, `1..=n` are the customers.
    pub coordinates: Vec<(f64, f64)>,
    /// Node demands; `demands[0] == 0` (the depot).
    pub demands: Vec<i64>,
    /// Vehicle capacity `Q`.
    pub capacity: i64,
    /// Fixed fleet size (number of routes in every solution).
    pub num_vehicles: usize,
    /// Whether distances are rounded to the nearest integer (`EUC_2D`).
    pub rounded: bool,
    /// Lazily built `nodes × nodes` distance matrix (row-major); only populated
    /// when `nodes <= VRP_DIST_MATRIX_MAX_N`.
    dist_matrix: OnceLock<Vec<f64>>,
    /// Lazily computed penalty weight (see [`Vrp::penalty_weight`]).
    penalty_weight: OnceLock<f64>,
}

impl Vrp {
    /// Creates a CVRP instance with plain (non-rounded) Euclidean distances.
    ///
    /// `coordinates[0]` / `demands[0]` are the depot (demand ignored). If
    /// `num_vehicles == 0` it defaults to `ceil(total_demand / capacity)`.
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
        Self::build(name.into(), coordinates, demands, capacity, num_vehicles, false)
    }

    /// Like [`Vrp::new`] but with nearest-integer `EUC_2D` distances.
    pub fn with_rounding(
        name: impl Into<String>,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
    ) -> Self {
        Self::build(name.into(), coordinates, demands, capacity, num_vehicles, true)
    }

    fn build(
        name: String,
        coordinates: Vec<(f64, f64)>,
        demands: Vec<i64>,
        capacity: i64,
        num_vehicles: usize,
        rounded: bool,
    ) -> Self {
        assert!(
            !coordinates.is_empty(),
            "VRP requires at least the depot node"
        );
        assert_eq!(
            coordinates.len(),
            demands.len(),
            "coordinates and demands must have the same length"
        );
        assert!(capacity > 0, "capacity must be positive");

        let num_vehicles = if num_vehicles == 0 {
            let total: i64 = demands.iter().sum();
            (total.max(0) as usize).div_ceil(capacity as usize).max(1)
        } else {
            num_vehicles
        };

        Self {
            name,
            coordinates,
            demands,
            capacity,
            num_vehicles,
            rounded,
            dist_matrix: OnceLock::new(),
            penalty_weight: OnceLock::new(),
        }
    }

    /// Number of customers (excludes the depot).
    pub fn get_n(&self) -> usize {
        self.coordinates.len() - 1
    }

    /// Number of nodes (depot + customers).
    fn num_nodes(&self) -> usize {
        self.coordinates.len()
    }

    /// Distance between nodes `i` and `j` (either may be the depot, index `0`).
    ///
    /// O(1) lookup once the lazily built distance matrix is populated (instances
    /// with at most [`VRP_DIST_MATRIX_MAX_N`] nodes); larger instances compute on
    /// the fly.
    #[inline]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let n = self.num_nodes();
        if n <= VRP_DIST_MATRIX_MAX_N {
            let matrix = self.dist_matrix.get_or_init(|| {
                let mut m = Vec::with_capacity(n * n);
                for a in 0..n {
                    for b in 0..n {
                        m.push(self.compute_distance(a, b));
                    }
                }
                m
            });
            return matrix[i * n + j];
        }
        self.compute_distance(i, j)
    }

    fn compute_distance(&self, i: usize, j: usize) -> f64 {
        let (x1, y1) = self.coordinates[i];
        let (x2, y2) = self.coordinates[j];
        let dx = x1 - x2;
        let dy = y1 - y2;
        let d = (dx * dx + dy * dy).sqrt();
        if self.rounded { d.round() } else { d }
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
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, OptError> {
        use crate::common::InstanceLines;

        let path = path.as_ref();
        let mut lines = InstanceLines::open(path)?;

        let mut name: Option<String> = None;
        let mut dimension: Option<usize> = None;
        let mut capacity: Option<i64> = None;
        let mut num_vehicles: usize = 0;

        // Header: parse until the first *_SECTION keyword.
        let first_section = loop {
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
                    capacity = Some(
                        value
                            .parse::<i64>()
                            .map_err(|e| lines.err(format!("failed to parse CAPACITY value: {e}")))?,
                    );
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
                            return Err(lines.err(format!("node index {idx} out of range 1..={dim}")));
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
                            return Err(lines.err(format!("node index {idx} out of range 1..={dim}")));
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
                                let v: i64 =
                                    l.split_whitespace().next().unwrap_or("-1").parse().unwrap_or(-1);
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

        Ok(Self::build(
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
            let r = feasible.unwrap_or_else(|| {
                (0..v).min_by_key(|&r| loads[r]).expect("num_vehicles >= 1")
            });
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
    fn get_n_excludes_depot() {
        assert_eq!(square_vrp().get_n(), 4);
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
    fn solution_from_routes_computes_fields() {
        let vrp = square_vrp();
        let sol = vrp.solution_from_routes(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(sol.route_loads, vec![2, 2]);
        assert_eq!(sol.overload, 0);
        assert!((sol.objective - sol.distance).abs() < 1e-9);
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
    fn num_vehicles_defaults_to_demand_over_capacity() {
        let vrp = Vrp::new(
            "d",
            vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            vec![0, 2, 2, 2],
            3,
            0,
        );
        // total demand 6 / capacity 3 = 2 vehicles
        assert_eq!(vrp.num_vehicles, 2);
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
        assert!(vrp.rounded);
        assert_eq!(vrp.demands, vec![0, 1, 1, 1, 1]);
        // depot at origin, customer 1 at (1,0): distance 1
        assert_eq!(vrp.distance(0, 1), 1.0);
        let _ = std::fs::remove_file(&path);
    }
}
