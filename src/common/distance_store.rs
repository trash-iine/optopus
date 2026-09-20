//! The pairwise distances a routing instance is built on, kept either as the
//! full matrix or as the nearest neighbours of every node, and the TSPLIB
//! formulas that turn coordinates into them.

use crate::error::OptError;

/// TSPLIB `EDGE_WEIGHT_TYPE` variants a [`DistanceStore`] can compute from
/// coordinates.
///
/// Each variant selects the distance formula applied to the coordinates.
/// `Continuous` is the default for programmatically constructed instances and
/// preserves the library's historical plain-Euclidean behavior; the remaining
/// variants match the integer-rounded formulas specified by TSPLIB so objective
/// values are comparable to published optima.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeWeightType {
    /// Plain Euclidean distance (no rounding).
    Continuous,
    /// `nint(sqrt(dx^2 + dy^2))`, TSPLIB `EUC_2D`.
    Euc2d,
    /// `ceil(sqrt(dx^2 + dy^2))`, TSPLIB `CEIL_2D`.
    Ceil2d,
    /// TSPLIB pseudo-Euclidean distance (`ATT`): `r = sqrt((dx^2+dy^2)/10); t = nint(r); d = if t<r {t+1} else {t}`.
    Att,
    /// TSPLIB geographical distance (`GEO`). Coordinates are `DDD.MM` → radians;
    /// great-circle distance with Earth radius 6378.388 km, truncated.
    Geo,
}

/// The coordinates a store was built from and the formula that turns a pair
/// of them into a distance. Only stores built from coordinates have one, so
/// [`DistanceStore::from_matrix`] stores have no geometry at all.
#[derive(Debug, Clone)]
struct Geometry {
    coordinates: Vec<(f64, f64)>,
    edge_weight_type: EdgeWeightType,
}

/// Converts a `DDD.MM` TSPLIB coordinate to radians (GEO distance helper).
#[inline]
fn geo_ddmm_to_rad(xy: f64) -> f64 {
    let deg = xy.trunc();
    let min = xy - deg;
    std::f64::consts::PI * (deg + 5.0 * min / 3.0) / 180.0
}

impl Geometry {
    /// Computes the distance formula directly from the coordinates.
    fn distance(&self, i: usize, j: usize) -> f64 {
        let (x1, y1) = self.coordinates[i];
        let (x2, y2) = self.coordinates[j];
        match self.edge_weight_type {
            EdgeWeightType::Continuous => {
                let dx = x1 - x2;
                let dy = y1 - y2;
                (dx * dx + dy * dy).sqrt()
            }
            EdgeWeightType::Euc2d => {
                let dx = x1 - x2;
                let dy = y1 - y2;
                (dx * dx + dy * dy).sqrt().round()
            }
            EdgeWeightType::Ceil2d => {
                let dx = x1 - x2;
                let dy = y1 - y2;
                (dx * dx + dy * dy).sqrt().ceil()
            }
            EdgeWeightType::Att => {
                let dx = x1 - x2;
                let dy = y1 - y2;
                let r = ((dx * dx + dy * dy) / 10.0).sqrt();
                let t = r.round();
                if t < r { t + 1.0 } else { t }
            }
            EdgeWeightType::Geo => {
                // TSPLIB GEO: column 1 = latitude, column 2 = longitude, encoded DDD.MM.
                let lat_i = geo_ddmm_to_rad(x1);
                let long_i = geo_ddmm_to_rad(y1);
                let lat_j = geo_ddmm_to_rad(x2);
                let long_j = geo_ddmm_to_rad(y2);
                const RRR: f64 = 6378.388;
                let q1 = (long_i - long_j).cos();
                let q2 = (lat_i - lat_j).cos();
                let q3 = (lat_i + lat_j).cos();
                (RRR * (0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)).acos() + 1.0).trunc()
            }
        }
    }

    /// The `k` nearest other nodes of every node with their distances, in
    /// ascending distance. Ties are broken by node index, since the sort is
    /// stable over an ascending index scan, which is what keeps a seeded run
    /// reproducible on instances with repeated distances.
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

#[derive(Debug, Clone)]
enum Store {
    /// The full `n × n` matrix, row-major. The geometry is present only when
    /// the matrix was built from coordinates.
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

/// The pairwise distances of `n` nodes, answering [`DistanceStore::distance`]
/// for any pair whatever it keeps in memory.
///
/// A store is built from 2D coordinates and an [`EdgeWeightType`], either as
/// the full matrix ([`DistanceStore::from_coordinates`]) or as the `k`
/// nearest neighbours of every node with the remaining pairs computed from
/// the coordinates on demand ([`DistanceStore::nearest_from_coordinates`]),
/// or from a matrix given directly ([`DistanceStore::from_matrix`]). The
/// routing problems hold one and add what is theirs on top.
#[derive(Debug, Clone)]
pub struct DistanceStore {
    n: usize,
    store: Store,
}

impl DistanceStore {
    /// The full `n × n` matrix of the distances between the coordinates.
    pub fn from_coordinates(
        coordinates: Vec<(f64, f64)>,
        edge_weight_type: EdgeWeightType,
    ) -> Self {
        let geometry = Geometry {
            coordinates,
            edge_weight_type,
        };
        let n = geometry.coordinates.len();
        let mut matrix = Vec::with_capacity(n * n);
        for a in 0..n {
            for b in 0..n {
                matrix.push(geometry.distance(a, b));
            }
        }
        Self {
            n,
            store: Store::Full {
                matrix,
                geometry: Some(geometry),
            },
        }
    }

    /// Only the `k` nearest neighbours of every node, `n × k` distances
    /// instead of `n²`. Any other pair is computed from the coordinates when
    /// asked for, so the store answers exactly like the full matrix would,
    /// only slower on pairs outside the lists. `k` larger than `n - 1` keeps
    /// every other node.
    pub fn nearest_from_coordinates(
        coordinates: Vec<(f64, f64)>,
        edge_weight_type: EdgeWeightType,
        k: usize,
    ) -> Self {
        let geometry = Geometry {
            coordinates,
            edge_weight_type,
        };
        let n = geometry.coordinates.len();
        let rows = geometry.nearest_rows(k.min(n.saturating_sub(1)));
        Self {
            n,
            store: Store::Nearest { rows, geometry },
        }
    }

    /// A matrix given directly, so the store has no coordinates.
    /// `matrix[i][j]` is the distance between nodes `i` and `j`. Returns an
    /// error unless the matrix is square.
    pub fn from_matrix(matrix: Vec<Vec<f64>>) -> Result<Self, OptError> {
        let n = matrix.len();
        let mut flat = Vec::with_capacity(n * n);
        for (i, row) in matrix.into_iter().enumerate() {
            if row.len() != n {
                return Err(OptError::InvalidState(format!(
                    "distance matrix row {} has {} entries, expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
            flat.extend(row);
        }
        Ok(Self {
            n,
            store: Store::Full {
                matrix: flat,
                geometry: None,
            },
        })
    }

    /// The number of nodes.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the store has no nodes.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The coordinates the store was built from, `None` for a store built
    /// from a matrix.
    pub fn coordinates(&self) -> Option<&[(f64, f64)]> {
        self.geometry().map(|g| g.coordinates.as_slice())
    }

    /// The formula applied to the coordinates, `None` for a store built from
    /// a matrix.
    pub fn edge_weight_type(&self) -> Option<EdgeWeightType> {
        self.geometry().map(|g| g.edge_weight_type)
    }

    fn geometry(&self) -> Option<&Geometry> {
        match &self.store {
            Store::Full { geometry, .. } => geometry.as_ref(),
            Store::Nearest { geometry, .. } => Some(geometry),
        }
    }

    /// The distance between nodes `i` and `j`.
    ///
    /// A full matrix answers with one lookup. A nearest-neighbour store scans
    /// the short list of `i` and falls back to the coordinates for any pair
    /// outside it, so the value is the same either way.
    #[inline]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        match &self.store {
            Store::Full { matrix, .. } => matrix[i * self.n + j],
            Store::Nearest { rows, geometry } => rows[i]
                .iter()
                .find(|(c, _)| *c == j)
                .map_or_else(|| geometry.distance(i, j), |(_, d)| *d),
        }
    }

    /// For each node, its `k` nearest other nodes in ascending distance, ties
    /// by node index. A nearest-neighbour store that already holds at least
    /// `k` neighbours per node answers from its rows, any other store costs
    /// O(n² log n).
    pub fn nearest_neighbors(&self, k: usize) -> Vec<Vec<usize>> {
        let n = self.n;
        let k = k.min(n.saturating_sub(1));
        if let Store::Nearest { rows, .. } = &self.store
            && rows.iter().all(|row| row.len() >= k)
        {
            return rows
                .iter()
                .map(|row| row.iter().take(k).map(|(j, _)| *j).collect())
                .collect();
        }
        (0..n)
            .map(|i| {
                let mut nbrs: Vec<(f64, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (self.distance(i, j), j))
                    .collect();
                nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                nbrs.truncate(k);
                nbrs.into_iter().map(|(_, j)| j).collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Twenty nodes on a lattice, so repeated distances exercise the tie
    /// breaking of every nearest-neighbour path.
    fn lattice() -> Vec<(f64, f64)> {
        (0..20).map(|i| ((i % 5) as f64, (i / 5) as f64)).collect()
    }

    #[test]
    fn from_matrix_rejects_a_ragged_matrix() {
        let ragged = vec![vec![0.0, 1.0], vec![1.0, 0.0, 2.0]];
        assert!(DistanceStore::from_matrix(ragged).is_err());
    }

    #[test]
    fn from_matrix_reads_the_matrix_as_given() {
        let matrix = vec![
            vec![0.0, 3.0, 7.0],
            vec![2.0, 0.0, 5.0],
            vec![9.0, 4.0, 0.0],
        ];
        let store = DistanceStore::from_matrix(matrix).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.distance(0, 1), 3.0);
        assert_eq!(store.distance(1, 0), 2.0);
        assert!(store.coordinates().is_none());
        assert!(store.edge_weight_type().is_none());
        assert_eq!(
            store.nearest_neighbors(2),
            vec![vec![1, 2], vec![0, 2], vec![1, 0]]
        );
    }

    #[test]
    fn nearest_store_answers_like_the_full_matrix() {
        let full = DistanceStore::from_coordinates(lattice(), EdgeWeightType::Euc2d);
        let sparse = DistanceStore::nearest_from_coordinates(lattice(), EdgeWeightType::Euc2d, 4);
        assert_eq!(sparse.edge_weight_type(), Some(EdgeWeightType::Euc2d));
        assert_eq!(sparse.coordinates(), full.coordinates());
        for i in 0..20 {
            for j in 0..20 {
                assert_eq!(
                    sparse.distance(i, j),
                    full.distance(i, j),
                    "pair ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn nearest_store_lists_match_the_full_matrix_below_and_above_its_k() {
        let full = DistanceStore::from_coordinates(lattice(), EdgeWeightType::Continuous);
        let sparse =
            DistanceStore::nearest_from_coordinates(lattice(), EdgeWeightType::Continuous, 4);
        for k in [1, 3, 4, 5, 19, 30] {
            assert_eq!(
                sparse.nearest_neighbors(k),
                full.nearest_neighbors(k),
                "k = {k}"
            );
        }
    }

    #[test]
    fn nearest_store_clamps_k_to_the_other_nodes() {
        let store = DistanceStore::nearest_from_coordinates(
            vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)],
            EdgeWeightType::Continuous,
            10,
        );
        assert_eq!(
            store.nearest_neighbors(10),
            vec![vec![1, 2], vec![0, 2], vec![0, 1]]
        );
        assert_eq!(store.distance(1, 2), 5f64.sqrt());
    }

    #[test]
    fn att_formula() {
        // r = sqrt(100 / 10) = sqrt(10) ≈ 3.162; t = 3; 3 < 3.162 → d = 4
        let store =
            DistanceStore::from_coordinates(vec![(0.0, 0.0), (10.0, 0.0)], EdgeWeightType::Att);
        assert_eq!(store.distance(0, 1), 4.0);
    }

    #[test]
    fn geo_formula() {
        // Identical locations give acos(1) = 0, plus 1 truncated.
        let store =
            DistanceStore::from_coordinates(vec![(0.0, 0.0), (0.0, 0.0)], EdgeWeightType::Geo);
        assert_eq!(store.distance(0, 1), 1.0);
        let store =
            DistanceStore::from_coordinates(vec![(0.0, 0.0), (10.0, 10.0)], EdgeWeightType::Geo);
        assert!(store.distance(0, 1) > 1.0);
    }

    #[test]
    fn ceil_2d_rounds_up() {
        let store =
            DistanceStore::from_coordinates(vec![(0.0, 0.0), (1.0, 1.0)], EdgeWeightType::Ceil2d);
        assert_eq!(store.distance(0, 1), 2.0);
    }
}
