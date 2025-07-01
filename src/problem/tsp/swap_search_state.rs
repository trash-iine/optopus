use rand::{seq::SliceRandom, thread_rng};

use super::definition::{calculate_tour_length, TspTour, TspWithCoordinates};
use crate::algorithm::{Comparable, Evaluable, Hashable, SearchState};

fn calculate_gain(tsp: &TspWithCoordinates, tour: &TspTour, ind_i: usize, ind_j: usize) -> f64 {
    let n = tsp.get_n();

    let i_minus = tour[(ind_i + n - 1) % n];
    let i = tour[ind_i];
    let i_plus = tour[(ind_i + 1) % n];

    let j_minus = tour[(ind_j + n - 1) % n];
    let j = tour[ind_j];
    let j_plus = tour[(ind_j + 1) % n];

    // check are i and j adjacent
    if (ind_i < ind_j && ind_i + 1 == ind_j) || (ind_i == n - 1 && ind_j == 0) {
        tsp.distance(i_minus, j) + tsp.distance(i, j_plus)
            - tsp.distance(i_minus, i)
            - tsp.distance(j, j_plus)
    } else if (ind_i > ind_j && ind_i == ind_j + 1) || (ind_j == n - 1 && ind_i == 0) {
        tsp.distance(j_minus, i) + tsp.distance(j, i_plus)
            - tsp.distance(j_minus, j)
            - tsp.distance(i, i_plus)
    } else {
        tsp.distance(i_minus, j)
            + tsp.distance(j, i_plus)
            + tsp.distance(j_minus, i)
            + tsp.distance(i, j_plus)
            - tsp.distance(i_minus, i)
            - tsp.distance(i, i_plus)
            - tsp.distance(j_minus, j)
            - tsp.distance(j, j_plus)
    }
}

#[derive(Debug, Clone)]
pub struct TspSwapNeighbour {
    i: usize,
    j: usize,
    gain: f64,
}

impl Hashable for TspSwapNeighbour {
    type HashType = (usize, usize);

    fn hash(&self) -> Self::HashType {
        if self.i < self.j {
            (self.i, self.j)
        } else {
            (self.j, self.i)
        }
    }
}
impl Comparable for TspSwapNeighbour {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}
impl Evaluable for TspSwapNeighbour {
    fn evaluate(&self) -> f64 {
        self.gain
    }
}

#[derive(Debug, Clone)]
pub struct TspSwapSearchState {
    pub started_at: std::time::Instant,
    pub timeout: Option<std::time::Duration>,
    pub iter: usize,
    pub max_iter: Option<usize>,
    pub tsp: TspWithCoordinates,
    pub current_tour: TspTour,
    pub current_energy: f64,
    pub best_tour: TspTour,
    pub best_energy: f64,
}

impl SearchState for TspSwapSearchState {
    type Neighbor = TspSwapNeighbour;

    fn is_done(&self) -> bool {
        let elapsed = if let Some(timeout) = self.timeout {
            self.started_at.elapsed() >= timeout
        } else {
            false
        };

        let overed = if let Some(max_iter) = self.max_iter {
            self.iter >= max_iter
        } else {
            false
        };

        elapsed || overed
    }

    fn get_iteration(&self) -> usize {
        self.iter
    }

    fn iter_of_neighbors(&self) -> impl Iterator<Item = Self::Neighbor> {
        (0..self.tsp.get_n()).flat_map(move |i| {
            (0..i).map(move |j| TspSwapNeighbour {
                i,
                j,
                gain: calculate_gain(&self.tsp, &self.current_tour, i, j),
            })
        })
    }

    fn apply_move(&mut self, neighbor: Self::Neighbor) {
        self.current_tour.swap(neighbor.i, neighbor.j);

        self.current_energy += neighbor.gain;

        if self.current_energy < self.best_energy {
            tracing::info!("update best: {}", self.current_energy);
            self.best_tour = self.current_tour.clone();
            self.best_energy = self.current_energy;
        }

        self.iter += 1
    }

    fn is_updating_best(&self, neighbor: &Self::Neighbor) -> bool {
        self.current_energy + neighbor.gain < self.best_energy
    }

    fn is_updating_current(&self, neighbor: &Self::Neighbor) -> bool {
        neighbor.gain < 0.0
    }
}

impl TspSwapSearchState {
    pub fn new(
        tsp: TspWithCoordinates,
        max_iter: Option<usize>,
        timeout: Option<std::time::Duration>,
        tour: Option<TspTour>,
    ) -> Self {
        if max_iter.is_none() && timeout.is_none() {
            panic!("Either max_iter or timeout must be specified");
        }

        let iter = 0;
        let current_tour = {
            if let Some(tour) = tour {
                tour
            } else {
                let mut tour: Vec<usize> = (0..tsp.get_n()).collect();
                tour.shuffle(&mut thread_rng());
                tour
            }
        };
        let current_energy = calculate_tour_length(&tsp, &current_tour);
        let best_tour = current_tour.clone();
        let best_energy = current_energy;

        Self {
            iter,
            max_iter,
            started_at: std::time::Instant::now(),
            timeout,
            tsp,
            current_tour,
            current_energy,
            best_tour,
            best_energy,
        }
    }
}
