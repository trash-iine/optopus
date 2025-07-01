use rand::{thread_rng, Rng};

use super::definition::{Sat, SatSolution};
use crate::algorithm::{Comparable, Evaluable, Hashable, SearchState};

#[derive(Debug, Clone)]
pub struct FlipNeighbour {
    i: i64,
    gain: i64,
}

impl Hashable for FlipNeighbour {
    type HashType = i64;

    fn hash(&self) -> Self::HashType {
        return self.i;
    }
}

impl Comparable for FlipNeighbour {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluable for FlipNeighbour {
    fn evaluate(&self) -> f64 {
        self.gain as f64
    }
}

#[derive(Debug, Clone)]
pub struct SatFlipSearchState {
    pub started_at: std::time::Instant,
    pub timeout: Option<std::time::Duration>,
    pub iter: usize,
    pub max_iter: Option<usize>,
    pub sat: Sat,
    pub current_x: SatSolution,
    pub current_n_satisfied: usize,
    pub best_x: SatSolution,
    pub best_n_satisfied: usize,
}

impl SearchState for SatFlipSearchState {
    type Neighbor = FlipNeighbour;

    fn is_done(&self) -> bool {
        if self.sat.get_cnf() == self.current_n_satisfied {
            return true;
        }

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
        (0..self.sat.get_p()).map(move |i| FlipNeighbour {
            i,
            gain: self.gain_list[i],
        })
    }

    fn apply_move(&mut self, neighbor: Self::Neighbor) {
        self.current_x[neighbor.i as usize] = !self.current_x[neighbor.i as usize];
        self.current_n_satisfied = self
            .sat
            .get_clauses_containing(neighbor.i as usize)
            .filter(|clause| {
                clause.iter().any(|&v| {
                    let v = v.abs() as usize;
                    let sign = v as i64 / v;
                    self.current_x[v - 1] == (sign == 1)
                })
            })
            .count();

        self.iter += 1;

        if self.current_n_satisfied > self.best_n_satisfied {
            self.best_x = self.current_x.clone();
            self.best_n_satisfied = self.current_n_satisfied;
        }
    }

    fn is_updating_best(&self, neighbor: &Self::Neighbor) -> bool {
        self.current_n_satisfied + neighbor.gain < self.best_n_satisfied
    }

    fn is_updating_current(&self, neighbor: &Self::Neighbor) -> bool {
        neighbor.gain < 0
    }
}

impl SatFlipSearchState {
    pub fn new(
        sat: Sat,
        max_iter: Option<usize>,
        timeout: Option<std::time::Duration>,
        x: Option<SatSolution>,
    ) -> Self {
        if max_iter.is_none() && timeout.is_none() {
            panic!("Either max_iter or timeout must be specified");
        }

        let iter = 0;
        let current_x = {
            if let Some(x) = x {
                x
            } else {
                (0..sat.get_p())
                    .map(|_| thread_rng().gen::<bool>())
                    .collect()
            }
        };
        let current_energy = calc_energy(&qubo, &current_x);
        let best_x = current_x.clone();
        let best_energy = current_energy;
        let gain_list = (0..qubo.get_n())
            .map(|i| calc_gain(&qubo, &current_x, i))
            .collect();

        Self {
            iter,
            max_iter,
            started_at: std::time::Instant::now(),
            timeout,
            qubo,
            current_x,
            current_energy,
            best_x,
            best_energy,
            gain_list,
        }
    }

    fn update_best_x(&mut self) {
        if self.current_energy < self.best_energy {
            tracing::debug!("Update best energy: {}", self.current_energy);
            self.best_x = self.current_x.clone();
            self.best_energy = self.current_energy;
        }
    }

    pub fn flip(&mut self, i: usize) {
        self.current_x[i] = !self.current_x[i];
        self.current_energy += self.gain_list[i];

        // update gain_list
        self.gain_list[i] = -self.gain_list[i];
        for (&j, &q) in self.qubo.get_adjacent(i) {
            if self.current_x[j] ^ self.current_x[i] {
                self.gain_list[j] += q;
            } else {
                self.gain_list[j] -= q;
            }
        }

        self.iter += 1;

        self.update_best_x();
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        self.flip(i);
        self.flip(j);
    }
}
