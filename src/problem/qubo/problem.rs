use core::error;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::search_state::{ProblemTrait, Rankable};

/// Integer coefficient type used in the Q matrix.
pub type Coefficient = i32;

/// A solution to the QUBO problem.
///
/// - `x` — variable assignment (`x[i] = true` means variable `i` is set to 1)
/// - `gain` — incremental energy change for each variable flip (`gain[i] < 0` means flipping `i` improves the objective)
/// - `objective` — current energy value (minimized)
#[derive(Debug, Clone)]
pub struct QuboSolution {
    pub x: HashMap<usize, bool>,
    pub gain: HashMap<usize, Coefficient>,
    pub objective: Coefficient,
}
impl Rankable for QuboSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

/// QUBO problem instance.
///
/// Stores the upper-triangular Q matrix as a symmetric sparse adjacency map.
/// Diagonal entries `Q[i][i]` represent linear coefficients.
#[derive(Debug, Clone)]
pub struct Qubo {
    q: HashMap<usize, HashMap<usize, Coefficient>>,
}

impl Qubo {
    pub fn new() -> Qubo {
        Qubo { q: HashMap::new() }
    }

    pub fn set_q(&mut self, i: usize, j: usize, v: Coefficient) {
        self.q.entry(i).or_insert_with(HashMap::new).insert(j, v);
        self.q.entry(j).or_insert_with(HashMap::new).insert(i, v);
    }

    pub fn get_q(&self, i: usize, j: usize) -> Option<Coefficient> {
        if let Some(hm) = self.q.get(&i) {
            if let Some(&v) = hm.get(&j) {
                return Some(v);
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    pub fn iter_on_variables(&self) -> impl Iterator<Item = &usize> {
        self.q.keys()
    }

    pub fn iter_on_adjacency<'a>(
        &'a self,
        i: usize,
    ) -> Box<dyn Iterator<Item = (&'a usize, &'a Coefficient)> + 'a> {
        if let Some(hm) = self.q.get(&i) {
            return Box::new(hm.iter());
        } else {
            return Box::new(std::iter::empty());
        }
    }

    pub fn num_of_variables(&self) -> usize {
        self.q.len()
    }

    pub fn load_file_as_max_cut(filename: &str) -> Result<Self, Box<dyn error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut line_iter = reader.lines();
        let (_, _) = {
            let line = line_iter.next().ok_or("File is empty")??;
            let mut iter = line.split_whitespace();
            let n = iter.next().ok_or("Not found N")?.parse::<usize>()?;
            let m = iter.next().ok_or("Not found M")?.parse::<usize>()?;
            (n, m)
        };
        let mut qubo = Qubo::new();
        while let Some(Ok(line)) = line_iter.next() {
            let mut iter = line.split_whitespace();
            let i = iter.next().ok_or("Not found i")?.parse::<usize>()? - 1;
            let j = iter.next().ok_or("Not found j")?.parse::<usize>()? - 1;
            let v = iter.next().ok_or("Not found v")?.parse::<i32>()?;

            if let Some(old_v) = qubo.get_q(i, j) {
                qubo.set_q(i, j, old_v + 2 * v);
            } else {
                qubo.set_q(i, j, 2 * v);
            }

            if let Some(old_v) = qubo.get_q(i, i) {
                qubo.set_q(i, i, old_v - v);
            } else {
                qubo.set_q(i, i, -v);
            }

            if let Some(old_v) = qubo.get_q(j, j) {
                qubo.set_q(j, j, old_v - v);
            } else {
                qubo.set_q(j, j, -v);
            }
        }

        Ok(qubo)
    }

    /// Calculates the change in energy when variable `i` is flipped.
    ///
    /// Returns `gain` such that `E(x') = E(x) + gain` where `x'` is `x` with variable `i` flipped.
    /// A negative value indicates an improvement (energy decrease).
    pub fn calculate_gain(&self, x: &HashMap<usize, bool>, i: usize) -> Coefficient {
        let mut gain = 0;
        for (&j, &q) in self.iter_on_adjacency(i) {
            if i == j {
                gain += q;
            } else {
                let j_side = *x
                    .get(&j)
                    .expect(format!("{} is not found in solution", j).as_str());

                if j_side {
                    gain += q;
                }
            }
        }

        let i_side = *x
            .get(&i)
            .expect(format!("{} is not found in solution", i).as_str());
        if i_side { -gain } else { gain }
    }

    /// Calculates the total energy `E(x) = Σ Q[i][j] * x[i] * x[j]` for the given assignment.
    pub fn calculate_energy(&self, x: &HashMap<usize, bool>) -> Coefficient {
        let mut energy = 0;
        for &i in self.iter_on_variables() {
            let i_side = *x
                .get(&i)
                .expect(format!("{} is not found in solution", i).as_str());

            if !i_side {
                continue;
            }

            for (&j, &q) in self.iter_on_adjacency(i) {
                if i < j {
                    continue;
                }

                if i == j {
                    // Diagonal term: Q_ii * x_i^2 = Q_ii * x_i (x_i == 1 is guaranteed here)
                    energy += q;
                } else {
                    let j_side = *x
                        .get(&j)
                        .expect(format!("{} is not found in solution", j).as_str());

                    if j_side {
                        energy += q;
                    }
                }
            }
        }

        energy
    }
}

impl ProblemTrait for Qubo {
    type Solution = QuboSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution {
        let mut x = HashMap::new();
        for &i in self.iter_on_variables() {
            x.insert(i, rng.random_bool(0.5));
        }

        let mut gain = HashMap::new();
        for &i in self.iter_on_variables() {
            let g = self.calculate_gain(&x, i);
            gain.insert(i, g);
        }

        let objective = self.calculate_energy(&x);

        return QuboSolution { x, gain, objective };
    }
}

pub fn calc_xor_of_solutions(sol1: &QuboSolution, sol2: &QuboSolution) -> usize {
    sol1.x
        .values()
        .zip(sol2.x.values())
        .filter(|(i, j)| **i ^ **j)
        .count()
}

pub fn make_sub_problem_from(qubo: &Qubo, parents: &[&QuboSolution]) -> Qubo {
    let mut ind_set = HashSet::new();
    for &ind in qubo.iter_on_variables() {
        let base = *parents[0]
            .x
            .get(&ind)
            .expect(format!("{} is not found in solution", ind).as_str());
        let is_free = parents[1..].iter().any(|p| {
            let v = *p.x.get(&ind).expect(format!("{} is not found in solution", ind).as_str());
            v != base
        });
        if is_free {
            ind_set.insert(ind);
        }
    }

    let mut sub_qubo = Qubo::new();

    for &ind in ind_set.iter() {
        for (&j, &v) in qubo.iter_on_adjacency(ind) {
            if ind_set.contains(&j) {
                if ind < j {
                    sub_qubo.set_q(ind, j, v);
                } else if ind == j {
                    if let Some(old_v) = sub_qubo.get_q(ind, ind) {
                        sub_qubo.set_q(ind, ind, old_v + v);
                    } else {
                        sub_qubo.set_q(ind, ind, v);
                    }
                }
            } else if parents[0].x[&j] {
                if let Some(old_v) = sub_qubo.get_q(ind, ind) {
                    sub_qubo.set_q(ind, ind, old_v + v);
                } else {
                    sub_qubo.set_q(ind, ind, v);
                }
            }
        }
    }

    return sub_qubo;
}

#[cfg(test)]
mod qubo_tests {
    use super::*;

    #[test]
    fn test_qubo() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 1);
        qubo.set_q(1, 2, 2);
        qubo.set_q(2, 0, 3);
        assert!(qubo.get_q(0, 1).is_some_and(|v| v == 1));
        assert!(qubo.get_q(1, 2).is_some_and(|v| v == 2));
        assert!(qubo.get_q(0, 2).is_some_and(|v| v == 3));
        assert_eq!(qubo.num_of_variables(), 3);
    }

    #[test]
    fn test_load_file_as_max_cut() {
        let qubo_result = Qubo::load_file_as_max_cut("data/max_cut/test_data.txt");
        assert!(qubo_result.is_ok());
        let qubo = qubo_result.unwrap();
        println!("{:?}", qubo.q);
        assert!(qubo.get_q(0, 1).is_some_and(|v| v == 2));
        assert!(qubo.get_q(1, 2).is_some_and(|v| v == -6));
        assert!(qubo.get_q(0, 0).is_some_and(|v| v == -1));
        assert!(qubo.get_q(1, 1).is_some_and(|v| v == 2));
        assert!(qubo.get_q(2, 2).is_some_and(|v| v == 3));
        assert_eq!(qubo.num_of_variables(), 3);
    }

    #[test]
    fn test_calc_energy() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 1);
        qubo.set_q(1, 2, 2);
        qubo.set_q(2, 0, 3);
        qubo.set_q(2, 2, 1);
        {
            let x = HashMap::from([(0, true), (1, false), (2, true)]);
            assert_eq!(qubo.calculate_energy(&x), 4);
        }
        {
            let x = HashMap::from([(0, false), (1, true), (2, false)]);
            assert_eq!(qubo.calculate_energy(&x), 0);
        }
        {
            let x = HashMap::from([(0, true), (1, true), (2, true)]);
            assert_eq!(qubo.calculate_energy(&x), 7);
        }
        {
            let x = HashMap::from([(0, false), (1, false), (2, false)]);
            assert_eq!(qubo.calculate_energy(&x), 0);
        }
    }

    #[test]
    fn test_calc_gain() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 1);
        qubo.set_q(1, 2, 2);
        qubo.set_q(2, 0, 3);
        qubo.set_q(2, 2, 1);
        {
            let x = HashMap::from([(0, true), (1, false), (2, true)]);
            assert_eq!(qubo.calculate_energy(&x), 4);
            for i in 0..3 {
                let mut flipped = x.clone();
                flipped.insert(i, !flipped[&i]);

                assert_eq!(
                    qubo.calculate_gain(&x, i) + qubo.calculate_energy(&x),
                    qubo.calculate_energy(&flipped)
                );
            }
        }
        {
            let x = HashMap::from([(0, false), (1, true), (2, false)]);
            assert_eq!(qubo.calculate_energy(&x), 0);
            for i in 0..3 {
                let mut flipped = x.clone();
                flipped.insert(i, !flipped[&i]);

                assert_eq!(
                    qubo.calculate_gain(&x, i) + qubo.calculate_energy(&x),
                    qubo.calculate_energy(&flipped)
                );
            }
        }
    }
}
