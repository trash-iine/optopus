use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::search_state::{ProblemTrait, Rankable};

pub type TspTour = Vec<usize>;

/// ツアーとその目的関数値（巡回路の総距離）を保持する解
#[derive(Debug, Clone)]
pub struct TspSolution {
    pub tour: TspTour,
    pub objective: f64,
}

impl Rankable for TspSolution {
    // TSP は最小化問題なので、距離が短い方が優れた解
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

impl ProblemTrait for TspWithCoordinates {
    type Solution = TspSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> TspSolution {
        use rand::seq::SliceRandom;
        let mut tour: Vec<usize> = (0..self.get_n()).collect();
        tour.shuffle(rng);
        let objective = calculate_tour_length(self, &tour);
        TspSolution { tour, objective }
    }
}

#[derive(Debug, Clone)]
pub struct TspWithCoordinates {
    pub name: String,
    pub coordinates: Vec<(f64, f64)>,
}

impl TspWithCoordinates {
    pub fn new(name: String, coordinates: Vec<(f64, f64)>) -> Self {
        Self { name, coordinates }
    }

    pub fn load_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let name = lines
            .next()
            .ok_or("File is empty")??
            .split_whitespace()
            .last()
            .ok_or("Not found name")?
            .to_string();

        // skip TYPE
        lines.next();
        // skip COMMENT
        lines.next();
        // skip DIMENSION
        let n = lines
            .next()
            .ok_or("Not found DIMENSION")??
            .split_whitespace()
            .last()
            .ok_or("Not found DIMENSION")?
            .parse::<usize>()?;
        // skip EDGE_WEIGHT_TYPE
        lines.next();
        // skip NODE_COORD_SECTION
        lines.next();

        let mut coord = vec![];
        for _ in 0..n {
            let line = lines.next().ok_or("Not found line")??;
            let mut iter = line.split_whitespace();

            // skip index
            let _ = iter.next();

            let x = iter.next().ok_or("Not found x")?.parse::<f64>()?;
            let y = iter.next().ok_or("Not found y")?.parse::<f64>()?;

            coord.push((x, y));
        }

        Ok(Self::new(name, coord))
    }

    pub fn get_n(&self) -> usize {
        self.coordinates.len()
    }

    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let (x1, y1) = self.coordinates[i];
        let (x2, y2) = self.coordinates[j];
        ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
    }
}

pub fn calculate_tour_length(tsp: &TspWithCoordinates, tour: &TspTour) -> f64 {
    if tsp.get_n() != tour.len() {
        panic!("The size of TSP and length of tour are not matched");
    }

    let mut arrived = HashSet::new();

    let mut length = 0.0;
    for ind in 0..tsp.get_n() {
        let i = tour[ind];

        if arrived.contains(&i) {
            panic!("The city {} is already arrived", i);
        }

        let j = tour[(ind + 1) % tsp.get_n()];
        length += tsp.distance(i, j);

        arrived.insert(i);
    }

    length
}

#[cfg(test)]
mod tsp_coord_tests {
    use super::*;

    #[test]
    fn test_load_file() {
        let tsp_result = TspWithCoordinates::load_file("data/tsp/test_data.txt");
        assert!(tsp_result.is_ok());
        let tsp = tsp_result.unwrap();
        assert_eq!(tsp.name, "test");
        assert_eq!(tsp.coordinates.len(), 4);
    }

    #[test]
    fn test_distance() {
        let tsp_result = TspWithCoordinates::load_file("data/tsp/test_data.txt");
        assert!(tsp_result.is_ok());
        let tsp = tsp_result.unwrap();
        assert_eq!(tsp.distance(0, 1), 1.0);
        assert_eq!(tsp.distance(1, 2), 2.0);
    }

    #[test]
    #[should_panic]
    fn test_not_match_size_tour() {
        let tsp_result = TspWithCoordinates::load_file("data/tsp/test_data.txt");
        assert!(tsp_result.is_ok());
        let tsp = tsp_result.unwrap();
        let invalid_tour = vec![0, 1];

        calculate_tour_length(&tsp, &invalid_tour);
    }

    #[test]
    #[should_panic]
    fn test_duplicated_tour() {
        let tsp_result = TspWithCoordinates::load_file("data/tsp/test_data.txt");
        assert!(tsp_result.is_ok());
        let tsp = tsp_result.unwrap();
        let invalid_tour = vec![0, 1, 0, 2];

        calculate_tour_length(&tsp, &invalid_tour);
    }

    #[test]
    fn test_tour_length() {
        let tsp_result = TspWithCoordinates::load_file("data/tsp/test_data.txt");
        assert!(tsp_result.is_ok());
        let tsp = tsp_result.unwrap();
        let tour = vec![0, 1, 2, 3];
        assert_eq!(calculate_tour_length(&tsp, &tour), 6.0);
    }
}
