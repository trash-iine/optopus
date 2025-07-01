use core::error;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub type SatSolution = Vec<bool>;

#[derive(Debug, Clone)]
pub struct Sat {
    p: i64,
    cnf: usize,
    clauses_map: Vec<Vec<HashSet<i64>>>,
}

impl Sat {
    pub fn new(p: i64) -> Self {
        let up: usize = p.try_into().unwrap();
        Self {
            p,
            cnf: 0,
            clauses_map: vec![vec![]; up],
        }
    }

    pub fn add_clause(&mut self, clause: HashSet<i64>) {
        for i in 0..clause.len() {
            self.clauses_map[i].push(clause.clone());
        }
        self.cnf += 1;
    }

    pub fn get_p(&self) -> i64 {
        self.p
    }

    pub fn get_cnf(&self) -> usize {
        self.cnf
    }

    pub fn get_clauses_containing(&self, i: usize) -> impl Iterator<Item = &HashSet<i64>> {
        self.clauses_map[i].iter()
    }

    pub fn load_file(filename: &str) -> Result<Self, Box<dyn error::Error>> {
        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file);
        let mut line_iter = reader.lines();
        let (p, cnf) = {
            let line = line_iter.next().ok_or("File is empty")??;
            let mut iter = line.split_whitespace();
            // p
            iter.next();
            // cnf
            iter.next();
            let p = iter.next().ok_or("Not found N")?.parse::<i64>()?;
            let cnf = iter.next().ok_or("Not found M")?.parse::<usize>()?;
            (p, cnf)
        };
        let mut sat = Self::new(p);
        for _ in 0..cnf {
            let line = line_iter.next().ok_or("Not found clause")??;
            let mut hs = HashSet::new();
            let mut iter = line.split_whitespace();
            loop {
                let v = iter.next().ok_or("Not found v")?.parse::<i64>()?;
                if v == 0 {
                    break;
                }
                hs.insert(v);
            }

            sat.add_clause(hs);
        }

        Ok(sat)
    }

    pub fn calc_gain(&self, sol: &SatSolution, i: usize) -> i64 {
        if sol[i] {
        for set in self.get_clauses_containing(i) {
        }
        }
        else {
        for set in self.get_clauses_containing(i) {
        }

        }
        return 0;
    }
}
