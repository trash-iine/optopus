use crate::search_state::ProblemTrait;
use std::{collections::HashMap, io::BufRead};

/// The MaxCut problem.
pub struct MaxCut {
    adj: HashMap<usize, HashMap<usize, f32>>,
}

#[derive(Debug, Clone)]
pub struct MaxCutSolution {
    pub cut: HashMap<usize, bool>,
    pub gain: HashMap<usize, f32>,
}

impl MaxCut {
    /// Creates a new [`MaxCut`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mc = optopus::problem::max_cut::problem::MaxCut::new();
    /// ```
    pub fn new() -> Self {
        Self {
            adj: HashMap::new(),
        }
    }

    /// Returns the number of vertices in the max cut graph.
    ///
    /// # Examples
    ///
    /// ```
    /// let mc = optopus::problem::max_cut::problem::MaxCut::new();
    /// assert_eq!(mc.len(), 0);
    ///
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// assert_eq!(mc.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.adj.len()
    }

    /// Returns the iterator visiting all vertices in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::max_cut::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// assert_eq!(mc.len(), 3);
    ///
    /// for i in mc.iter_on_vertices() {
    ///    println!("{}", i); // 0, 1, 2 (in any order)
    /// }
    /// ```
    pub fn iter_on_vertices(&self) -> impl Iterator<Item = &usize> {
        return self.adj.keys();
    }

    /// Returns the iterator visiting all edges of the vertex `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::max_cut::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// mc.add_weight(1, 2, 1.0);
    /// for (&j, &w) in mc.iter_on_adjacency(&0) {
    ///     println!("{} {}", j, w); // 1 1.0, 2 1.0
    /// }
    /// ```
    pub fn iter_on_adjacency<'a>(
        &'a self,
        i: &usize,
    ) -> Box<dyn Iterator<Item = (&'a usize, &'a f32)> + 'a> {
        if let Some(hm) = self.adj.get(i) {
            return Box::new(hm.iter());
        } else {
            return Box::new(std::iter::empty());
        }
    }

    /// Adds the weight `w` between the node `i` and `j`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::max_cut::problem::MaxCut::new();
    /// mc.add_adj(0, 1, 1.0);
    /// mc.add_adj(0, 2, 1.0);
    /// mc.add_adj(0, 1, 2.0);
    /// ```
    pub fn add_weight(&mut self, i: usize, j: usize, w: f32) {
        *self
            .adj
            .entry(i)
            .or_insert(HashMap::new())
            .entry(j)
            .or_insert(0.0) += w;
        *self
            .adj
            .entry(j)
            .or_insert(HashMap::new())
            .entry(i)
            .or_insert(0.0) += w;
    }

    /// Gets the weight between the node `i` and `j`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::max_cut::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    ///
    /// assert_eq!(mc.get_weight(0, 1), 1.0);
    ///
    /// mc.add_weight(0, 1, 2.0); // allows to add weight to existing edge
    /// assert_eq!(mc.get_weight(0, 1), 3.0);
    /// ```
    pub fn get_weight(&self, i: usize, j: usize) -> f32 {
        if let Some(hm) = self.adj.get(&i) {
            if let Some(&w) = hm.get(&j) {
                return w;
            } else {
                return 0.0;
            }
        } else {
            return 0.0;
        }
    }

    /// Calculates the cut size of the given cut.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::max_cut::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// mc.add_weight(1, 2, 1.0);
    ///
    /// let cut = std::collections::HashMap::from([(0, false), (1, false), (2, true)]);
    /// assert_eq!(mc.calculate_cut_size(&cut), 2.0);
    /// ```
    pub fn calculate_cut_size(&self, cut: &HashMap<usize, bool>) -> f32 {
        let mut ret = 0.0;

        for i in self.iter_on_vertices() {
            let i_side = *cut
                .get(i)
                .expect(format!("{} is not found in solution", *i).as_str());
            for (j, &w) in self.iter_on_adjacency(i) {
                let j_side = *cut
                    .get(j)
                    .expect(format!("{} is not found in solution", *j).as_str());

                if i_side ^ j_side {
                    ret += w;
                }
            }
        }

        return ret / 2.0;
    }

    pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn core::error::Error>> {
        let file = std::fs::File::open(filename).unwrap();
        let reader = std::io::BufReader::new(file);
        let mut line_iter = reader.lines();

        // parse the number of vertices and edges (not used)
        let (_, _) = {
            let line = line_iter.next().ok_or("File is empty")??;
            let mut iter = line.split_whitespace();
            let n = iter.next().ok_or("Not found N")?.parse::<usize>()?;
            let m = iter.next().ok_or("Not found M")?.parse::<usize>()?;
            (n, m)
        };

        let mut mc = MaxCut::new();
        while let Some(Ok(line)) = line_iter.next() {
            let mut iter = line.split_whitespace();
            let i = iter.next().ok_or("Not found i")?.parse::<usize>()? - 1;
            let j = iter.next().ok_or("Not found j")?.parse::<usize>()? - 1;
            let w = iter.next().ok_or("Not found w")?.parse::<f32>()?;

            mc.add_weight(i, j, w);
        }

        Ok(mc)
    }

    /// Calculates the gain of flipping the cut of vertex `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// mc.add_weight(1, 2, 1.0);
    ///
    /// let cut = std::collections::HashMap::from([(0, false), (1, false), (2, true)]);
    /// assert_eq!(mc.calculate_gain(&cut, 0), -2.0);
    /// ```
    pub fn calculate_gain(&self, cut: &HashMap<usize, bool>, i: usize) -> f32 {
        let i_side = *cut
            .get(&i)
            .expect(format!("{} is not found in solution", i).as_str());
        self.iter_on_adjacency(&i)
            .map(|(j, &w)| {
                let j_side = *cut
                    .get(j)
                    .expect(format!("{} is not found in solution", j).as_str());

                if i_side ^ j_side {
                    -w
                } else {
                    w
                }
            })
            .sum()
    }
}

impl ProblemTrait for MaxCut {
    type Solution = MaxCutSolution;
    type Objective = f32;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution {
        let mut cut = HashMap::new();
        for &i in self.iter_on_vertices() {
            cut.insert(i, rng.random_bool(0.5));
        }

        let mut gain = HashMap::new();
        for &i in self.iter_on_vertices() {
            let g = self.calculate_gain(&cut, i);
            gain.insert(i, g);
        }

        return MaxCutSolution { cut, gain };
    }

    fn calculate_objective(&self, sol: &Self::Solution) -> Self::Objective {
        self.calculate_cut_size(&sol.cut)
    }

    fn is_first_objective_better_than_second(
        &self,
        first: &Self::Objective,
        second: &Self::Objective,
    ) -> bool {
        first > second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blank_graph() {
        let mc = MaxCut::new();
        assert_eq!(mc.len(), 0);
    }

    #[test]
    fn test_set_and_get_weight() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 1.0);
        mc.add_weight(0, 1, 2.0);

        assert_eq!(mc.len(), 3);

        assert_eq!(mc.get_weight(0, 1), 3.0);
        assert_eq!(mc.get_weight(0, 2), 1.0);
        assert_eq!(mc.get_weight(1, 2), 0.0);
    }

    #[test]
    fn test_calculate_cut_size() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 2.0);
        mc.add_weight(1, 2, 3.0);

        {
            let cut = HashMap::from([(0, false), (1, false), (2, false)]);
            assert_eq!(mc.calculate_cut_size(&cut), 0.0);
        }
        {
            let cut = HashMap::from([(0, true), (1, false), (2, false)]);
            assert_eq!(mc.calculate_cut_size(&cut), 3.0);
        }
        {
            let cut = HashMap::from([(0, true), (1, false), (2, true)]);
            assert_eq!(mc.calculate_cut_size(&cut), 4.0);
        }
        {
            let cut = HashMap::from([(0, true), (1, true), (2, false)]);
            assert_eq!(mc.calculate_cut_size(&cut), 5.0);
        }
    }

    #[test]
    fn test_calculate_gain_list() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 2.0);
        mc.add_weight(1, 2, 3.0);

        let cut = HashMap::from([(0, false), (1, false), (2, false)]);
        assert_eq!(mc.calculate_gain(&cut, 0), 3.0);
        assert_eq!(mc.calculate_gain(&cut, 1), 4.0);
        assert_eq!(mc.calculate_gain(&cut, 2), 5.0);
    }
}
