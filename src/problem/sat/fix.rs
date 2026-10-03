use crate::trait_defs::{Evaluable, FixVariables, FixedVariables, Placed};

use super::problem::Sat;

impl FixVariables for Sat {
    /// The free variables, numbered in order. A clause a fixed literal
    /// satisfies goes to the offset, a fixed literal that is false is dropped
    /// from its clause, and a clause left with no literal can no longer be
    /// satisfied and is dropped. The folding is exact for every solution.
    fn fix(&self, fixed: &[Option<bool>]) -> FixedVariables<Sat> {
        let mut free = 0;
        let placed: Vec<Placed> = (0..self.n_vars())
            .map(|i| match fixed[i] {
                Some(value) => Placed::Fixed(value),
                None => {
                    free += 1;
                    Placed::Free(free - 1)
                }
            })
            .collect();
        let mut target = Sat::new(free);
        let mut offset = 0.0;
        for clause in self.all_clauses() {
            let mut literals = Vec::with_capacity(clause.len());
            let mut satisfied = false;
            for &lit in clause {
                match placed[lit.unsigned_abs() as usize - 1] {
                    Placed::Fixed(value) => satisfied |= value == (lit > 0),
                    Placed::Free(t) => literals.push((t as i64 + 1) * lit.signum()),
                }
            }
            if satisfied {
                offset += 1.0;
            } else if !literals.is_empty() {
                target.add_clause(literals);
            }
        }
        FixedVariables {
            target,
            placed,
            offset,
            reference: None,
        }
    }

    /// Every clause satisfied.
    fn trivial_bound(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.n_clauses() as f64)
    }
}
