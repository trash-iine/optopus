use crate::trait_defs::{Evaluable, FixVariables, FixedVariables, Placed};

use super::problem::Qubo;

impl FixVariables for Qubo {
    /// The free variables under their own indices. A product of a free
    /// variable with one fixed at `1` becomes a linear term of the free one,
    /// and a product of fixed variables at `1` goes to the offset, so the
    /// folding is exact for every solution.
    fn fix(&self, fixed: &[Option<bool>]) -> FixedVariables<Qubo> {
        let mut placed = vec![Placed::Fixed(false); fixed.len()];
        for &i in self.iter_on_variables() {
            placed[i] = match fixed[i] {
                Some(value) => Placed::Fixed(value),
                None => Placed::Free(i),
            };
        }
        let mut target = Qubo::new();
        let mut offset = 0.0;
        for (i, j, q) in self.entries() {
            match (placed[i], placed[j]) {
                (Placed::Free(a), Placed::Free(b)) => target.add_q(a, b, q),
                (Placed::Free(a), Placed::Fixed(true)) | (Placed::Fixed(true), Placed::Free(a)) => {
                    target.add_q(a, a, q)
                }
                (Placed::Fixed(true), Placed::Fixed(true)) => offset += f64::from(q),
                _ => {}
            }
        }
        FixedVariables {
            target,
            placed,
            offset,
            reference: None,
        }
    }

    /// Every negative coefficient taken and no positive one.
    fn trivial_bound(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.entries().map(|(_, _, q)| f64::from(q.min(0))).sum())
    }
}
