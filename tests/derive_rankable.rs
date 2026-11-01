//! What `#[derive(Rankable)]` writes, for each way of naming the order.

use optopus::trait_defs::{Evaluable, Evaluate, Rankable, rank_cmp};
use std::cmp::Ordering;

/// No attribute, so the order is the objective, here maximized.
#[derive(Rankable)]
struct Cut(f64);

impl Evaluate for Cut {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.0)
    }
}

#[test]
fn no_attribute_ranks_by_the_objective() {
    assert!(Cut(2.0).is_better_than(&Cut(1.0)));
    assert!(!Cut(1.0).is_better_than(&Cut(2.0)));
    assert!(!Cut(1.0).is_better_than(&Cut(1.0)));
}

/// Keys on the fields, compared in declaration order, with a field between
/// them that is not a key.
#[derive(Rankable)]
struct Assignment {
    #[rank(min)]
    violations: u32,
    _label: &'static str,
    #[rank(max)]
    value: f64,
}

fn assignment(violations: u32, value: f64) -> Assignment {
    Assignment {
        violations,
        _label: "",
        value,
    }
}

#[test]
fn field_keys_compare_in_declaration_order() {
    // Fewer violations wins whatever the value.
    assert!(assignment(0, 1.0).is_better_than(&assignment(1, 100.0)));
    // At equal violations, more value wins.
    assert!(assignment(2, 5.0).is_better_than(&assignment(2, 4.0)));
    assert!(!assignment(2, 4.0).is_better_than(&assignment(2, 5.0)));
    // Equal on every key is neither better.
    assert_eq!(
        rank_cmp(&assignment(1, 3.0), &assignment(1, 3.0)),
        Ordering::Equal
    );
}

/// Keys on the type, a method and then a field.
#[derive(Rankable)]
#[rank(min = violation(), min = cost)]
struct Routes {
    overload: i64,
    lateness: f64,
    cost: f64,
}

impl Routes {
    fn violation(&self) -> f64 {
        self.overload as f64 + self.lateness
    }
}

#[test]
fn type_keys_read_methods_and_fields() {
    let feasible = Routes {
        overload: 0,
        lateness: 0.0,
        cost: 900.0,
    };
    let cheap = Routes {
        overload: 1,
        lateness: 0.0,
        cost: 100.0,
    };
    let cheaper_feasible = Routes {
        overload: 0,
        lateness: 0.0,
        cost: 800.0,
    };
    assert!(feasible.is_better_than(&cheap));
    assert!(cheaper_feasible.is_better_than(&feasible));
}

/// A tuple struct with a const parameter, which the macro_rules form could not
/// name.
#[derive(Rankable)]
struct Pair<const K: usize>(#[rank(max)] u32, #[rank(min)] u32);

#[test]
fn tuple_fields_and_generics() {
    assert!(Pair::<3>(2, 9).is_better_than(&Pair::<3>(1, 0)));
    assert!(Pair::<3>(2, 1).is_better_than(&Pair::<3>(2, 4)));
}

/// A generic type ranked by its objective gets the impl wherever it has one.
#[derive(Rankable)]
struct Scored<T> {
    score: f64,
    _tag: T,
}

impl<T> Evaluate for Scored<T> {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.score)
    }
}

#[test]
fn a_generic_type_ranks_by_its_objective() {
    let a = Scored {
        score: 1.0,
        _tag: 'a',
    };
    let b = Scored {
        score: 2.0,
        _tag: 'b',
    };
    assert!(a.is_better_than(&b));
}
