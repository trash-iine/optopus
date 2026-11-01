//! What the derives write: `Rankable` for each way of naming the order, and
//! `Evaluate` and `Distance` from one marked field.

use optopus::trait_defs::{Distance, Evaluable, Evaluate, Rankable, rank_cmp};
use std::cmp::Ordering;

/// No attribute, so the order is the objective, here maximized.
#[derive(Rankable, Evaluate)]
struct Cut(#[objective(maximize)] f64);

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

/// A minimized integer objective, read as an `f64`.
#[derive(Evaluate, Rankable)]
struct Makespan {
    _sequence: Vec<usize>,
    #[objective(minimize)]
    makespan: u32,
}

/// A maximized gain on a move, in a tuple struct.
#[derive(Evaluate, Rankable)]
struct Gain(#[objective(maximize)] i64);

/// A generic type gets the impl for every parameter.
#[derive(Evaluate)]
struct Tagged<T> {
    _tag: T,
    #[objective(minimize)]
    cost: f64,
}

#[test]
fn evaluate_wraps_the_marked_field_in_its_direction() {
    let short = Makespan {
        _sequence: vec![],
        makespan: 7,
    };
    let long = Makespan {
        _sequence: vec![],
        makespan: 9,
    };
    assert!(matches!(short.evaluate(), Evaluable::Minimize(v) if v == 7.0));
    assert!(short.is_better_than(&long));

    assert!(matches!(Gain(-3).evaluate(), Evaluable::Maximize(v) if v == -3.0));
    assert!(Gain(2).is_better_than(&Gain(1)));

    let tagged = Tagged {
        _tag: "a",
        cost: 1.5,
    };
    assert!(matches!(tagged.evaluate(), Evaluable::Minimize(v) if v == 1.5));
}

/// Hamming distance over the marked field, the others ignored.
#[derive(Distance)]
struct Assigned {
    #[distance(hamming)]
    colors: Vec<usize>,
    _used: usize,
}

#[derive(Distance)]
struct Bits(#[distance(hamming)] Vec<bool>);

#[test]
fn distance_counts_differing_positions_of_the_marked_field() {
    let a = Assigned {
        colors: vec![0, 1, 2, 1],
        _used: 3,
    };
    let b = Assigned {
        colors: vec![0, 2, 2, 0],
        _used: 99,
    };
    assert_eq!(a.distance(&b), 2);
    assert_eq!(a.distance(&a), 0);

    assert_eq!(
        Bits(vec![true, false, true]).distance(&Bits(vec![false, false, false])),
        2
    );
}

mod tabu {
    use optopus::building_blocks::search::TabuMemory;
    use optopus::rand::SeedableRng;
    use optopus::rand::rngs::SmallRng;
    use optopus::trait_defs::EnabledTabu;

    /// A memory where everything forbidden at iteration 0 stays forbidden
    /// through iteration 3 and is free from 4.
    fn memory() -> (TabuMemory, SmallRng) {
        let mut tabu = TabuMemory::default();
        tabu.set_tenure((3, 3));
        (tabu, SmallRng::seed_from_u64(0))
    }

    #[derive(EnabledTabu)]
    #[tabu(dense(i), dense(j))]
    struct TwoDense {
        i: usize,
        j: usize,
    }

    #[test]
    fn every_key_must_be_free_and_every_key_is_forbidden() {
        let (mut tabu, mut rng) = memory();
        TwoDense { i: 1, j: 2 }.add_to_tabu_map(&mut tabu, 0, &mut rng);
        assert!(!TwoDense { i: 1, j: 5 }.is_move_enabled(&tabu, 1));
        assert!(!TwoDense { i: 5, j: 2 }.is_move_enabled(&tabu, 1));
        assert!(TwoDense { i: 5, j: 6 }.is_move_enabled(&tabu, 1));
        assert!(
            TwoDense { i: 1, j: 2 }.is_move_enabled(&tabu, 4),
            "freed after the tenure"
        );
    }

    #[derive(EnabledTabu)]
    #[tabu(pair(a, b))]
    struct Ordered {
        a: usize,
        b: usize,
    }

    #[derive(EnabledTabu)]
    #[tabu(unordered(a, b))]
    struct Unordered {
        a: usize,
        b: usize,
    }

    #[test]
    fn pair_is_ordered_and_unordered_is_not() {
        let (mut tabu, mut rng) = memory();
        Ordered { a: 1, b: 2 }.add_to_tabu_map(&mut tabu, 0, &mut rng);
        assert!(!Ordered { a: 1, b: 2 }.is_move_enabled(&tabu, 1));
        assert!(Ordered { a: 2, b: 1 }.is_move_enabled(&tabu, 1));

        let (mut tabu, mut rng) = memory();
        Unordered { a: 1, b: 2 }.add_to_tabu_map(&mut tabu, 0, &mut rng);
        assert!(!Unordered { a: 2, b: 1 }.is_move_enabled(&tabu, 1));
    }

    #[derive(EnabledTabu)]
    #[tabu(triple(0, 1, 2))]
    struct Triple(usize, usize, usize);

    #[derive(EnabledTabu)]
    #[tabu(var(0))]
    struct Large(usize);

    #[test]
    fn triple_and_var_keys_on_tuple_fields() {
        let (mut tabu, mut rng) = memory();
        Triple(1, 2, 3).add_to_tabu_map(&mut tabu, 0, &mut rng);
        assert!(!Triple(1, 2, 3).is_move_enabled(&tabu, 1));
        assert!(Triple(1, 3, 2).is_move_enabled(&tabu, 1));

        Large(1 << 40).add_to_tabu_map(&mut tabu, 0, &mut rng);
        assert!(!Large(1 << 40).is_move_enabled(&tabu, 1));
        assert!(Large(7).is_move_enabled(&tabu, 1));
    }
}
