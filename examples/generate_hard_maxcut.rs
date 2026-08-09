//! Generates the planted MaxCut suite, where the optimum is known exactly.
//!
//! The other two generated suites (`generate_dense_maxcut.rs`,
//! `generate_sparse_maxcut.rs`) vary structure and density, but share the
//! G-set's central limitation: nobody knows what the right answer is, so a
//! result can only ever be compared against another result. These instances are
//! built around a solution that provably cannot be beaten, so a run's gap is
//! measured against the truth and "solved it" becomes a statement rather than a
//! guess.
//!
//! Two families, chosen because they are the two that map to weighted MaxCut
//! without auxiliary variables and cover opposite density regimes:
//!
//! - **Tile planting**, on the square and cubic lattice tori. Degree 4 and 6,
//!   which puts the 2D set in the same band as the G-set `toroidal` group, so
//!   it extends an existing comparison instead of starting a new one.
//! - **Wishart planting**, on the complete graph. Small by MaxCut standards and
//!   brutal for its size — the original paper reports parallel tempering
//!   failing to reach even a loose approximation at `N = 64` after eleven hours.
//!
//! Parameters come from a hardness sweep rather than from the papers: where
//! each family becomes hard moves with the instance size, and the published
//! values were measured at sizes far below these. The sweep ran two MaxCut
//! heuristics over a grid of each knob at a 10s budget, three instances by five
//! runs per point, and scored by how often a run reached the planted optimum —
//! a question only a planted instance can answer. Each parameter below records
//! what it showed. The sweep harness itself lands separately, with the second
//! of those two heuristics.
//!
//! Every instance comes from a fixed seed, so re-running reproduces the files
//! byte for byte and they never need to be committed. `manifest.toml` records
//! the optimum of each one; nothing else does, and no heuristic or benchmark
//! config ever reads it — it is for analysing results, not for steering a run.
//!
//! ```text
//! cargo run --release --example generate_hard_maxcut
//! ```

use std::fmt::Write as _;
use std::path::Path;

use optopus::common::seeded_rng;
use optopus::prelude::*;
use optopus::problem::{PlantedMaxCut, TileProbs2d, TileProbs3d, WishartCouplers};

/// Which planted construction an instance comes from.
enum Family {
    /// Square lattice torus, degree 4.
    Tile2d { l: usize, p: TileProbs2d },
    /// Cubic lattice torus, degree 6.
    Tile3d { l: usize, p: TileProbs3d },
    /// Complete graph.
    Wishart {
        n: usize,
        alpha: f64,
        couplers: WishartCouplers,
    },
}

impl Family {
    fn build(&self, seed: u64) -> PlantedMaxCut {
        let mut rng = seeded_rng(seed);
        match *self {
            Family::Tile2d { l, p } => PlantedMaxCut::tile_planting_2d(l, p, &mut rng),
            Family::Tile3d { l, p } => PlantedMaxCut::tile_planting_3d(l, p, &mut rng),
            Family::Wishart { n, alpha, couplers } => {
                PlantedMaxCut::wishart(n, alpha, couplers, &mut rng)
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Family::Tile2d { .. } => "TilePlanting2d",
            Family::Tile3d { .. } => "TilePlanting3d",
            Family::Wishart { .. } => "Wishart",
        }
    }

    /// The construction parameters, as a TOML inline table body.
    fn params(&self) -> String {
        match *self {
            Family::Tile2d { l, p } => {
                format!("l = {l}, p1 = {}, p2 = {}, p3 = {}", p.p1, p.p2, p.p3)
            }
            Family::Tile3d { l, p } => {
                format!("l = {l}, p_2fp = {}, p_4fp = {}", p.p_2fp, p.p_4fp)
            }
            Family::Wishart { n, alpha, couplers } => format!(
                "n = {n}, alpha = {alpha}, couplers = \"{}\"",
                match couplers {
                    WishartCouplers::Gaussian => "Gaussian",
                    WishartCouplers::Discrete => "Discrete",
                }
            ),
        }
    }
}

fn main() {
    let out_dir = Path::new("data/instances/max_cut/hard");
    std::fs::create_dir_all(out_dir).expect("create output directory");

    let specs = suite();

    println!(
        "{:<26} {:>7} {:>9} {:>8} {:>14} {:>7}",
        "instance", "n", "m", "avg deg", "optimum", "exact"
    );

    let mut manifest = String::from(
        "# Planted MaxCut suite — the optimum of every instance below is exact by\n\
         # construction, not a best-known value. Regenerate with\n\
         # `cargo run --release --example generate_hard_maxcut`.\n\
         #\n\
         # `exact_optimum = false` marks a real-weighted instance, where the f32\n\
         # objective cannot represent the cut value exactly and a run can only be\n\
         # scored against `planted_cut` up to rounding.\n\
         #\n\
         # Nothing in the library reads this file. It is for analysing results.\n\n",
    );
    manifest.push_str("generator = \"examples/generate_hard_maxcut\"\n");

    for (stem, seed, family) in &specs {
        let planted = family.build(*seed);
        planted
            .verify()
            .unwrap_or_else(|e| panic!("{stem} failed its own optimality check: {e}"));

        let path = out_dir.join(format!("{stem}.txt"));
        planted
            .problem
            .graph
            .write_to_file(&path)
            .expect("write instance");

        // A written instance must load back as the same problem, and the
        // planted cut must still be worth what the manifest is about to claim.
        let reloaded = MaxCut::load_file(&path).expect("reload as MaxCut");
        assert_eq!(
            reloaded.graph.num_edges(),
            planted.problem.graph.num_edges()
        );
        assert_eq!(
            reloaded.calculate_cut_size(&planted.planted),
            planted.optimum,
            "{stem}: the optimum does not survive a write / load round trip"
        );

        let n = planted.problem.graph.num_vertices();
        let m = planted.problem.graph.num_edges();
        let exact = planted.has_exact_optimum();
        println!(
            "{stem:<26} {n:>7} {m:>9} {:>8.1} {:>14} {:>7}",
            2.0 * m as f64 / n as f64,
            planted.optimum,
            exact
        );

        write!(
            manifest,
            "\n[[instance]]\n\
             file = \"{stem}.txt\"\n\
             family = \"{}\"\n\
             seed = {seed}\n\
             n = {n}\n\
             m = {m}\n\
             planted_cut = {:?}\n\
             exact_optimum = {exact}\n\
             params = {{ {} }}\n",
            family.name(),
            planted.optimum,
            family.params(),
        )
        .expect("format manifest entry");
    }

    let manifest_path = out_dir.join("manifest.toml");
    std::fs::write(&manifest_path, &manifest).expect("write manifest");
    // Parsing it back is the only thing that proves the hand-written TOML is
    // TOML.
    toml::from_str::<toml::Value>(&manifest).expect("manifest is valid TOML");

    println!(
        "\nwritten to {} ({} instances + manifest.toml)",
        out_dir.display(),
        specs.len()
    );
}

/// The suite. Sizes are zero-padded so the benchmark glob's lexicographic order
/// stays meaningful.
fn suite() -> Vec<(String, u64, Family)> {
    let mut specs: Vec<(String, u64, Family)> = Vec::new();

    // -- Tile planting, square lattice (degree 4) ----------------------------
    //
    // Three sizes bracketing the G-set toroidal group, whose largest member is
    // n = 20000. The mixture is calibrated at the smallest of them, so
    // difficulty rises across the three — see `tile_2d_mix`.
    for (i, l) in [40usize, 100, 160].into_iter().enumerate() {
        specs.push((
            format!("tp2d_l{l:03}"),
            3000 + i as u64,
            Family::Tile2d {
                l,
                p: tile_2d_mix(),
            },
        ));
    }

    // -- Tile planting, cubic lattice (degree 6) -----------------------------
    for (i, l) in [10usize, 16, 20].into_iter().enumerate() {
        specs.push((
            format!("tp3d_l{l:02}"),
            3100 + i as u64,
            Family::Tile3d {
                l,
                p: tile_3d_mix(),
            },
        ));
    }

    // -- Wishart planting, complete graph ------------------------------------
    //
    // Integer couplers, so the cut value is exact and reaching the optimum is a
    // comparison rather than a judgement about rounding. The scaling grows as
    // n³, which is what caps this family at n = 96 — see `WishartCouplers`.
    for (i, n) in [48usize, 64, 96].into_iter().enumerate() {
        specs.push((
            format!("wp_n{n:03}"),
            3200 + i as u64,
            Family::Wishart {
                n,
                alpha: wishart_alpha(n),
                couplers: WishartCouplers::Discrete,
            },
        ));
    }
    // One larger instance for scale, at the cost of exactness. Kept because the
    // integer family stops well below any density this repository otherwise
    // benchmarks at, and dropping the dense band entirely would be worse.
    specs.push((
        "wp_n256_gaussian".to_string(),
        3210,
        Family::Wishart {
            n: 256,
            alpha: wishart_alpha(256),
            couplers: WishartCouplers::Gaussian,
        },
    ));

    specs
}

/// Square-lattice class mixture: class 1 against class 4, at the point where
/// the sweep put the boundary.
///
/// Measured at `l = 40`: below `p1 = 0.3` both heuristics solve every draw, and
/// from `p1 = 0.4` up the better of them solves 1 in 15. `p1 = 0.35` is the last
/// point that still separates them — 0/15 for BLS against 10/15 for the stronger
/// method — which is what a suite wants, since larger lattices are harder at a
/// fixed mixture and the two bigger sizes have to stay on the reachable side of
/// the boundary too.
///
/// The class-1-against-class-3 line was swept as well and is uniformly harder:
/// BLS never reached an optimum at any mixture on it. Rejected for exactly that
/// reason — a family no method ever solves ranks nothing.
fn tile_2d_mix() -> TileProbs2d {
    TileProbs2d::new(0.35, 0.0, 0.0)
}

/// Cubic-lattice class mixture, at `p_6fp = 0.75`.
///
/// The reference implementation's comment names `p_6fp = 1` as the hardest
/// regime, and the sweep agrees — too well. At `l = 10` that corner leaves both
/// heuristics at 0/15, while `p_6fp = 0.75` leaves the stronger one at 11/15 and
/// BLS at 0/15. Everything at or below `p_6fp = 0.5` is solved on every run.
fn tile_3d_mix() -> TileProbs3d {
    TileProbs3d::new(0.125, 0.125)
}

/// Wishart hardness parameter for `n`, placed at the measured boundary.
///
/// The sweep found the boundary at `alpha = 0.35 / 0.50 / 0.65 / 0.90` for
/// `n = 48 / 64 / 96 / 256` — no single `alpha` works, and `chook`'s default of
/// `0.75` is easy at every one of those sizes. What is constant is not the
/// ratio but **`n - M`, the dimension of the kernel the planted vector lives
/// in**: the four boundaries sit at `n - M` = 31, 32, 34 and 26. So the knob is
/// expressed as a kernel dimension and `alpha` is derived from it.
///
/// The mechanism reads directly off the construction. `W` is built so that
/// `Wᵀt = 0`, and the energy above the ground state is `|Wᵀs|² / 2n`, which is
/// zero exactly on `ker Wᵀ`. A wide kernel is a large space to find a `±1`
/// vector in; a narrow one is not. The literature's easy–hard–easy profile in
/// `alpha` does not appear here at all — **small `alpha` is the hard side
/// throughout**, which is what a fixed kernel dimension predicts and what
/// "reach the exact optimum" measures, as opposed to the physics criterion the
/// original study used.
const WISHART_KERNEL_DIM: usize = 32;

fn wishart_alpha(n: usize) -> f64 {
    (n.saturating_sub(WISHART_KERNEL_DIM)).max(1) as f64 / n as f64
}
