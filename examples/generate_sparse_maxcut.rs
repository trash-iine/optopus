//! Generates the sparse MaxCut benchmark set, where kernelization can fire.
//!
//! Exact data reduction ([`MaxCutKernel`](optopus::problem::MaxCutKernel))
//! works on vertices of degree at most two, so it does nothing at all on the
//! dense suite written by `generate_dense_maxcut.rs` (minimum degree 50+).
//! These instances live in the opposite regime — average degree 1 to 5, the
//! same range as the G-set instances that reduce (`G70` at degree 2, `G55` and
//! `G60` at degree 5).
//!
//! Every instance comes from a fixed seed, so re-running reproduces the files
//! byte for byte and they never need to be committed.
//!
//! ```text
//! cargo run --release --example generate_sparse_maxcut
//! ```

use optopus::prelude::*;
use optopus::problem::MaxCutKernel;
use std::path::Path;

/// Structural model to sample from.
enum Model {
    /// `p` is derived from the target average degree.
    ErdosRenyi {
        n: usize,
        degree: f64,
    },
    BarabasiAlbert {
        n: usize,
        m: usize,
    },
}

/// Weight regime applied on top of a structural model.
enum Weights {
    Unit,
    /// Inclusive integer range; `0` is never drawn.
    Range(i64, i64),
}

fn main() {
    let out_dir = Path::new("data/instances/max_cut/generated_sparse");
    std::fs::create_dir_all(out_dir).expect("create output directory");

    let specs = [
        (
            // A tree: every vertex reduces away, so the kernel is empty and
            // the offset is the exact optimum.
            "ba_n10000_m001_uw",
            2001,
            Model::BarabasiAlbert { n: 10000, m: 1 },
            Weights::Unit,
        ),
        (
            "ba_n10000_m002_uw",
            2002,
            Model::BarabasiAlbert { n: 10000, m: 2 },
            Weights::Unit,
        ),
        (
            // The G70 regime: average degree 2, over half the vertices
            // immediately reducible.
            "er_n10000_d002_uw",
            2003,
            Model::ErdosRenyi {
                n: 10000,
                degree: 2.0,
            },
            Weights::Unit,
        ),
        (
            "er_n10000_d004_uw",
            2004,
            Model::ErdosRenyi {
                n: 10000,
                degree: 4.0,
            },
            Weights::Unit,
        ),
        (
            // The G55 / G60 regime with weights, so the reduction has to carry
            // weighted edges rather than unit ones.
            "er_n05000_d005_w110",
            2005,
            Model::ErdosRenyi {
                n: 5000,
                degree: 5.0,
            },
            Weights::Range(1, 10),
        ),
        (
            // Signed and large: the path rule produces negative weights, so
            // this checks the whole pipeline on an already-signed instance.
            "er_n20000_d003_pm10",
            2006,
            Model::ErdosRenyi {
                n: 20000,
                degree: 3.0,
            },
            Weights::Range(-10, 10),
        ),
    ];

    println!(
        "{:<24} {:>7} {:>8} {:>8} {:>10} {:>9} {:>10}",
        "instance", "n", "m", "avg deg", "kernel n", "kernel m", "offset"
    );
    for (stem, seed, model, weights) in specs {
        let mut rng = seeded_rng(seed);
        let graph = match model {
            Model::ErdosRenyi { n, degree } => {
                Graph::erdos_renyi(n, degree / (n - 1) as f64, &mut rng)
            }
            Model::BarabasiAlbert { n, m } => Graph::barabasi_albert(n, m, &mut rng),
        };
        let graph = match weights {
            Weights::Unit => graph,
            Weights::Range(lo, hi) => graph.with_random_weights((lo, hi), &mut rng),
        };

        let path = out_dir.join(format!("{stem}.txt"));
        graph.write_to_file(&path).expect("write graph");

        let reloaded = MaxCut::load_file(&path).expect("reload as MaxCut");
        assert_eq!(reloaded.graph.num_edges(), graph.num_edges());

        // Report what the reduction achieves, which is the whole point of
        // this suite.
        let kernel = MaxCutKernel::reduce(&reloaded);
        println!(
            "{stem:<24} {:>7} {:>8} {:>8.2} {:>10} {:>9} {:>10.0}",
            graph.num_vertices(),
            graph.num_edges(),
            2.0 * graph.num_edges() as f64 / graph.num_vertices() as f64,
            kernel.kernel().graph.num_vertices(),
            kernel.kernel().graph.num_edges(),
            kernel.offset(),
        );
    }
    println!("\nwritten to {}", out_dir.display());
}
