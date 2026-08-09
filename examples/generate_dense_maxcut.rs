//! Generates the dense random MaxCut benchmark set.
//!
//! The G-set tops out around 6% density, so it says little about how a
//! heuristic behaves when every vertex touches hundreds of others. This writes
//! a fixed suite of dense (plus a few deliberately non-dense control)
//! instances covering all three structural models and three weight regimes:
//! unweighted, positive integer, and signed.
//!
//! Every instance has **at least 2000 vertices**, up to 5000. The earlier
//! 800-1000 vertex version of this suite saturated: at a 60-second budget most
//! of its instances returned an identical solution on every run, which makes
//! any comparison unmeasurable. Size is what buys headroom.
//!
//! Every instance comes from a fixed seed, so re-running reproduces the files
//! byte for byte and they never need to be committed. The suite is ~250 MB and
//! takes a few minutes to write.
//!
//! Run with (release is not optional — `Graph::from_edges` inserts into sorted
//! adjacency vectors, which takes minutes per dense instance in a debug build):
//! ```text
//! cargo run --release --example generate_dense_maxcut
//! ```

use optopus::prelude::*;
use std::path::Path;

/// Structural model to sample from.
enum Model {
    ErdosRenyi { n: usize, p: f64 },
    BarabasiAlbert { n: usize, m: usize },
    WattsStrogatz { n: usize, k: usize, beta: f64 },
}

/// Weight regime applied on top of a structural model.
enum Weights {
    /// Leave every edge at 1.0 — plateaus are widest here, which is what the
    /// objective-preserving perturbations exploit.
    Unit,
    /// Inclusive integer range; `0` is never drawn.
    Range(i64, i64),
}

fn main() {
    let out_dir = Path::new("data/instances/max_cut/generated");
    std::fs::create_dir_all(out_dir).expect("create output directory");

    // Zero-padded stems keep the benchmark glob's lexicographic order
    // meaningful.
    let specs = [
        (
            "er_n2000_p030_uw",
            1001,
            Model::ErdosRenyi { n: 2000, p: 0.30 },
            Weights::Unit,
        ),
        (
            "er_n2000_p030_w110",
            1002,
            Model::ErdosRenyi { n: 2000, p: 0.30 },
            Weights::Range(1, 10),
        ),
        (
            "er_n2000_p050_uw",
            1003,
            Model::ErdosRenyi { n: 2000, p: 0.50 },
            Weights::Unit,
        ),
        (
            "er_n2000_p050_pm10",
            1004,
            Model::ErdosRenyi { n: 2000, p: 0.50 },
            Weights::Range(-10, 10),
        ),
        (
            "er_n3000_p030_uw",
            1005,
            Model::ErdosRenyi { n: 3000, p: 0.30 },
            Weights::Unit,
        ),
        (
            // Same edge count as the n=3000 instance at a third of the
            // density: separates "many edges" from "dense".
            "er_n5000_p010_uw",
            1006,
            Model::ErdosRenyi { n: 5000, p: 0.10 },
            Weights::Unit,
        ),
        (
            "ba_n2000_m050_uw",
            1007,
            Model::BarabasiAlbert { n: 2000, m: 50 },
            Weights::Unit,
        ),
        (
            "ba_n3000_m100_w110",
            1008,
            Model::BarabasiAlbert { n: 3000, m: 100 },
            Weights::Range(1, 10),
        ),
        (
            // Average degree 100 at 2000 vertices: the sparse control, close
            // to G-set territory in density but far larger.
            "ws_n2000_k100_b010_uw",
            1009,
            Model::WattsStrogatz {
                n: 2000,
                k: 100,
                beta: 0.1,
            },
            Weights::Unit,
        ),
        (
            "ws_n3000_k200_b030_w110",
            1010,
            Model::WattsStrogatz {
                n: 3000,
                k: 200,
                beta: 0.3,
            },
            Weights::Range(1, 10),
        ),
    ];

    println!(
        "{:<26} {:>6} {:>9} {:>8} {:>9}",
        "instance", "n", "m", "avg deg", "density"
    );
    for (stem, seed, model, weights) in specs {
        let mut rng = seeded_rng(seed);
        let graph = match model {
            Model::ErdosRenyi { n, p } => Graph::erdos_renyi(n, p, &mut rng),
            Model::BarabasiAlbert { n, m } => Graph::barabasi_albert(n, m, &mut rng),
            Model::WattsStrogatz { n, k, beta } => Graph::watts_strogatz(n, k, beta, &mut rng),
        };
        let graph = match weights {
            Weights::Unit => graph,
            Weights::Range(lo, hi) => graph.with_random_weights((lo, hi), &mut rng),
        };

        let path = out_dir.join(format!("{stem}.txt"));
        graph.write_to_file(&path).expect("write graph");

        // A written instance must load back as the same MaxCut problem.
        let reloaded = MaxCut::load_file(&path).expect("reload as MaxCut");
        assert_eq!(reloaded.graph.num_edges(), graph.num_edges());
        assert_eq!(reloaded.graph.num_vertices(), graph.num_vertices());

        let n = graph.len() as f64;
        let m = graph.num_edges() as f64;
        println!(
            "{stem:<26} {:>6} {:>9} {:>8.1} {:>8.1}%",
            graph.len(),
            graph.num_edges(),
            2.0 * m / n,
            100.0 * m / (n * (n - 1.0) / 2.0),
        );
    }
    println!("\nwritten to {}", out_dir.display());
}
