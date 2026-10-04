//! Single-query (`nq = 1`) scan latency vs. index size, unfiltered and
//! allowlist-filtered, against the production `TurbovecIndex` adapter.
//!
//! Exists to re-check the partitioned-search premise "a single search uses a
//! single core" (docs/partitioned-search-plan.md) under turbovec 1.0, which
//! added a block-parallel single-query kernel gated on
//! `n_blocks >= SINGLE_QUERY_PARALLEL_MIN_BLOCKS` (1024 blocks = 32768 vectors).
//! Also times a full index rebuild (`new` + `add_with_ids`), the cost every
//! long-lived reader pays after any write.
//!
//! Run it twice and compare — the thread count is fixed per process:
//!
//!     cargo run --release -p turbovecdb-core --example scan_parallelism
//!     RAYON_NUM_THREADS=1 cargo run --release -p turbovecdb-core --example scan_parallelism
//!
//! Vectors are random Gaussian, so this measures scan cost only, not recall.

use std::time::Instant;

use rand::{rngs::StdRng, Rng, SeedableRng};
use turbovecdb_core::index::{TurbovecIndex, VectorIndex};

const DIM: usize = 384;
const BIT_WIDTH: usize = 4;
const K: usize = 10;
const QUERIES: usize = 200;
const SIZES: &[usize] = &[8_192, 16_384, 32_768, 65_536, 131_072, 262_144];
/// Allowlist covers this fraction of the index — one "wing".
const WING_FRACTION: f64 = 0.10;

fn gaussian(rng: &mut StdRng, n: usize) -> Vec<f32> {
    // Box-Muller; rand 0.8 has no Normal without rand_distr.
    (0..n)
        .map(|_| {
            let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
            let u2: f32 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect()
}

/// Median per-query latency in microseconds.
fn time_queries(idx: &TurbovecIndex, queries: &[f32], allowlist: Option<&[u64]>) -> f64 {
    // Warm-up: the first search builds the blocked-codes cache.
    idx.search(&queries[..DIM], K, allowlist).unwrap();
    let mut us: Vec<f64> = queries
        .chunks_exact(DIM)
        .map(|q| {
            let t = Instant::now();
            idx.search(q, K, allowlist).unwrap();
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    us[us.len() / 2]
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("threads={threads} dim={DIM} bit_width={BIT_WIDTH} k={K} queries={QUERIES} wing={WING_FRACTION}");
    println!(
        "{:>8} {:>7} {:>11} {:>13} {:>13} {:>11}",
        "n", "blocks", "unfilt_us", "wing_contig_us", "wing_rand_us", "rebuild_ms"
    );

    let mut rng = StdRng::seed_from_u64(42);
    let queries = gaussian(&mut rng, QUERIES * DIM);

    for &n in SIZES {
        let vectors = gaussian(&mut rng, n * DIM);
        let ids: Vec<u64> = (0..n as u64).collect();

        let t = Instant::now();
        let mut idx = TurbovecIndex::new(DIM, BIT_WIDTH).unwrap();
        idx.add_with_ids(&vectors, &ids).unwrap();
        let rebuild_ms = t.elapsed().as_secs_f64() * 1e3;

        let wing_n = ((n as f64) * WING_FRACTION) as usize;
        // mempalace mines wing by wing, so a wing's uids are mostly contiguous.
        let contiguous: Vec<u64> = (0..wing_n as u64).collect();
        // Worst case for block skipping: the wing's uids scattered everywhere.
        let mut scattered = ids.clone();
        for i in 0..wing_n {
            let j = rng.gen_range(i..n);
            scattered.swap(i, j);
        }
        scattered.truncate(wing_n);

        let unfilt = time_queries(&idx, &queries, None);
        let contig = time_queries(&idx, &queries, Some(&contiguous));
        let scat = time_queries(&idx, &queries, Some(&scattered));

        println!(
            "{:>8} {:>7} {:>11.0} {:>13.0} {:>13.0} {:>11.0}",
            n,
            n.div_ceil(32),
            unfilt,
            contig,
            scat,
            rebuild_ms
        );
    }
}
