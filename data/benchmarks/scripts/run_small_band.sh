#!/usr/bin/env bash
# Sequentially runs the small-band benchmark TOMLs and moves each result to
# docs/benchmarks/data/<problem>/<curated_name>.toml.
#
# Output: one "PROGRESS" line per stage (start/done/fail) to stdout, full
# benchmark logs to /tmp/phase4_runner.<n>.log per stage.
#
# Total wall-clock at 11 cores: ~10 hours.
set -u

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
BIN="$REPO/target/release/optopus"
RESULT_DIR="$REPO/result"
CURATED_DIR="$REPO/docs/benchmarks/data"
LOG_DIR="/tmp/phase4_logs"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Tuples: <source TOML> <curated subdir> <curated filename>
# Ordered shortest first for early failure detection.
STAGES=(
  "tsp/lkh_small.toml                  tsp           lkh_tsplib_small.toml"
  "tsp/general_small.toml              tsp           general_tsplib_small.toml"
  "jssp/general_small.toml             jssp          general_orlib_small.toml"
  "qubo/cdcl_small.toml                qubo          cdcl_bqp_small.toml"
  "sat/cdcl_small.toml                 sat           cdcl_uf_small.toml"
  "maxcut/cdcl_small.toml              maxcut        cdcl_gset_small.toml"
  "qubo/general_small.toml             qubo          general_bqp_small.toml"
  "sat/general_small.toml              sat           general_uf_small.toml"
  "vertex_cover/general_small.toml     vertex_cover  general_gset_small.toml"
  "maxcut/general_small.toml           maxcut        general_gset_small.toml"
)

cd "$REPO" || exit 2

stage_idx=0
for entry in "${STAGES[@]}"; do
  read -r src subdir name <<< "$entry"
  stage_idx=$((stage_idx + 1))
  cfg="data/benchmarks/$src"
  log="$LOG_DIR/${stage_idx}_${subdir}_$(basename "$name" .toml).log"
  dst_dir="$CURATED_DIR/$subdir"
  dst="$dst_dir/$name"
  mkdir -p "$dst_dir"

  start_time=$(date +%s)
  echo "PROGRESS stage=$stage_idx/${#STAGES[@]} START cfg=$cfg t=$(date -u +%FT%TZ)"

  # Snapshot existing results to identify the new one afterwards.
  before=$(ls -t "$RESULT_DIR" 2>/dev/null | head -1 || true)

  if ! "$BIN" "$cfg" > "$log" 2>&1; then
    end_time=$(date +%s)
    echo "PROGRESS stage=$stage_idx/${#STAGES[@]} FAIL cfg=$cfg dur=$((end_time-start_time))s log=$log"
    continue
  fi

  # The newest result/*.toml is ours.
  newest=$(ls -t "$RESULT_DIR" | head -1)
  if [[ -z "$newest" || "$newest" == "$before" ]]; then
    echo "PROGRESS stage=$stage_idx/${#STAGES[@]} FAIL cfg=$cfg reason=no-result-file"
    continue
  fi
  mv "$RESULT_DIR/$newest" "$dst"
  end_time=$(date +%s)
  echo "PROGRESS stage=$stage_idx/${#STAGES[@]} DONE cfg=$cfg dur=$((end_time-start_time))s dst=$dst"
done

echo "PROGRESS ALL_DONE total_stages=${#STAGES[@]}"
