#!/usr/bin/env bash
# Sequentially runs the medium-band benchmark TOMLs and moves each result to
# docs/benchmarks/data/<problem>/<curated_name>.toml.
#
# Output: one "PROGRESS" line per stage (start/done/fail) to stdout, full
# benchmark logs to /tmp/phase4_medium.<n>.log per stage.
#
# Wall-clock estimate at 11 cores: ~4-5 hours.
set -u

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
BIN="$REPO/target/release/optopus"
RESULT_DIR="$REPO/result"
CURATED_DIR="$REPO/docs/benchmarks/data"
LOG_DIR="/tmp/phase4_medium_logs"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Tuples: <source TOML> <curated subdir> <curated filename>
# Ordered shortest first for early failure detection.
STAGES=(
  "tsp/lkh_medium.toml                  tsp           lkh_tsplib_medium.toml"
  "tsp/general_medium.toml              tsp           general_tsplib_medium.toml"
  "qubo/general_medium.toml             qubo          general_bqp_medium.toml"
  "sat/general_medium.toml              sat           general_uf_medium.toml"
  "maxcut/general_medium.toml           maxcut        general_gset_medium.toml"
  "vertex_cover/general_medium.toml     vertex_cover  general_gset_medium.toml"
  "jssp/general_medium.toml             jssp          general_orlib_medium.toml"
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

  before=$(ls -t "$RESULT_DIR" 2>/dev/null | head -1 || true)

  if ! "$BIN" "$cfg" > "$log" 2>&1; then
    end_time=$(date +%s)
    echo "PROGRESS stage=$stage_idx/${#STAGES[@]} FAIL cfg=$cfg dur=$((end_time-start_time))s log=$log"
    continue
  fi

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
