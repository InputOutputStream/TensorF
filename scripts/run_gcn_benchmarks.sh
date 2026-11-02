#!/usr/bin/env bash
# Run bin/coratrain on every converted dataset over several seeds and print a summary.
#
# Usage: scripts/run_gcn_benchmarks.sh [-s SEEDS] [-e EPOCHS] [-p PATIENCE] [-v] [dataset ...]
#   default datasets: cora citeseer pubmed   default: 5 seeds, 200 epochs, patience 20
#   -p 0 disables early stopping (always trains the full EPOCHS)
#   -v   also print per-seed results (best epoch, val, test, ms/step for each seed)
#
# Needs: make bin/coratrain, scripts/download_datasets.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=5; EPOCHS=200; PATIENCE=20; VERBOSE=0
while getopts "s:e:p:v" o; do
  case "$o" in
    s) SEEDS="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    p) PATIENCE="$OPTARG" ;;
    v) VERBOSE=1 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
sets=("$@"); [ ${#sets[@]} -eq 0 ] && sets=(cora citeseer pubmed)

[ -x "$ROOT/bin/coratrain" ] || { echo "bin/coratrain missing: run 'make bin/coratrain'" >&2; exit 1; }

# ── helpers ──────────────────────────────────────────────────────────────────
# stdin: one number per line  →  stdout: "mean std min max" (space separated)
stats() {
    awk '
        { n++; s += $1; if (n == 1 || $1 < mn) mn = $1; if (n == 1 || $1 > mx) mx = $1; a[n] = $1 }
        END {
            if (n == 0) { print "n/a n/a n/a n/a"; exit }
            m = s / n; q = 0
            for (i = 1; i <= n; i++) q += (a[i] - m) ^ 2
            printf "%.1f %.1f %.1f %.1f\n", m, sqrt(q / n), mn, mx
        }'
}

# ── header ───────────────────────────────────────────────────────────────────
cat <<EOF

═════════════════════════════════════════════════════════════════════════════════════
GCN training benchmark — ${SEEDS} seeds × ${EPOCHS} epochs (patience ${PATIENCE})
═════════════════════════════════════════════════════════════════════════════════════

DATASET     GRAPH                                  RUNS  VAL @ BEST VAL       TEST @ BEST VAL      BEST EPOCH   SPEED
            nodes   feats   cls   edges                  mean ± std           mean ± std           mean ± std   ms/step
                                                          (min / max)
─────────────────────────────────────────────────────────────────────────────────────
EOF

# ── per-dataset ──────────────────────────────────────────────────────────────
for d in "${sets[@]}"; do
    c="$ROOT/Datasets/$d/$d.content"
    e="$ROOT/Datasets/$d/$d.cites"
    if [ ! -s "$c" ] || [ ! -s "$e" ]; then
        printf '%-10s  MISSING — run scripts/download_datasets.sh %s\n' "$d" "$d" >&2
        continue
    fi

    log="$ROOT/bin/.run_$d.log"
    : > "$log"
    for s in $(seq 1 "$SEEDS"); do
        (cd "$ROOT" && ./bin/coratrain "$c" "$e" "$EPOCHS" "$s" "$PATIENCE") >> "$log"
    done

    # ── graph info ───────────────────────────────────────────────────────────
    nodes="" feats="" classes="" edges=""
    while IFS= read -r line; do
        if [[ "$line" =~ loaded\ ([0-9]+)\ nodes,\ ([0-9]+)\ features,\ ([0-9]+)\ classes,\ ([0-9]+)\ raw\ edges ]]; then
            nodes=${BASH_REMATCH[1]}; feats=${BASH_REMATCH[2]}
            classes=${BASH_REMATCH[3]}; edges=${BASH_REMATCH[4]}
            break
        fi
    done < "$log"
    [ -z "$nodes" ] && { printf '%-10s  NO GRAPH INFO — see %s\n' "$d" "$log" >&2; continue; }

    # ── per-seed series ──────────────────────────────────────────────────────
    epochs=(); vals=(); tests=(); ms=()
    while IFS= read -r line; do
        if [[ "$line" =~ best\ val\ \(epoch\ ([0-9]+)\):\ val\ ([0-9.]+)%\ test\ ([0-9.]+)% ]]; then
            epochs+=("${BASH_REMATCH[1]}")
            vals+=("${BASH_REMATCH[2]}")
            tests+=("${BASH_REMATCH[3]}")
        fi
    done < <(grep '^best val' "$log")

    while IFS= read -r line; do
        if [[ "$line" =~ avg\ train\ step:\ ([0-9.]+)\ ms ]]; then
            ms+=("${BASH_REMATCH[1]}")
        fi
    done < <(grep '^avg train step' "$log")

    n=${#tests[@]}
    [ "$n" -eq 0 ] && { printf '%-10s  NO RESULTS — see %s\n' "$d" "$log" >&2; continue; }

    # ── optional per-seed breakdown ──────────────────────────────────────────
    if [ "$VERBOSE" -eq 1 ]; then
        echo
        echo "── ${d}: per-seed results ──"
        printf '  %-5s  %-11s  %-11s  %-12s  %-9s\n' seed "best epoch" "val @ best" "test @ best" "ms/step"
        for i in "${!tests[@]}"; do
            printf '  %-5d  %-11d  %10.2f%%  %11.2f%%  %9.2f\n' \
                $((i+1)) "${epochs[$i]}" "${vals[$i]}" "${tests[$i]}" "${ms[$i]}"
        done
        echo
    fi

    # ── aggregate ────────────────────────────────────────────────────────────
    read -r v_mean v_std _ _        < <(printf '%s\n' "${vals[@]}"   | stats)
    read -r t_mean t_std t_min t_max < <(printf '%s\n' "${tests[@]}"  | stats)
    read -r ep_mean ep_std _ _      < <(printf '%s\n' "${epochs[@]}" | stats)
    read -r ms_mean _ _ _           < <(printf '%s\n' "${ms[@]}"     | stats)

    val_cell=$(printf '%.1f ± %.1f'  "$v_mean"  "$v_std")
    test_cell=$(printf '%.1f ± %.1f' "$t_mean"  "$t_std")
    epoch_cell=$(printf '%.1f ± %.1f' "$ep_mean" "$ep_std")
    mm_cell=$(printf '(%5.1f / %5.1f)' "$t_min" "$t_max")

    printf '%-10s  %5s   %5s  %4s  %5s   %2d/%-2d  %-20s  %-20s  %-12s  %6.1f\n' \
        "$d" "$nodes" "$feats" "$classes" "$edges" "$n" "$SEEDS" \
        "$val_cell" "$test_cell" "$epoch_cell" "$ms_mean"
    printf '%-10s  %5s   %5s  %4s  %5s   %5s  %-20s  %-20s\n' \
        "" "" "" "" "" "" "" "$mm_cell"
done

# ── legend ───────────────────────────────────────────────────────────────────
cat <<'EOF'

─────────────────────────────────────────────────────────────────────────────────────
Legend
  GRAPH               Nodes, features, classes, and edges of the input graph.
  RUNS                Seeds that produced a result (a crashed run shows fewer than SEEDS).
  VAL @ BEST VAL      Validation accuracy reached at the best validation epoch,
                      averaged over seeds (mean ± std).
  TEST @ BEST VAL     Test accuracy measured at that same epoch.  Second line shows
                      min / max across seeds so a single bad seed is visible at a glance.
  BEST EPOCH          Epoch at which each seed's best validation accuracy occurred.
  SPEED               Mean training time per epoch across all runs.
EOF