#!/usr/bin/env bash
# Federated GCN on Cora across K simulated devices (separate processes, real sockets).
#   ./scripts/fed_gcn_demo.sh [K=4] [ROUNDS=30] [LOCAL_EPOCHS=5] [STRATEGY=fedavg] [PARTITION=bfs]
# Needs: make fed-mobile fed_gcn_baseline ; ./scripts/download_cora.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
K=${1:-4}; ROUNDS=${2:-30}; LE=${3:-5}; STRAT=${4:-fedavg}; PART=${5:-bfs}; PORT=${PORT:-9760}
OPTS=(--opt data=Datasets/cora --opt parts=$K --opt partition=$PART --opt local_epochs=$LE --opt eval_full=1 --opt verbose=0)
LOG=$(mktemp -d); echo "logs in $LOG"
./bin/fed_mobile_server --algo gcn --opt data=Datasets/cora --port $PORT --min-clients $K --rounds $ROUNDS --strategy "$STRAT" --grace-ms 300 > $LOG/server.log 2>&1 &
SRV=$!; sleep 1
pids=()
for ((i=0;i<K;i++)); do
  ./bin/fed_mobile_client --algo gcn --port $PORT --id dev$i "${OPTS[@]}" --opt part=$i --rounds $ROUNDS > $LOG/dev$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
wait $SRV || true
echo "--- federated GCN: $K devices, $ROUNDS rounds x $LE local epochs, partition=$PART, strategy=$STRAT"
for ((i=0;i<K;i++)); do grep "global round $ROUNDS:" $LOG/dev$i.log | tail -1; done
echo "--- bytes:"; grep -h "done:" $LOG/dev0.log
echo "--- reference (same split, same total local epochs = $((ROUNDS*LE))):"
./bin/fed_gcn_baseline --opt data=Datasets/cora --opt parts=$K --opt partition=$PART --epochs $((ROUNDS*LE))
