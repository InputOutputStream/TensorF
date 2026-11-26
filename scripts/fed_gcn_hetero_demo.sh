#!/usr/bin/env bash
# Heterogeneous federated GCN with SERVER-SIDE data shards.
#   ./scripts/fed_gcn_hetero_demo.sh [POLICY=capacity|equal] [ROUNDS=20] [LOCAL_EPOCHS=5] [PARTITIONER=auto]
# Env: DATA=Datasets/cora (or Datasets/ogbn-arxiv, ...)   SHARDS="partition partition partition partition"
#      (one of auto|partition|halo:K|full per device)   SERVER_SHARD_MODE=auto (server-side override)
# 4 simulated phones (score/speed 95/1.0, 80/0.8, 40/0.4, 15/0.15). The server owns the dataset, partitions it by
# capacity and ships each phone its shard; clients run in an empty directory (no dataset access) and are told only
# features/classes (which fix the model layout). The server evaluates every global model on the test nodes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY=${1:-capacity}; ROUNDS=${2:-20}; LE=${3:-5}; PART=${4:-auto}; PORT=${PORT:-$((9800 + RANDOM % 150))}
DATA=${DATA:-$ROOT/Datasets/cora}; read -r -a SHARDS <<< "${SHARDS:-partition partition partition partition}"
SCORES=(95 80 40 15); SPEEDS=(1.0 0.8 0.4 0.15); K=4
DIMS=$("$ROOT/bin/fed_mobile_server" --algo gcn --opt data=$DATA --dims); FEAT=${DIMS#features=}; FEAT=${FEAT%% *}; CLS=${DIMS##*classes=}
LOG=$(mktemp -d); WORK=$(mktemp -d)
t0=$(date +%s.%N)
"$ROOT/bin/fed_mobile_server" --algo gcn --opt data=$DATA --opt shards=1 --opt server_eval=1 --opt policy=$POLICY --opt partitioner=$PART \
  --opt shard_mode=${SERVER_SHARD_MODE:-auto} --port $PORT --min-clients $K --rounds $ROUNDS --grace-ms 300 --enroll-grace-ms 500 > $LOG/server.log 2>&1 &
SRV=$!; sleep 1; pids=()
cd "$WORK"          # empty directory: the clients cannot read the dataset
for ((i=0;i<K;i++)); do
  "$ROOT/bin/fed_mobile_client" --algo gcn --port $PORT --id dev$i --score ${SCORES[$i]} --rounds $ROUNDS --shard ${SHARDS[$i]:-partition} \
     --opt source=shard --opt features=$FEAT --opt classes=$CLS --opt local_epochs=$LE --opt speed=${SPEEDS[$i]} --opt verbose=1 > $LOG/dev$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done; wait $SRV || true
t1=$(date +%s.%N)
echo "=== data=$(basename $DATA) policy=$POLICY partitioner=$PART rounds=$ROUNDS (logs: $LOG)"
grep -o '"type":"fed_partition".*' $LOG/server.log | python3 -c "
import sys,json
for l in sys.stdin:
    j=json.loads('{'+l.strip().replace('@@TFEVT ',''))
    print('partitioner=%s cut=%.1f%% comm_volume=%d max_imbalance=%.2f partition_ms=%.1f'%(j['partitioner'],100*j['cut'],j['volume'],j['max_imbalance'],j['ms']))
    for d in j['devices']: print('  %-5s score %3.0f  target %4.1f%%  owned %6d  shard=%-9s (%d nodes) %s'%(d['id'],d['score'],100*d['target'],d['nodes'],d['shard'],d['shard_nodes'],('downgraded: '+d['reason']) if d['reason'] else ''))
"
echo "final server-side evaluation: $(grep fed_eval $LOG/server.log | tail -1 | sed 's/@@TFEVT //')"
printf "wall time: %.1fs\n" $(echo "$t1 - $t0" | bc)
