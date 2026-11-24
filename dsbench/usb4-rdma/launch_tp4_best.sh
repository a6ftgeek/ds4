#!/bin/bash
# DS4 4-node TP=4 over the switchless USB4 RDMA ring -- best verified setup.
#
# Verified 2026-11-24 on feat/rocm-tp-ring @ 79a9247, 81 GiB Q2 weights
# (DeepSeek-V4-Flash-...-0731.gguf), greedy, all four ranks 192.168.1.101-104:
#
#   ctx 2048   : prefill 64.25 t/s, generation 9.65 t/s, GTT peak 37.82 GiB/rank,
#                resident 26.34 GiB/rank; generated text md5
#                a13ba035b8347fa92e602fd8f751ca91, byte-identical to the TP=1
#                reference (the canonical 16-token check).
#   ctx 1048576: allocates and runs. KV 13.79 GiB (raw 0.36 + compressed 13.44)
#                + 8.00 GiB buffers + 26.34 GiB resident = 48.13 GiB planned,
#                GTT peak 47.88 GiB/rank, MemAvailable min 42.85 GiB/64 GiB on
#                node1. A 30,474-token story prompt (fact deep inside, task at
#                the end) answered Bob=34, Alice=52, Clara=71 -- the exact
#                written-out assignments -- so compressed KV retrieval works.
#
# Vision: OFF. ./download_model.sh ds41f-vision is the V4.1 sidecar; the 0731
#   checkpoint is rejected at ds4.c:74106 ("--vision requires ... the pinned
#   DeepSeek V4 Flash Vision-Exp or V4.1 Flash model"), and V4.1 vision itself
#   is Metal-only (ds4.c:74034). The 0731 weights carry no vision projection.
# DSpark: OFF by default (DSPARK=1 turns it on). The N-rank verify gate is
#   generalized (tp_batch_rows is set for world>=2), so a world=4 verify block
#   now uses the same batched per-layer gate as world=2 instead of the per-layer
#   bulk exchange: verify 742 -> ~320 ms/cycle on coding and TP=4 DSpark goes
#   4.00-4.18 -> 7.22-7.98 t/s. It is still a net loss against 9.4-9.5 t/s:
#     TP=4 coding  9.45 -> 7.22 t/s (-24%, accept 82.50%, 33/40)
#     TP=4 prose   9.35 -> 7.98 t/s (-15%, accept 90.00%, 9/10)
#   The canonical 16-token greedy md5 (a13ba035...) and the coding 64-token md5
#   (4a10b554...) match the non-speculative run with seed batches on or off.
#   The prose continuation differs (de52e3b2 vs e728672c): that is the batched
#   verifier's reduction order and is identical to TP=2 DSpark, which already
#   behaves this way. TP=2 coding remains the one net win (10.04 -> 14.45 t/s).
#   Remaining gap and the audit are in dsbench/usb4-rdma/FEATURE_STATUS.md.
#
# usage: bash launch_tp4_best.sh [prompt-file] [n_tokens]
# env:   CTX (default 1048576), TP_CORE (2), PORT (19730), TIMEOUT (3600),
#        DSPARK (0)
#
# rank 0 = node1 (coordinator, started last). Ranks 1-3 = node2-4.
# --tp-cpu-core is mandatory: unpinned busy-poll ring threads lose ~40% decode.

set -u

REPO=${REPO:-$HOME/dev/ds4-v41-pp/src}
PROMPT=${1:-/tmp/prompt_now.txt}
NTOK=${2:-16}
CTX=${CTX:-1048576}
TP_CORE=${TP_CORE:-2}
PORT=${PORT:-19730}
TIMEOUT=${TIMEOUT:-3600}
# Speculative decode is opt-in while its net effect is measured: DSPARK=1 adds
# --dspark and the 0731 support drafter plus DS4_DSPARK_STATS=1 so the leader
# prints cycles/proposed/accepted/accept_rate. The default stays the verified
# non-speculative configuration.
DSPARK=${DSPARK:-0}
DSPARK_MODEL=${DSPARK_MODEL:-gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}

M='gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
IPS=(192.168.1.101 192.168.1.102 192.168.1.103 192.168.1.104)
HOSTS=(node1 node2 node3 node4)
TAG=${TAG:-tp4best}

[ -f "$REPO/$M" ] || { echo "missing model: $REPO/$M" >&2; exit 1; }
[ -f "$PROMPT" ] || { echo "missing prompt file: $PROMPT" >&2; exit 1; }

ring=()
for ip in "${IPS[@]}"; do ring+=(--tp-ring-ip "$ip"); done

common=(-m "$M" -c "$CTX" --temp 0 -n "$NTOK" --prompt-file "$PROMPT" --rocm
        --tensor-parallel --transport usb4-rdma --tp-world-size 4
        "${ring[@]}" --tp-cpu-core "$TP_CORE")
if [ "$DSPARK" = 1 ]; then
    common+=(--dspark --mtp-model "$DSPARK_MODEL")
fi
DS4_ENV="USB4_RING_QP_TIMEOUT=14"
[ "$DSPARK" = 1 ] && DS4_ENV="$DS4_ENV DS4_DSPARK_STATS=1"

cleanup() {
    for ((r=1; r<4; r++)); do
        ssh -o BatchMode=yes "${HOSTS[$r]}" 'pkill -x ds4 2>/dev/null' >/dev/null 2>&1
    done
    pkill -x ds4 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "== DS4 TP=4 USB4 ring: ctx=$CTX n=$NTOK prompt=$PROMPT"
cleanup; sleep 1

for ((r=1; r<4; r++)); do
    h=${HOSTS[$r]}
    ssh -o BatchMode=yes "$h" \
        "cd \$HOME/dev/ds4-v41-pp/src && exec toolbox run -c vllm-gfx1151 -- env $DS4_ENV ./ds4 ${common[*]} --role worker --tp-rank $r --coordinator ${IPS[0]} $PORT" \
        > "/tmp/${TAG}_r${r}.log" 2>&1 &
    echo "started rank $r on $h"
done
sleep 6

cd "$REPO" || exit 1
timeout "$TIMEOUT" toolbox run -c vllm-gfx1151 -- env $DS4_ENV \
    ./ds4 "${common[@]}" \
    --role coordinator --listen 0.0.0.0 "$PORT" \
    > "/tmp/${TAG}_r0.log" 2>&1
rc=$?
echo "coordinator exit=$rc"
sleep 2
cleanup

echo "== result"
grep -ao 'prefill: [0-9.]* t/s, generation: [0-9.]* t/s' "/tmp/${TAG}_r0.log" | tail -1
grep -av '^ds4' "/tmp/${TAG}_r0.log" | grep -av '^tp worker' | \
    grep -av '^tensor parallelism' | grep -av '^$' | md5sum | \
    sed 's/^/generated-text md5: /'
echo "logs: /tmp/${TAG}_r0.log .. /tmp/${TAG}_r3.log"
exit $rc
