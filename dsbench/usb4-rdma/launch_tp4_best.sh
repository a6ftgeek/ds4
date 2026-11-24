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
# DSpark: OFF. Matching 0731 drafter weights exist and run on ROCm single-host,
#   but the mirrored speculative-verify worker path is rank-1-only (ds4.c:85410)
#   and the lockstep drafter split is compiled out on ROCm (ds4.c:75261,
#   ds4.c:80511); ranks 2 and 3 abort at the first spec cycle. Do not add
#   --dspark to a 4-rank launch until that path is generalized.
#
# usage: bash launch_tp4_best.sh [prompt-file] [n_tokens]
# env:   CTX (default 1048576), TP_CORE (2), PORT (19730), TIMEOUT (3600)
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
        "cd \$HOME/dev/ds4-v41-pp/src && exec env USB4_RING_QP_TIMEOUT=14 toolbox run -c vllm-gfx1151 -- ./ds4 ${common[*]} --role worker --tp-rank $r --coordinator ${IPS[0]} $PORT" \
        > "/tmp/${TAG}_r${r}.log" 2>&1 &
    echo "started rank $r on $h"
done
sleep 6

cd "$REPO" || exit 1
timeout "$TIMEOUT" env USB4_RING_QP_TIMEOUT=14 \
    toolbox run -c vllm-gfx1151 -- ./ds4 "${common[@]}" \
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
