#!/bin/bash
# DS4 4-node TP=4 over the switchless USB4 RDMA ring -- best verified setup.
#
# Default model: gguf/DeepSeek-V4.1-Flash-Q4.gguf
#   518,596,067,328 B, sha256
#   a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e,
#   byte-identical on node1-4 (ranks 0-3).
#
# Verified 2026-11-24 on feat/rocm-tp-full-stack @ 6d75052, greedy --temp 0,
# ranks 192.168.1.101-104, `--tensor-parallel --transport usb4-rdma
# --tp-world-size 4 --tp-cpu-core 2`:
#
#   ctx 2048 (271-token coding prompt):
#     CLI prefill 11.12 t/s, generation 5.12 t/s
#     CLIP 12.09/8.04 and 9.94/5.95 on repeat runs (first-token warmup varies)
#     TP expert shard 80.56 GiB of 294.14 GiB per rank, 82.11 GiB planned,
#     GTT peak 82.13 GiB/rank, MemAvailable min 22.15 GiB
#     coherent answer that traces patience sorting correctly through
#     [3,1,4,1,5,9,2,6,5,3,5] (tails 3 -> 1 -> 1,4 -> ... -> 1,4,5,9)
#   ctx 1048576:
#     KV 6.26 GiB (compressed) + 12.35 GiB buffers + 80.56 GiB resident
#     = 99.17 GiB planned; GTT peak 97.83 GiB/rank (79% of the 124 GiB pool);
#     MemAvailable min node1 5.61 GiB, node2 18.59, node3 23.06, node4 23.09.
#     CLI prefill 10.74 t/s, generation 7.30 t/s (n=32).
#     HTTP API (ds4-server, same 4-rank ring): /v1/models reports
#     context_length 1048576; a 74-token chat completion TTFT 6.72 s
#     (measured: the server logs "prompt done 6.715s"), decode 9.84 t/s over
#     160 tokens, 22.975 s total. Answer was coherent and correct.
#
# Performance vs the 0731 V4 Q2 reference at TP=4 (ctx 2048): prefill
# 64.25 -> 11.12 t/s (-83%) because V4.1 prefill here runs the per-token
# scalar path with a ring gate per layer (the batched V4.1 prefill path is
# not wired at world=4 yet), plus Engram rows read from disk; decode
# 9.65 -> 9.84 t/s through the server (5.1-8.0 through the CLI, which
# flushes per token). Resident grows 26.34 -> 80.56 GiB/rank because Q4
# weights are 6x the Q2 file.
#
# Vision: OFF and not wired. ./download_model.sh ds41f-vision is the V4.1
#   encoder on all four nodes, but V4.1 vision is hard-gated to Metal
#   (ds4.c:74034, "ds4: V4.1 vision requires Metal") and the ROCm backend has
#   no V4.1 vision kernels. This launcher never passes --vision.
# DSpark: OFF by default (DSPARK=1 enables --dspark with the 0731 drafter).
#   There is NO V4.1 drafter: download_model.sh has no ds41f-dspark target, the
#   HF repo ships only Q2/Q4/Vision, and the repo README says drafter weights
#   are not interchangeable across checkpoints. The only drafter on disk,
#   gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf, belongs to 0731, so DSpark
#   cannot be measured on Q4 -- expect DSPARK=1 to be meaningless here.
#
# Rank fan-out: node1 -> node2 ssh is unreliable on this fabric (shared host
#   keys + auth). The script tries ssh nodeN from the coordinator with retries;
#   if that fails it prints the exact commands and stops before the
#   coordinator starts, so no half-boot is left behind. From the Windows host
#   (where `ssh nodeN` works) start ranks 1-3 with e.g.
#     ssh node2 'nohup bash launch_worker_cmd.sh 1 > /tmp/tp4_r1.log 2>&1 &'
#   or the equivalent `toolbox run ... --role worker --tp-rank N` line, then
#   run this script with WORKERS_STARTED=1 to start only the coordinator.
#
# usage: bash launch_tp4_best.sh [prompt-file] [n_tokens]
# env:   MODEL (q4 default; 0731 for the verified Q2 path; or a literal path),
#        CTX (1048576), TP_CORE (2), PORT (19730), TIMEOUT (3600),
#        DSPARK (0), WORKERS_STARTED (0)

set -u

REPO=${REPO:-$HOME/dev/ds4-v41-pp/src}
PROMPT=${1:-/tmp/prompt_now.txt}
NTOK=${2:-16}
CTX=${CTX:-1048576}
TP_CORE=${TP_CORE:-2}
PORT=${PORT:-19730}
TIMEOUT=${TIMEOUT:-3600}
WORKERS_STARTED=${WORKERS_STARTED:-0}
# Speculative decode is opt-in while its net effect is measured: DSPARK=1 adds
# --dspark and the 0731 support drafter plus DS4_DSPARK_STATS=1 so the leader
# prints cycles/proposed/accepted/accept_rate. The default stays the verified
# non-speculative configuration. There is no V4.1 drafter -- see the header.
DSPARK=${DSPARK:-0}
DSPARK_MODEL=${DSPARK_MODEL:-gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}

MODEL=${MODEL:-q4}
case "$MODEL" in
    q4|ds41f-q4)  M='gguf/DeepSeek-V4.1-Flash-Q4.gguf' ;;
    0731|q2|v4)   M='gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf' ;;
    *)            M=$MODEL ;;
esac

IPS=(192.168.1.101 192.168.1.102 192.168.1.103 192.168.1.104)
HOSTS=(node1 node2 node3 node4)
SSHOPT=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
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
    case "$M" in
        *V4.1*) echo "warning: DSPARK=1 on a V4.1 model uses the 0731 drafter;" \
                       "no V4.1 drafter exists (see header)" >&2 ;;
    esac
fi
DS4_ENV="USB4_RING_QP_TIMEOUT=14"
[ "$DSPARK" = 1 ] && DS4_ENV="$DS4_ENV DS4_DSPARK_STATS=1"

cleanup() {
    for ((r=1; r<4; r++)); do
        ssh "${SSHOPT[@]}" "${HOSTS[$r]}" 'pkill -x ds4 2>/dev/null' >/dev/null 2>&1
    done
    pkill -x ds4 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "== DS4 TP=4 USB4 ring: model=$M ctx=$CTX n=$NTOK prompt=$PROMPT"
cleanup; sleep 1

if [ "$WORKERS_STARTED" != 1 ]; then
    for ((r=1; r<4; r++)); do
        h=${HOSTS[$r]}
        started=0
        for attempt in 1 2 3 4; do
            nohup ssh "${SSHOPT[@]}" "$h" \
                "cd \$HOME/dev/ds4-v41-pp/src && exec toolbox run -c vllm-gfx1151 -- env $DS4_ENV ./ds4 ${common[*]} --role worker --tp-rank $r --coordinator ${IPS[0]} $PORT" \
                > "/tmp/${TAG}_r${r}.log" 2>&1 &
            sleep 5
            if ! grep -qa 'Permission denied\|Host key verification' "/tmp/${TAG}_r${r}.log" 2>/dev/null; then
                started=1; echo "started rank $r on $h (attempt $attempt)"; break
            fi
            echo "rank $r: ssh to $h failed (attempt $attempt)" >&2
            sleep 2
        done
        if [ "$started" != 1 ]; then
            cat >&2 <<EOF
cannot reach $h over ssh from this node; start the ranks from the host that can:
  ssh $h 'cd \$HOME/dev/ds4-v41-pp/src && toolbox run -c vllm-gfx1151 -- env $DS4_ENV ./ds4 ${common[*]} --role worker --tp-rank $r --coordinator ${IPS[0]} $PORT' > /tmp/${TAG}_r${r}.log 2>&1 &
then rerun: WORKERS_STARTED=1 bash $0 $PROMPT $NTOK
EOF
            exit 3
        fi
    done
    sleep 6
else
    echo "WORKERS_STARTED=1: ranks 1-3 are expected to be running already"
fi

cd "$REPO" || exit 1
timeout "$TIMEOUT" toolbox run -c vllm-gfx1151 -- env $DS4_ENV \
    ./ds4 "${common[@]}" \
    --role coordinator --listen 0.0.0.0 "$PORT" \
    > "/tmp/${TAG}_r0.log" 2>&1
rc=$?
echo "coordinator exit=$rc"
sleep 2

if [ "$WORKERS_STARTED" = 1 ]; then
    for ((r=1; r<4; r++)); do
        ssh "${SSHOPT[@]}" "${HOSTS[$r]}" 'pkill -x ds4 2>/dev/null' >/dev/null 2>&1
    done
fi
cleanup

echo "== result"
grep -ao 'prefill: [0-9.]* t/s, generation: [0-9.]* t/s' "/tmp/${TAG}_r0.log" | tail -1
grep -av '^ds4' "/tmp/${TAG}_r0.log" | grep -av '^tp worker' | \
    grep -av '^tensor parallelism' | grep -av '^$' | md5sum | \
    sed 's/^/generated-text md5: /'
echo "logs: /tmp/${TAG}_r0.log .. /tmp/${TAG}_r3.log"
exit $rc
