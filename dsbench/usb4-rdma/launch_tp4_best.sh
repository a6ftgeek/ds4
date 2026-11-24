#!/bin/bash
# DS4 4-node TP=4 over the switchless USB4 RDMA ring -- service launcher.
#
# Starts rank 0 (coordinator, ds4-server, OpenAI-compatible HTTP API) on this
# node and ranks 1-3 (workers, ./ds4) on node2-4, then stays in the foreground
# so a systemd unit can supervise it.
#
# Model: gguf/DeepSeek-V4.1-Flash-Q4.gguf (V4.1 Flash Q4), ctx 1048576,
#        --vision gguf/DeepSeek-V4.1-Flash-Vision.gguf
# Ring : --transport usb4-rdma --tp-world-size 4, rank=node1..node4,
#        --tp-ring-ip 192.168.1.101..104, --tp-cpu-core 2.
#
# Inter-node fan-out uses passwordless ssh node_one@node1 -> node_two@node2,
# node_three@node3, node_four@node4 (see dsbench/usb4-rdma/ROCM_TP_REPORT.md).
# `ssh nodeN` from node1 is configured in ~/.ssh/config with the right User.
#
# usage: bash launch_tp4_best.sh {start|stop|status|restart}
#
# env: CTX (1048576)      NTOK (256)         MODEL (V4.1 Q4 path)
#      BIND (0.0.0.0)     PORT (8000)        RING_PORT (19730)
#      VISION (1)         VIS_MODEL (V4.1 vision GGUF)
#      TP_CORE (2)        QP_TIMEOUT (14)    ENVV (extra in-container env)
#      LOGDIR (/tmp/ds4-tp4)                 LOG_MODE (file|journal)
#      START_TIMEOUT (900 seconds)
#
# LOG_MODE=journal (set by ds4-tp4.service) leaves the coordinator's stdout on
# the unit's stdout so journald owns it; otherwise it is redirected to
# $LOGDIR/r0.log and the workers to /tmp/ds4-tp4_rN.log on their own nodes.

set -u

ACTION=${1:-start}
REPO=${REPO:-$HOME/dev/ds4-v41-pp/src}
CTX=${CTX:-1048576}
NTOK=${NTOK:-256}
BIND=${BIND:-0.0.0.0}
PORT=${PORT:-8000}
RING_PORT=${RING_PORT:-19730}
VISION=${VISION:-1}
TP_CORE=${TP_CORE:-2}
QP_TIMEOUT=${QP_TIMEOUT:-14}
ENVV=${ENVV:-}
LOGDIR=${LOGDIR:-/tmp/ds4-tp4}
LOG_MODE=${LOG_MODE:-file}
START_TIMEOUT=${START_TIMEOUT:-900}

MODEL=${MODEL:-gguf/DeepSeek-V4.1-Flash-Q4.gguf}
VIS_MODEL=${VIS_MODEL:-gguf/DeepSeek-V4.1-Flash-Vision.gguf}
IPS=(192.168.1.101 192.168.1.102 192.168.1.103 192.168.1.104)
HOSTS=(node1 node2 node3 node4)
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30)

ring=()
for ip in "${IPS[@]}"; do ring+=(--tp-ring-ip "$ip"); done

vis=()
if [ "$VISION" = 1 ]; then vis=(--vision "$VIS_MODEL"); fi

container() {  # container <rank> <binary> <extra args...>
    local r=$1 bin=$2; shift 2
    toolbox run -c vllm-gfx1151 -- env USB4_RING_QP_TIMEOUT="$QP_TIMEOUT" ${ENVV:-} \
        "$bin" -m "$MODEL" -c "$CTX" -n "$NTOK" --rocm \
        --tensor-parallel --transport usb4-rdma --tp-world-size 4 "${ring[@]}" \
        --tp-cpu-core "$TP_CORE" "$@" ${vis[@]+"${vis[@]}"}
}

worker_ok() {  # worker_ok <host>
    [ "$("${SSH[@]}" "$1" 'pgrep -x ds4 2>/dev/null | wc -l' 2>/dev/null)" -ge 1 ]
}

stop_ranks() {
    # Coordinator first: workers then see the control socket close and exit
    # instead of waiting inside a gate. Exact process names only.
    pkill -TERM -x ds4-server 2>/dev/null
    for h in node2 node3 node4; do
        "${SSH[@]}" "$h" 'pkill -TERM -x ds4 2>/dev/null' >/dev/null 2>&1
    done
    local i
    for i in $(seq 1 30); do
        local live
        live=$(pgrep -x ds4-server 2>/dev/null | wc -l)
        for h in node2 node3 node4; do
            live=$((live + $("${SSH[@]}" "$h" 'pgrep -x ds4 2>/dev/null | wc -l' 2>/dev/null)))
        done
        [ "$live" -eq 0 ] && return 0
        sleep 1
    done
    pkill -KILL -x ds4-server 2>/dev/null
    for h in node2 node3 node4; do
        "${SSH[@]}" "$h" 'pkill -KILL -x ds4 2>/dev/null' >/dev/null 2>&1
    done
    return 0
}

start_workers() {
    local r h
    for ((r = 1; r < 4; r++)); do
        h=${HOSTS[$r]}
        setsid "${SSH[@]}" "$h" \
            "cd \$HOME/dev/ds4-v41-pp/src && setsid nohup env USB4_RING_QP_TIMEOUT=$QP_TIMEOUT ${ENVV:-} toolbox run -c vllm-gfx1151 -- env USB4_RING_QP_TIMEOUT=$QP_TIMEOUT ${ENVV:-} ./ds4 -m $MODEL -c $CTX -n $NTOK --rocm --tensor-parallel --transport usb4-rdma --tp-world-size 4 ${ring[*]} --tp-cpu-core $TP_CORE --role worker --tp-rank $r --coordinator ${IPS[0]} $RING_PORT > /tmp/ds4-tp4_r${r}.log 2>&1 < /dev/null &" \
            >/dev/null 2>&1 < /dev/null &
        echo "ds4-tp4: starting rank $r on $h"
    done
    for ((r = 1; r < 4; r++)); do
        h=${HOSTS[$r]}
        local deadline=$((SECONDS + 60))
        until worker_ok "$h"; do
            if [ "$SECONDS" -ge "$deadline" ]; then
                echo "ds4-tp4: rank $r on $h did not start (see /tmp/ds4-tp4_r${r}.log)" >&2
                return 1
            fi
            sleep 1
        done
        echo "ds4-tp4: rank $r on $h is up"
    done
    sleep 4    # let the workers finish their ring handshake loop
    return 0
}

do_start() {
    if [ ! -x "$REPO/ds4-server" ]; then
        echo "ds4-tp4: missing $REPO/ds4-server (build with USE_USB4_RDMA=1)" >&2
        return 1
    fi
    stop_ranks
    start_workers || return 1

    mkdir -p "$LOGDIR"
    local log=/dev/null
    [ "$LOG_MODE" = file ] && log="$LOGDIR/r0.log"

    # Coordinator in the background so this process can wait for readiness,
    # trap TERM, and then wait() -- that keeps a systemd unit honest.
    if [ "$LOG_MODE" = journal ]; then
        # The engine runs inside the rootless podman container, so its stdout
        # would be attributed to the container scope in journald. Pipe it
        # through systemd-cat so Cockpit's ds4-tp4 unit log shows it too.
        ( cd "$REPO" && container 0 ./ds4-server --role coordinator \
            --listen 0.0.0.0 "$RING_PORT" --host "$BIND" --port "$PORT" 2>&1 |
            systemd-cat -t ds4-tp4 ) &
    else
        ( cd "$REPO" && container 0 ./ds4-server --role coordinator \
            --listen 0.0.0.0 "$RING_PORT" --host "$BIND" --port "$PORT" ) \
            >"$log" 2>&1 &
    fi
    local coord=$!

    local deadline=$((SECONDS + START_TIMEOUT))
    until curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
        if ! kill -0 "$coord" 2>/dev/null; then
            echo "ds4-tp4: coordinator exited before becoming ready" >&2
            stop_ranks
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "ds4-tp4: coordinator not ready after ${START_TIMEOUT}s" >&2
            stop_ranks
            return 1
        fi
        sleep 2
    done
    echo "ds4-tp4: ready at http://$BIND:$PORT/v1 (rank 0 pid $coord)"

    trap 'echo "ds4-tp4: stopping"; stop_ranks; exit 0' TERM INT
    wait "$coord"
    local rc=$?
    echo "ds4-tp4: coordinator exited rc=$rc"
    stop_ranks
    # SIGTERM/SIGINT means an explicit stop (systemctl stop/restart), not a
    # crash; do not turn it into Restart=on-failure churn.
    case "$rc" in 143|130) exit 0 ;; esac
    return "$rc"
}

do_status() {
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    echo "http 127.0.0.1:$PORT -> ${code:-000}"
    echo "listen: $(ss -ltn 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $4}' | head -1)"
    echo "local ds4-server: $(pgrep -x ds4-server 2>/dev/null | wc -l)"
    for r in 1 2 3; do
        echo "rank $r on ${HOSTS[$r]}: $("${SSH[@]}" "${HOSTS[$r]}" 'pgrep -x ds4 2>/dev/null | wc -l' 2>/dev/null)"
    done
    [ "$code" = 200 ] || return 1
}

case "$ACTION" in
    start)   do_start ;;
    stop)    stop_ranks; echo "ds4-tp4: stopped" ;;
    status)  do_status ;;
    restart) stop_ranks; exec "$0" start ;;
    *) echo "usage: $0 {start|stop|status|restart}" >&2; exit 2 ;;
esac
