# USB4/Thunderbolt RDMA ring collectives for DS4 tensor parallelism

## Problem

DS4's tensor parallelism is two ranks, `leader` + one `worker`, and every
collective is a single pairwise exchange of half the routed experts. That is a
fine fit for two Macs. It is not a fit for a four-node switchless Thunderbolt
ring, for two independent reasons:

1. **No switch.** Each node has exactly two USB4 RDMA HCAs, one per physical
   neighbour, and node *r* can only reach `(r+1) % world` and
   `(r-1+world) % world`. There is no fabric, so no transport that assumes
   any-to-any reachability can work.
2. **NCCL/RCCL cannot drive this topology.** Both build their topology from
   PCI paths and merge ranks that share one, which is exactly wrong here: two
   HCAs in one host look like one device, and the ring order
   `0->1->2->3->0` is not an ordering over the paths they infer. The result is
   a communicator that either refuses to initialise or routes over a link that
   does not exist.

## Architecture

`ds4_tp_ring.c` adds the collective seam: rank *r* only exchanges with its two
neighbours, and an all-reduce is a ring all-gather followed by a local
reduction, so the number of ring steps is `world-1` regardless of message size.

Two backends implement one contract:

* **USB4** (`misc/usb4/usb4_rdma_ring`, vendored): one-sided
  `IBV_WR_RDMA_WRITE` only. The payload is followed by a trailing 64-bit
  sequence marker inside the same logical message, and the receiver busy-polls
  that marker in its own memory. **No receive WQEs are ever posted.**
* **TCP**: the same ring algorithm over the management network, so the contract
  is testable on a cluster with no verbs device and a missing device degrades
  instead of failing.

### Why one-sided writes and not SEND/RECV

Upstream `thunderbolt-ibverbs` has an open driver bug on RNR-NAKed sends
(PR #90). Any design that consumes receive WQEs hits sporadic
`min_rnr_timer x retry` stalls, 10.24 ms each at the usual setting. A plain
`RDMA_WRITE` consumes no receive WQE, so RNR is **structurally impossible**
rather than merely unlikely. The responder still needs a queue pair, because RC
requires a peer QPN, but it never posts a receive.

### Backend selection and fallback

Each rank probes locally for an active device exposing a non-zero IPv4-mapped
GID. The probes are exchanged over the ring **before any verbs resource is
opened**, and unless every rank is ready all of them take the TCP ring, with a
one-line warning. A probe that succeeds everywhere but then fails to open is a
hard error rather than a silent split, because asymmetric backends would
deadlock. Devices are discovered at runtime; control addresses come from
`--tp-ring-ip`. Nothing about the cluster is compiled in.

### Size generality

The collectives take an arbitrary runtime element count (0 and 1 included, no
world-size multiple, no 4096-byte alignment requirement) and an explicit dtype,
and they reduce in the dtype they were handed. Prefill batches, variable-length
speculative verification and vision embeddings pass their real counts straight
through; the USB4 backend chunks internally. F16/BF16 sums round through
float32 per ring step rather than a float64 accumulator, so they can differ in
the last bit from a sequential sum.

## Collective microbenchmark, 4 nodes

All-reduce latency, identical on all four ranks, 20 iterations per size after
one unmeasured round. Real hardware, real ring.

| Size | USB4 RDMA | TCP ring (5GbE) |
|---|---|---|
| 4 KB | 0.104 ms / 0.32 Gbps | 0.165 ms / 0.20 Gbps |
| 16 KB | 0.135 ms / 0.97 Gbps | 0.330 ms / 0.40 Gbps |
| 64 KB | 0.226 ms / 2.32 Gbps | 0.570 ms / 0.93 Gbps |
| 256 KB | 0.551 ms / 3.81 Gbps | 1.63 ms / 1.29 Gbps |
| 1 MB | 1.898 ms / 4.42 Gbps | 5.72 ms / 1.47 Gbps |

The USB4 column agrees with the earlier standalone ring probe (64 KB
0.227 ms/2.31 Gbps, 256 KB 0.552 ms/3.80 Gbps, 1 MB 2.088 ms/4.02 Gbps), which
independently validates the adapter.

### The 67 ms idle-wake stall is our own ACK timeout

The local ACK timeout is `4.096 us * 2^timeout`, so the IB default `timeout=14`
is **67 ms**. A cold path pays it as a single stall whenever two collectives are
separated by real work. 256 KB all-reduce with 500 ms between calls, ten rounds:

| `USB4_RING_QP_TIMEOUT` | Runs passing | Idle-separated max |
|---|---|---|
| 14 (IB default) | 3/3 | 74.2 / 74.2 / 74.8 ms |
| 10 | 1/3 | 5.2 ms (two init failures) |
| 8 (default here) | 3/3 | 3.7 / 2.3 / 2.9 ms |

Back-to-back collectives are unaffected at every setting (~0.63 ms max), and
with a 5 ms gap — the scale a layer actually exhibits — **no setting stalls**.
So the 67 ms hit requires a genuinely cold path, i.e. init or a long idle, not
the between-layer spacing of decode.

At init the short timeout is occasionally too short: one isolated run at
`timeout=8` failed its init warmup (`TIMEOUT waiting for marker`). The module's
init warmup still wants a retry. This is reported rather than hidden. No
keepalive machinery was added, because measurement did not show it is needed at
layer-scale gaps.

## DS4 inference: NOT MEASURED, and why

> **Update 2026-11-23:** the backend gate described in this section was lifted
> (`fdf2222`, `792508c`) and 2-node TP now runs on this cluster. The transport
> comparison and its numbers are in **World=2: the degenerate ring as the TP=2
> payload plane** at the end of this document. The history below is kept because
> it explains why the first, transport-only change stopped where it did.

A 2-node TCP-vs-RDMA inference comparison was attempted and **could not run**:
`ds4_tp_validate_engine_options` rejects network tensor parallelism on any
backend that is not Metal or CUDA (`ds4_tp.c:778-785`):

```c
bool supported_backend = opt->backend == DS4_BACKEND_METAL;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
supported_backend |= opt->backend == DS4_BACKEND_CUDA;
#endif
if (!supported_backend) {
    tp_set_err(err, errlen, "network tensor parallelism requires Metal or supported CUDA models");
```

This cluster builds `make strix-halo` (`DS4_ROCM_BUILD`), so the CUDA branch is
compiled out and the only accepted backend is Metal, which is not present. The
run exits before loading weights:

```
ds4-bench: network tensor parallelism requires Metal or supported CUDA models
```

So **there are no DS4 tok/s or TTFT numbers in this document**, for TP or
otherwise. The weights themselves are fine (V4-Flash Q2, 86.7 GB, present on
both nodes); the *backend* does not participate in TP. Adding a non-Metal,
non-CUDA TP backend is a prerequisite for measuring tensor parallelism on this
hardware at all, and it is a larger change than the transport.

Note also that a 4–8 GB model does not exist for DS4: the loaders validate
exact shapes against compile-time `DS4_N_EXPERT`/`DS4_N_LAYER`, so only the
project's own checkpoints can load, and the smallest full one is ~81 GB.

## N-rank sharding is still 2-rank

The transport is general. The sharding around it is not, which is why
`--tp-world-size` above 2 is refused with a reason instead of running a wrong
split:

* `ds4.h:104-108` — `ds4_tp_role` has `NONE/LEADER/WORKER`, no rank; `ds4_tp.c:2437`
  derives `rank = role == LEADER ? 0 : 1`.
* `ds4_tp.c:2458-2466` — the leader `accept()`s exactly one worker.
* `ds4_cuda.cu:4854` — `if ((world != 1u && world != 2u) || rank >= world) return 0;`
* `ds4_cuda.cu:5019-5021` — `cuda_build_derived_artifacts(..., rank, 2)`,
  a literal 2.
* `ds4_cuda.cu:474-477`, `5023-5028` — routed experts are split as **halves
  along the intermediate dimension** (`mid/2`, `K/64`), not per-expert
  ownership, so "each rank keeps half of every expert".
* `ds4.c:24253`, `33382`, `26809`, `42294` — attention, FFN and shared-expert
  splits all gated on `g->tp_world == 2`.
* `ds4.c:42343`, `43250`, `43285` — `g->tp_rank == (il & 1u)`: shared-expert
  ownership alternates on layer parity, which has no N-rank meaning.
* `ds4_tp.h:126-138` — the slab holds one peer slot per gate; there is no
  per-peer dimension.
* `ds4_tp.c:3724` — the logits split is a fixed two-way halving.

318 `tp_world`/`tp_rank` references exist; roughly 40 are explicit
`tp_world == 2` graph-shape decisions. That is why this PR lands the transport
and stops.

## Reproduction

```sh
# Build both ways; only the second needs libibverbs.
make strix-halo -j32
make strix-halo -j32 USE_USB4_RDMA=1

# Ring collective tests, no weights needed.
make test-tp-ring                                   # four ranks on loopback
./tests/test_tp_ring <rank> 4 <ip0,ip1,ip2,ip3> <port>
# A/B the ACK timeout:
USB4_RING_QP_TIMEOUT=14 TEST_RING_GAP_MS=500 ./tests/test_tp_ring 0 4 ... <port>

# Existing 2-rank transport tests.
make tests/test_tp_rdma tests/test_tp_tcp tests/test_tp_link
```

Options: `--transport usb4-rdma`, `--tp-world-size N`, `--tp-ring-ip ADDR`
(repeat per rank, in rank order), `--tp-cpu-core N`. Defaults are unchanged;
`world_size` defaults to 2 and every new field is inert unless set.

## NOT verified

* **Any DS4 inference metric** — this document's original numbers predate the
  ROCm TP backend; the 2-node numbers are in the world=2 section below.
* **N-rank TP end to end** — the sharding is still 2-rank.
* **`--transport usb4-rdma` above world=2** — allowed and measured at world=2
  only; larger worlds remain refused by the sharding gate.
* **Full `make strix-halo` on the other nodes** — the ring seam builds on all
  four nodes (0 warnings, both modes); the full tree build was done on one host
  only, because that is where the checkout lives.
* **Mixed `USE_USB4_RDMA` settings across ranks** — unsupported; every rank
  must use the same build setting.
* **Long-run stability** — the microbenchmarks are 20 iterations per size and
  10 rounds per gap setting, not a soak test, and the TP=2 run is 16 generated
  tokens.

## World=2: the degenerate ring as the TP=2 payload plane

`--transport usb4-rdma` is allowed at world=2, which is the only world the
sharding supports (larger worlds are still refused for the reasons above). Rank
0's right and left neighbour are both rank 1, so both ring directions resolve
to the same HCA. `usb4_rdma_ring.c` opens that device once and shares the
`ibv_context` and PD, then creates two QPs — `out` for the writes to the peer
and `in` for the peer's writes — each with its own CQ. The discovery path
already accepts one XDomain playing both roles. The TCP control ring, the gate
headers and the command mailbox stay on the 5GbE sockets; only the bulk gate
payload moves over the ring, which is the split the driver supports.

Isolated two-node (`node1`+`node2`) test, 512 KiB chunk; `correct=YES` on both
ranks at every size, including the unaligned 1,000,003-element F32 case:

| Size | all_reduce |
|---|---|
| 4 KB | 0.042 ms / 0.79 Gbps |
| 16 KB | 0.059 ms / 2.21 Gbps |
| 64 KB | 0.119 ms / 4.42 Gbps |
| 256 KB | 0.332 ms / 6.32 Gbps |
| 1 MB | 1.125 ms / 7.46 Gbps |

DS4 TP=2, `ctx 2048`, greedy, 16 tokens, `--tp-cpu-core 8`:

| | TP=1 | TP=2 TCP | TP=2 USB4 RDMA |
|---|---|---|---|
| prefill t/s | 92.12 | 88.59 | 89.31 |
| decode t/s | 16.93 | 10.17 | 10.21 |
| ms/token | 59.1 | 98.3 | 97.9 |
| whole gate exchange | — | 129.8 us | 104.7 us |
| payload per gate | — | ~110 us | 60.5 us |
| resident model/rank | 80.76 GiB | 44.48 GiB | 44.48 GiB |
| GTT peak/rank | 87.88 GiB | 55.97 GiB | 55.97 GiB |

The greedy 16-token text is byte-identical across all three configurations.

The ~900 us/layer implied by the 10.17 t/s vs 16.93 t/s delta is **not** the
exchange: the measured whole gate exchange is 130 us/layer over TCP and 105
us/layer over USB4, so ~780 us/layer sits outside the exchange (per-rank
compute / MoE efficiency at the 50/50 expert split, not the link). USB4 RDMA
removes ~25 us/layer, about 1 ms/token — real, but far too small to close the
gap. Two practical requirements came out of this: the RDMA busy-poll thread
must be pinned (`--tp-cpu-core`), because otherwise scheduling gaps stretch the
tail to ~10 ms and decode falls to 6.43 t/s; and 512 KiB is the largest chunk
that fits the container's 8 MiB `RLIMIT_MEMLOCK` (4 MiB fails `ibv_reg_mr` with
`ENOMEM`).

## Two cluster-facing caveats

* `unique_id` in sysfs is per-port/router, **not** a per-node identity;
  `device_name` is the stable identity.
* The vendored module differs from upstream in exactly two places: the
  `#ifndef _GNU_SOURCE` guard and the QP ACK timeout default. Both are
  documented at their call sites and should be upstreamed.
