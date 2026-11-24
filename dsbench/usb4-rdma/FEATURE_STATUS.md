# DS4 feature status under 4-node TP=4 over the USB4 RDMA ring

Verified 2026-11-24 on `feat/rocm-tp-full-stack`, four nodes
(192.168.1.101-104, switchless USB4 ring, RoCE v2 GID index 1), one 81 GiB
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
per host, ranks 0-3 on node1-4, `--tp-cpu-core 2`.

| feature | status | evidence | blocker |
|---|---|---|---|
| TP=4 USB4 RDMA ring | **verified** | 64.25 t/s prefill, 9.65 t/s decode, GTT 37.82 GiB/rank at ctx 2048; text md5 `a13ba035b8347fa92e602fd8f751ca91` == TP=1 | none |
| 1M context (compressed KV) | **verified** | `-c 1048576`: KV 13.79 GiB + 8.00 GiB buffers + 26.34 GiB resident = 48.13 GiB/rank, GTT 47.88 GiB, MemAvailable >= 42.85 GiB; 30,474-token story answered `Bob=34, Alice=52, Clara=71` | none |
| Vision (`--vision`) | **blocked** | all four ranks exit 1 with the rejection below; GTT stays 16.4 MiB | `ds4.c:74106`; `ds4.c:74034`; `ds4.c:6707`; `ds4.c:7458` |
| DSpark / MTP (`--dspark --mtp-model`) | **runs at N ranks on the batched verify gate; net loss at TP=4** | the N-rank verify gate is generalized (`tp_batch_rows` at world>=2), so a world=4 verify block costs ~320 ms/cycle instead of the per-layer bulk exchange's 742-870 ms and DSpark improves from 4.00-4.18 to 7.22-7.98 t/s. Against 9.35-9.45 t/s without DSpark that is still -15% to -24%, so `--dspark` stays off in `launch_tp4_best.sh` | performance: the 4-rank ring all-reduce costs 6 serialized round trips per layer gate plus ~1 software-HCA stall per block; the batched verifier's reduction order also changes the prose continuation |

## Vision

`./download_model.sh ds41f-vision` downloads `DeepSeek-V4.1-Flash-Vision.gguf`
(970,555,552 bytes, SHA-256
`cc283f032b3e8b8d78aeb5fccaa14e97b859b0c53aae3cd6bffa690ddf0e9e15`). It is the
V4.1 Flash encoder; its GGUF metadata is
`general.architecture=deepseek4-vision`,
`deepseek4-vision.checkpoint_variant=v4.1-flash`.

The running 0731 checkpoint is the plain V4 Flash model. Its metadata has no
`deepseek4.checkpoint_variant=vision-exp` marker, so
`g_ds4_flash_vision_exp` stays false (`ds4.c:6707-6726`) and `--vision` is
rejected before any encoder is opened (`ds4.c:74106-74112`):

```
ds4: --vision requires GLM-5.3, Qwen3.8-Flash-Next or the pinned DeepSeek V4
Flash Vision-Exp or V4.1 Flash model
```

Even with the gate passed, a V4 model would reject this sidecar at the
revision/variant check (`ds4.c:7458`, encoder variant must be `vision-exp` or
`v4.1-flash` matching the target family), and the V4.1 vision path itself is
hard-gated to Metal on ROCm at `ds4.c:74034` ("ds4: V4.1 vision requires
Metal"). The 0731 weights also lack the visual router biases the V4.1 path
requires.

This is a checkpoint/backend gap, not a TP=4-specific one: the correct pairing
for this encoder is `ds41f-q2` (341 GiB) on Metal, and the correct pairing for
0731 is the separate Vision-Exp language checkpoint plus its own encoder, which
is not the 81 GiB Q2 file.

## DSpark / MTP

Matching weights exist and download cleanly:
`./download_model.sh ds4f-dspark` ->
`gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` (5,989,114,272 bytes,
`stages=3 block=5 markov_rank=256 tensors=81 missing=0 invalid=0`). Its
`general.architecture` is `deepseek4-dspark`, so the 0731 target selects the
DeepSeek-4 DSpark path, not the `ds41_*` V4.1 path.

### Why N ranks used to fail

The protocol was written for one leader and one worker:

* `ds4_session_tp_spec_cycle` rejected every rank except 1 (`rank != 1`), so at
  world=4 ranks 2 and 3 aborted on the first verify with
  `tp worker verify: tp: spec cycle outside worker mode`. They closed their
  control sockets; the leader then failed `tp: V4.1 verification or prefix
  commit failed` / `tp: verify commit send failed`, and rank 1 read the closed
  socket as `bad verify-commit frame (type 0 bytes 0)`.
* The two-rank *lockstep* drafter (`e->tp.dspark_split`, each rank keeps half of
  every drafter expert) is CUDA-only: built under
  `#if !__APPLE__ && !defined(DS4_ROCM_BUILD)`, and `tp_dspark_split_eligible`
  additionally requires `DS4_BACKEND_CUDA` and `ds4_gpu_device_is_spark`
  (`ds4.c:73593`). It is not required for N ranks: every rank already holds the
  whole 5.98 GiB drafter, and at world>2 the target vocabulary is replicated
  (`vocab_split` is `world == 2` only, `ds4.c:76274`). The leader can therefore
  draft locally while every worker mirrors only the target verification.

### What changed

`ds4.c`:

* `ds4_session_tp_spec_cycle` (`ds4.c:85455`) - the worker-mode check is now
  `rank <= 0`, so ranks 1..N-1 all mirror the leader's block. The commit frame
  already fans out to every peer and `ds4_tp_recv_verify_commit` already reads
  each worker's own socket, so no frame change was needed.
* `ds4_session_tp_dspark_draft` (`ds4.c:85669`) - accepts any worker rank; the
  V4.1 vocab-split drafter keeps its rank-1-only half exchange, world>2
  replicates the full head.
* `metal_graph_eval_dspark_stage_chain` (`ds4.c:36172`) and the legacy MTP
  drafter (`ds4.c:37473`) - the leader-local drafter disarms TP
  (`g->tp_world = 0`); it must also suspend expert sharding on *every* rank
  count, not just world=2, or the ROCm MoE launcher filters the drafter's 6
  experts to rank 0's contiguous 1/4 slice
  (`rocm/ds4_rocm_moe_launch.cuh:610`). Changed `== 2` to `>= 2`.
* `dspark_verify_head_split_wanted` (`ds4.c:84342`) - the verify-head split is
  a two-way vocabulary halving (`head_row0 = rank * V/2`), so it is now gated
  to `world == 2`. It was already dead at N=4 because the RDMA command mailbox
  is only opened at world=2, but the guard makes the constraint explicit.

The CUDA lockstep drafter split remains compiled out on ROCm. Porting it would
mean a ROCm `ds4_gpu_build_dspark_tp_split` plus the half-expert MMQ kernels
(`ds4_cuda.cu:5271`, `g_dspark_tp_split_active`); it would only parallelize the
*proposal*, which is 0.5-0.6 s of a 3-4 s generation here, so it is not on the
N-rank correctness path and was left alone.

### The N-rank verify gate (this change)

A verify block is `n_tokens` rows (the seed plus the drafted tokens), and every
layer's routed-MoE partial is a full-row float32 matrix. At world=2 that block
always ran through the batched verify gate: `g->tp_batch_rows = n_tokens`
selects one `DS4_TP_BATCH_MAX_ROWS`-row slab exchange per layer, the ring
reduces it across the ranks, and the consumer adds `own + sum(others)` on the
device. At world>2 `tp_batch_rows` was forced to 0 and the block fell back to
the per-layer bulk exchange (`ds4_gpu_tp_big_gate_encode`: D2H into pinned
staging, ring, H2D) - the ~870 ms/cycle path measured on `42809c2`.

Audit of every reader/setter at `42809c2`:

| site | what it does | why it excluded N ranks |
|---|---|---|
| `ds4.c:39470` | sets `tp_batch_rows` | literal `tp_world == 2`; the flag selects the batch path for the whole block |
| `ds4.c:33776` | gates `tp_split_batch_moe` | a second literal `tp_world == 2` in the FFN consumer, so setting the flag alone would not route the exchange |
| `ds4.c:33515` | gates `tp_shared_slice` | a genuine two-rank fold: rank 0 writes the block, rank 1 adds the peer's copy in rank order |
| `ds4.c:31380`, `35729` | verify-head / drafter head split | two-way vocabulary/head halving on CUDA/Metal-only kernels; dead on ROCm |
| `ds4.c:31427` | attention row split | two-half swap, needs a gather at N ranks and the ROCm row-range attention kernels |
| `ds4_tp.c:3468` | `ds4_tp_batch_gate_exchange` | already N-rank: header fan-out, payload through `tp_ring_exchange`, whose world>2 branch all-reduces and returns `total - own` |
| `ds4_tp.c:3374` | `ds4_tp_batch_block_begin_gates` | shape negotiation already fans out; the posted-receive window is world=2 only but is skipped for the ring |
| `ds4_tp.c:1233` | slab sizing | `n_layer * BATCH_MAX_ROWS * vec_bytes` per rank, world-independent |

So the restriction was an admission predicate, not a layout, socket/frame,
view/slot-table or exchange-shape assumption for the FFN gate. The payload is a
whole-row all-reduce (`rows * vec_bytes`, any row count, no single-token
buffer), and the only genuinely two-rank machinery is the head split, the
shared-slice fold and the attention row split - none of which is on the ROCm
verify path.

`ds4.c` changes:

* `metal_graph_verify_suffix_tops_impl` (`ds4.c:39473`) - `tp_batch_rows` is
  set for `tp_world >= 2` instead of `== 2`.
* `metal_graph_encode_layer_ffn_batch` (`ds4.c:33781`) - `tp_split_batch_moe`
  accepts `tp_world >= 2` instead of `== 2`.
* `tp_shared_slice` (`ds4.c:33515`) - pinned to `tp_world == 2` so the
  two-rank shared fold cannot be picked up at N ranks.

The exchange itself is the existing collective, unmodified: the slab payload
goes through `ds4_tp_batch_gate_exchange`, whose ring path all-reduces
`rows * vec_bytes` over `ds4_tp_ring_all_reduce` (USB4 RDMA, TCP-ring
fallback). world=2 is unchanged by construction (`>= 2` and `== 2` are the same
set there, and the added `world == 2` term is always true).

### Measurements (after the gate generalization)

Same binary (md5 `225468b0670bb93bc2e045f6b64a14e6`) on all four nodes,
`-c 2048 --temp 0 -n 64`, USB4 RDMA ring, `--tp-cpu-core 2`, greedy. Pairs are
adjacent runs on the same four nodes; prefill 40-62 t/s in this session, decode
is the stable metric.

| prompt | no DSpark | `--dspark` | delta | accept | proposed / accepted | verify ms/cycle |
|---|---:|---:|---:|---:|---:|---:|
| coding | 9.45 (9.43/9.45/9.46) | **7.65** (7.10/8.62/7.22) | **-19%** | 82.50% | 40 / 33 | 320 (352/268/339) |
| prose | 9.35 (9.13/9.51/9.42) | **7.99** (7.44/8.54/7.98) | **-15%** | 90.00% | 10 / 9 | 61 (49/76/59) |
| coding, seed batch off | 9.45 | 6.97 | -26% | 69.70% | 33 / 23 | 155 |
| prose, seed batch off | 9.35 | 8.52 | -9% | 76.47% | 17 / 13 | 33 |

Before this change the same pairs were 4.00/4.18 t/s with verify ~742 ms/cycle
(`verify=11871 ms` over 16 cycles). The prose verify number is low per cycle
because the DSpark scheduler declines 32 of 41 cycles (no draft); per verify
block it is ~240 ms.

`DS4_TP_XCHG_STATS=1` on the after run: 1333 payload exchanges, 50.0 MB,
mean 3.2 ms (p50 <= 256 us, p99 <= 131 ms, max 322 ms) - the same payloads that
used to ride the bulk gate. `DS4_TP_RING_VERBOSE=1` shows 20 `STALL` step
reports in a 19-cycle run (post p50 0.1 ms / p90 122 ms, marker p50 74 ms): a
software-HCA wake-up tail, not bandwidth.

### Greedy parity (after)

* Non-speculative TP=4: coding `4a10b554b2ea29a6513b6739e61562b2`, prose
  `e728672ca2b65a182f580d227838d889`, canonical 16-token
  `a13ba035b8347fa92e602fd8f751ca91`; TP=1 and TP=2 reproduce the canonical md5
  and decode 16.91 / 10.15 t/s.
* DSpark coding (seed batches on **and** off) is byte-identical to
  non-speculative: `4a10b554b2ea29a6513b6739e61562b2`.
* DSpark canonical 16-token with the default seed batch is also byte-identical:
  `a13ba035b8347fa92e602fd8f751ca91`.
* DSpark prose is **not** byte-identical: both seed modes emit
  `de52e3b25a178c7b1dfeb3a6f3608d93` against the non-speculative
  `e728672ca2b65a182f580d227838d889`. This is the batched verifier's
  floating-point reduction order, not an N-rank defect: TP=2 DSpark emits the
  same `de52e3b2...` on this prompt (measured with the same binary), so TP=2
  and TP=4 DSpark now agree bitwise. `docs/SPECULATIVE_DECODING.md` already
  documents that accepted tokens keep the batched verifier's order and long
  greedy continuations need not be byte-identical.

### Remaining gap

Even with the gate, a world=4 verify block is ~320 ms/cycle on coding against
~146 ms at world=2: the ring all-reduce is `2*(world-1)` = 6 serialized RDMA
round trips per layer gate (2 at world=2), and one software-HCA wake-up of
~100 ms per block survives. TP=2 DSpark pays 140 ms/cycle on the same block and
wins only because its cycle still emits ~3.7 tokens; at TP=4 the same block
plus a 95 ms target step needs to fall below ~200 ms to beat 9.5 t/s. That is
ring-latency work (fewer collectives per block or a shorter-stall transport),
not more drafter work.

### Decision and toggle

`--dspark` stays **off by default** in `dsbench/usb4-rdma/launch_tp4_best.sh`:
the N-rank gate makes TP=4 DSpark 15-24% faster than it was but it is still
15-24% slower than no DSpark, and prose is not byte-identical. TP=2 coding
remains the one net win (10.04 -> 14.45 t/s with this binary).

```
DSPARK=1 CTX=2048 bash dsbench/usb4-rdma/launch_tp4_best.sh /tmp/prompt_now.txt 16
```

`DSPARK=1` adds `--dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf`
and `DS4_DSPARK_STATS=1`; `DS4_DSPARK_SEED_BATCH=0` turns the fused seed batch
off.
