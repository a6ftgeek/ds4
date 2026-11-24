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
| DSpark / MTP (`--dspark --mtp-model`) | **runs at N ranks, net loss at TP=4** | world=4: 4.00-4.18 t/s vs 9.15-9.23 without (coding/prose); output text identical to non-speculative TP=4 on both 64-token prompts. The canonical 16-token parity md5 matches only with `DS4_DSPARK_SEED_BATCH=0`. `--dspark` therefore stays off in `launch_tp4_best.sh` | performance, not correctness; fused seed batch changes greedy output (documented FP reduction-order effect) |

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

### Measurements

Same binary (md5 `a7730c130f13a485861080f8b5c6daf7`) on all four nodes,
`-c 2048 --temp 0 -n 64`, `--transport usb4-rdma`, USB4 RDMA ring, greedy.
Prefill in this session read 46-61 t/s; the pairs below are adjacent runs on
the same four nodes, so the decode comparison is direct. Two prompts:
`prompt_coding.txt` (thread-safe TTL LRU cache implementation) and
`prompt_prose.txt` (train journey narration).

| TP | prompt | no DSpark | `--dspark` | delta | accept_rate | proposed/accepted |
|---:|---|---:|---:|---:|---:|---:|
| 4 | coding | 9.15 | **4.00** | **-56.3%** | 80.95% | 42 / 34 |
| 4 | prose | 9.23 | **4.18** | **-54.7%** | 96.00% | 25 / 24 |
| 4 | coding, seed batch off | 9.15 | 4.00 | -56.3% | 74.29% | 35 / 26 |
| 4 | prose, seed batch off | 9.23 | 6.51 | -29.5% | 82.35% | 17 / 14 |
| 2 | coding | 10.04 | 14.53 | +44.7% | 92.50% | 40 / 37 |
| 2 | prose | 10.04 | 9.53 | -5.1% | 58.33% | 12 / 7 |
| 1 | coding | 16.20 | 16.35 | +0.9% | 79.07% | 43 / 34 |
| 1 | prose | 16.20 | 13.75 | -15.1% | 84.62% | 13 / 11 |

The N-rank verifier is the bottleneck. At world=2 the verify block uses the
two-rank batched gate path (`g->tp_batch_rows == n_tokens`), costing about
140 ms per cycle; at world=4 `tp_batch_rows` is 0 (`ds4.c:39470`) and the same
block falls back to the per-layer bulk exchange, costing about 870 ms per cycle
(`verify=13869 ms` over 16 cycles on coding). That is roughly 3x an ordinary
decode per verified row, so a block of 2.6 accepted rows cannot repay its cost.
This is the concrete follow-up for a TP=4 win: an N-rank verify-block gate
batch, not more drafter work.

### Greedy parity

* Of 64 tokens, TP=4 DSpark output is byte-identical to TP=4 without DSpark on
  both prompts: md5 `4a10b554b2ea29a6513b6739e61562b2` (coding) and
  `e728672ca2b65a182f580d227838d889` (prose), with seed batches on and off.
  TP=1/TP=2 match their own non-speculative md5 too.
* Canonical 16-token check (`/tmp/prompt_now.txt`): TP=4 without DSpark gives
  `a13ba035b8347fa92e602fd8f751ca91`. With the default `--dspark` the text
  differs by one word (`... a long, reflective passage ...` vs `... essay ...`,
  md5 `7045543659e276f5d3ed39e717e2f953`). The divergence is inside a fused
  **seed batch**, where the seed token's own logits come from the batched
  verifier instead of ordinary decode; `DS4_DSPARK_SEED_BATCH=0` reproduces
  `a13ba035...` exactly. `docs/SPECULATIVE_DECODING.md` already documents that
  accepted tokens keep the batched verifier's floating-point reduction order
  and long greedy continuations need not be byte-identical.
* Note the TP=4 non-speculative baseline itself is not bit-identical to TP=1 on
  every prompt (prose: `e728672c...` vs `de52e3b2...`), because N-way expert
  ownership changes the sum order. The canonical 16-token md5 is the check that
  does match.

### Decision and toggle

`--dspark` stays **off by default** in `dsbench/usb4-rdma/launch_tp4_best.sh`:
at TP=4 it is -55% to -56% on both prompt types, and its only parity-clean
configuration is still -30% to -56%. TP=2 coding is the one configuration where
DSpark pays (+44.7%), but that is not the verified 4-rank launch.

```
DSPARK=1 CTX=2048 bash dsbench/usb4-rdma/launch_tp4_best.sh /tmp/prompt_now.txt 16
```

`DSPARK=1` adds `--dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf`
and `DS4_DSPARK_STATS=1`; `DS4_DSPARK_SEED_BATCH=0` additionally makes the
canonical greedy md5 match at the cost of speed.
