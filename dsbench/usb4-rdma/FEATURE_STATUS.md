# DS4 feature status under 4-node TP=4 over the USB4 RDMA ring

Verified 2026-11-24 on `feat/rocm-tp-full-stack`, four nodes
(192.168.1.101-104, switchless USB4 ring, RoCE v2 GID index 1), one 81 GiB
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
per host, ranks 0-3 on node1-4, `--tp-cpu-core 2`.

> **Update (2026-11-24):** the final `ds41f-q4` 4-node section at the end of this file
> supersedes the vision and DSpark rows below. Vision is now **verified** on `ds41f-q4`
> and text prefill at TP=4 is **7.34x** faster.

| feature | status | evidence | blocker |
|---|---|---|---|
| TP=4 USB4 RDMA ring | **verified** | 64.25 t/s prefill, 9.65 t/s decode, GTT 37.82 GiB/rank at ctx 2048; text md5 `a13ba035b8347fa92e602fd8f751ca91` == TP=1 | none |
| 1M context (compressed KV) | **verified** | `-c 1048576`: KV 13.79 GiB + 8.00 GiB buffers + 26.34 GiB resident = 48.13 GiB/rank, GTT 47.88 GiB, MemAvailable >= 42.85 GiB; 30,474-token story answered `Bob=34, Alice=52, Clara=71` | none |
| Vision (`--vision`) | **blocked** | all four ranks exit 1 with the rejection below; GTT stays 16.4 MiB | `ds4.c:74106`; `ds4.c:74034`; `ds4.c:6707`; `ds4.c:7458` |
| DSpark / MTP (`--dspark --mtp-model`) | **runs at N ranks on the batched verify gate; net loss at TP=4** | the N-rank verify gate is generalized (`tp_batch_rows` at world>=2), so a world=4 verify block costs ~320 ms/cycle instead of the per-layer bulk exchange's 742-870 ms and DSpark improves from 4.00-4.18 to 7.22-7.98 t/s. Against 9.35-9.45 t/s without DSpark that is still -15% to -24%, so `--dspark` stays off in `launch_tp4_best.sh` | performance: the 4-rank ring all-reduce costs 6 serialized round trips per layer gate plus ~1 software-HCA stall per block; the batched verifier's reduction order also changes the prose continuation |

## V4.1 Flash Q4 at TP=4 (ROCm)

Verified 2026-11-24 on `feat/rocm-tp-full-stack` @ `6d75052`, four nodes, one
518,596,067,328 B `DeepSeek-V4.1-Flash-Q4.gguf` per host, `--temp 0`,
`--tensor-parallel --transport usb4-rdma --tp-world-size 4 --tp-cpu-core 2`.

V4.1 originally refused network TP on ROCm: `ds4.c:74141` required
`--tp-world 1` under `DS4_ROCM_BUILD`, and when that gate was forced open the
first prefill token failed in `ds41_attention_expand`:

```
ds4: v41 trace: layer 0 attention_expand failed
ds4: v41 trace: layer 0 attention failed
ds4: v41 trace: layer 0 before_moe failed (stage -1)
ds4: V4.1 layer 0 failed at position 0
```

`ds41_attention_expand` -> `ds41_attention_b` -> `metal_graph_matmul_dense_quant_kslice`
-> `ds4_gpu_matmul_quant_kslice_tensor`, which was a `ROCM_UNAVAILABLE_INT`
stub in `ds4_rocm_unavailable.cu` and returned 0 with no message (a clean
failure, not a hang or garbage). The group-sliced attention output needs that
K-slice projection because each rank keeps only `DS4_N_OUT_GROUP / world` = 2
of the 8 output groups, i.e. `k_off = rank * 2048`, `k_cnt = 2048` of the
8192-wide `attn_output_b` row. `ds4_gpu_matmul_q8_0_kslice_tensor` was also a
stub (it printed "tensor parallelism is Metal-only").

Ported 1:1 from `ds4_cuda.cu`: `ds4_gpu_matmul_q8_0_kslice_rows_tensor` (new
`matmul_q8_0_kslice_preq_warp8_kernel`), `ds4_gpu_matmul_q8_0_kslice_tensor`,
`ds4_gpu_matmul_quant_kslice_tensor`, and
`ds4_gpu_routed_moe_batch_owned_tensor` (delegates to the existing
`routed_moe_launch` ownership window). Two 2-rank assumptions were also
fixed: the N-rank gate combine is now `own + in` (the ring hands back
`total - own`, `ds4_tp.c:566`), and the attention partial is all-reduced at
`world > 2` instead of only at `world == 2`.

| item | ctx 2048 | ctx 1048576 |
|---|---|---|
| prefill | 11.12 t/s (CLI), 11.03 t/s (API) | 10.74 t/s |
| decode | 5.12 t/s (CLI), 9.84 t/s (API) | 7.30 t/s (CLI) |
| TTFT | - | 6.72 s measured for 74 prompt tokens (API) |
| resident/rank | 80.56 GiB | 80.56 GiB |
| planned | 82.11 GiB | 99.17 GiB (KV 6.26 + buffers 12.35) |
| GTT peak/rank | 82.13 GiB | 97.83 GiB of the 124 GiB pool |
| MemAvailable min | 22.15 GiB (node1) | 5.61 GiB (node1), 18.59/23.06/23.09 |

Against the 0731 V4 Q2 reference at TP=4 / ctx 2048 (64.25 prefill, 9.65
decode, 37.82 GTT, 26.34 resident) the Q4 model trades memory for quality:
resident 3.1x and GTT 2.6x higher, prefill 5.8x slower (V4.1 prefill runs the
per-token scalar path with a ring gate per layer and reads Engram rows from
disk), decode about the same through the serving path.

`ds4-server` serves the same 4-rank ring: `/v1/models` reports
`context_length: 1048576`, and a 74-token chat completion produced a coherent,
correct answer (2*(N-1) = 6 hops for N=4) in 22.975 s at 9.84 t/s decode.

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

---

# FINAL STATE — `ds41f-q4` 4-node TP=4 over the USB4 RDMA ring

Verified 2026-11-24 on `feat/rocm-tp-full-stack`, HEAD `d6c3f84`, tree clean, nothing
pushed. Nodes 192.168.1.101-104 (node1-4), ranks 0-3, switchless USB4 ring, RoCE v2.
**This section supersedes the earlier per-phase results above where they differ:** vision
now works, and text prefill at TP=4 is 7.34x faster.

## Config

| item | value |
|---|---|
| model | DS4 `ds41f-q4`, 518,596,067,328 B, V4.1 Flash |
| topology | 4-node TP=4, `--transport usb4-rdma`, ring 192.168.1.101-104 |
| context | `CTX=1048576` |
| vision | `--vision gguf/DeepSeek-V4.1-Flash-Vision.gguf`, enabled |
| binary | `ds4` md5 `edacd28798141a279293024641e61800`, identical on all four nodes |

## Vision — VERIFIED

Request: `POST /v1/chat/completions` with a 3308-byte PNG (4412 b64) as an `image_url`
data URI. The 560x560 image contains a red circle (top-left), a blue square (top-right),
"CAT" in black, "347" in green.

Response (HTTP 200, 49.45 s, 254 prompt / 185 completion): *"A red circle, a blue
square, and the text "CAT 347" in black and green."* -- all four elements correct.

Encoder on rank 0:
`rows=196 dim=5120 elems=1003520 min=-0.785 max=0.305 mean=1.03e-4 nonfinite=0 zeros=0`,
14x14 grid; 196 x 5120 x 4 = 4,014,080 B exact. 212/212 image positions substituted,
`token_start=43`. Rank 1 and rank 2 logged identical row values to rank 0, proving the
world=4 broadcast.

Text-only control: *"No image was provided, so I can't identify any shapes or text."*

Root cause of the earlier failure: the shared encoder hardcoded the V4-Flash aligner
width **4096** while V4.1's aligner is **5120** wide (`aligner.w1 (5120,9216)`,
`aligner.w2 (5120,5120)`), leaving 1024 of every 5120 lanes uninitialised. Fixed to use
`weights->projection_dim` (`087bb3f`).

## Prefill at TP=4 — 7.34x

Root cause, two independent defects:

1. `ds41_tp_batch_enabled` (`ds4.c:43694`) admitted only `tp_world` 1 or 2, so world=4
   ran the token-major scalar path with **80 ring exchanges per token**.
2. `ds4_gpu_attention_output_low_q8_rows_exact_tensor` was a `ROCM_UNAVAILABLE_INT` stub
   (`ds4_rocm_unavailable.cu:13`).

Both fixed in `d6c3f84`. Same binary, same 282-token prompt, ctx 1048576:

| path | prefill | prompt time | decode |
|---|---:|---:|---:|
| token-major scalar | 9.80 t/s | 28.763 s | 5.20 t/s |
| chunked (new) | **71.96 t/s** | **3.919 s** | **5.32 t/s** |

7.34x on the same prompt; against the inherited 10.74 t/s baseline that is 6.7x.

Caveat: prompts **<256 tokens use the 8-row short path (~13 t/s)**, and **image prompts
stay token-major by design (~11.7 t/s)** -- the win is text prompts >=256 tokens.

## Memory

- GTT peak **98.74 GiB of 124** with vision loaded; resident **80.56 GiB/rank**.
- **node1 memory resolved:** it was not coordinator overhead. `/tmp` (63 GB tmpfs) held
  **1,981 stale debug tensor dumps totalling 18 GB**, plus 10.5 GB shmem; deleting them
  took MemAvailable **4.95 -> 23.29 GiB**. The remaining node1 delta is the 0.90 GiB
  vision encoder, expected.

## Decode breakdown

Derived from two A/B points: **~84% USB4 ring latency, ~16% compute/Engram/head**.
R ~= 89 ms/token across 80 exchanges => **~1.11 ms per gate exchange**; C ~= 12.9 ms/row.
Engram I/O is tiny (~12.7 KB/token) and hidden by async prefetch.

Larger unbuilt opportunities to log: fuse the two gates per layer to halve the exchange
count, Engram row caching, a faster link.

## Logging and DSpark

- **No `v41 trace:` logging exists at this HEAD** -- nothing to strip.
- **DSpark: not applicable.** There is no V4.1 drafter artifact (only the 0731 drafter),
  so speculative decode was not measured on Q4. For the record, the TP=4 DSpark
  measurement on 0731 was 7.65/7.99 t/s against 9.45/9.35 non-speculative (i.e. a loss),
  and `--dspark` is off by default.

## Q2 reference

The Q2 reference is **no longer reproducible**: the `DeepSeek-V4-Flash-IQ2XXS-...-0731`
GGUF was deleted on all four nodes during the 2026-11-24 cleanup. Numbers recorded
earlier in this document are historical only.

## Operations

Inter-node SSH **works** as of 2026-11-24: node1 (`node_one`) has a passwordless
ed25519 key in `authorized_keys` on node2/3/4, and `~/.ssh/config` maps
`node2 -> node_two@192.168.1.102`, `node3 -> node_three@192.168.1.103`,
`node4 -> node_four@192.168.1.104`. Verified from node1:
`for h in node2 node3 node4; do ssh -o BatchMode=yes $h 'id -un; hostname'; done`
returns rc=0 with the right username on each. `sudo` is passwordless on node1.

### API endpoint

- Local (on node1): **`http://127.0.0.1:8000/v1`** -- OpenAI-compatible
  (`/v1/models`, `/v1/chat/completions`), served model id `deepseek-v4.1-flash`,
  `context_length: 1048576`, `max_completion_tokens: 256`, `--vision` on.
- LAN: **`http://192.168.1.101:8000/v1`** -- `ds4-server` now binds `0.0.0.0:8000`
  and `8000/tcp` is open permanently in the `FedoraServer` firewalld zone.
  Verified with `curl` from the Windows host (HTTP 200).

### LAN exposure and services (updated 2026-11-24)

- `ds4-server` binds **`0.0.0.0:8000`** (`--host 0.0.0.0 --port 8000`); `ss -ltnp`
  shows `0.0.0.0:8000` on node1. The LAN URL is **http://192.168.1.101:8000/v1**.
- `firewalld` is running; default zone `FedoraServer` (interfaces `tbv0`, `enp191s0`).
  **Port `8000/tcp` is open permanently** (`firewall-cmd --permanent --add-port=8000/tcp
  && firewall-cmd --reload`; `firewall-cmd --query-port=8000/tcp` -> `yes`).
- Cockpit is installed (`cockpit-368-1.fc45`); `cockpit.socket` is enabled and active on
  `*:9090`, and the `cockpit` service is allowed through the firewall, so
  **https://192.168.1.101:9090** is reachable from the LAN.
- The service is **`ds4-tp4.service`** (see the update section), visible in Cockpit
  **Services** at <https://192.168.1.101:9090>; the unit's journal carries the engine
  output through `systemd-cat`.

### Start / stop / status (systemd; manual fallback below)

```bash
# start all four ranks; the API answers when the engine is loaded (~40-90 s)
ssh -o BatchMode=yes node1 'sudo systemctl start ds4-tp4.service'

# stop coordinator then workers (exact process names)
ssh -o BatchMode=yes node1 'sudo systemctl stop ds4-tp4.service'

# restart; verified to leave 0 orphaned processes
ssh -o BatchMode=yes node1 'sudo systemctl restart ds4-tp4.service'

# unit health + journal
ssh -o BatchMode=yes node1 'systemctl status ds4-tp4.service --no-pager'
ssh -o BatchMode=yes node1 'journalctl -u ds4-tp4.service -n 50 --no-pager'
```

The launcher is `dsbench/usb4-rdma/launch_tp4_best.sh` (`start|stop|status|restart`);
it fans out to node2-4 over the passwordless ssh above. The unit runs as `node_one`
because rootless `toolbox`/podman live in that user's storage; `loginctl enable-linger
node_one` keeps `/run/user/1000` available without a login. The manual wrappers
`/tmp/start_coord.sh` and `/tmp/start_worker.sh` are preserved as a fallback, and the
old CLI benchmark launcher is kept as `launch_tp4_cli_bench.sh`.

### Stop (manual fallback)

```bash
# coordinator first, then the three workers; exact names only
ssh -o BatchMode=yes node1 'pkill -TERM -x ds4-server'
for h in node2 node3 node4; do ssh -o BatchMode=yes $h 'pkill -TERM -x ds4'; done
```

Under systemd use `systemctl stop ds4-tp4.service` instead.

### Status

```bash
# HTTP health (expect 200)
ssh -o BatchMode=yes node1 'curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/models'

# model card
ssh -o BatchMode=yes node1 'curl -s http://127.0.0.1:8000/v1/models'

# rank processes (1 coordinator + 3 workers)
ssh -o BatchMode=yes node1 'pgrep -af "role coordinator"'
ssh -o BatchMode=yes node2 'pgrep -af "role worker"'
ssh -o BatchMode=yes node3 'pgrep -af "role worker"'
ssh -o BatchMode=yes node4 'pgrep -af "role worker"'

# listener / bind address
ssh -o BatchMode=yes node1 'ss -ltnp | grep 8000'
```

Smoke request (returns `51`, HTTP 200, ~4.6 s):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 17*3? Answer with the number only."}],"max_tokens":400,"temperature":0}'
```

### git log (all three branches, `--oneline`)

```
feat/rocm-tp-full-stack (HEAD d6c3f84, 26 commits)
d6c3f84 rocm: run the V4.1 chunk prefill at TP=4 instead of per token
087bb3f rocm: project the V4.1 vision aligner at the bound width, not 4096
fbb11e5 chore: drop the stray grep output accidentally added with the vision commit
507908a rocm: allow the V4.1 vision encoder on ROCm (gate opened, image path unverified)
68a44a2 rocm: 1M context + HTTP API for V4.1 Q4 at TP=4; launcher + gate stats
6d75052 rocm: run V4.1 Flash Q4 at TP=4 over the USB4 RDMA ring
02f98a5 tp: run the verify-block batch gate at N ranks (DSpark still a loss)
42809c2 tp: port DSpark speculative decode to N ranks; off by default (no TP=4 win)
49d6ce9 tp: generalize the prefill shared-expert row split to N ranks
318f399 tp: document the vision and DSpark gaps at world=4
fefb7ae tp: document the world=4 ring and 1M context verification
8bcfbca tp: add the verified four-node TP=4 launch script
79a9247 tp: run the ROCm expert split at N ranks over the USB4 ring
a8021ff tp: restore sorted IQ2 prefill on expert-owning ranks
123bf97 tp: keep the USB4 ring warm across the prefill cold start
402cb48 docs: drop the unsupported TCP payload-only row from the world=2 table
fe839d6 tp: run the USB4 RDMA ring at world=2 as the TP=2 payload plane
fdf2222 tp: implement routed-MoE expert ownership on ROCm so TP=2 produces tokens
792508c tp: open the ROCm collective seam and refuse at the expert-split wall
e26ee49 tp: do not treat an unset --tp-cpu-core as a request
10141e2 docs: describe the USB4 RDMA ring transport and what it does not cover
0cafa0d vendor: sync usb4_rdma_ring and bound the QP ACK timeout
79b0482 gitignore: ignore the ring collective test binary
61b8475 tp: add opt-in N-rank ring collective transport
7a46b01 rocm: fix host-side rsqrtf call in the vision attention site
b3f5018 Read half-aligned Qwen Q2 weight rows correctly on Metal

feat/rocm-tp-ring (49d6ce9, 18 commits, untouched)
49d6ce9 tp: generalize the prefill shared-expert row split to N ranks
318f399 tp: document the vision and DSpark gaps at world=4
fefb7ae tp: document the world=4 ring and 1M context verification
8bcfbca tp: add the verified four-node TP=4 launch script
79a9247 tp: run the ROCm expert split at N ranks over the USB4 ring
a8021ff tp: restore sorted IQ2 prefill on expert-owning ranks
123bf97 tp: keep the USB4 ring warm across the prefill cold start
402cb48 docs: drop the unsupported TCP payload-only row from the world=2 table
fe839d6 tp: run the USB4 RDMA ring at world=2 as the TP=2 payload plane
fdf2222 tp: implement routed-MoE expert ownership on ROCm so TP=2 produces tokens
792508c tp: open the ROCm collective seam and refuse at the expert-split wall
e26ee49 tp: do not treat an unset --tp-cpu-core as a request
10141e2 docs: describe the USB4 RDMA ring transport and what it does not cover
0cafa0d vendor: sync usb4_rdma_ring and bound the QP ACK timeout
79b0482 gitignore: ignore the ring collective test binary
61b8475 tp: add opt-in N-rank ring collective transport
7a46b01 rocm: fix host-side rsqrtf call in the vision attention site
b3f5018 Read half-aligned Qwen Q2 weight rows correctly on Metal

feat/usb4-rdma-ring (e26ee49, 7 commits, untouched)
e26ee49 tp: do not treat an unset --tp-cpu-core as a request
10141e2 docs: describe the USB4 RDMA ring transport and what it does not cover
0cafa0d vendor: sync usb4_rdma_ring and bound the QP ACK timeout
79b0482 gitignore: ignore the ring collective test binary
61b8475 tp: add opt-in N-rank ring collective transport
7a46b01 rocm: fix host-side rsqrtf call in the vision attention site
b3f5018 Read half-aligned Qwen Q2 weight rows correctly on Metal
```

## State at handoff (2026-11-24)

- 4-node server **up and serving**: `/v1/models` -> HTTP 200; smoke completion ->
  HTTP 200, 43 prompt / 20 completion, 4.58 s, answer `51`.
- `ds4` md5 `edacd28798141a279293024641e61800` on node1/2/3/4.
- `feat/rocm-tp-full-stack` @ `d6c3f84` + this docs commit, `git status --porcelain`
  empty; `feat/rocm-tp-ring` `49d6ce9` and `feat/usb4-rdma-ring` `e26ee49` unchanged.
- Launchers `/tmp/start_coord.sh` and `/tmp/start_worker.sh` preserved on all nodes.
- Q2 GGUF deleted on node2/3/4 (~81 GiB freed each); Q4 + vision + 0731 drafter remain.


# UPDATE — gate audit, LAN API and `ds4-tp4.service` (2026-11-24)

## Step 1 — the two per-layer ring gates

### What the two exchanges are

V4.1 has `DS4_N_LAYER = 40` and two gates per layer (`DS4_TP_GATES_PER_LAYER = 2`,
`ds4_tp.h:30-32`: `DS4_TP_GATE_ATTN = 0`, `DS4_TP_GATE_FFN = 1`). Both fire from the
decode graph and reach the same seam, `ds4_tp_gate_exchange()` (`ds4_tp.c:3321`). At
world=4 over `--transport usb4-rdma` each gate moves **5120 x f32 = 20,480 B** per rank
(`tp_ring_exchange`, `ds4_tp.c:556` -> `ds4_tp_ring_all_reduce` ->
`usb4_rdma_ring_all_reduce`, `misc/usb4/usb4_rdma_ring.c:1312`).

| gate | id | fired at | reduces | dtype / rows |
|---|---|---|---|---|
| attention output | `DS4_TP_GATE_ATTN` (0) | `ds4.c:42115` in `ds41_attention_expand()`, reached from `ds41_attention()` (`ds4.c:42365`) | this rank's `attn_output_b` group slice: a partial sum of the layer's attention output | F32, 1 row x 5120 |
| MoE / FFN output | `DS4_TP_GATE_FFN` (1) | `ds4.c:42699` in `ds41_moe_finish()`, reached from `ds41_moe()` (`ds4.c:42706`) -> `ds41_graph_layer()` (`ds4.c:43332`) | this rank's routed/shared expert partial | F32, 1 row x 5120 |

(The two `ds41_sum_partial(..., DS4_TP_GATE_ATTN)` sites at `ds4.c:42378` and
`ds4.c:43374` are the world=2 decode-island path; at world=4 the
`ds41_attention_expand` site above is the one that runs.)

### Data dependency: strictly serial

`ds41_attention_expand` (`ds4.c:42103-42118`) reduces the ATTN partials into `g->block`
and then `ds41_hc_expand`s it. `ds41_graph_before_moe` (`ds4.c:42953`) calls that before
`ds41_moe`. `ds41_hc_mix` (`ds4.c:42032`) computes
`g->x = ds41_hc_sum(residual, pre, ffn)` and `g->norm = ds41_norm(g->x)`, and
`ds41_moe_partial` (`ds4.c:42690`) computes the router row with
`ds41_matmul(g->route_logits, ffn_gate_inp, g->norm)`. The MoE gate/router therefore
consumes the **reduced** attention output: the FFN all-reduce cannot start until the
attention all-reduce has finished. All 80 exchanges per token form one serial chain
(layer L+1's attention also depends on layer L's FFN residual), so there is no
independent pair to fuse or overlap either. Fusing the two collectives would require
routing on an unreduced hidden state or sending an incomplete attention partial — both
change model math. **Not fuseable; not attempted.**

### Cheaper wins that do not touch model math

1. **Per-gate TCP header (implemented, `f2a3863`).** At world>2 the ring path still sent
   a 16-byte layer/gate/seq header to every peer over TCP and waited for the replies
   (`tp_tcp_exchange_multi`, `ds4_tp.c:860`) before the ring all-reduce. The ring already
   barriers at every step and its payload carries an absolute seq marker, so the header
   was diagnostic, not a correctness requirement. Measured on the live stack: **199 us of
   a 491.7 us whole gate through the API (40.5%)**, 291 us of 461 us on the CLI.
   `f2a3863` skips it on the ring decode path (`DS4_TP_RING_HEADER=1` restores the
   cross-check); the TCP gate path and the prefill big-gate arrival barrier are
   unchanged. An ordering bug still fails safe: `wait_marker` accepts only the expected
   seq and times out after 5 s.
2. **Overlap instead of fuse (rejected).** There is no consumer that can run between the
   ATTN reduce and the FFN reduce: the router is that consumer, and the 80 gates are a
   serial chain.
3. **Software-HCA wake-up / polling.** The ring busy-polls its completion marker with
   `cpu_pause()` and a 5 s deadline (`misc/usb4/usb4_rdma_ring.c:775`). The QP ACK
   timeout is tunable with `USB4_RING_QP_TIMEOUT` (`misc/usb4/usb4_rdma_ring.c:73`): the
   library default is 8 (1.05 ms), the launcher pins 14 (67 ms) to survive the one cold
   prefill gap. In the 5040-gate CLI sample only 4 steps exceeded 1 ms (max 41 ms, i.e.
   one wake-up); decode gates are back-to-back, so the ms-scale idle gap that causes the
   stall is absent. A full 8-vs-14 A/B was not run in this session; lowering the timeout
   trades the cold-gate retry margin for a shorter worst-case stall and is the first knob
   to try if the stall returns.

### Measured result (same prompts, ctx 1048576, greedy)

| metric | before | after |
|---|---:|---:|
| gate whole / payload (API) | 491.7 / 292.7 us | 360.6 / 360.6 us |
| gate whole / payload (CLI) | 461.2 / 170.2 us | -- |
| 99-token decode | 8.88 t/s | 10.35-11.60 t/s (avg 10.98, +23.6%) |
| 99-token TTFT | 8.74 s | 6.87-7.43 s |
| 36-token TTFT | 3.38 s | 2.63 s (1 run) |
| 390-token decode | 4.29/4.49 t/s | 4.27/4.55 t/s (unchanged) |
| 390-token prefill | 78.5 t/s | 77.6-79.8 t/s (unchanged) |

The 390-token decode is attention/indexer-compute-bound, so the ~16 ms/token removed
from 80 gates is inside run-to-run noise there; the short/mid path (<256-token prompts
run the token-major gates) is where it shows. Prefill >=256 tokens keeps the big-gate
arrival header and is unchanged.

Gate count: `DS4_TP_GATE_STATS=1` (now an alias of `DS4_TP_XCHG_STATS`) prints
`n=118240` = 80 x 1478 token positions (740 decoded + 738 prefilled on the token-major
short path) = **80 exchanges per position exactly** — the count is unchanged, as
expected: this is a latency cut, not a fusion.

Vision re-verified on the same 560x560 PNG (md5 `c18f122eb25273583aa5f197f6532439`):
*"A red circle and a blue square above the text "CAT 347" (with "CAT" in black and "347"
in green)."* HTTP 200, 29.7 s, `finish=stop`.

## Step 2 — LAN API and `ds4-tp4.service`

### Bind and firewall

`ds4-server` now starts with `--host 0.0.0.0 --port 8000` (launcher default;
`CTX=1048576`, `--vision` on). `ss -ltnp` on node1:

```
LISTEN 0 128 0.0.0.0:8000 0.0.0.0:* users:(("ds4-server",pid=...,fd=28))
```

`firewall-cmd --permanent --add-port=8000/tcp && firewall-cmd --reload`;
`firewall-cmd --query-port=8000/tcp` -> **yes** (zone `FedoraServer`, interfaces
`tbv0 enp191s0`).

### Inter-node SSH (the constraint that decided the unit design)

The unit can only restart all four ranks if node1 can reach node2-4. That is **working**:
node1 (`node_one`) has an ed25519 key trusted in `authorized_keys` on each worker, and
`~/.ssh/config` maps the per-node usernames. Evidence from node1:

```
$ for h in node2 node3 node4; do ssh -o BatchMode=yes $h 'id -un; hostname'; done
node_two
node2
node_three
node3
node_four
node4
```

rc=0 on all three, no password prompt, no `enp191s0` change, no new install. The unit
therefore manages the **full four-rank cluster**, not just the coordinator.

### Unit

`dsbench/usb4-rdma/ds4-tp4.service` (installed at `/etc/systemd/system/ds4-tp4.service`)
runs as `node_one` (rootless `toolbox`/podman live in that user's storage;
`loginctl enable-linger node_one` keeps `/run/user/1000` present), with
`Restart=on-failure`, `KillMode=mixed` and journald output:

```
[Service]
User=node_one
Environment=HOME=/home/node_one
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=CTX=1048576
Environment=BIND=0.0.0.0
Environment=PORT=8000
Environment=VISION=1
ExecStart=/usr/bin/bash /home/node_one/dev/ds4-v41-pp/src/dsbench/usb4-rdma/launch_tp4_best.sh start
ExecStop=/usr/bin/bash /home/node_one/dev/ds4-v41-pp/src/dsbench/usb4-rdma/launch_tp4_best.sh stop
Restart=on-failure
```

The engine runs inside the container, so the launcher pipes the coordinator through
`systemd-cat -t ds4-tp4`; without that, journald attributes the engine to the container
scope and Cockpit's unit log only shows the launcher. `ExecStart` fans out to node2-4
and then waits for `/v1/models`; `ExecStop` SIGTERMs `ds4-server` on node1 then `ds4` on
node2-4 (exact process names only).

Cockpit: <https://192.168.1.101:9090> -> **Services** -> `ds4-tp4.service`
(Logs tab = `journalctl -u ds4-tp4.service`).

### Verified

- `systemctl status ds4-tp4.service` -> `active (running)`, `NRestarts=0`.
- LAN: `curl -s -o /dev/null -w '%{http_code}' http://192.168.1.101:8000/v1/models`
  from the Windows host -> **200**; `/v1/models` returns `deepseek-v4.1-flash`,
  `context_length: 1048576`, modalities text+image.
- A LAN chat completion from the Windows host returned `51` for `17*3` in 4.40 s
  (HTTP 200).
- `systemctl restart ds4-tp4.service`: **PASS**. Before: node1 `ds4-server` pid
  345430, node2-4 `ds4` pids 587527/313409/312359. After 42 s:
  345973/588063/313909/312818 — **0 orphaned processes**, exact counts node1
  `1 ds4-server + 0 ds4`, node2-4 `1 ds4 + 0 ds4-server`.
- A second restart after the journald routing fix also came back with
  `rank 0/4 connected` in `journalctl -u ds4-tp4.service`.

## Audit (2026-11-24)

- Branch `feat/rocm-tp-full-stack`, HEAD `75eb65e` plus this docs commit (code
  `f2a3863`), clean working tree.
- `feat/rocm-tp-ring` = `49d6ce9`, `feat/usb4-rdma-ring` = `e26ee49` (untouched).
- `ds4` md5 **6f3c1fe471ddced353807a65bab5c5d9** and `ds4-server` md5
  **7db66c88681a710475a75077a82d9a6c** on node1-node4.

## Final read-only verification (2026-11-24)

Live 4-node TP=4 stack, model `deepseek-v4.1-flash`, `ds4-tp4.service`
`active (running)` with `NRestarts=0` before and after the run. Every
request went to `http://127.0.0.1:8000/v1` on node1 with `max_tokens: 256`
and `temperature: 0`. Prompts were synthetic numbered-paragraph filler with
a unique vault-code needle (`QX-4417`) near the start and a relay-identifier
needle (`BR-9032`) near the middle; the closing question asked for both.

### Context-depth sweep

| depth | prompt tokens | prefill t/s | prefill s | TTFT s | decode t/s | gen | needle recall |
|---|---|---|---|---|---|---|---|
| 4K | 4005 | 99.86 | 40.11 | 40.34 | 13.05 | 57 | vault PASS, relay PASS |
| 16K | 15882 | 133.21 | 119.24 | 119.48 | 12.39 | 62 | vault PASS, relay PASS |
| 64K | 63538 | 150.46 | 422.30 | 422.56 | 12.62 | 69 | vault PASS, relay PASS |

Targets were 4096/16384/65536 tokens; the paragraphs tokenize at ~37.0
tokens each at scale (measured from `usage.prompt_tokens`), so the actual
counts above are 4005 / 15882 / 63538. Each depth answered exactly
`VAULT=QX-4417` and `RELAY=BR-9032`.

How each column was obtained:

- prompt tokens: response `usage.prompt_tokens`, identical to the coordinator
  journal `chat ctx=0..N:N`.
- prefill t/s and prefill s: coordinator journal, last
  `prefill chunk ... avg=X t/s` and `prompt done Xs` (engine-measured);
  prompt_tokens / prompt_done_s reproduces the same rate.
- TTFT: client wall clock on node1, POST send to first streamed delta
  (reasoning or content). It sits 0.23-0.27 s above the engine's
  `prompt done`, i.e. network + first-token dispatch.
- decode t/s: coordinator journal `decoding ... avg=X t/s`, engine-measured
  over all generated tokens (reasoning + content); `gen` matches
  `usage.completion_tokens`.

Peak GTT and minimum MemAvailable per node, sampled every 8 s on all four
nodes for the whole request (GTT from
`/sys/class/drm/card1/device/mem_info_gtt_used`, MemAvailable from
`/proc/meminfo`; the 8 s interval can in principle miss a sub-interval spike):

| depth | node1 GTT / MemAvail | node2 | node3 | node4 |
|---|---|---|---|---|
| 4K | 98.80 GiB / 21.92 GiB | 97.91 GiB / 22.66 GiB | 97.91 GiB / 22.65 GiB | 97.90 GiB / 22.69 GiB |
| 16K | 98.88 GiB / 21.71 GiB | 97.99 GiB / 22.46 GiB | 97.99 GiB / 22.45 GiB | 97.98 GiB / 22.48 GiB |
| 64K | 99.06 GiB / 21.29 GiB | 98.16 GiB / 22.02 GiB | 98.17 GiB / 22.00 GiB | 98.15 GiB / 22.04 GiB |

GTT rises only ~0.3 GiB from 4K to 64K; the 63.5K KV is compressed.

### KV-cache prefix reuse

- Turn 1: 4005 prompt tokens, TTFT **40.61 s**, prefill 99.45 t/s,
  `cached_tokens: 0`.
- Turn 2: the identical message list plus the turn-1 assistant reply and one
  short follow-up user message, 4078 prompt tokens, TTFT **1.56 s**,
  engine `prompt done 1.33 s`, `cached_tokens: 4062` of 4078 (only 16 new
  tokens written).

TTFT fell 40.61 -> 1.56 s (-96%). Verdict: prefix KV reuse is active.
The timing drop alone is an inference; the `cached_tokens` accounting is
direct evidence that 99.6% of the turn-2 prompt reused cached KV rather
than being re-prefilled.

### Tool calling

A request with the `get_weather` function schema and the prompt
"What's the weather in Paris?" returned HTTP 200 with one structured call
and `finish_reason: tool_calls`:

```json
"tool_calls": [
  {
    "id": "call_9220a7a442cabb3b2d7ce8a34e09e17f",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\":\"Paris\"}"
    }
  }
]
```

`function.name` is `get_weather`; `arguments` is a string that parses as
JSON to the object `{"location": "Paris"}` (valid JSON, `location`
present); `content` is empty and `reasoning_content` is 62 chars, as
expected for a tool call.

### RNR / timeout / restart evidence

`journalctl -u ds4-tp4.service` over the full run window (11.44 min, 360
journal lines): `grep -aiE 'RNR|TIMEOUT|retry|flushed|bad|desync'` -> **0
matches**. `systemctl show -p NRestarts ds4-tp4.service` ->
`NRestarts=0`; `systemctl is-active` -> `active`.

Raw driver: `/tmp/fb/results.json` on node1 (not committed).

## 2026-11-25 — `feat/rocm-tp-perf` (Tier 1-4 on top of `b90e0b1`)

Branch `feat/rocm-tp-perf`, four nodes, `ds41f-q4`, TP=4 USB4 RDMA ring,
`CTX=1048576`, vision on. Binary `ds4` md5 `cc86676a…` on all four ranks.
Full detail and measurements: `ROCM_TP_REPORT.md` (Tier 1-4 + wrap-up sections).

| tier | change | status |
|---|---|---|
| 1 | V4.1 DSpark drafter built from the pinned checkpoint (8.04 GiB, 81 tensors verified) and wired to all four ranks as an opt-in flag | **off by default**: 0/160 acceptance, verify 1.5-4.2 s per 6-row block, decode -41% when five drafts are forced; does not fit beside 1M ctx + vision (110.18 vs 109.44 GiB) |
| 2 | short-prefill widening: batched sweep from 8 tokens at resident ROCm TP | **shipped**: <256-token prefill 4.7-5.4x (13.66 -> 64-65 t/s at 124; 12.58 -> 68.3 at 229); >=256-token prompts unchanged |
| 2 | bit-exact row-batched split-K attention-output HIP kernel | **dropped**: identical output, no speedup (+0.6% at 4K, -1.1% at 16K) |
| 3 | decode HIP-graph islands / MoE-router fusion | **no change**: ~84% of decode is ring round trips; decode islands are `tp_world<=1` by design (island 1 wraps both gates); MoE/router already fused on gfx1151 |
| 4 | USB4 ring BF16/WHT6 payload compression + prefill overlap | **analysed, not implemented**: payload is 6-11% of prefill wall time, so 2x compression is worth ~3-5% prefill and 0% decode; overlap impossible (FFN waits on the attention-output reduce, one in-flight block) |

Final end-to-end (shipped binary):

| ctx | prompt tok | prefill | TTFT | decode | needle |
|---|---:|---:|---:|---:|---|
| 4K | 3061 | 81.45 t/s | 37.59 s | 12.72 t/s | recalled |
| 16K | 11843 | 126.84 t/s | 93.38 s | 13.18 t/s | recalled |
| 64K | 44781 | 144.48 t/s | 309.96 s | 12.86 t/s | recalled |

Vision PNG (red circle / blue square / CAT 347) correct; OpenAI tool calling
returns two structured calls; `NRestarts=0`; zero RNR/timeout/desync lines;
GTT 101.4 GiB on node1 and 100.5 GiB on node2-4, MemAvailable 20-22 GiB.

Open item: multi-turn KV prefix reuse did not engage in this sweep
(`cached_tokens=0`, server `reason=token-mismatch` with the common prefix one
token short of the live checkpoint) although `b90e0b1` recorded a working case.
Pre-existing server behaviour, flagged with the `--trace` diagnostic to use
next; not caused by Tiers 1-4.

# 2026-11-25 (second session) — Avenue 1-3 on `feat/rocm-tp-perf`

Continued from `d6a8ae7`. Final binary `ds4` md5 `1d92b58c0f83722534f10e05de0a8d8a`,
`ds4-server` `9324cffc43033fb11830ac2545580a27`, identical on all four ranks.
Service runs its pristine unit config (`CTX=1048576`, vision on, no DSpark, no
drop-ins), `NRestarts=0`.

## Avenue 1 — DSpark at TP=4: one real bug fixed, acceptance still 0

**Drafter identity verified first:** all four ranks load the V4.1 support file
(sha256 `14f04ebf7314342d78921e2b9e28d8806fedc93a6551b72079fccd2f9ec5497b`,
8,633,942,016 B, `stages=3 block=5 markov_rank=256`), not the older V4 0731 drafter.

**Bug found and fixed:** `ds4_deepseek41_dspark.h` suspended ROCm expert sharding
for the leader-local drafter only when `tp_world == 2`, so at TP=4 the drafter's
routed MoE still ran with rank 0's contiguous quarter of the expert tables — the
same `== 2` -> `>= 2` defect fixed for the V4 Flash drafter. After the fix the
draft cost rose 60 ms -> 92 ms per 5-token proposal, confirming the drafter now
sweeps all 128 experts.

**Acceptance is still 0/160**, so a second defect remains in the ROCm drafter
path. It has never been exercisable at TP>=2 (the model needs all four nodes to
be resident, so TP=1/TP=2 cannot be run for comparison). The instrumented probe
shows the verifier is correct (its row-0 argmax becomes the next cycle's `t0`)
while the proposals are unrelated (`t0=109405` -> drafts 305/367/418/463 against
the target's own 127869/33372). Not a verify-row offset, not a draft-broadcast
divergence: at world>2 no draft frame is exchanged and only rank 0 drafts.

**Verify block stays 1.5-4.2 s** for a 5-6 row block, i.e. ~20-25 ms per gate.
`ds41_dspark_verify` gates the pre-posted small-row window on `tp_world == 2`;
at N ranks every one of the 80 gates pays the bulk-prefill handshake
(`ds4_tp_batch_block_begin_gates` pre-posts one layer's receives against a
single peer, which the ring collective does not provide). Porting that window
to the N-rank ring is the fix, and it only pays once acceptance works.

DSpark therefore stays **off by default**; the expert-sharding fix is
correctness-only and cannot affect the shipped stack.

## Avenue 2 — the KV prefix `token-mismatch` was a server-side rendering bug

`--trace` (now wired to the launcher as `TRACE_PATH`) printed the divergence:

```
919 == live 128804 "<|Assistant|>"  prompt 128804 "<|Assistant|>"
920 != live 128821 "<think>"        prompt 128822 "</think>"
```

The engine generates `<|Assistant|><think>{reasoning}</think>`, but both chat
renderers emitted a bare `</think>` for any assistant turn that was not the live
tail and not in a tool context (`tool_context || i > last_user_idx`,
`keep_reasoning || i > last_user`). Every reply from this model carries
reasoning, so **prefix reuse was impossible on essentially every multi-turn
request**, which is why the earlier sweep saw `cached_tokens=0`.

Fix: render the reasoning block whenever the client supplied one
(`DS4_CHAT_NO_REPLAY_REASONING` restores the old rendering).

| case | prefill before | prefill after | cached |
|---|---:|---:|---|
| plain multi-turn | 11.524 s | **1.071 s** (-90.7%) | 929/978 |
| tool-calling turn | 11.083 s | **0.864 s** (-92.2%) | 942/970 |

## Avenue 3 — prefill tile sweep: no win, default kept

`DS41_PREFILL_CAP` is now tunable (`DS4_V41_PREFILL_CAP`, 2048..32768, default
8192 unchanged). Matched A/B, 832K context, 16K prompt, same binary:

| tile | prefill | graph buffers | planned |
|---|---:|---:|---:|
| 8192 (default) | 126.96 t/s | 11.43 GiB | 97.08 GiB |
| 16384 | 127.26 t/s | 19.56 GiB | 105.21 GiB |

+0.24% for +8.13 GiB, and the wide tile no longer fits admission at 1M context
(110.05 vs 109.44 GiB). Prefill is weight-bandwidth bound per chunk, not step
bound. Default stays 8192; the knob is kept as a documented diagnostic.

## Final sweep (shipped binary, pristine unit)

| context | prompt tok | prefill | TTFT | decode | needle |
|---|---:|---:|---:|---:|---|
| short | 124 | 64.16 t/s | 1.94 s | — | — |
| short | 229 | 67.02 t/s | 3.42 s | — | — |
| 4K | 3061 | 81.43 t/s | 37.60 s | 12.72 t/s | recalled |
| 16K | 11843 | 125.50 t/s | 94.37 s | 13.18 t/s | recalled |
| 64K | 44781 | 145.17 t/s | 308.48 s | 12.86 t/s | recalled |

* Multi-turn KV reuse: `cached_tokens=929/978`, prefill 11.52 s -> 1.07 s.
* Vision PNG: red circle, blue square, "CAT 347" — correct.
* Tool calling: two structured calls with valid JSON arguments.
* `NRestarts=0`; zero RNR / retry / timeout / desync / QP-error lines for the
  whole run (the only grep hit is the benign `gate-timeout=750ms` startup line).

### 2026-11-25 (fifth session) — layer-by-layer bisect: verify_prefix corruption found and fixed

DSpark stays **off** (speed gate not met), but the correctness bug is now
identified, fixed and measured. `ds4` `243ae576e013e15a8971cd205cf9fd55` plus the
gate counter.

#### Harness

`DS4_BISECT_DUMP=1` prints, per layer, `sum`, `l2`, `max_abs` and the first four
floats of one row's post-attention tensor (`after_attn`, where the attention
output lives at TP>=2) and post-FFN residual. Comparing one 2-row verify block
(`DS4_DSPARK_VERIFY_CAP=1`) against two single-token decode steps of the same
prompt and tokens (`1, 2, 3, 4, 5, 6, 7,` -> `We`, ` need`).

Harness validation: the ordinary decode step that precedes the block is
byte-identical in both runs (`dsum=0`, `dl2=0` through layer 39), so the two
sessions really do share a starting state.

#### Result: row 0, verify block vs decode step 1

| stage | il | reference | verify | dsum | dl2 |
|---|---:|---:|---:|---:|---:|
| pre-layer0 | 0 | 28.034391 | 28.034391 | 0 | 0 |
| post-attn | 0 | 18.688898 | 18.691813 | **2.9e-3** | **4.8e-3** |
| post-attn | 1 | 0.342505 | 0.352706 | 1.0e-2 | 6.9e-2 |
| post-attn | 2 | 14.178909 | 14.069027 | 0.11 | 5.5e-3 |
| post-attn | 3 | 32.844519 | 38.046677 | 5.2 | 171.8 |
| post-attn | 4 | 45.196963 | 84.943732 | 39.7 | 1073 |

Row 1 behaves identically (first divergence also at layer-0 post-attention,
dsum 1.0e-2). The input embeddings are exact, so the block starts from the right
state.

**Interpretation.** The first divergence is a ~1.6e-4 *relative* difference at
layer 0, i.e. a floating-point reduction-order difference between the batched
verify projections and the single-token decode kernels — not a layout or
position bug. The model then amplifies it ~50x per layer from layer 3 onwards,
which is the MoE router flipping experts on near-boundary logits. By mid-stack
the verified trajectory is unrelated to the target's, so the verifier's argmax
is garbage and every correct draft is rejected. That is exactly the 0/160
acceptance and the "Wearelli是非..." output.

#### Fix

Each verify row now runs through the *exact* decode layer
(`ds41_graph_layer`) inside the block, so verified logits are bit-identical to
single-token decode. `DS4_VERIFY_BATCHED_LAYER=1` restores the old batched
pipeline for A/B.

| | before | after |
|---|---|---|
| output on the counting prompt | `Wearelli是非 main oj oj` | `We need answer?` (identical to plain decode) |
| acceptance | 0/160 (0%) | up to **4/4** on a block (observed 4/4, 4/2, 1/1) |
| verify block latency | 1.5-4.2 s | **250-540 ms** |
| net decode, 4K prompt | — | 7.87 t/s vs 12.72 plain |
| net decode, 896K short prompts | — | 4.47 / 4.77 / 4.00 vs 5.26 / 4.61 / 3.06 plain |

Phase timing (`DS4_VERIFY_PHASE_TIMING=1`) attributes the whole block to the
layer sweep, with a ring-gate counter: `setup=0.4 ms layers=445-540 ms
rest=0.0 ms` for a 6-row block — 40 layers x 6 rows of decode-path gates, i.e.
480 exchanges instead of 80.

#### Verdict and next step

DSpark still **loses** (7.87 vs 12.72 t/s at 4K) and stays **off**: the gate is
"faster than plain decode and 100% stable", and it is not faster yet. The
correctness blocker is gone, though, and the remaining gap is purely the
exchange count. Because a gate is a per-row all-reduce, batching the *exchange*
over the block's rows (one payload of `rows` rows per layer) does not change any
row's arithmetic — so per-row compute + one batched exchange per layer should
take 480 exchanges back to 80 while keeping bit-identical logits. That is the
next step, not a numerical compromise.

### 2026-11-25 (sixth session) — avenues 1-5: measured ceilings, no code change

Starting from `5b83d79`. Nothing in this session warranted a code change, so the
shipped binary is unchanged (`ds4` `39704dbbc86e56a62171569baf1fe1d9`); the
service was healthy throughout (`NRestarts=0`, no ring errors) and the close-out
sweep below re-confirms it.

#### Avenue 1 — batching the verify gate exchange: analysis, not implemented

Measured baseline (`DS4_VERIFY_PHASE_TIMING=1`): a 6-row block fires
**480 gates (80 per row) at 0.93 ms each = 446 ms**, which is the entire block
cost (`setup=0.3 ms`, `layers=446 ms`, `rest=0.0 ms`).

The intended refactor is to keep the per-row decode compute (bit-identical, see
`75f5613`) and batch only the exchange, since a gate is a per-row all-reduce and
one payload of `rows` rows leaves every row's arithmetic unchanged. The code
audit shows what that costs: the gate lives *inside* `ds41_attention_expand`
(attention, `ds4.c:42130`) and `ds41_moe_finish` (`ds4.c:42719`), so batching
requires splitting both into gated/ungated phases plus per-row destination
buffers, because `ds41_moe_partial` writes its partial into the single-row
`g->tp_out[il*2+FFN]` slot when the slab gate is active. The partial/finish split
already exists (`ds41_moe_partial` / `ds41_moe_finish`, used by the TP=2 decode
island), so the work is contained but not small: four phase boundaries per layer,
two new "finish without gate" paths, and the engram/`engram_rows_b` row-offset
handling has to stay correct across the split.

Expected outcome per the phase order: FFN-only batching gives 280 gates
(40 layers x 1 + 40 layers x 6), both gates batched gives 80. At 0.93 ms/gate
that is 446 -> ~260 ms -> ~75 ms per block, which is what would put a
5-accepted-token block under the plain-decode cost. Not attempted this session,
because a half-finished split in the verify path risks the correctness that
`75f5613` just established.

#### Avenue 2 — per-gate latency: the link is already at its ceiling

**Every USB4 port on all four nodes reports `10.0 Gb/s x 2 lanes`**
(`/sys/bus/thunderbolt/devices/*/rx_speed`, `tx_speed`), not the 20 Gb/s x 2 the
avenue hoped for. That matches the bring-up note in
`dsbench/usb4-rdma/readiness.md`: without the H2H enhancement each DMA path is
capped near 10 Gb/s. So 2.5 GB/s per link is the hardware ceiling here, and
0.93 ms per gate is consistent with a ring round trip at that rate plus CQ
wait/wakeup; there is no down-trained link to fix.

OS/transport state is already tuned: `thunderbolt_ibverbs` runs
`profile=linux_perf bind_services=Y allocate_rings=Y start_rings=Y`, and the CPU
is `amd_pstate=active` with governor `performance`.

Two parameters differ from the published multi-port tuning
([pleite/thunderbolt-ibverbs MULTI-PORT-TOPOLOGY](https://raw.githubusercontent.com/pleite/thunderbolt-ibverbs/refs/heads/main/docs/MULTI-PORT-TOPOLOGY.md)):
`native_fragment_striping=N` (guide suggests `1`) and `zcopy_min_bytes=0`
(guide suggests `4096`). Both are bandwidth/striping knobs rather than latency
knobs, so they would target prefill, and changing them needs a module reload that
tears down the ring for all four nodes - a maintenance-window item, not something
to do under a serving stack.

#### Avenue 3 — pipeline parallelism: not implemented

`--pp`, `--pipeline`, `pp_stage`, `pp_world` and `DS4_PP` do not exist anywhere
in the tree (0 symbols). Per the instruction, confirmed and moved on; there is no
TP=4 vs PP=4 A/B to run.

#### Avenue 4 — multi-slot KV checkpoints: single live slot today

Measured A -> B -> A with two different 4K documents:

```
A1: wall=39.77s prompt=3061 cached=0
B : wall=12.67s prompt= 809 cached=0
A2: wall=39.50s prompt=3061 cached=0    <- returning to A re-prefills everything
```

So the engine keeps exactly one live KV checkpoint (the last conversation) and
reuse only works when a prompt *extends* it - which is the 95% reuse already
measured inside one conversation. `slot_count` in `ds4_server.c` is server
concurrency (concurrent sessions), not retained prefix checkpoints. Multi-slot
warm prefixes would need engine-level work (one checkpointed session graph per
slot), not a flag; at ~5.5 GiB KV per 4K slot inside a 20 GiB `MemAvailable`
budget, two extra slots might fit, but that is a design change.

#### Avenue 5 — upstream and ecosystem delta

`git fetch origin` (read-only) shows **0 commits** on `antirez/ds4` beyond
`b90e0b1`; nothing to cherry-pick.

Ecosystem, read-only:
* [Shoestring-dev/StrixFold](https://upstar.dev/r/Shoestring-dev/StrixFold) is a
  Zig port of TensorFold running V4.1-Flash across **two** Strix Halo boxes. Its
  HIP/Strix Halo kernels are interesting but its topology is 2-node, so it does
  not map onto our 4-node TP=4 ring.
* The [multi-port topology guide](https://raw.githubusercontent.com/pleite/thunderbolt-ibverbs/refs/heads/main/docs/MULTI-PORT-TOPOLOGY.md)
  is the source of the two module params above.
* Sven's thunderbolt fixes ("lost ring interrupts" and an XDomain use-after-free,
  found between two Strix Halo hosts running an out-of-tree USB4 service driver)
  are candidates for the host kernel
  ([patch](https://patchew.org/linux/1794520303-5758-1-git-send-email-sven-kernel@sven.de/)).
* Strix Halo system tuning (ODS) recommends the `performance` governor and
  disabling idle states for 5-8% prompt processing; the governor is already
  `performance`.

#### Close-out sweep (unchanged binary, plain decode, DSpark off)

| context | prompt tok | prefill | TTFT | decode | needle |
|---|---:|---:|---:|---:|---|
| short | 124 | 64.75 t/s | 1.92 s | — | — |
| short | 229 | 67.40 t/s | 3.40 s | — | — |
| 4K | 3061 | 79.69 t/s | 38.42 s | 13.41 t/s | recalled |
| 16K | 11843 | 126.50 t/s | 93.63 s | 13.18 t/s | recalled |
| 64K | 44781 | 144.06 t/s | 310.86 s | 12.86 t/s | recalled |

Vision PNG (red circle / blue square / CAT 347) correct; two structured tool
calls; multi-turn KV reuse `cached 929/978`; `NRestarts=0`; zero
RNR/timeout/desync.

### 2026-11-25 (seventh session) — Avenue 1, Stage 1: implemented, measured, gated off

Implemented the Stage 1 slice exactly as specified: per-row decode compute is
unchanged (still bit-identical to plain decode), the `count` FFN partials are
staged into their own batch slices and reduced by **one** batched gate per layer.
It works and hits the gate target, but it is a **pessimization**, so it ships
opt-in (`DS4_VERIFY_BATCH_FFN_GATE=1`). Stage 2 was not attempted.

#### What the change does

Inside the `verify_prefix` per-row loop: `ds41_graph_before_moe` (attention plus
its own gate, unchanged) -> `ds41_moe_partial(..., shared_down=true)` -> copy the
partial out of the shared single-row `g->tp_out[il*2+FFN]` slot into
`rows_view[i].block` -> once per layer `ds41_sum_partial_batch(g, active.block,
il, DS4_TP_GATE_FFN, rows)` -> per row `ds41_bf16(row.block)` then
`ds41_graph_after_moe`. `DS4_VERIFY_BATCH_FFN_GATE=0` (default) keeps the
per-row gate.

#### Measured (6-row block, `DS4_VERIFY_PHASE_TIMING=1`)

| config | gates | layers |
|---|---:|---:|
| Stage 1 off (default) | **480** (80/row) | **446 ms** |
| Stage 1 on (FFN batched) | **280** (240 row + 40 batch) | **1700 ms** |

Output is identical in both cases (`We need answer? User just wrote "` on
`1, 2, 3, 4, 5, 6, 7,`), so the batching is arithmetically sound - it is purely
a latency regression.

#### Why

The per-row gate and the batch gate are different transport paths. A row gate
costs **~0.93 ms** (`ds4_gpu_tp_gate_encode` -> the fast single-shot row
exchange). A batch gate costs **~37 ms** (`ds4_gpu_tp_batch_gate_encode` ->
`ds4_tp_batch_gate_exchange` -> `tp_rdma_big_gate_exchange`, which posts a
`tp_rdma_window_barrier` plus a windowed big-gate transfer on every call).

So 40 batched gates cost 1477 ms where the 240 row gates they replace cost
223 ms - a 40x per-exchange penalty that swamps the 6x fewer exchanges.

#### Consequence for Stage 2

Batching the attention gate would use the same `ds41_sum_partial_batch` path, so
80 batched gates would cost ~3 s against ~450 ms of row gates: Stage 2 would be
the same pessimization, and was therefore not attempted.

The real lever is now precise: **make the fast row-gate path carry a multi-row
payload** (a single post/send pair of `rows * vec_bytes`, no window barrier) so a
batched gate costs ~1 ms instead of ~37 ms. The slab already has per-slot
`DS4_TP_BATCH_MAX_ROWS * vec_bytes` buffers, so the storage side exists; what is
missing is the low-latency exchange entry point. With that, the Stage 1/2
structure above becomes a clean win (480 -> 80 gates at ~1 ms each, ~446 ->
~75 ms).

Regression gate on the shipped binary (`ds4` `96f6f14801915b9eacef315340b3e502`):
Vision correct, two structured tool calls, KV reuse `cached 929/978`,
`NRestarts=0`, zero RNR/timeout/desync. DSpark remains off in the service.
