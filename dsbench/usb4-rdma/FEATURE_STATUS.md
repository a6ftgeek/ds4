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

Inter-node SSH is broken (`Permission denied (publickey)`), so always fan out from the
Windows host with `ssh nodeN`. Write scripts locally, `scp` them, and run
`bash /tmp/script.sh`. `sudo` is passwordless on node1 (used only for read-only inspection here).

### API endpoint

- Local (on node1): **`http://127.0.0.1:8000/v1`** -- OpenAI-compatible
  (`/v1/models`, `/v1/chat/completions`), served model id `deepseek-v4.1-flash`,
  `context_length: 1048576`, `max_completion_tokens: 256`.
- LAN: **not reachable as configured.** `ds4-server` is started with `--host 127.0.0.1`
  and listens on `127.0.0.1:8000` only. See "LAN exposure and services" below.

### LAN exposure and services (read-only findings, 2026-11-24)

- `ds4-server` listens on `127.0.0.1:8000` only (`--host 127.0.0.1`), so the API is
  **not reachable from another machine** today. The LAN URL would be
  `http://192.168.1.101:8000/v1` if it were started with `--host 0.0.0.0` (or
  `--host 192.168.1.101`).
- `firewalld` is running; default zone `FedoraServer` (interfaces `enp191s0`, `tbv0`)
  allows services `cockpit dhcpv6-client ssh`. **Port 8000/tcp is not open.** Exposing
  the API would need `firewall-cmd --add-port=8000/tcp` (not done).
- Cockpit is installed (`cockpit-368-1.fc45`); `cockpit.socket` is enabled and active on
  `*:9090`, and the `cockpit` service is allowed through the firewall, so
  **https://192.168.1.101:9090** is reachable from the LAN.
- No "Halostrix"/AI-cockpit **service registry** exists on node1: no systemd unit, no
  registry directory under `/etc` or `~/dev`. The only AI-cockpit artifact is the
  pipx-installed `ai-toolbox-cockpit` TUI (v2026.11.22.1536) with
  `~/.config/ai-toolbox-cockpit/config.json` = `{"active_platform": "strix-halo"}`.

### Start (coordinator first, then the three workers)

```bash
# node1 -- coordinator, rank 0 (keeps /tmp/start_coord.sh intact)
ssh -o BatchMode=yes node1 'CTX=1048576 VISION=1 nohup bash /tmp/start_coord.sh >/tmp/srv_final.log 2>&1 &'

# node2 -- worker rank 1
ssh -o BatchMode=yes node2 'nohup bash /tmp/start_worker.sh 1 >/tmp/srv_r1.log 2>&1 &'

# node3 -- worker rank 2
ssh -o BatchMode=yes node3 'nohup bash /tmp/start_worker.sh 2 >/tmp/srv_r2.log 2>&1 &'

# node4 -- worker rank 3
ssh -o BatchMode=yes node4 'nohup bash /tmp/start_worker.sh 3 >/tmp/srv_r3.log 2>&1 &'
```

`/tmp/start_coord.sh` and `/tmp/start_worker.sh` must be preserved -- the server cannot be
relaunched without them. Both are plain `toolbox run -c vllm-gfx1151 -- ./ds4-server|./ds4`
wrappers over `$HOME/dev/ds4-v41-pp/src`; env knobs are `CTX`, `VISION=1`, `EXTRA`,
`ENVV`.

### Stop (SIGTERM; workers first, then the coordinator)

```bash
ssh -o BatchMode=yes node2 'pkill -TERM -f "role worker"'
ssh -o BatchMode=yes node3 'pkill -TERM -f "role worker"'
ssh -o BatchMode=yes node4 'pkill -TERM -f "role worker"'
ssh -o BatchMode=yes node1 'pkill -TERM -f "role coordinator"'
```

This was **not** executed during the handoff: the 4-node server must stay running.

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
