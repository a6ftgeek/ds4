# DS4 prefill row split: 2 ranks -> N ranks (TP=4 report)

Branch `feat/rocm-tp-ring`, this commit (parent `318f399`).
Four Strix Halo nodes (gfx1151), one 81 GiB
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
per node, ranks 0-3 on node1-4, greedy `--temp 0 -n 16 -c 2048`, 369-token
prompt, `--transport usb4-rdma` unless noted.

## What changed

The prefill row split of the replicated shared expert was hard-wired to two
ranks: `tp_world == 2`, rank 0 got `(n+1)/2` rows, rank 1 the rest. At TP=4
every rank therefore evaluated the full shared expert for every chunk row.

`ds4.c`:

* `metal_graph_tp_row_partition()` (`ds4.c:31034`) - balanced, disjoint N-way
  partition: `base = n/world`, `rem = n%world`, rank `r` owns
  `base + (r < rem)` rows starting at `r*base + min(r, rem)`.
* `tp_row_split_ffn` (`ds4.c:33524`) - gate is now `tp_world >= 2 &&
  n_tokens >= tp_world`, split is computed by the helper (`ds4.c:33531`).
* `metal_graph_tp_prefill_ffn_split_min()` (`ds4.c:31067`) - the FFN split has
  its own threshold so it no longer inherits the ROCm clamp that exists for the
  attention row split (whose row-range attention kernels are CUDA/Metal-only;
  see the audit below). `DS4_TP_PREFILL_SPLIT_MIN` overrides both thresholds.

At `world == 2` the partition is exactly the old one: for odd `n` rank 0 gets
`n/2 + 1 == (n+1)/2` rows and rank 1 starts at `(n+1)/2`; for even `n` it is
`n/2` / `n/2`. A standalone check over `n = 2..20000` and `world = 2..8` verified
that the ranges are contiguous, disjoint, complete, and that `world == 2`
reproduces `(n+1)/2`.

## Why the exchange stays correct: it is an all-reduce, not a gather

The bulk exchange is `ds4_gpu_tp_big_gate_encode(il, n_tokens, routed_out,
ffn_out, bytes)` followed by `routed_out = own + peer`, where the transport
returns, for `world > 2`, `ring_all_reduce(own) - own` (SUM over the full
payload, then subtract this rank's contribution). That is an **all-reduce of
full-row routed partials**: every rank holds a real partial for every row
(because each rank evaluates all rows against its resident expert slice), and
the sum is the full routed output.

The shared-expert term is different: it is computed by one rank per row. It is
folded into the matching rows of that rank's routed partial before the
exchange (`ds4.c:33938`). For a row owned by rank `r`, the reduce therefore adds
the shared term exactly once (from `r`) and zeros from the other ranks' shared
work, because they never contributed it for that row. A naive "sum of N partials
with no ownership" would multiply the shared expert by N; that is the trap this
design avoids, and it is why the fold uses a row-range view at `tp_row0`.

Consequences that are handled explicitly:

* `n_tokens % world != 0` - the remainder rows go to the lowest ranks, so no row
  is dropped or double-owned. The canonical prompt is 369 tokens = 4x92+1, so
  rank 0 owns 93 rows and ranks 1-3 own 92 in every TP=4 run below.
* `n_tokens < world` - the split is disabled by `n_tokens >= tp_world`; every
  rank then runs the replicated shared expert and the shared term is added after
  the exchange on the non-split path, which is also correct. No rank ever gets a
  zero-row range, and no zero-length matmul/add/exchange is encoded.
* Transport - the payload is the full `n_tokens * 4096` float matrix over
  `ds4_tp_ring_all_reduce` (arbitrary count, no single-token buffers) with the
  TCP-ring fallback; no direct peer assumption. Both transports are exercised
  below.

## Audit of the other prefill `tp_world == 2` sites

| site (ds4.c) | what it does | decision |
|---|---|---|
| `31425` `tp_row_split_attn` | attention chunk row split for large zero-prefix chunks | **left 2-rank-only.** Its exchange is a pairwise half swap of disjoint row ranges, not a reduce; at N ranks that would need a gather (and a reduce only if non-owned rows are zero-filled). It also needs the row-range prefill attention kernels, which do not exist on ROCm (`ds4_rocm_unavailable.cu:15` stubs `ds4_gpu_attention_prefill_raw_heads_range_tensor`), so the ROCm clamp stays on this threshold. Not trivially safe. |
| `31379` `tp_head_split` | verify-block head split + attention-output K-slice | **left 2-rank-only.** Requires `tp_verify_heads`, which is CUDA/Metal-only, and `tp_batch_rows`, which is 0 whenever `tp_world != 2` (`39467`). Dead at N=4, and the K-slice kernel is a ROCm stub. |
| `33776` `tp_split_batch_moe` | verify-block routed exchange | **left 2-rank-only.** Same `tp_batch_rows == n_tokens` precondition, which is 0 at N=4. Dead at N=4. |
| `33511` `tp_split_ffn` | enables the bulk FFN exchange | already `tp_world >= 2`; unchanged (the exchange itself is world-generic). |
| `34286`, `34291` logits half / readback | two-way vocab split at logits readback | **left 2-rank-only.** Decode/logits readback, not prefill; `tp_logits_half` is never set at world 4. |
| `24179`, `24304`, `24867`, `25793`, `26363`, `26864`, `27317`, `27730`, `27904`, `30262-30415` | decode/streaming token path | out of scope; TP=4 decode is verified working (9.4-9.7 t/s) and does not take these paths. |
| `42036-42058`, `42471-42674` | DeepSeek V4.1 family (`ds41_*`) graph | not compiled into this checkpoint's path (it is V4 Flash). These sites already use `tp_world >= 2` and `il % tp_world` owner rotation, so they are not 2-rank-hardcoded. |
| `51640`, `52908`, `53405`, `53562` | GLM graph prefill | different model family; not reachable with the 0731 checkpoint. GLM's TP prefill still uses its 50/50 pairwise bounce and was not touched. |

Attention (q path, KV path, attention itself, output projection) stays
replicated at N=4: the attention head split is out of scope, and the attention
row split is blocked by the missing ROCm row-range kernels. See the timing
breakdown for what that costs.

## Verification

Builds (inside `toolbox run -c vllm-gfx1151`):

* `make strix-halo -j32` - **0 warnings, 0 errors**.
* `USE_USB4_RDMA=1 make strix-halo -j32` - **0 warnings, 0 errors**.
  Final `ds4` md5 `4d1aeb64f6643845d573e4e9d7888c12` (36,535,528 bytes), pushed to
  node2/3/4; md5 equals on all four nodes.

Tests: `make test-tp-ring` - **4/4 PASS, 0 failures** (4 ranks on loopback:
`all_reduce` count 0/1/1000003, `all_gather` 777, u32 4099).

Parity, TP=4, both transports, split on and off - greedy 16-token text
byte-identical to the TP=1 reference:

```
The user has provided a long, reflective essay about the history of computing, focusing
```

md5 `a13ba035b8347fa92e602fd8f751ca91` in all of: TP=1, TP=2 (split on/off),
TP=4 USB4 (split on/off), TP=4 TCP (split on/off).

## Performance

### Same-binary interleaved A/B (isolates this patch)

TP=4, USB4 ring, 6 pairs alternating split-off / split-on, `--tp-cpu-core 2`:

| arm | prefill t/s (6 runs) | generation t/s | GTT peak/rank |
|---|---:|---:|---:|
| before (split off = the pre-patch path) | 86.60 | 9.43 | 37.82 GiB |
| after (split on) | 86.56 | 9.37 | 37.82 GiB |
| delta | -0.05% | -0.6% (noise) | 0 |

Earlier in the same session, at lower GPU clocks: off 75.60 / on 78.20
(+3.4%, 2 pairs). TP=4 TCP, 2 interleaved pairs: off 73.58 / on 73.93 (+0.5%).

So the generalized split is correctness- and memory-neutral and worth at most a
few percent of prefill: the shared expert is only 2.7% of TP=4 prefill and is
weight-bandwidth-bound (its Q8 weights are replicated, so splitting rows does
not reduce the bytes each rank reads). The profile below shows
`shared_gate_up` actually getting slightly slower at 92 rows/rank (44 -> 55 ms)
while `shared_down` improves (66 -> 35 ms).

### Like-for-like TP=1 / TP=2 / TP=4, same session, warm

Interleaved rounds, same prompt, ctx, flags:

| config | prefill t/s | generation t/s | GTT peak/rank | resident/rank |
|---|---:|---:|---:|---:|
| TP=1 | 83.96 (83.91/83.67/84.30) | 16.95 | 87.0 GiB | 80.76 GiB |
| TP=2 USB4, core 2 | 82.32 (82.12/82.52) | 10.21 | 55.96 GiB | 44.48 GiB |
| TP=2 USB4, core 8 | 81.88 | 10.23 | 55.96 GiB | 44.48 GiB |
| TP=4 USB4, core 2 | 88.73 (88.54/88.91) | 9.51 | 37.82 GiB | 26.34 GiB |
| TP=4 TCP, core 2 | 73.93 (73.90/73.95) | 9.19 | 37.82 GiB | 26.34 GiB |

TP=4 prefill is **7.8% faster** than TP=2 here, and TP=4 decode is 6.8% slower -
the expected trade (the 4-node ring all-reduce costs more than the extra expert
split saves in tokens/s, and prefill is bounded by replicated attention work).
There is no TP=4 prefill regression to fix in this session.

### Why the published `45.33 vs 93.84` pair does not reproduce

Prefill on this prompt/host is dominated by the GPU clock ramp, not by the
build. The same binary and flags, TP=4 USB4, run back to back after an idle
period:

| time | prefill t/s |
|---|---:|
| 05:55 (first run after 11 h idle) | 54.05 |
| 06:02 | 71.88 |
| 06:03 / 06:04 | 75.12 / 76.07 |
| 06:07-06:11 (6 runs) | 85.36 - 88.43 |
| 06:17-06:20 (2 runs) | 88.54 / 88.91 |

That is +64% from warm-up alone. The parent-table `45.33` TP=4 and `93.84` TP=2
were taken at different clock states (and the TP=2 number with
`--tp-cpu-core 8`); a same-session cold pair was 54.05 (TP=4) vs 54.31 (TP=2).
Decode is the stable metric: TP=2 10.21-10.23, TP=4 9.37-9.62, TP=1 16.94-16.96,
in every arm.

### Where the TP=4 prefill time goes

`DS4_ROCM_LAYER_STAGE_PROFILE=1` on rank 0, 43 layers, 369-token chunk
(profile total 4099 ms vs 4158 ms unprofiled):

| stage | ms | share |
|---|---:|---:|
| attn/output_proj (replicated, all ranks, all rows) | 2508.7 | 61.2% |
| ffn/hc_post (contains the 503 ms FFN collective + HC expand) | 607.8 | 14.8% |
| ffn/routed_moe (expert-sliced) | 341.0 | 8.3% |
| attn/q_path | 146.6 | 3.6% |
| attn/attention | 131.6 | 3.2% |
| attn/hc_pre + ffn/hc_pre | 121.8 | 3.0% |
| ffn/shared_gate_up + shared_down | 109.8 | 2.7% |
| attn/indexer_setup + compressor_prefill + misc | ~116 | 2.8% |
| ffn/router | 10.4 | 0.3% |

Collective (from `DS4_TP_XCHG_STATS=1`): 43 payload exchanges of 6,045,696 B
each. TP=2: 7.79 ms each = 335 ms (7.4% of 4.51 s). TP=4 USB4: 11.70 ms each =
503 ms (12.1% of 4.16 s), 4.1 Gbps effective. TP=4 TCP is slower per exchange,
which is why TCP prefill is 73.9 vs 88.7 t/s.

The bottleneck is the **replicated attention output projection**: ~50 GFLOP per
layer per rank (A: 369x8x4096x1024, B: 369x8192x4096) that every rank recomputes
for every row. At 2.51 s it is 4x less efficient than the routed-MoE kernels and
is untouched by any FFN row split. The next real TP=4 prefill win is to row-split
that projection (or its K-slice) across ranks, which needs a gather (or a
zero-fill + all-reduce) and is deliberately out of this change's scope; the
attention head split stays replicated.

## Caveats

* Prefill t/s is only comparable within a session / within minutes. Always
  report decode and GTT with it.
* The FFN row split is correct and cheap to keep, but on this hardware it is not
  a prefill lever (weight-bandwidth-bound shared expert, 2.7% of the time).
* The `DS4_TP_PREFILL_SPLIT_MIN=1000000000` / split-off arm used for the A/B is
  the exact pre-patch path at TP=4 (the old gate was `tp_world == 2`), so the
  "before" rows above are the old behavior without rebuilding the old binary.

## Commit

`feat/rocm-tp-ring`, one commit on top of `318f399`:

```
git log --oneline feat/usb4-rdma-ring..feat/rocm-tp-ring
<this commit> tp: generalize the prefill shared-expert row split to N ranks
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
```
