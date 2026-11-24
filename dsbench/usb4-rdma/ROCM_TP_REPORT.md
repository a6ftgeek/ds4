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
