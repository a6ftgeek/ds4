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

# TIER 1 — DSpark speculative decoding at TP=4 (`feat/rocm-tp-perf`)

Worked 2026-11-25 on `feat/rocm-tp-perf` (parent `b90e0b1`), same four nodes,
`ds41f-q4`, `--transport usb4-rdma`, ranks 0-3. Outcome: the drafter is built,
verified and wired, but speculative decode is **not a win at TP=4 and stays opt-in
(`DSPARK=0`)**. Both blockers are measured below.

## 1. The artifact had to be converted, not downloaded

There is no published V4.1 DSpark file: `antirez/deepseek-v4.1-flash-gguf` ships
only the language GGUFs and the vision encoder, and its README says the weights
"are not interchangeable with DeepSeek V4 Flash weights or its DSpark drafter".
Upstream's V4.1 section (`docs/SPECULATIVE_DECODING.md`) says the support file is
built with `gguf-tools/README.md#convert-v41-dspark`; the converter step is
`gguf-tools/deepseek41_dspark.py`.

Built from the pinned source `deepseek-ai/DeepSeek-V4.1-Flash` revision
`df42c109f1defefcbfcedbe7d905718a12266e40`, using only the `mtp.*` shards
(`model-00044/45/46-of-00048`, 7.4 GiB):

| | |
|---|---|
| file | `gguf/DeepSeek-V4.1-Flash-DSpark.gguf` |
| size | 8,633,942,016 B (**8.04 GiB**, matches upstream) |
| md5 | `e1a5b6e9ea2a92f257080f970a444011` |
| audit | `--verify`: all 81 tensor layouts and payloads match the source |
| metadata | `stages=3 block=5 markov_rank=256 tensors=78 missing=0 invalid=0`, target layers 37-39 |

Copied to all four ranks with matching md5 (rsync, ~590 MB/s).

## 2. Upstream audit: the N-rank verify path already exists

At `b90e0b1` the N-rank work is present (`42809c2`, `02f98a5`): the leader drafts
locally, every worker mirrors the verify (`ds4_session_tp_spec_cycle` accepts any
rank > 0 and dispatches `ds41_session_tp_verify`), the commit frame fans out to
all peers (`tp_peer_send_frame` loops `n_peer`), and worker acks fan in with all
peers required to agree (`tp_peer_recv_frame`). A verify block is already a
**single batched forward**: `ds41_dspark_verify` runs `ds41_graph_step_batch` with
`count` rows. So Tier 1 step 3 is a *measurement* question, not a missing pass.

## 3. Memory: the resident drafter does not fit beside 1M context

Each rank must hold the drafter (the verify is mirrored), +8.10 GiB/rank:

| | no DSpark | DSpark |
|---|---:|---:|
| GTT peak (node1) | 101,447 MiB | 108,001 MiB |
| MemAvailable | 20 GiB | 14 GiB |
| planned (ctx 896K) | 99.17 GiB | 105.88 GiB |

At `CTX=1048576` the engine refuses admission: `V4.1 needs 110.18 GiB before the
expert cache; safe budget 109.44 GiB` (the 7/8-RAM rule, `ds41_memory_budget`).
1M context + vision + the 8.04 GiB drafter is 0.74 GiB over. DSpark therefore runs
at `CTX=917504` (896K), which fits with margin. This is a config constraint, not
a defect: the weights themselves fit comfortably in GTT.

## 4. Measurements (matched context 917,504, `--temp 0 -n 128`, one binary)

Alternating arms on the same four nodes; decode t/s is the server's own
`decoding … avg=` over the request.

| prompt | no DSpark | `--dspark --dspark-confidence 0.7` | delta |
|---|---:|---:|---:|
| coding (271 tok) | 5.26 | 5.11 | -2.9% |
| prose (262 tok) | 4.61 | 4.80 | +4.1% |
| tool calling (309 tok) | 3.06 | 3.46 | +13.1% |
| coding, five drafts forced (`--dspark-confidence 0`) | 5.26 | 3.12 | **-41%** |

Acceptance over 32 instrumented cycles: **proposed 160 drafts, accepted 0
(0.00%)**. Draft cost 60 ms per 5-token proposal; **verify 1.5-4.2 s per 6-row
block**, against ~78 ms for one ordinary decode token. The +4%/+13% rows are the
scheduler detecting the loss and backing off to ordinary decode, not a win.
Tool-calling output was byte-identical in both arms (`text md5 7a3279b7…`).

## 5. Root causes

**A. The drafter's proposals are unrelated to the target (the real blocker).**
A new probe (`DS4_DSPARK_SPEC_DEBUG=1`) prints the block tokens, the verifier's
argmax per row and the drafter's own argmax:

```
t0=89801  t1=14    v0=49489   draft_row0=14
t0=49489  t1=304   v0=109405  draft_row0=304
t0=109405 t1=305   v0=109405  draft_row0=305
```

The verifier is *correct*: its row-0 argmax becomes the next cycle's `t0`
(49489 → 109405 → 109405), i.e. it predicts the target's actual continuation.
The drafter proposes a different token every time (14, 304, 305, 463 …) while the
target continues 109405/127869/33372. So acceptance fails because the V4.1
drafter's proposals are not aligned with the target on this backend/topology —
not because of a verify-row offset (that hypothesis was tested and rejected).

**B. A 6-row verify costs 20-54x one decode token.** `ds41_dspark_verify` gates
the pre-posted small-row window on `g->tp_world == 2`; at world=4 the block runs
without it and every one of the 80 gates pays the bulk-prefill handshake instead
of a pre-posted ring exchange. Fix path: port
`ds4_tp_batch_block_begin_gates`/`ds4_tp_batch_block_end` to the N-rank ring.
This is worth doing only after (A), because a fast verify of rejected drafts
still loses.

## 6. Decision

`--dspark` stays **off by default**; `launch_tp4_best.sh` gains opt-in flags:

```sh
DSPARK=1 CTX=917504 DSPARK_CONF=0.7 DSPARK_STATS=1 bash launch_tp4_best.sh start
# DSPARK_EXACT=1 adds --mtp-exact-sampling; MTP_MODEL overrides the support file
```

The flags are applied to **all four ranks** (the verify is mirrored by every
rank). Baseline configuration and performance are unchanged.

# TIER 2 — short-prefill widening (kept) and a batched attention-output kernel (measured, dropped)

Worked 2026-11-25 on `feat/rocm-tp-perf` after Tier 1. Outcome: **short prompts
are 4.7-5.4x faster to prefill**; the batched 2D kernel was implemented, proved
bit-exact, showed no measurable gain, and was reverted.

## 1. The short-prefill cliff (<256 tokens)

`ds41_prefill_count` (`ds4.c`) returned a batch width of **1** whenever the
pending append was under 256 tokens, so a short prompt ran one gated pass per
row: 80 USB4 ring exchanges *per token* instead of once per chunk. The
server's 128-token mixed quantum hit this on every short agentic turn.

Resident ROCm TP now keeps short appends on the batched sweep:

```c
#if defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
    if (!g->streaming && g->tp_world >= 2 &&
        !getenv("DS4_ROCM_DISABLE_V41_TP_SMALL_PREFILL")) minimum = 8u;
#endif
```

`DS4_ROCM_DISABLE_V41_TP_SMALL_PREFILL=1` restores the old behaviour for A/B.
The threshold is 8 (not 2) so a one-token append still takes the exact
single-row path, and image prompts are untouched (they stay token-major by
design through the separate `g->image_count` gate).

### Same-binary A/B (`DS4_ROCM_DISABLE_V41_TP_SMALL_PREFILL`), one binary, one ctx

| prompt | token-major (old) | batched (new) | speedup | TTFT old -> new |
|---|---:|---:|---:|---:|
| 124 tokens | 13.66 t/s | **64.2-65.0 t/s** | **4.7x** | 9.08 s -> 1.93 s |
| 229 tokens | 12.58 t/s | **68.3 t/s** | **5.4x** | 18.21 s -> 3.36 s |

Against the pre-Tier-2 baseline recorded earlier (115 tokens 14.22 t/s, 220
tokens 12.71 t/s) the shape is identical; the A/B above is the rigorous pair
because only the env flag differs.

### Long prompts are unchanged

| context | before | after |
|---|---:|---:|
| 276 tokens | 75.97 t/s | 77.19 t/s |
| 4K (3061 tok) | 81.31 t/s | 80.44 t/s |
| 16K (11843 tok) | 127.48 t/s | 127.29 t/s |
| 64K (44781 tok) | 145.98 t/s | 144.21 t/s |

Prompts >= 256 tokens take the identical code path with an identical threshold,
so their output is byte-for-byte unchanged by construction.

### Parity of the widened path

For <256-token prompts the batched path produces **different but coherent**
greedy text than the old token-major path (124 tok: `2db15dd6` vs `16bf1de4`;
229 tok: `0f8d29ae` vs `a13af534`). That is the documented floating-point
reduction-order effect, not a defect: **TP=1 resident already used the batched
path for every append >= 8 tokens** (`minimum = 8u` on the `tp_world == 1`
branch), so before this change a 124-token prompt produced one answer at TP=4
and a different one at TP=1. The widening removes that topology-dependent
switch. Tool calling and vision work unchanged (section 3).

## 2. Batched 2D attention-output kernel — built, bit-exact, no gain, reverted

Tier 2's kernel item: `ds4_gpu_attention_output_low_q8_rows_exact_tensor`
(`rocm/ds4_rocm_attention_launch.cuh`) looped `n_rows` times over the scalar
split-K16x8 entry point, i.e. two launches plus one 256 KB staging allocation
per row, per layer. A chunked prefill therefore paid 2 x rows x 40 launches per
chunk (e.g. 3.6M launches for a 64K prefill).

Implemented: two new HIP kernels in `rocm/ds4_rocm_q8.cuh`
(`grouped_q8_0_a_partial16_w32_rows_kernel` with the row in `blockIdx.z`, and
`q8_partial_sum8_rows_kernel`), driven in tiles of 8 rows so the split-K staging
buffer stayed at 2 MB. Same split slice, same 16-block order, same warp
reduction and same 8-way partial sum as the scalar kernel, so it is bit-exact
by construction.

Measured (same binary, `DS4_ROCM_DISABLE_ATTN_OUT_ROWS_BATCH` A/B):

| context | per-row loop | batched rows | delta |
|---|---:|---:|---:|
| 124 tokens | 64.19 t/s | 64.23 t/s | +0.06% |
| 4K | 79.97 t/s | 80.44 t/s | +0.6% |
| 16K | 127.29 t/s | 125.92 t/s | -1.1% |

Bit-exactness confirmed: identical generated text both arms
(`2db15dd6` at 124 tokens, `844924a5` at 4K).

**Dropped.** Launch overhead is not on the prefill critical path: the HIP
launches are already queued asynchronously behind the GPU work, so replacing
16k launches with 2k changes nothing measurable. Keeping 106 lines of extra
kernel code for a result inside the noise band is not justified; reverted, with
the measurement recorded here.

## 3. Gate after Tier 2 (binary `ds4` md5 `cc86676a…`, identical on all four nodes)

* Text coherence: correct (`17*23` -> `391`).
* Vision PNG: "a large, solid red circle … a large, solid blue square … the text
  "CAT 347"" — all three elements correct.
* Tool calling: two structured `tool_calls` with correct names and JSON
  arguments (`read_file {"/etc/hostname"}`, `run_tests {"unit"}`).
* `NRestarts=0`, zero RNR / retry / timeout / desync / QP-error lines in the
  coordinator or any worker log.
* GTT 101,427 MiB (node1) / 100,515 / 100,515 / 100,501 MiB; MemAvailable
  20-22 GiB per node.

# TIER 3 — decode compute: audit and why no code changed

Worked 2026-11-25 on `feat/rocm-tp-perf` after Tier 2. Outcome: **nothing was
changed**, and this section records the measurements that justify it. Both of
Tier 3's items are bounded by the share of decode time that is compute at all.

## 1. The budget: decode is ring-bound, not compute-bound

The measured decode rates on this stack are 12.46 t/s at 4K and 12.09 t/s at
64K (82.6 ms/token). The two per-layer ring gates are strictly serial with the
compute (`attn out -> DS4_TP_GATE_ATTN -> FFN -> DS4_TP_GATE_FFN -> next layer`,
`ds4.c:26002` and `ds4.c:27748`), i.e. 80 round trips per token. At the
~0.9-1.1 ms per exchange measured at `b90e0b1` that is 72-88 ms of the 82.6 ms
token: the report's own A/B split stands (**~84% ring, ~16% compute+head**).

Everything Tier 3 can touch lives inside that ~16%, and host launch overhead is
a fraction of it. That is the ceiling for the whole tier: **<=1.19x decode**,
and realistically a few percent.

## 2. Per-layer HIP graph capture: the mechanism already exists, and TP disables it

There is already a decode-island graph capture in the tree
(`ds4.c:24370-24440`, "design ported from the Entrpi/ds4 batched-serving fork").
Two position-independent islands of each decode layer are captured and replayed;
a replayed graph re-runs the captured kernels byte-for-byte.

It is disabled for this deployment by construction:

```c
const bool decode_graphs_common_ok =
    !g->placement && g->tp_world <= 1 && !g->cuda_tp_decode && ...;
```

`tp_world <= 1` is not incidental. Island 0 is
`METAL_DECODE_LAYER_TO_QKV` (hc-pre -> QKV) and is gate-free, but **island 1 is
`METAL_DECODE_LAYER_FROM_ATTN_TO_FFN` (attention output -> layer end) and
contains both ring gates** (`ds4.c:26002-26056` attention-output gate,
`ds4.c:27704-27748` FFN gate). Capturing it would freeze a host-driven
collective into a device graph.

Porting this to TP=4 therefore means capturing only gate-free segments. The one
candidate is island 0 (hc-pre through QKV); the segment it covers is a minority
of the 16% compute slice, and the decode-island machinery is exactly the kind
that can freeze a pointer or a rank-dependent buffer key. The expected gain is
well under 5% of decode against a real risk of a deadlock or silent divergence
in a 4-node lockstep ring.

Decision: **not enabled**, per the Tier 3 instruction to document and drop
HIP-graph work that has no measurable gain on `gfx1151`. Measured basis: the
84/16 split above plus the fact that at TP=4 the only capturable island is the
smaller one.

## 3. MoE / router fusion: already fused for gfx1151 decode

The audit found the decode hot path is already fused and wave32-tuned:

| stage | current implementation |
|---|---|
| routed MoE gate+up+SwiGLU | `moe_gate_up_mid_decode_lut_qwarp32_kernel` and `..._ptrs_kernel` (`rocm/ds4_rocm_moe.cuh`), plus `moe_gate_up_mid_sorted_p2_qwarp32_kernel` |
| shared expert gate+up+SwiGLU | `shared_gate_up_swiglu_fuse` is on by default (`g->shared_gate_up_swiglu_fuse`, `ds4.c:19231`) |
| router / top-k | `router_select_warp_topk_kernel` (`rocm/ds4_rocm_router.cuh`), one launch per layer |
| warp shuffles | `warp_sum_f32_oldhip_w32`, `q8_0_scale_broadcast_oldhip_w32` (`rocm/ds4_rocm_q8.cuh`) |
| QKV + rope | `g->cuda_qkv_kv_rope_fuse`, `g->cuda_q_norm_rope_fuse` |
| attention-out -> HC | `metal_graph_hc_expand_fusion_eligible`; `DS4_CUDA_TP_ATTN_OUT_HC_FUSE` can also fuse the TP reduce into it |
| HC pre norm+mix | fused path in `metal_graph_encode_decode_layer_phase` |

46 fused/vectorized decode kernel sites were counted across the four files. The
gate+up+SwiGLU fusion Tier 3 asks for is the `moe_gate_up_mid_decode_lut_*`
family, which already computes gate, up and the SwiGLU product in one launch;
the router is one wave-level top-k kernel per layer. There is no unfused
intermediate left to merge without changing the reduction shape and therefore
the numerics.

Decision: **no kernel change**. Fusing further would only shave launches inside
a 16% slice that is itself dominated by two 1 ms collectives per layer.

## 4. What this means for throughput

The measured Tier 3 "no change" state is decode 12.46 t/s at 4K and 12.09 t/s at
64K, with DSpark off (Tier 1 keeps `DSPARK=0`: at TP=4 the drafter accepts 0% of
its proposals, so combining it with Tier 3 would only add the verify cost).

The single largest remaining decode lever is not in this tier: it is the 80
serial ring exchanges themselves. Merging the two gates per layer is not
possible without changing the math (the FFN input depends on the attention
output), so the realistic options are a faster link or a lower-latency
collective, not more kernel work.

# TIER 4 — USB4 ring payload compression and prefill overlap: analysis, not implemented

Worked 2026-11-25 on `feat/rocm-tp-perf` after Tier 3. Outcome: **no code
changed**. The payload arithmetic below puts the ceiling at a few percent of
prefill and zero on decode, which is below the risk of changing the wire format
of a lockstep ring; the overlap question is answered from the dependency graph
and is negative.

## 1. What the ring actually carries, and what compressing it would buy

The two exchanges are the same 4096-wide float row matrix that the model
all-reduces (`ds4_gpu_tp_gate_encode` -> `ds4_tp_ring_all_reduce` ->
`usb4_rdma_ring_all_reduce`), so per layer and chunk:

```
attn gate payload = rows x 4096 x 4 B
ffn  gate payload = rows x 4096 x 4 B
per chunk          = rows x 1.31 MB   (40 layers, 2 gates)
```

| prompt | payload per rank | ring time at ~2.5 GB/s/link | share of measured prefill |
|---|---:|---:|---:|
| 4K (3061 rows) | 4.0 GB | ~2.4 s | ~6% of 37.7 s |
| 16K (11843 rows) | 15.5 GB | ~9.3 s | ~10% of 92.9 s |
| 64K (44781 rows) | 58.7 GB | ~35 s | ~11% of 306.8 s |

The link figure is the one measured during bring-up: each USB4 path negotiates
`10.0 Gb/s x 2 lanes` without H2H (`dsbench/usb4-rdma/readiness.md:95`,
`findings.md:162`), i.e. ~2.5 GB/s per link, and a ring all-reduce moves roughly
`2 x payload x (N-1)/N` across the four links.

So BF16 payloads (2x fewer bytes) would be worth **~3% at 4K and ~5% at 64K
prefill**, and **0% on decode**: a decode gate is 20 KB and is latency-bound
(80 round trips per token), not bandwidth-bound. A 6-bit Walsh-Hadamard variant
would double that again, but only on prefill.

That is the whole upside. Against it: the gate payload is the model's partial
sum, so a lossy encoding is a numerical change in the all-reduce that has to be
verified end to end (needle recall, vision, tool calling) on every rank at every
chunk size, and the ring has no framing slack for a mixed-precision payload
without a both-ends agreement step. With Tiers 1-3 already spending the
available time and the end-to-end sweep still to run, changing the wire format
for ~3-5% of prefill was not a good trade. **Not implemented**; the measurement
above is the record of why.

## 2. Can chunked prefill compute overlap with ring transfers?

No, within a chunk, and the code makes the reason explicit. Per layer:

```
hc/attn compute -> ds4_gpu_tp_gate_encode(ATTN) -> ring all-reduce -> attn out
               -> FFN compute -> ds4_gpu_tp_gate_encode(FFN) -> ring all-reduce -> routed out
               -> next layer
```

`ds4.c:26002-26056` (attention-output gate) and `ds4.c:27704-27748` (FFN gate)
sit between the two compute blocks that consume their results, and the FFN
cannot start before the attention-output reduce lands. Layers are sequential for
the same reason. The only way to overlap would be two independent row chunks in
flight, which the ring's single in-flight collective does not permit
(`ds4_tp_batch_block_begin_gates` refuses a second window while
`r->block_active`).

The one place overlap already happens is Engram: its table reads are issued
asynchronously ahead of the layer that consumes them
(`ds4_engram_read_batch_start` / `_finish`). That is why Engram I/O is not
visible in the profile.

## 3. Accuracy

Nothing changed, so there is no accuracy delta to report for this tier. The
end-to-end gate for the shipped binary is in the wrap-up section below.

# WRAP-UP — Tier 1-4 before/after on `feat/rocm-tp-perf`

Final state 2026-11-25, branch `feat/rocm-tp-perf` (parent `b90e0b1`), four nodes,
`ds41f-q4`, TP=4 over the USB4 RDMA ring, `CTX=1048576`, vision on, DSpark off.
Binary `ds4` md5 `cc86676a131817300c2bc15aa19ca1b9`, identical on all four
ranks; `ds4-server` `693ecfd9940093d560620a66f3f8ac76`.

## What shipped

| tier | change | verdict |
|---|---|---|
| 1 | V4.1 DSpark built (8.04 GiB, 81 tensors verified) and wired to all ranks via opt-in flags | **kept off** (`DSPARK=0`): 0/160 draft acceptance, verify 1.5-4.2 s/block |
| 2a | short-prefill widening: resident ROCm TP uses the batched sweep from 8 tokens | **shipped**, 4.7-5.4x on <256-token prompts |
| 2b | bit-exact row-batched split-K attention-output kernel | **dropped**: identical output, +0.6%/-1.1% (noise) |
| 3 | decode HIP-graph islands + MoE/router fusion audit | **no change**: decode is 84% ring-bound, hot kernels already fused |
| 4 | ring payload BF16/WHT6 compression + prefill overlap | **analysed, not implemented**: ~3-5% prefill ceiling, 0% decode |

## Tier 1 — speculative decode (gated off)

| arm (ctx 917,504, n=128, temp 0) | coding | prose | tool calling |
|---|---:|---:|---:|
| no DSpark | 5.26 t/s | 4.61 t/s | 3.06 t/s |
| `--dspark --dspark-confidence 0.7` | 5.11 (-2.9%) | 4.80 (+4.1%) | 3.46 (+13.1%) |
| `--dspark-confidence 0` (five drafts forced) | 3.12 (-41%) | — | — |

Acceptance **0/160 drafts (0.00%)** over 32 instrumented cycles; draft 60 ms per
5-token proposal; verify 1.5-4.2 s per 6-row block against ~78 ms for one ordinary
decode token; GTT +6.6 GiB/rank (101,447 -> 108,001 MiB) and MemAvailable
20 -> 14 GiB. At `CTX=1048576` the resident drafter does not fit the engine's
admission budget (needs 110.18 GiB vs 109.44 GiB safe), hence 896K for the
DSpark arms. The positive prose/tool deltas are the scheduler backing off to
ordinary decode, not a speculative win.

## Tier 2 — prefill

| context | before | after | change |
|---|---:|---:|---:|
| 124 tokens | 13.66 t/s | **64.2-65.0 t/s** | **4.7x** |
| 229 tokens | 12.58 t/s | **68.3 t/s** | **5.4x** |
| 276 tokens | 75.97 t/s | 77.19 t/s | flat |
| 4K | 81.31 t/s | 80.44 t/s | flat |
| 16K | 127.48 t/s | 127.29 t/s | flat |
| 64K | 145.98 t/s | 144.21 t/s | flat |

TTFT for short turns fell 9.08 -> 1.93 s (124 tokens) and 18.21 -> 3.36 s
(229 tokens). The batched 2D kernel produced byte-identical text
(`2db15dd6`/`844924a5`) but no speedup, so it was reverted; the measurement is
in the Tier 2 section above.

## Tier 3 / Tier 4 — measured ceilings

Decode is `80 x ~0.9-1.1 ms` of ring round trips in an 82.6 ms token, i.e. ~84%
ring / ~16% compute, so every compute-side optimisation (HIP graphs, kernel
fusion) shares that 16% ceiling. The tree already contains a decode-island graph
capture, disabled at `tp_world > 1` because island 1 wraps both ring gates, and
the decode MoE/router path is already fused (`moe_gate_up_mid_decode_lut_qwarp32`,
`router_select_warp_topk`, wave32 shuffles). The ring payload is `rows x 1.31 MB`
per chunk (58.7 GB at 64K) against ~2.5 GB/s per USB4 link, so BF16 payloads
would be worth ~3% at 4K and ~5% at 64K prefill and nothing on decode, which is
latency-bound; prefill compute cannot overlap the reduces because the FFN waits
on the attention-output reduce and the ring allows one in-flight block.

## Final end-to-end verification (shipped binary)

| context | prompt tokens | prefill | TTFT | decode | needle (`ZX-4419-QQ`) |
|---|---:|---:|---:|---:|---|
| 4K | 3061 | 81.45 t/s | 37.59 s | 12.72 t/s | recalled |
| 16K | 11843 | 126.84 t/s | 93.38 s | 13.18 t/s | recalled |
| 64K | 44781 | 144.48 t/s | 309.96 s | 12.86 t/s | recalled |

* Vision PNG (red circle, blue square, "CAT 347"): **correct**, all three elements.
* OpenAI tool calling: two structured `tool_calls` with valid JSON arguments
  (`read_file {"/etc/hostname"}`, `run_tests {"suite":"unit"}`).
* Ring health: `NRestarts=0`, no RNR / retry / timeout / desync / QP-error line
  in the coordinator or any worker log for the whole final run (the only grep hit
  is the benign `gate-timeout=750ms` startup line).
* GTT 101,427 MiB (node1) / 100,515 / 100,515 / 100,501 MiB; MemAvailable
  20-22 GiB per node.

### KV prefix reuse — not reproduced (open item)

The `b90e0b1` report recorded turn-2 TTFT 40.61 -> 1.56 s with
`cached_tokens=4062` of 4078. In this final sweep **every** multi-turn probe
re-prefilled: `cached_tokens=0`, and the server logged
`live kv cache miss ... common=<one below live> ... reason=token-mismatch`
(for example `live=929 prompt=969 common=920`). `trace_cache_miss_reason()`
requires `common == old_pos` (the whole live checkpoint must be a prefix of the
incoming prompt), so a one-token divergence at the end of the cached span
rejects the whole reuse. The tree already has a trace facility for exactly this
(`trace_cache_capture`, a 17-token window around the first mismatch, written when
the server runs with `--trace`); running the service with `--trace` and following
one multi-turn request is the way to close this out. Flagged rather than fixed:
it is a pre-existing server-side behaviour, not a change from Tiers 1-4.

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

## 2026-11-25 (third session) — isolating the ROCm TP=4 DSpark defects

Continues `7cec491`. DSpark remains **off**; the shipped default path is
unchanged (the additions here are env-gated diagnostics). Final binary
`ds4` `7edb69a0fede69e82bf8cb22c44a2697`, `ds4-server` `999ba99a912e82f0dd304e3dc7894c6e`,
identical on all four ranks; service healthy, `NRestarts=0`, no ring errors.

### 1. Drafter identity (all four ranks)

`sha256 14f04ebf7314342d78921e2b9e28d8806fedc93a6551b72079fccd2f9ec5497b`,
8,633,942,016 B, `stages=3 block=5 markov_rank=256 tensors=78`, i.e. the V4.1
support file and not the older V4 0731 drafter.

### 2. What the new diagnostics showed

`DS4_DSPARK_SPEC_DEBUG=1` now prints, per block, the drafter's **base** argmax
(before the Markov bias) next to its post-bias argmax (`DSPARK_MKV`), and the
decoded text of the drafts and of the target's verified tokens.

**The draft network itself is sane.** Base logits sit at 10-13 with varied,
plausible tokens and no NaN:

```
DSPARK_MKV row=0 prev=101309 base_top=8905(12.469) post_top=8905(12.469)
DSPARK_MKV row=0 prev=36098  base_top=62838(10.949) post_top=62838(10.949)
```

**The Markov bias is oversized and overrides it.** With the bias on, the same
rows become:

```
base_top=22483(11.945) -> post_top=22663(58.462)     (+46 logits)
base_top=8905(12.469)  -> post_top=418(9.975)        (suppresses the base top)
```

A CPU reference reading `markov_w1`/`markov_w2` straight out of the GGUF
(`perf/markov_ref.py`) reproduces this exactly — bias `max=+31.57`,
`mean=-6.55`, top tokens 2..6 for `prev=101309` — so it is inherent to the
converted head, not a ROCm kernel. `DS4_DSPARK_NO_MARKOV=1` removes it; the
drafter then proposes its own distribution but acceptance stays **0**.

**The verify forward is wrong from the very first block.** Same prompt, temp 0:

| run | output |
|---|---|
| DSpark off | `We need answer?` |
| DSpark on, 6-row blocks | `Wearelli是非 candid candid …` |
| DSpark on, 2-row blocks | `Wearelli是非 candid candid …` |
| after continued use | `<｜begin▁of▁sentence｜>` repeated |

With `keep=1` the emitted token is `argmax(verified row 0)`, so the model is
emitting the verify's own logits: only the drafter-and-verify pair is involved.
The first verify already disagrees with ordinary decode (`argmax` 8905 vs the
target's 107838), and it happens with 2-row blocks too, so this is **not** the
rollback leaving uncommitted rows behind — it is the verify forward's arithmetic
at TP=4.

### 3. Why a verify block costs 1.5-4.2 s

The selector at `ds4.c:44587` enables `ds41_attention_batch` only for Apple:
on ROCm `batch_attention = false`, so a 6-row verify block runs the **whole
single-token graph once per row**. Every row therefore pays its own 80 ring
gates — 480 gates per block instead of 80 — which is the 1.5-4.2 s.

Enabling the batched path for verify blocks was tried and reverted: the block
degraded into a `<｜begin▁of▁sentence｜>` loop and the verify stayed 1.6-2.5 s,
so neither symptom is fixed by that selector alone.

### 4. Where this leaves Avenue 1

Two concrete defects are now isolated, and neither is where the earlier
hypotheses pointed:

1. **Verify-forward correctness at TP=4.** The batch is evaluated per row
   through the single-token graph but with `verify_prefix` set, so the gate
   path (`ds41_sum_partial_batch`'s slot branch) and KV publication differ from
   the prefill rows that are known to be correct. Comparing a verify block's
   row-0 logits against one ordinary decode step, per gate, is the next
   measurement.
2. **Per-row verify cost.** `ds41_attention_batch` needs to be validated for
   ROCm (it exists, but no ROCm path has ever called it), which is what turns
   480 gates into 80.

The Markov head is a third, independent defect: the converted bias is 5-10x the
base distribution's spread, which is worth revisiting against the upstream
converter's intended scaling even after the verify is fixed.

### 2026-11-25 (fourth session) — the drafter is correct; the verifier is not

Continues the Step 1/2 isolation. DSpark remains **off**; the default path is
unchanged. `ds4` `978c1fee902b88e343ead95fcd84814c`, `ds4-server`
`62350363466930ed72f124fb6f6d62c4`, identical on all four ranks, service
healthy (`NRestarts=0`, no ring errors).

#### The decisive observation

The first speculative block of a fresh session on `"1, 2, 3, 4, 5, 6, 7,"`,
`--temp 0`, confidence 0:

```
DSPARK_DBG count=6 | t0=2581 'We'  t1=1309 ' need'  t2=3287 ' answer'
DSPARK_DBG        | v0=112763 'arelli'  v1=127869 ' pronto'  | draft0=1309 ' need'
```

* `draft0 = 1309 ' need'` is **exactly** what plain decode produces after
  `We` ("We need answer?") — **the drafter's proposal is correct**.
* `v0 = 112763 'arelli'` is the verifier's row-0 argmax — **the target's own
  verification logits are wrong**.

So acceptance is 0 because the *verifier* rejects a correct draft. The drafter,
its capture/seed path and the Markov head are not the acceptance blocker (the
Markov bias does add up to +15 on the top token but it did not change the
argmax here).

The corrupted verify state is not random: it is prompt-dependent and degrades
into repetition, which is the signature of a wrong KV/position state rather than
a stale buffer:

| prompt | verify output |
|---|---|
| `1, 2, 3, 4, 5, 6, 7,` | `Wearelli是非 main oj oj constructive constructive` |
| `Write a haiku about the sea.` | `We tonal tones tonesfgfgfgfg` |
| `What is 17*23?` | `We answering answeringansw Expl expl` |

Because `keep=1` commits `argmax(verified row 0)` every cycle, the server emits
the verifier's wrong logits — which is why enabling DSpark degraded the whole
response, not just the speculative part.

#### Ruled out by direct A/B (each produced byte-identical output)

| hypothesis | test | result |
|---|---|---|
| verify gate exchange (slot vs bulk) | route world>2 verify through `ds4_gpu_tp_big_gate_encode` | identical |
| attention selector (per-row vs batched) | `batch_attention` enabled for verify | **worse**: BOS loop, verify still 1.6-2.5 s |
| N-rank batch pipeline / mid-loop device sync | `DS4_METAL_DISABLE_V41_BATCH_DECODE_PIPELINE=1` | identical |
| batched output head (only verify uses it; prefill heads one row per-row) | `DS4_METAL_DISABLE_V41_BATCH_HEAD=1` | identical |
| stale/uninitialised buffer | three different prompts | prompt-dependent, so no |
| draft broadcast divergence / expert sharding | earlier sessions | ruled out / fixed |

#### Verify cost, restated

`batch_attention` is Apple-only (`ds4.c:44587`), so a 6-row verify runs the whole
single-token graph once per row: **480 ring gates per block instead of 80**.
Enablement attempts above did not reduce the time, so the per-row path is doing
more than the gate count suggests (each row also re-runs the layer's
`ds41_before_attention`/publish work).

#### Next step, precisely

The verifier's row-0 logits must be compared against one ordinary decode step at
the same position, with per-stage dumps (post-attention residual, post-FFN
residual, pre-head norm). Everything above is consistent with the verify rows
being evaluated against a KV window that differs from the decode window; that
comparison is the shortest path to the fix, and it no longer needs the drafter in
the loop at all — the drafter already proposes the right token.

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
