# DS4 feature status under 4-node TP=4 over the USB4 RDMA ring

Verified 2026-11-24 on `feat/rocm-tp-ring` @ `79a9247`, four nodes
(192.168.1.101-104, switchless USB4 ring, RoCE v2 GID index 1), one 81 GiB
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
per host, ranks 0-3 on node1-4, `--tp-cpu-core 2`.

| feature | status | evidence | blocker |
|---|---|---|---|
| TP=4 USB4 RDMA ring | **verified** | 64.25 t/s prefill, 9.65 t/s decode, GTT 37.82 GiB/rank at ctx 2048; text md5 `a13ba035b8347fa92e602fd8f751ca91` == TP=1 | none |
| 1M context (compressed KV) | **verified** | `-c 1048576`: KV 13.79 GiB + 8.00 GiB buffers + 26.34 GiB resident = 48.13 GiB/rank, GTT 47.88 GiB, MemAvailable >= 42.85 GiB; 30,474-token story answered `Bob=34, Alice=52, Clara=71` | none |
| Vision (`--vision`) | **blocked** | all four ranks exit 1 with the rejection below; GTT stays 16.4 MiB | `ds4.c:74106`; `ds4.c:74034`; `ds4.c:6707`; `ds4.c:7458` |
| DSpark / MTP (`--dspark --mtp-model`) | **blocked at N>2** | world=4: ranks 2/3 die with `tp: spec cycle outside worker mode`; single-host ROCm works but is slower (13.67 vs 16.18 t/s) | `ds4.c:85410`; `ds4.c:75261`; `ds4.c:80511` |

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
`stages=3 block=5 markov_rank=256 tensors=81 missing=0 invalid=0`).

Single host (TP=1, ROCm, ctx 2048, greedy, n=64, same prompt):

| | decode t/s | DSpark stats |
|---|---:|---|
| no DSpark | 16.18 | -- |
| `--dspark --mtp-model ...` | 13.67 | cycles=34 proposed=15 accepted_draft=13 accept_rate=86.67% avg_accept=0.382 scheduler_skips=18 |

So the drafter is compatible with ROCm and accepts most of what it proposes,
but on this short generation the proposal and verify overhead outweighs the
savings (net -15.5%). `docs/SPECULATIVE_DECODING.md` already warns that this
can happen; Do not treat 13.67 as a TP=4 number, it is single-host.

Under TP=4 the same command fails. On ranks 2 and 3:

```
tp worker verify: tp: spec cycle outside worker mode
```

The cause is structural, not a tuning problem:

* `ds4_session_tp_spec_cycle` rejects every rank except 1:
  `if (!e || !e->tp.active || e->tp.rank != 1)` -> `ds4.c:85409-85412`.
  At world=4 the workers are ranks 1, 2, 3, so two of them abort at the first
  speculative verify cycle, close the ring, and the leader then fails the next
  collective with an RDMA transport-retry error.
* The worker verify path additionally requires the lockstep drafter split
  (`ds4.c:85450`), and the draft-frame handler is compiled out for ROCm:
  `ds4.c:85628-85640` ends in `tp: the lockstep DSpark drafter is not compiled
  in`.
* That split is only built for non-ROCm targets (`ds4.c:75261-75272`), and
  `ds4_session_dspark_tp_step` is under
  `#if ... && !defined(DS4_ROCM_BUILD)` (`ds4.c:80498`). On this build
  `e->tp.dspark_split` can never become true, so the 2-rank lockstep path is
  unavailable as well; the current worker protocol is hard-wired to exactly
  one worker at rank 1.

Closing the gap needs the speculative verify/commit protocol generalized to N
ranks (rank set instead of `rank == 1`, per-rank proposals or a broadcast
proposal), and a ROCm build of the drafter split. Until then `--dspark` must
stay out of the 4-rank launch script.
