/* tests/test_tp_ring.c — weight-free test of the N-rank ring collective seam.
 *
 * Exercises ds4_tp_ring_all_reduce()/all_gather() with the shapes the TP path
 * actually produces: arbitrary element counts that are not multiples of the
 * world size, unaligned buffers, zero-length calls, and multi-megabyte prefill
 * payloads. No model weights and no engine are needed, so this runs on a
 * cluster that has no GGUF at all.
 *
 * Usage:
 *   ./tests/test_tp_ring <rank> <world> <ip0,ip1,...[,ipN]> <base_port> [chunk]
 *
 * Every rank must be started with the same world/port/chunk. Rank r listens on
 * base_port + 100 + r, so a single host can self-test with all-127.0.0.1.
 * Exit status 0 means every check passed on this rank.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4_tp_ring.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int split_ips(const char *in, const char **out, int max) {
    int n = 0;
    char *copy = strdup(in);
    if (!copy) return 0;
    char *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        if (n >= max) { n = -1; break; }
        out[n++] = tok;
    }
    /* The token storage must outlive the ring, so hand back interned copies. */
    return n;
}

/* Element-wise sum is exact for the small integers used here, so equality (not
 * tolerance) is the right check: a wrong reduction order over integers still
 * has to land on the same value. */
static int check_f32_sum(const float *buf, size_t count, int world) {
    const float want = (float)(world * (world + 1) / 2);
    for (size_t i = 0; i < count; i++) {
        if (buf[i] != want) {
            fprintf(stderr, "  MISMATCH at %zu: got %.9g want %.9g\n", i,
                    (double)buf[i], (double)want);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <rank> <world> <ip0,ip1,...> <base_port> [chunk]\n",
                argv[0]);
        return 2;
    }
    const int rank = atoi(argv[1]);
    const int world = atoi(argv[2]);
    const int port = atoi(argv[4]);
    const size_t chunk = argc > 5 ? (size_t)strtoull(argv[5], NULL, 10) : 0;

    static const char *ips[DS4_TP_RING_MAX_WORLD];
    const int nips = split_ips(argv[3], ips, DS4_TP_RING_MAX_WORLD);
    if (nips != world) {
        fprintf(stderr, "rank %d: got %d control addresses for world %d\n", rank,
                nips, world);
        return 2;
    }

    ds4_tp_ring_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.rank = rank;
    opt.world = world;
    for (int i = 0; i < world; i++) opt.control_ip[i] = ips[i];
    opt.port = port;
    opt.gid_index = -1;
    opt.chunk_bytes = chunk;
    opt.cpu_core = -1;
    opt.verbose = 1;

    ds4_tp_ring *ring = NULL;
    char err[256] = "";
    if (ds4_tp_ring_create(&ring, &opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "rank %d: create failed: %s\n", rank, err);
        return 1;
    }
    printf("rank %d/%d backend=%s\n", rank, world,
           ds4_tp_ring_backend_name(ds4_tp_ring_get_backend(ring)));
    fflush(stdout);

    int failures = 0;

    /* 1. Unaligned, prime-sized, not a multiple of world. */
    {
        const size_t count = 1000003u;
        unsigned char *raw = malloc(count * sizeof(float) + 3u);
        if (!raw) return 1;
        float *buf = (float *)(void *)(raw + 1);
        for (size_t i = 0; i < count; i++) buf[i] = (float)(rank + 1);
        if (ds4_tp_ring_all_reduce(ring, buf, count, DS4_TP_RING_DTYPE_F32,
                                   DS4_TP_RING_OP_SUM, err, sizeof(err)) != 0) {
            fprintf(stderr, "  all_reduce(prime, unaligned) failed: %s\n", err);
            failures++;
        } else if (check_f32_sum(buf, count, world) != 0) {
            failures++;
        } else {
            printf("  ok: f32 all_reduce count=%zu (unaligned, %s)\n", count,
                   count % (size_t)world ? "not a world multiple" : "aligned size");
        }
        free(raw);
    }

    /* 2. Single element, and zero elements (must be a no-op success). */
    {
        float one = (float)(rank + 1);
        if (ds4_tp_ring_all_reduce(ring, &one, 1, DS4_TP_RING_DTYPE_F32,
                                   DS4_TP_RING_OP_SUM, err, sizeof(err)) != 0 ||
            one != (float)(world * (world + 1) / 2)) {
            fprintf(stderr, "  all_reduce(count=1) failed: %s got %.9g\n", err,
                    (double)one);
            failures++;
        } else {
            printf("  ok: f32 all_reduce count=1\n");
        }
        if (ds4_tp_ring_all_reduce(ring, NULL, 0, DS4_TP_RING_DTYPE_F32,
                                   DS4_TP_RING_OP_SUM, err, sizeof(err)) != 0) {
            fprintf(stderr, "  all_reduce(count=0) failed: %s\n", err);
            failures++;
        } else {
            printf("  ok: f32 all_reduce count=0 is a no-op\n");
        }
    }

    /* 3. all_gather: every rank must end up with all world copies in rank
     *    order, using a segment layout that is not a multiple of world. */
    {
        const size_t count = 777u;
        float *send = malloc(count * sizeof(float));
        float *recv = malloc(count * (size_t)world * sizeof(float));
        if (!send || !recv) return 1;
        for (size_t i = 0; i < count; i++) send[i] = (float)(rank + 1);
        if (ds4_tp_ring_all_gather(ring, send, recv, count,
                                   DS4_TP_RING_DTYPE_F32, err, sizeof(err)) != 0) {
            fprintf(stderr, "  all_gather failed: %s\n", err);
            failures++;
        } else {
            int bad = 0;
            for (int r = 0; r < world && !bad; r++)
                for (size_t i = 0; i < count; i++)
                    if (recv[(size_t)r * count + i] != (float)(r + 1)) {
                        fprintf(stderr, "  all_gather MISMATCH rank %d at %zu: %.9g\n",
                                r, i, (double)recv[(size_t)r * count + i]);
                        bad = 1;
                        break;
                    }
            if (bad) failures++;
            else printf("  ok: all_gather count=%zu world=%d\n", count, world);
        }
        free(send);
        free(recv);
    }

    /* 4. Integer dtype, so the seam is shown not to be float-only. */
    {
        const size_t count = 4099u;
        uint32_t *buf = malloc(count * sizeof(uint32_t));
        if (!buf) return 1;
        for (size_t i = 0; i < count; i++) buf[i] = (uint32_t)(rank + 1);
        const uint32_t want = (uint32_t)(world * (world + 1) / 2);
        if (ds4_tp_ring_all_reduce(ring, buf, count, DS4_TP_RING_DTYPE_U32,
                                   DS4_TP_RING_OP_SUM, err, sizeof(err)) != 0) {
            fprintf(stderr, "  u32 all_reduce failed: %s\n", err);
            failures++;
        } else {
            int bad = 0;
            for (size_t i = 0; i < count; i++)
                if (buf[i] != want) { bad = 1; break; }
            if (bad) { fprintf(stderr, "  u32 all_reduce MISMATCH\n"); failures++; }
            else printf("  ok: u32 all_reduce count=%zu\n", count);
        }
        free(buf);
    }

    /* 5. Timing sweep over the prefill/decode payload range. Reported, never
     *    used as a pass/fail gate: a slow link is not a correctness failure. */
    static const size_t sizes[] = {4u << 10, 16u << 10, 64u << 10, 256u << 10,
                                   1u << 20};
    const int iters = 20;
    printf("rank %d: all_reduce latency (%d iterations each)\n", rank, iters);
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        const size_t bytes = sizes[s];
        const size_t count = bytes / sizeof(float);
        float *buf = malloc(bytes);
        if (!buf) return 1;
        for (size_t i = 0; i < count; i++) buf[i] = (float)(rank + 1);
        /* One unmeasured round: the ring is already warm, but the first touch
         * of a fresh buffer is not representative. */
        (void)ds4_tp_ring_all_reduce(ring, buf, count, DS4_TP_RING_DTYPE_F32,
                                     DS4_TP_RING_OP_SUM, err, sizeof(err));
        double t0 = now_sec(), worst = 0;
        for (int i = 0; i < iters; i++) {
            double a = now_sec();
            if (ds4_tp_ring_all_reduce(ring, buf, count, DS4_TP_RING_DTYPE_F32,
                                       DS4_TP_RING_OP_SUM, err, sizeof(err)) != 0) {
                fprintf(stderr, "  timed all_reduce failed: %s\n", err);
                failures++;
                break;
            }
            double el = now_sec() - a;
            if (el > worst) worst = el;
        }
        const double dt = (now_sec() - t0) / iters;
        printf("  %6zu B count=%6zu: %.3f ms  %.2f Gbps  max %.3f ms\n", bytes,
               count, dt * 1e3, (double)bytes * 8.0 / dt / 1e9, worst * 1e3);
        free(buf);
    }

    /* 6. Idle-separated collectives. Real inference puts GPU compute between
     *    consecutive all-reduces, so the ring goes cold between calls. A stall
     *    bound by the QP ACK timeout shows up here and nowhere in the
     *    back-to-back sweep above, which is the whole point of this block. */
    {
        const size_t count = (256u << 10) / sizeof(float);
        float *buf = malloc(count * sizeof(float));
        if (!buf) return 1;
        for (size_t i = 0; i < count; i++) buf[i] = (float)(rank + 1);
        const int rounds = 10;
        /* Gap between collectives, in ms. 5 ms is the shape of a layer; larger
         * values test whether a genuinely cold Thunderbolt path behaves
         * differently. Override with TEST_RING_GAP_MS. */
        long gap_ms = 5;
        {
            const char *g = getenv("TEST_RING_GAP_MS");
            if (g && g[0]) {
                char *end = NULL;
                long v = strtol(g, &end, 10);
                if (end != g && *end == '\0' && v >= 0 && v <= 5000) gap_ms = v;
            }
        }
        double warm_sum = 0, warm_worst = 0, back_sum = 0, back_worst = 0;
        for (int i = 0; i < rounds; i++) {
            struct timespec gap = {.tv_sec = gap_ms / 1000,
                                   .tv_nsec = (gap_ms % 1000) * 1000 * 1000};
            nanosleep(&gap, NULL);
            const double a = now_sec();
            if (ds4_tp_ring_all_reduce(ring, buf, count, DS4_TP_RING_DTYPE_F32,
                                       DS4_TP_RING_OP_SUM, err, sizeof(err)) != 0) {
                fprintf(stderr, "  separated all_reduce failed: %s\n", err);
                failures++;
                break;
            }
            const double el = now_sec() - a;
            warm_sum += el;
            if (el > warm_worst) warm_worst = el;
        }
        for (int i = 0; i < rounds; i++) {
            const double a = now_sec();
            if (ds4_tp_ring_all_reduce(ring, buf, count, DS4_TP_RING_DTYPE_F32,
                                       DS4_TP_RING_OP_SUM, err, sizeof(err)) != 0) {
                failures++;
                break;
            }
            const double el = now_sec() - a;
            back_sum += el;
            if (el > back_worst) back_worst = el;
        }
        printf("  256 KB all_reduce: idle-separated mean %.3f ms max %.3f ms | "
               "back-to-back mean %.3f ms max %.3f ms (%d rounds each)\n",
               warm_sum / rounds * 1e3, warm_worst * 1e3,
               back_sum / rounds * 1e3, back_worst * 1e3, rounds);
        free(buf);
    }

    ds4_tp_ring_destroy(ring);
    printf("rank %d: %s (%d failure%s)\n", rank, failures ? "FAIL" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
