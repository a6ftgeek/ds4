/*
 * usb4_rdma_ring.c — implementation of the USB4 RDMA ring collective API.
 *
 * Data plane from the known-good ring_allreduce.c prototype: pure
 * IBV_WR_RDMA_WRITE, payload + trailing 64-bit marker in the same logical
 * message, receiver busy-polls the marker, responder posts no receive WQEs.
 *
 * Arbitrary sizes: each call is processed in ring passes. Within a pass the
 * payload is split into balanced per-rank segments (remainder to the lowest
 * ranks), each segment is padded to seg_elems internally, and one
 * reduce-scatter + all-gather ring pass produces the result. The trailer
 * sequence number is strictly monotonic across passes and calls.
 *
 * Device selection:
 *   - explicit params->right_dev / left_dev, or
 *   - sysfs discovery: /sys/class/infiniband/<ibdev>/device -> PCI function ->
 *     <pci>/domainN -> /sys/bus/thunderbolt/devices/<domainN>-<port> (an
 *     XDomain, identified by its rx_speed attribute) -> device_name (peer
 *     hostname) / unique_id.  Peer hostname is matched to a rank through
 *     params->peer_name[] or the reverse DNS of params->control_ip[].
 *
 * GID selection: params->gid_index, or the first non-zero IPv4-mapped
 * (::ffff:a.b.c.d) GID in the device's port-1 GID table (RoCE v2).
 *
 * Diagnostics-only env knobs:
 *   RING_TB_IGNORE=dev[,dev]   skip devices (used by the no-device self-test)
 *   RING_TB_UID_MAP=path       "<unique_id> <name>" fallback when device_name
 *                              is empty
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "usb4_rdma_ring.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define CHUNK_ALIGN 4096u
#define NSLOT 8
#define MARKER_SZ 8
#define MAX_XFER_SOFT (256u << 10)
#define MAX_HCA 32
#define PATHBUF (PATH_MAX + 512)

#define TB_DEV_DIR "/sys/bus/thunderbolt/devices"
#define IB_CLASS_DIR "/sys/class/infiniband"

/* Local ACK timeout is 4.096 us * 2^timeout. The IB default of 14 is 67 ms,
 * which is far more tolerance than a point-to-point Thunderbolt cable needs,
 * and a cold path pays it as a single ~67 ms stall (the driver logs "native
 * retransmit reason=1") whenever two collectives are separated by real work --
 * exactly the shape of a TP data path, where GPU compute sits between
 * all-reduces. 8 is 1.05 ms. USB4_RING_QP_TIMEOUT overrides the value so the
 * two settings can be compared on the same hardware. */
static int usb4_ring_qp_ack_timeout(void) {
    const char *v = getenv("USB4_RING_QP_TIMEOUT");
    if (v && v[0]) {
        char *end = NULL;
        long t = strtol(v, &end, 10);
        if (end != v && *end == '\0' && t >= 0 && t <= 31) return (int)t;
    }
    return 8;
}

struct peerinfo {
    uint32_t qpn;
    uint32_t rkey;
    uint64_t addr;
    uint8_t gid[16];
};

struct usb4_rdma_ring {
    int rank, world, right, left;
    int port, verbose, connect_timeout_ms;
    char control_ip[USB4_RDMA_RING_MAX_WORLD][64];
    char peer_name[USB4_RDMA_RING_MAX_WORLD][USB4_RDMA_RING_PEER_NAME_MAX];
    int have_peer_name[USB4_RDMA_RING_MAX_WORLD];

    char right_dev[USB4_RDMA_RING_DEV_NAME_MAX];
    char left_dev[USB4_RDMA_RING_DEV_NAME_MAX];
    char right_peer[USB4_RDMA_RING_PEER_NAME_MAX];
    char left_peer[USB4_RDMA_RING_PEER_NAME_MAX];
    int discovered;

    struct ibv_context *rc, *lc;
    struct ibv_pd *rpd, *lpd;
    struct ibv_cq *ocq, *icq;
    struct ibv_qp *out, *in;
    struct ibv_mr *smr, *rmr;

    char *sbuf, *rbuf;
    size_t chunk_bytes;     /* effective per-rank payload chunk (8-aligned) */
    size_t stride;          /* chunk_bytes + MARKER_SZ */
    size_t max_xfer;
    size_t sbuf_sz, rbuf_sz;

    int right_gid_index, left_gid_index;
    union ibv_gid right_gid, left_gid;
    enum ibv_mtu mtu;

    struct peerinfo right_in, left_out;
    uint64_t seq;           /* strictly monotonic across every ring step */

    int lfd, send_fd, acc_fd;

    cpu_set_t saved_affinity;
    int have_saved_affinity;
    int pinned_core;
};

/* ------------------------------------------------------------------ */
/* logging / small helpers                                            */
/* ------------------------------------------------------------------ */

static void rerr(int rank, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "[usb4_rdma_ring rank %d] ERROR: ", rank);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

static void rlog(int rank, int verbose, const char *fmt, ...) {
    if (!verbose) return;
    va_list ap;
    fprintf(stderr, "[usb4_rdma_ring rank %d] ", rank);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

/* CPU pause hint for busy-poll loops (x86 PAUSE, portable fallback). */
static inline void cpu_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__)
    __builtin_ia32_pause();
#else
    __asm__ __volatile__("pause");
#endif
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

static void copy_str(char *dst, size_t n, const char *src) {
    if (n == 0) return;
    if (!src) { dst[0] = '\0'; return; }
    size_t l = strlen(src);
    if (l >= n) l = n - 1;
    memcpy(dst, src, l);
    dst[l] = '\0';
}

/* ------------------------------------------------------------------ */
/* dtype arithmetic                                                   */
/* ------------------------------------------------------------------ */

static size_t dtype_size(enum usb4_rdma_ring_dtype d) {
    switch (d) {
    case USB4_RDMA_RING_DTYPE_U8:
    case USB4_RDMA_RING_DTYPE_I8: return 1;
    case USB4_RDMA_RING_DTYPE_U16:
    case USB4_RDMA_RING_DTYPE_I16:
    case USB4_RDMA_RING_DTYPE_F16:
    case USB4_RDMA_RING_DTYPE_BF16: return 2;
    case USB4_RDMA_RING_DTYPE_U32:
    case USB4_RDMA_RING_DTYPE_I32:
    case USB4_RDMA_RING_DTYPE_F32: return 4;
    case USB4_RDMA_RING_DTYPE_U64:
    case USB4_RDMA_RING_DTYPE_I64:
    case USB4_RDMA_RING_DTYPE_F64: return 8;
    default: return 0;
    }
}

static float f16_to_f32(uint16_t h) {
    uint32_t h_exp = (h & 0x7c00u) >> 10;
    uint32_t h_sig = h & 0x03ffu;
    uint32_t f_sgn = (uint32_t)(h & 0x8000u) << 16;
    uint32_t f_exp, f_sig;
    if (h_exp == 0) {
        if (h_sig == 0) { f_exp = 0; f_sig = 0; }
        else {
            uint32_t e = 127 - 15 + 1;
            while (!(h_sig & 0x0400u)) { h_sig <<= 1; e--; }
            h_sig &= 0x03ffu;
            f_exp = e << 23;
            f_sig = h_sig << 13;
        }
    } else if (h_exp == 0x1f) {
        f_exp = 0xffu << 23;
        f_sig = h_sig << 13;
    } else {
        f_exp = (h_exp + (127 - 15)) << 23;
        f_sig = h_sig << 13;
    }
    uint32_t bits = f_sgn | f_exp | f_sig;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t f_sgn = (x >> 16) & 0x8000u;
    uint32_t f_exp = (x >> 23) & 0xffu;
    uint32_t f_sig = x & 0x7fffffu;
    if (f_exp == 0xff)
        return (uint16_t)(f_sgn | 0x7c00u | (f_sig ? 0x0200u : 0));
    int32_t exp = (int32_t)f_exp - 127 + 15;
    if (exp >= 0x1f) return (uint16_t)(f_sgn | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)f_sgn;
        f_sig |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t sig = f_sig >> shift;
        uint32_t rem = f_sig & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (sig & 1))) sig++;
        return (uint16_t)(f_sgn | sig);
    }
    uint32_t sig = f_sig >> 13;
    uint32_t rem = f_sig & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (sig & 1))) {
        sig++;
        if (sig == 0x400u) {
            sig = 0;
            exp++;
            if (exp >= 0x1f) return (uint16_t)(f_sgn | 0x7c00u);
        }
    }
    return (uint16_t)(f_sgn | ((uint16_t)exp << 10) | sig);
}

static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu))
        return (uint16_t)((u >> 16) | 1u);           /* keep NaN a NaN */
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;                              /* round to nearest even */
    return (uint16_t)(u >> 16);
}

#define REDUCE_UNSIGNED(T)                                                     \
    do {                                                                       \
        T *d = (T *)dst;                                                       \
        const T *s = (const T *)src;                                           \
        for (size_t i = 0; i < n; i++) {                                       \
            if (op == USB4_RDMA_RING_OP_SUM) d[i] = (T)(d[i] + s[i]);           \
            else if (op == USB4_RDMA_RING_OP_MAX) { if (s[i] > d[i]) d[i] = s[i]; } \
            else { if (s[i] < d[i]) d[i] = s[i]; }                              \
        }                                                                      \
    } while (0)

#define REDUCE_SIGNED(T, U)                                                    \
    do {                                                                       \
        T *d = (T *)dst;                                                       \
        const T *s = (const T *)src;                                           \
        for (size_t i = 0; i < n; i++) {                                       \
            if (op == USB4_RDMA_RING_OP_SUM) d[i] = (T)((U)d[i] + (U)s[i]);     \
            else if (op == USB4_RDMA_RING_OP_MAX) { if (s[i] > d[i]) d[i] = s[i]; } \
            else { if (s[i] < d[i]) d[i] = s[i]; }                              \
        }                                                                      \
    } while (0)

#define REDUCE_FLOAT(T)                                                        \
    do {                                                                       \
        T *d = (T *)dst;                                                       \
        const T *s = (const T *)src;                                           \
        for (size_t i = 0; i < n; i++) {                                       \
            if (op == USB4_RDMA_RING_OP_SUM) d[i] = (T)(d[i] + s[i]);           \
            else if (op == USB4_RDMA_RING_OP_MAX) { if (s[i] > d[i]) d[i] = s[i]; } \
            else { if (s[i] < d[i]) d[i] = s[i]; }                              \
        }                                                                      \
    } while (0)

/* dst = op(dst, src), n elements. Returns 0 on success, -1 if unsupported. */
static int reduce_elem(enum usb4_rdma_ring_dtype dt, enum usb4_rdma_ring_op op,
                       void *dst, const void *src, size_t n) {
    if (op != USB4_RDMA_RING_OP_SUM && op != USB4_RDMA_RING_OP_MAX &&
        op != USB4_RDMA_RING_OP_MIN)
        return -1;
    switch (dt) {
    case USB4_RDMA_RING_DTYPE_U8: REDUCE_UNSIGNED(uint8_t); break;
    case USB4_RDMA_RING_DTYPE_U16: REDUCE_UNSIGNED(uint16_t); break;
    case USB4_RDMA_RING_DTYPE_U32: REDUCE_UNSIGNED(uint32_t); break;
    case USB4_RDMA_RING_DTYPE_U64: REDUCE_UNSIGNED(uint64_t); break;
    case USB4_RDMA_RING_DTYPE_I8: REDUCE_SIGNED(int8_t, uint8_t); break;
    case USB4_RDMA_RING_DTYPE_I16: REDUCE_SIGNED(int16_t, uint16_t); break;
    case USB4_RDMA_RING_DTYPE_I32: REDUCE_SIGNED(int32_t, uint32_t); break;
    case USB4_RDMA_RING_DTYPE_I64: REDUCE_SIGNED(int64_t, uint64_t); break;
    case USB4_RDMA_RING_DTYPE_F32: REDUCE_FLOAT(float); break;
    case USB4_RDMA_RING_DTYPE_F64: REDUCE_FLOAT(double); break;
    case USB4_RDMA_RING_DTYPE_F16: {
        uint16_t *d = (uint16_t *)dst;
        const uint16_t *s = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            float a = f16_to_f32(d[i]), b = f16_to_f32(s[i]), r;
            if (op == USB4_RDMA_RING_OP_SUM) r = a + b;
            else if (op == USB4_RDMA_RING_OP_MAX) r = b > a ? b : a;
            else r = b < a ? b : a;
            d[i] = f32_to_f16(r);
        }
        break;
    }
    case USB4_RDMA_RING_DTYPE_BF16: {
        uint16_t *d = (uint16_t *)dst;
        const uint16_t *s = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            float a = bf16_to_f32(d[i]), b = bf16_to_f32(s[i]), r;
            if (op == USB4_RDMA_RING_OP_SUM) r = a + b;
            else if (op == USB4_RDMA_RING_OP_MAX) r = b > a ? b : a;
            else r = b < a ? b : a;
            d[i] = f32_to_bf16(r);
        }
        break;
    }
    default:
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* sysfs peer discovery                                               */
/* ------------------------------------------------------------------ */

struct hca {
    char dev[USB4_RDMA_RING_DEV_NAME_MAX];
    char pci[PATH_MAX];
    char domain[64];
    char domainpath[PATHBUF];
    char xdomain[64];
    char peer[USB4_RDMA_RING_PEER_NAME_MAX];
    char uid[USB4_RDMA_RING_PEER_NAME_MAX];
};

static int read_trim(const char *path, char *buf, size_t n) {
    if (n == 0) return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t got = fread(buf, 1, n - 1, f);
    int bad = ferror(f);
    fclose(f);
    if (bad) return -1;
    buf[got] = '\0';
    size_t len = strnlen(buf, got);
    buf[len] = '\0';
    while (len > 0 && isspace((unsigned char)buf[len - 1])) buf[--len] = '\0';
    return 0;
}

static int find_domain_dir(const char *pci, char *out, size_t n) {
    DIR *d = opendir(pci);
    if (!d) return -1;
    struct dirent *e;
    int found = 0;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "domain", 6) != 0) continue;
        const char *p = e->d_name + 6;
        if (*p == '\0') continue;
        int digits = 1;
        for (const char *q = p; *q; q++)
            if (!isdigit((unsigned char)*q)) digits = 0;
        if (!digits) continue;
        if (found) { closedir(d); return -2; }
        snprintf(out, n, "%s/%s", pci, e->d_name);
        found = 1;
    }
    closedir(d);
    return found ? 0 : -1;
}

static int find_xdomain(const char *domdir, char *xname, size_t xn,
                        char *peer, size_t pn, char *uid, size_t un) {
    DIR *d = opendir(TB_DEV_DIR);
    if (!d) return -1;
    size_t dl = strlen(domdir);
    char best[256] = "";
    int best_depth = 1 << 30;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char path[PATHBUF], real[PATH_MAX];
        snprintf(path, sizeof path, "%s/%s", TB_DEV_DIR, e->d_name);
        if (!realpath(path, real)) continue;
        if (strncmp(real, domdir, dl) != 0 || real[dl] != '/') continue;
        int dfd = open(path, O_RDONLY | O_DIRECTORY);
        if (dfd < 0) continue;
        int is_xdomain = (faccessat(dfd, "rx_speed", R_OK, 0) == 0);
        close(dfd);
        if (!is_xdomain) continue;
        int depth = 0;
        for (const char *q = real + dl + 1; *q; q++)
            if (*q == '/') depth++;
        if (depth < best_depth) {
            best_depth = depth;
            snprintf(best, sizeof best, "%s", e->d_name);
        }
    }
    closedir(d);
    if (best[0] == '\0') return -1;
    snprintf(xname, xn, "%s", best);

    char path[PATHBUF];
    snprintf(path, sizeof path, "%s/%s/device_name", TB_DEV_DIR, best);
    if (read_trim(path, peer, pn) != 0) peer[0] = '\0';
    snprintf(path, sizeof path, "%s/%s/unique_id", TB_DEV_DIR, best);
    if (read_trim(path, uid, un) != 0) uid[0] = '\0';
    return 0;
}

static void first_label(const char *in, char *out, size_t n) {
    size_t i = 0;
    if (!in) { out[0] = '\0'; return; }
    while (in[i] && in[i] != '.' && i + 1 < n) {
        out[i] = (char)tolower((unsigned char)in[i]);
        i++;
    }
    out[i] = '\0';
}

static int peer_name_matches(const char *peer, const char *want) {
    char a[USB4_RDMA_RING_PEER_NAME_MAX], b[USB4_RDMA_RING_PEER_NAME_MAX];
    first_label(peer, a, sizeof a);
    first_label(want, b, sizeof b);
    return a[0] != '\0' && b[0] != '\0' && strcmp(a, b) == 0;
}

static int ignored_dev(const char *name) {
    const char *env = getenv("RING_TB_IGNORE");
    if (!env || *env == '\0') return 0;
    char buf[512];
    snprintf(buf, sizeof buf, "%s", env);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save))
        if (strcmp(tok, name) == 0) return 1;
    return 0;
}

static int uid_map_lookup(const char *uid, char *out, size_t n) {
    const char *path = getenv("RING_TB_UID_MAP");
    if (!path || *path == '\0' || uid[0] == '\0') return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    int rc = -1;
    while (fgets(line, sizeof line, f)) {
        char u[USB4_RDMA_RING_PEER_NAME_MAX], nm[USB4_RDMA_RING_PEER_NAME_MAX];
        if (sscanf(line, "%127s %127s", u, nm) == 2 && strcmp(u, uid) == 0) {
            copy_str(out, n, nm);
            rc = 0;
            break;
        }
    }
    fclose(f);
    return rc;
}

static int expected_peer(const struct usb4_rdma_ring_params *p, int rank,
                         char *out, size_t n) {
    if (p->peer_name[rank] && p->peer_name[rank][0]) {
        copy_str(out, n, p->peer_name[rank]);
        return 0;
    }
    const char *ip = p->control_ip[rank];
    if (!ip || !*ip) return -1;
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    if (inet_pton(AF_INET, ip, &sa.sin_addr) != 1) return -1;
    char host[NI_MAXHOST];
    if (getnameinfo((struct sockaddr *)&sa, sizeof sa, host, sizeof host,
                    NULL, 0, NI_NAMEREQD) != 0)
        return -1;
    copy_str(out, n, host);
    return 0;
}

static void dump_hcas(int rank, const struct hca *h, int nh) {
    rerr(rank, "local USB4 RDMA HCAs and their Thunderbolt peers:\n");
    for (int i = 0; i < nh; i++)
        fprintf(stderr, "    %-14s pci=%s %s/%s peer=%-12s uid=%s\n",
                h[i].dev, h[i].pci, h[i].domain, h[i].xdomain,
                h[i].peer[0] ? h[i].peer : "<unknown>",
                h[i].uid[0] ? h[i].uid : "<none>");
}

static int discover_devices(const struct usb4_rdma_ring_params *p, int rank, int world,
                            char *rdev, size_t rn, char *ldev, size_t ln,
                            char *rpeer, size_t rpn, char *lpeer, size_t lpn) {
    int right = (rank + 1) % world, left = (rank - 1 + world) % world;
    char want_r[USB4_RDMA_RING_PEER_NAME_MAX], want_l[USB4_RDMA_RING_PEER_NAME_MAX];
    if (expected_peer(p, right, want_r, sizeof want_r) != 0 ||
        expected_peer(p, left, want_l, sizeof want_l) != 0) {
        rerr(rank, "cannot derive peer names: set params->peer_name[] or make "
                   "control_ip[] reverse-resolvable via DNS/hosts\n");
        return USB4_RDMA_RING_ERR_DISCOVERY;
    }

    int ndev = 0;
    struct ibv_device **list = ibv_get_device_list(&ndev);
    if (!list) {
        rerr(rank, "ibv_get_device_list failed: %s\n", strerror(errno));
        return USB4_RDMA_RING_ERR_SYS;
    }

    struct hca h[MAX_HCA];
    int nh = 0;
    for (int i = 0; i < ndev && nh < MAX_HCA; i++) {
        char name[USB4_RDMA_RING_DEV_NAME_MAX];
        snprintf(name, sizeof name, "%s", ibv_get_device_name(list[i]));
        if (ignored_dev(name)) {
            rlog(rank, p->verbose, "ignoring %s (RING_TB_IGNORE)\n", name);
            continue;
        }
        char link[PATHBUF], pci[PATH_MAX], dom[PATHBUF];
        snprintf(link, sizeof link, "%s/%s/device", IB_CLASS_DIR, name);
        if (!realpath(link, pci)) continue;
        if (find_domain_dir(pci, dom, sizeof dom) != 0) continue;

        struct hca *x = &h[nh];
        memset(x, 0, sizeof *x);
        snprintf(x->dev, sizeof x->dev, "%s", name);
        snprintf(x->pci, sizeof x->pci, "%s", pci);
        const char *base = strrchr(dom, '/');
        base = base ? base + 1 : dom;
        copy_str(x->domain, sizeof x->domain, base);
        snprintf(x->domainpath, sizeof x->domainpath, "%s", dom);
        if (find_xdomain(dom, x->xdomain, sizeof x->xdomain,
                         x->peer, sizeof x->peer, x->uid, sizeof x->uid) != 0)
            rlog(rank, p->verbose, "%s: %s has no attached XDomain\n", x->dev, x->domain);
        if (x->peer[0] == '\0') {
            char mapped[USB4_RDMA_RING_PEER_NAME_MAX];
            if (uid_map_lookup(x->uid, mapped, sizeof mapped) == 0)
                copy_str(x->peer, sizeof x->peer, mapped);
        }
        nh++;
    }
    ibv_free_device_list(list);

    if (nh == 0) {
        rerr(rank, "no USB4 RDMA HCA found (%d other verbs device(s)); "
                   "this host cannot use the USB4 ring transport\n", ndev);
        return USB4_RDMA_RING_ERR_NO_DEVICE;
    }

    int ri = -1, li = -1;
    for (int i = 0; i < nh; i++) {
        if (h[i].peer[0] == '\0') continue;
        if (peer_name_matches(h[i].peer, want_r)) {
            if (ri >= 0) {
                rerr(rank, "ambiguous right peer %s: both %s and %s report it\n",
                     want_r, h[ri].dev, h[i].dev);
                return USB4_RDMA_RING_ERR_DISCOVERY;
            }
            ri = i;
        }
    }
    if (right != left) {
        for (int i = 0; i < nh; i++) {
            if (h[i].peer[0] == '\0') continue;
            if (peer_name_matches(h[i].peer, want_l)) {
                if (li >= 0) {
                    rerr(rank, "ambiguous left peer %s: both %s and %s report it\n",
                         want_l, h[li].dev, h[i].dev);
                    return USB4_RDMA_RING_ERR_DISCOVERY;
                }
                li = i;
            }
        }
    } else {
        li = ri;
    }

    if (ri < 0 || li < 0) {
        if (ri < 0)
            rerr(rank, "no local HCA faces right rank %d (peer %s)\n", right, want_r);
        if (li < 0)
            rerr(rank, "no local HCA faces left rank %d (peer %s)\n", left, want_l);
        dump_hcas(rank, h, nh);
        return USB4_RDMA_RING_ERR_DISCOVERY;
    }

    copy_str(rdev, rn, h[ri].dev);
    copy_str(ldev, ln, h[li].dev);
    copy_str(rpeer, rpn, h[ri].peer);
    copy_str(lpeer, lpn, h[li].peer);
    rlog(rank, p->verbose,
         "discovered right=rank%d dev=%s peer=%s (%s/%s) left=rank%d dev=%s peer=%s (%s/%s)\n",
         right, h[ri].dev, h[ri].peer, h[ri].domain, h[ri].xdomain,
         left, h[li].dev, h[li].peer, h[li].domain, h[li].xdomain);
    return USB4_RDMA_RING_OK;
}

/* ------------------------------------------------------------------ */
/* GID helpers                                                        */
/* ------------------------------------------------------------------ */

static int gid_is_ipv4_mapped(const uint8_t *g) {
    static const uint8_t prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    if (memcmp(g, prefix, 12) != 0) return 0;
    return !(g[12] == 0 && g[13] == 0 && g[14] == 0 && g[15] == 0);
}

static int resolve_gid(struct ibv_context *ctx, int explicit_idx,
                       int *out_idx, union ibv_gid *out_gid) {
    struct ibv_port_attr pa;
    if (ibv_query_port(ctx, 1, &pa) != 0) return -1;
    if (explicit_idx >= 0) {
        if (explicit_idx >= pa.gid_tbl_len) return -1;
        if (ibv_query_gid(ctx, 1, explicit_idx, out_gid) != 0) return -1;
        *out_idx = explicit_idx;
        return 0;
    }
    for (int i = 0; i < pa.gid_tbl_len; i++) {
        union ibv_gid g;
        if (ibv_query_gid(ctx, 1, i, &g) != 0) continue;
        if (gid_is_ipv4_mapped(g.raw)) {
            *out_gid = g;
            *out_idx = i;
            return 0;
        }
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* TCP out-of-band handshake                                          */
/* ------------------------------------------------------------------ */

static void set_sock_timeout(int fd, int ms) {
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
}

static int tcp_listen(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    if (s < 0) return -1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons((uint16_t)port);
    if (bind(s, (void *)&a, sizeof a) || listen(s, 4)) {
        close(s);
        return -1;
    }
    return s;
}

static int accept_timeout(int lfd, int timeout_ms) {
    long deadline = now_ms() + timeout_ms;
    for (;;) {
        long left = deadline - now_ms();
        if (left <= 0) return -1;
        struct pollfd pfd = {.fd = lfd, .events = POLLIN};
        int pr = poll(&pfd, 1, (int)(left > 200 ? 200 : left));
        if (pr > 0 && (pfd.revents & POLLIN))
            return accept(lfd, NULL, NULL);
        if (pr < 0 && errno != EINTR) return -1;
    }
}

static int tcp_connect(const char *ip, int port, int timeout_ms) {
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) return -1;

    long deadline = now_ms() + timeout_ms;
    while (now_ms() < deadline) {
        int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
        if (s < 0) return -1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int fl = fcntl(s, F_GETFL, 0);
        fcntl(s, F_SETFL, fl | O_NONBLOCK);
        int rc = connect(s, (void *)&a, sizeof a);
        if (rc == 0) {
            fcntl(s, F_SETFL, fl);
            set_sock_timeout(s, timeout_ms);
            return s;
        }
        if (errno == EINPROGRESS) {
            struct pollfd pfd = {.fd = s, .events = POLLOUT};
            int pr = poll(&pfd, 1, 200);
            if (pr > 0) {
                int err = 0;
                socklen_t el = sizeof err;
                if (getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &el) == 0 && err == 0) {
                    fcntl(s, F_SETFL, fl);
                    set_sock_timeout(s, timeout_ms);
                    return s;
                }
            }
        }
        close(s);
        usleep(200000);
    }
    return -1;
}

static int xfer(int fd, void *p, size_t n, int send_it) {
    size_t off = 0;
    while (off < n) {
        ssize_t r = send_it ? send(fd, (char *)p + off, n - off, 0)
                            : recv(fd, (char *)p + off, n - off, 0);
        if (r <= 0) return -1;
        off += (size_t)r;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* verbs helpers                                                      */
/* ------------------------------------------------------------------ */

static int wait_cq(struct ibv_cq *cq, const char *what) {
    struct ibv_wc wc;
    double deadline = now_s() + 5.0;
    long spins = 0;
    for (;;) {
        int n = ibv_poll_cq(cq, 1, &wc);
        if (n < 0) {
            fprintf(stderr, "[usb4_rdma_ring] poll error: %s\n", what);
            return USB4_RDMA_RING_ERR_SYS;
        }
        if (n == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "[usb4_rdma_ring] completion error: %s: %s (opcode %d)\n",
                        what, ibv_wc_status_str(wc.status), wc.opcode);
                return USB4_RDMA_RING_ERR_CORRUPT;
            }
            return USB4_RDMA_RING_OK;
        }
        cpu_pause();
        if ((++spins & 4095) == 0 && now_s() > deadline) {
            fprintf(stderr, "[usb4_rdma_ring] TIMEOUT waiting for completion: %s\n", what);
            return USB4_RDMA_RING_ERR_TIMEOUT;
        }
    }
}

static int wait_marker(char *slot, size_t marker_off, uint64_t seq, const char *what) {
    volatile uint64_t *m = (volatile uint64_t *)(slot + marker_off);
    double deadline = now_s() + 5.0;
    long spins = 0;
    for (;;) {
        if (*m == seq) {
            __sync_synchronize();
            return USB4_RDMA_RING_OK;
        }
        cpu_pause();
        if ((++spins & 4095) == 0 && now_s() > deadline) {
            fprintf(stderr, "[usb4_rdma_ring] TIMEOUT waiting for marker: %s "
                            "(want %lu got %lu)\n",
                    what, (unsigned long)seq, (unsigned long)*m);
            return USB4_RDMA_RING_ERR_TIMEOUT;
        }
    }
}

/* One logical message = payload + gap + 8-byte marker at marker_off. */
static int post_write(struct usb4_rdma_ring *r, uint64_t remote_addr, uint32_t rkey,
                      char *src, size_t total, uint64_t seq) {
    size_t off = 0;
    while (off < total) {
        size_t n = total - off;
        if (n > r->max_xfer) n = r->max_xfer;
        int last = (off + n == total);
        struct ibv_sge sge = {(uintptr_t)(src + off), (uint32_t)n, r->smr->lkey};
        struct ibv_send_wr wr = {
            .wr_id = seq, .sg_list = &sge, .num_sge = 1,
            .opcode = IBV_WR_RDMA_WRITE,
            .send_flags = last ? IBV_SEND_SIGNALED : 0,
        }, *bad = NULL;
        wr.wr.rdma.remote_addr = remote_addr + off;
        wr.wr.rdma.rkey = rkey;
        if (ibv_post_send(r->out, &wr, &bad) != 0) {
            rerr(r->rank, "ibv_post_send failed at offset %zu\n", off);
            return USB4_RDMA_RING_ERR_SYS;
        }
        off += n;
    }
    return wait_cq(r->ocq, "rdma_write");
}

/* ------------------------------------------------------------------ */
/* teardown / CPU pinning                                             */
/* ------------------------------------------------------------------ */

static void ring_free(struct usb4_rdma_ring *r) {
    if (!r) return;
    if (r->out) ibv_destroy_qp(r->out);
    if (r->in) ibv_destroy_qp(r->in);
    if (r->ocq) ibv_destroy_cq(r->ocq);
    if (r->icq) ibv_destroy_cq(r->icq);
    if (r->smr) ibv_dereg_mr(r->smr);
    if (r->rmr) ibv_dereg_mr(r->rmr);
    if (r->rpd) ibv_dealloc_pd(r->rpd);
    if (r->lpd) ibv_dealloc_pd(r->lpd);
    if (r->rc) ibv_close_device(r->rc);
    if (r->lc) ibv_close_device(r->lc);
    if (r->lfd >= 0) close(r->lfd);
    if (r->send_fd >= 0) close(r->send_fd);
    if (r->acc_fd >= 0) close(r->acc_fd);
    free(r->sbuf);
    free(r->rbuf);
    free(r);
}

int usb4_rdma_ring_destroy(struct usb4_rdma_ring *ring) {
    ring_free(ring);
    return USB4_RDMA_RING_OK;
}

const char *usb4_rdma_ring_strerror(int status) {
    switch (status) {
    case USB4_RDMA_RING_OK: return "success";
    case USB4_RDMA_RING_ERR_INVAL: return "invalid parameter";
    case USB4_RDMA_RING_ERR_NO_DEVICE: return "no USB4 RDMA HCA present";
    case USB4_RDMA_RING_ERR_DISCOVERY: return "HCA peer discovery failed";
    case USB4_RDMA_RING_ERR_SYS: return "verbs/socket/memory failure";
    case USB4_RDMA_RING_ERR_TIMEOUT: return "timeout during handshake or completion";
    case USB4_RDMA_RING_ERR_CORRUPT: return "unexpected completion/marker state";
    default: return "unknown status";
    }
}

static int apply_cpu(struct usb4_rdma_ring *r, int core, int verbose) {
    if (core < 0) {
        if (r->have_saved_affinity &&
            sched_setaffinity(0, sizeof r->saved_affinity, &r->saved_affinity) != 0)
            return USB4_RDMA_RING_ERR_SYS;
        r->pinned_core = -1;
        return USB4_RDMA_RING_OK;
    }
    if (core >= CPU_SETSIZE) return USB4_RDMA_RING_ERR_INVAL;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) {
        rerr(r->rank, "sched_setaffinity(cpu %d) failed: %s\n", core, strerror(errno));
        return USB4_RDMA_RING_ERR_SYS;
    }
    r->pinned_core = core;
    rlog(r->rank, verbose, "pinned to CPU %d\n", core);
    return USB4_RDMA_RING_OK;
}

int usb4_rdma_ring_set_cpu(struct usb4_rdma_ring *ring, int core) {
    if (!ring) return USB4_RDMA_RING_ERR_INVAL;
    return apply_cpu(ring, core, ring->verbose);
}

/* ------------------------------------------------------------------ */
/* init                                                               */
/* ------------------------------------------------------------------ */

static int qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr at;
    memset(&at, 0, sizeof at);
    at.qp_state = IBV_QPS_INIT;
    at.pkey_index = 0;
    at.port_num = 1;
    at.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    return ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                  IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int qp_to_rtr(struct ibv_qp *qp, uint32_t dest_qpn, const uint8_t *dgid,
                     int sgid_index, enum ibv_mtu mtu) {
    struct ibv_qp_attr at;
    memset(&at, 0, sizeof at);
    at.qp_state = IBV_QPS_RTR;
    at.path_mtu = mtu;
    at.dest_qp_num = dest_qpn;
    at.rq_psn = 0;
    at.max_dest_rd_atomic = 1;
    at.min_rnr_timer = 12;
    at.ah_attr.is_global = 1;
    at.ah_attr.port_num = 1;
    memcpy(at.ah_attr.grh.dgid.raw, dgid, 16);
    at.ah_attr.grh.sgid_index = (uint8_t)sgid_index;
    at.ah_attr.grh.hop_limit = 64;
    return ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                  IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                  IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr at;
    memset(&at, 0, sizeof at);
    at.qp_state = IBV_QPS_RTS;
    at.timeout = (uint8_t)usb4_ring_qp_ack_timeout();
    at.retry_cnt = 7;
    at.rnr_retry = 7;
    at.sq_psn = 0;
    at.max_rd_atomic = 1;
    return ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                  IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                  IBV_QP_MAX_QP_RD_ATOMIC);
}

int usb4_rdma_ring_init(struct usb4_rdma_ring **ring_out,
                        const struct usb4_rdma_ring_params *p,
                        struct usb4_rdma_ring_info *info) {
    if (!ring_out) return USB4_RDMA_RING_ERR_INVAL;
    *ring_out = NULL;
    if (info) memset(info, 0, sizeof *info);
    if (!p) return USB4_RDMA_RING_ERR_INVAL;

    if (p->world < 2 || p->world > USB4_RDMA_RING_MAX_WORLD) return USB4_RDMA_RING_ERR_INVAL;
    if (p->rank < 0 || p->rank >= p->world) return USB4_RDMA_RING_ERR_INVAL;
    if (p->port <= 0 || p->port > 65000) return USB4_RDMA_RING_ERR_INVAL;

    size_t chunk_req = p->chunk_bytes ? p->chunk_bytes : USB4_RDMA_RING_DEFAULT_CHUNK_BYTES;
    size_t chunk = chunk_req & ~(size_t)7;            /* marker needs 8-byte alignment */
    if (chunk < 8) return USB4_RDMA_RING_ERR_INVAL;
    if (chunk > (SIZE_MAX - MARKER_SZ) / (USB4_RDMA_RING_MAX_WORLD + NSLOT))
        return USB4_RDMA_RING_ERR_INVAL;

    struct usb4_rdma_ring *r = calloc(1, sizeof *r);
    if (!r) return USB4_RDMA_RING_ERR_SYS;
    r->rank = p->rank;
    r->world = p->world;
    r->right = (p->rank + 1) % p->world;
    r->left = (p->rank - 1 + p->world) % p->world;
    r->port = p->port;
    r->verbose = p->verbose;
    r->connect_timeout_ms = p->connect_timeout_ms > 0 ? p->connect_timeout_ms : 60000;
    r->lfd = r->send_fd = r->acc_fd = -1;
    r->chunk_bytes = chunk;
    r->stride = chunk + MARKER_SZ;

    for (int i = 0; i < p->world; i++) {
        if (!p->control_ip[i] || !p->control_ip[i][0]) {
            rerr(r->rank, "params->control_ip[%d] is required\n", i);
            ring_free(r);
            return USB4_RDMA_RING_ERR_INVAL;
        }
        snprintf(r->control_ip[i], sizeof r->control_ip[i], "%s", p->control_ip[i]);
        if (p->peer_name[i] && p->peer_name[i][0]) {
            copy_str(r->peer_name[i], sizeof r->peer_name[i], p->peer_name[i]);
            r->have_peer_name[i] = 1;
        }
    }

    /* Optional pinning; disabled by default (p->cpu_core = -1). */
    r->have_saved_affinity =
        (sched_getaffinity(0, sizeof r->saved_affinity, &r->saved_affinity) == 0);
    if (p->pin_cpu) {
        int st = apply_cpu(r, p->cpu_core, r->verbose);
        if (st != USB4_RDMA_RING_OK) {
            ring_free(r);
            return st;
        }
    }

    /* ---------- device selection ---------- */
    int r_auto = (!p->right_dev || !p->right_dev[0] || !strcmp(p->right_dev, "auto"));
    int l_auto = (!p->left_dev || !p->left_dev[0] || !strcmp(p->left_dev, "auto"));
    if (r_auto != l_auto) {
        rerr(r->rank, "right_dev and left_dev must both be set or both be \"auto\"\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_INVAL;
    }

    int st;
    if (r_auto) {
        st = discover_devices(p, r->rank, r->world,
                              r->right_dev, sizeof r->right_dev,
                              r->left_dev, sizeof r->left_dev,
                              r->right_peer, sizeof r->right_peer,
                              r->left_peer, sizeof r->left_peer);
        if (st != USB4_RDMA_RING_OK) {
            ring_free(r);
            return st;
        }
        r->discovered = 1;
    } else {
        copy_str(r->right_dev, sizeof r->right_dev, p->right_dev);
        copy_str(r->left_dev, sizeof r->left_dev, p->left_dev);
    }

    int ndev = 0;
    struct ibv_device **list = ibv_get_device_list(&ndev);
    if (!list) {
        rerr(r->rank, "ibv_get_device_list failed: %s\n", strerror(errno));
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    struct ibv_device *rd = NULL, *ld = NULL;
    for (int i = 0; i < ndev; i++) {
        const char *n = ibv_get_device_name(list[i]);
        if (!strcmp(n, r->right_dev)) rd = list[i];
        if (!strcmp(n, r->left_dev)) ld = list[i];
    }
    if (!rd || !ld) {
        rerr(r->rank, "HCA not found: right=%s %s left=%s %s (have %d device(s))\n",
             r->right_dev, rd ? "ok" : "MISSING", r->left_dev, ld ? "ok" : "MISSING", ndev);
        for (int i = 0; i < ndev; i++)
            fprintf(stderr, "    dev[%d]=%s\n", i, ibv_get_device_name(list[i]));
        ibv_free_device_list(list);
        ring_free(r);
        return USB4_RDMA_RING_ERR_NO_DEVICE;
    }
    r->rc = ibv_open_device(rd);
    r->lc = ibv_open_device(ld);
    ibv_free_device_list(list);
    if (!r->rc || !r->lc) {
        rerr(r->rank, "ibv_open_device failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    r->rpd = ibv_alloc_pd(r->rc);
    r->lpd = ibv_alloc_pd(r->lc);
    if (!r->rpd || !r->lpd) {
        rerr(r->rank, "ibv_alloc_pd failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }

    struct ibv_port_attr pattr;
    if (ibv_query_port(r->rc, 1, &pattr) != 0) {
        rerr(r->rank, "ibv_query_port failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    r->max_xfer = pattr.max_msg_sz;
    if (r->max_xfer == 0 || r->max_xfer > MAX_XFER_SOFT) r->max_xfer = MAX_XFER_SOFT;
    if (r->max_xfer < 64) r->max_xfer = 64;
    r->mtu = pattr.active_mtu;
    if (r->mtu < IBV_MTU_256 || r->mtu > IBV_MTU_4096) r->mtu = IBV_MTU_4096;

    r->sbuf_sz = r->stride * (size_t)r->world;
    r->rbuf_sz = r->stride * NSLOT;
    if (posix_memalign((void **)&r->sbuf, 4096, r->sbuf_sz) != 0 ||
        posix_memalign((void **)&r->rbuf, 4096, r->rbuf_sz) != 0) {
        rerr(r->rank, "posix_memalign failed (sbuf=%zu rbuf=%zu)\n", r->sbuf_sz, r->rbuf_sz);
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    memset(r->sbuf, 0, r->sbuf_sz);
    memset(r->rbuf, 0, r->rbuf_sz);

    r->smr = ibv_reg_mr(r->rpd, r->sbuf, r->sbuf_sz, IBV_ACCESS_LOCAL_WRITE);
    r->rmr = ibv_reg_mr(r->lpd, r->rbuf, r->rbuf_sz,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!r->smr || !r->rmr) {
        rerr(r->rank, "ibv_reg_mr failed (sbuf=%zu rbuf=%zu): %s\n",
             r->sbuf_sz, r->rbuf_sz, strerror(errno));
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }

    if (resolve_gid(r->rc, p->gid_index, &r->right_gid_index, &r->right_gid) != 0) {
        rerr(r->rank, "%s: no usable RoCE v2 IPv4-mapped GID (set params->gid_index)\n",
             r->right_dev);
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    if (resolve_gid(r->lc, p->gid_index, &r->left_gid_index, &r->left_gid) != 0) {
        rerr(r->rank, "%s: no usable RoCE v2 IPv4-mapped GID (set params->gid_index)\n",
             r->left_dev);
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    rlog(r->rank, r->verbose,
         "devices: right=%s (gid idx %d) left=%s (gid idx %d) chunk=%zu\n",
         r->right_dev, r->right_gid_index, r->left_dev, r->left_gid_index, r->chunk_bytes);

    /* ---------- QPs ---------- */
    struct ibv_qp_init_attr oa;
    memset(&oa, 0, sizeof oa);
    oa.cap.max_send_wr = 128;
    oa.cap.max_recv_wr = 8;
    oa.cap.max_send_sge = 1;
    oa.cap.max_recv_sge = 1;
    oa.qp_type = IBV_QPT_RC;
    oa.send_cq = ibv_create_cq(r->rc, 256, NULL, NULL, 0);
    if (!oa.send_cq) {
        rerr(r->rank, "ibv_create_cq(right) failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    oa.recv_cq = oa.send_cq;
    r->ocq = oa.send_cq;
    r->out = ibv_create_qp(r->rpd, &oa);
    if (!r->out) {
        rerr(r->rank, "ibv_create_qp(out) failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }

    struct ibv_qp_init_attr ia;
    memset(&ia, 0, sizeof ia);
    ia.cap.max_send_wr = 8;
    ia.cap.max_recv_wr = 8;
    ia.cap.max_send_sge = 1;
    ia.cap.max_recv_sge = 1;
    ia.qp_type = IBV_QPT_RC;
    ia.send_cq = ibv_create_cq(r->lc, 32, NULL, NULL, 0);
    if (!ia.send_cq) {
        rerr(r->rank, "ibv_create_cq(left) failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    ia.recv_cq = ia.send_cq;
    r->icq = ia.send_cq;
    r->in = ibv_create_qp(r->lpd, &ia);
    if (!r->in) {
        rerr(r->rank, "ibv_create_qp(in) failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }

    struct peerinfo mine;
    memset(&mine, 0, sizeof mine);
    mine.qpn = r->in->qp_num;
    mine.rkey = r->rmr->rkey;
    mine.addr = (uintptr_t)r->rbuf;
    memcpy(mine.gid, r->left_gid.raw, 16);

    struct peerinfo out_side;
    memset(&out_side, 0, sizeof out_side);
    out_side.qpn = r->out->qp_num;
    out_side.rkey = r->smr->rkey;
    out_side.addr = (uintptr_t)r->sbuf;
    memcpy(out_side.gid, r->right_gid.raw, 16);

    /* ---------- out-of-band handshake ---------- */
    r->lfd = tcp_listen(r->port + 100 + r->rank);
    if (r->lfd < 0) {
        rerr(r->rank, "listen on port %d failed: %s\n",
             r->port + 100 + r->rank, strerror(errno));
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    r->send_fd = tcp_connect(r->control_ip[r->right], r->port + 100 + r->right,
                             r->connect_timeout_ms);
    if (r->send_fd < 0) {
        rerr(r->rank, "connect to rank %d (%s:%d) failed within %d ms\n",
             r->right, r->control_ip[r->right], r->port + 100 + r->right,
             r->connect_timeout_ms);
        ring_free(r);
        return USB4_RDMA_RING_ERR_TIMEOUT;
    }
    if (xfer(r->send_fd, &out_side, sizeof out_side, 1) != 0) {
        rerr(r->rank, "send out-QP info to rank %d failed\n", r->right);
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    r->acc_fd = accept_timeout(r->lfd, r->connect_timeout_ms);
    if (r->acc_fd < 0) {
        rerr(r->rank, "accept from rank %d failed within %d ms\n",
             r->left, r->connect_timeout_ms);
        ring_free(r);
        return USB4_RDMA_RING_ERR_TIMEOUT;
    }
    set_sock_timeout(r->acc_fd, r->connect_timeout_ms);
    if (xfer(r->acc_fd, &mine, sizeof mine, 1) != 0) {
        rerr(r->rank, "send in-QP info to rank %d failed\n", r->left);
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }

    memset(&r->right_in, 0, sizeof r->right_in);
    memset(&r->left_out, 0, sizeof r->left_out);
    if (xfer(r->send_fd, &r->right_in, sizeof r->right_in, 0) != 0) {
        rerr(r->rank, "recv right in-QP info failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    if (xfer(r->acc_fd, &r->left_out, sizeof r->left_out, 0) != 0) {
        rerr(r->rank, "recv left out-QP info failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    rlog(r->rank, r->verbose,
         "handshake: right in qpn=%u rkey=0x%x addr=0x%lx | left out qpn=%u\n",
         r->right_in.qpn, r->right_in.rkey,
         (unsigned long)r->right_in.addr, r->left_out.qpn);

    if (!gid_is_ipv4_mapped(r->right_in.gid) || !gid_is_ipv4_mapped(r->left_out.gid))
        rlog(r->rank, r->verbose, "warning: peer GID is not IPv4-mapped\n");

    /* ---------- QP state transitions ---------- */
    if (qp_to_init(r->out) != 0 || qp_to_init(r->in) != 0) {
        rerr(r->rank, "QP -> INIT failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    if (qp_to_rtr(r->out, r->right_in.qpn, r->right_in.gid,
                  r->right_gid_index, r->mtu) != 0) {
        rerr(r->rank, "out QP -> RTR failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    if (qp_to_rts(r->out) != 0) {
        rerr(r->rank, "out QP -> RTS failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    if (qp_to_rtr(r->in, r->left_out.qpn, r->left_out.gid,
                  r->left_gid_index, r->mtu) != 0) {
        rerr(r->rank, "in QP -> RTR failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }
    if (qp_to_rts(r->in) != 0) {
        rerr(r->rank, "in QP -> RTS failed\n");
        ring_free(r);
        return USB4_RDMA_RING_ERR_SYS;
    }

    /* ---------- collective warmup (absorbs the ~40 ms path bring-up) ----------
     * All ranks must call init(); warmup_iters must agree across ranks because
     * each iteration is a real collective and advances the shared sequence. */
    int warm = p->warmup_iters;
    if (warm == 0) warm = 2;      /* default on */
    if (warm < 0) warm = 0;       /* explicit opt-out */
    if (warm > 0) {
        /* Warm up at the real per-rank chunk size: the one-off path bring-up
         * and the driver's per-peer credit/MTU state are size-sensitive, so a
         * tiny dummy message can leave the first full-size call exposed. */
        unsigned char *dummy = malloc(r->chunk_bytes);
        if (!dummy) {
            rerr(r->rank, "warmup buffer allocation failed\n");
            ring_free(r);
            return USB4_RDMA_RING_ERR_SYS;
        }
        memset(dummy, (unsigned char)(r->rank + 1), r->chunk_bytes);
        for (int i = 0; i < warm; i++) {
            double t0 = now_s();
            st = usb4_rdma_ring_all_reduce((struct usb4_rdma_ring *)r, dummy,
                                           r->chunk_bytes,
                                           USB4_RDMA_RING_DTYPE_U8,
                                           USB4_RDMA_RING_OP_SUM);
            if (st != USB4_RDMA_RING_OK) {
                rerr(r->rank, "warmup all_reduce %d/%d failed (status %d)\n",
                     i + 1, warm, st);
                free(dummy);
                ring_free(r);
                return st;
            }
            rlog(r->rank, r->verbose, "warmup %d/%d took %.1f ms\n",
                 i + 1, warm, (now_s() - t0) * 1e3);
        }
        free(dummy);
    }

    if (info) {
        copy_str(info->right_dev, sizeof info->right_dev, r->right_dev);
        copy_str(info->left_dev, sizeof info->left_dev, r->left_dev);
        copy_str(info->right_peer, sizeof info->right_peer, r->right_peer);
        copy_str(info->left_peer, sizeof info->left_peer, r->left_peer);
        info->right_gid_index = r->right_gid_index;
        info->left_gid_index = r->left_gid_index;
        memcpy(info->right_gid, r->right_gid.raw, 16);
        memcpy(info->left_gid, r->left_gid.raw, 16);
        info->discovered = r->discovered;
        info->chunk_bytes = r->chunk_bytes;
    }

    *ring_out = r;
    return USB4_RDMA_RING_OK;
}

/* ------------------------------------------------------------------ */
/* collectives                                                        */
/* ------------------------------------------------------------------ */

/* Balanced split of `total` elements: remainder goes to the lowest ranks. */
static void split_sizes(size_t total, int world, size_t *sizes, size_t *prefix) {
    size_t base = total / (size_t)world, rem = total % (size_t)world;
    prefix[0] = 0;
    for (int c = 0; c < world; c++) {
        sizes[c] = base + ((size_t)c < rem ? 1 : 0);
        prefix[c + 1] = prefix[c] + sizes[c];
    }
}

int usb4_rdma_ring_all_reduce(struct usb4_rdma_ring *r, void *buf, size_t count,
                              enum usb4_rdma_ring_dtype dtype,
                              enum usb4_rdma_ring_op op) {
    if (!r) return USB4_RDMA_RING_ERR_INVAL;
    if (count == 0) return USB4_RDMA_RING_OK;
    if (!buf) return USB4_RDMA_RING_ERR_INVAL;
    size_t es = dtype_size(dtype);
    if (es == 0) return USB4_RDMA_RING_ERR_INVAL;
    if (count > SIZE_MAX / es) return USB4_RDMA_RING_ERR_INVAL;
    size_t chunk_elems = r->chunk_bytes / es;
    if (chunk_elems == 0) {
        rerr(r->rank, "chunk_bytes=%zu too small for dtype size %zu\n",
             r->chunk_bytes, es);
        return USB4_RDMA_RING_ERR_INVAL;
    }
    unsigned char *ub = (unsigned char *)buf;
    size_t world = (size_t)r->world;
    size_t sizes[USB4_RDMA_RING_MAX_WORLD], prefix[USB4_RDMA_RING_MAX_WORLD + 1];

    for (size_t p0 = 0; p0 < count;) {
        size_t pass_elems = count - p0;
        if (pass_elems > world * chunk_elems) pass_elems = world * chunk_elems;
        size_t seg_elems = (pass_elems + world - 1) / world;
        size_t seg_bytes = seg_elems * es;
        if (seg_bytes > r->chunk_bytes || seg_bytes > r->stride - MARKER_SZ) {
            rerr(r->rank, "internal segment overflow (seg=%zu chunk=%zu)\n",
                 seg_bytes, r->chunk_bytes);
            return USB4_RDMA_RING_ERR_INVAL;
        }
        size_t marker_off = align_up(seg_bytes, 8);
        size_t total_msg = marker_off + MARKER_SZ;
        split_sizes(pass_elems, r->world, sizes, prefix);

        for (int c = 0; c < r->world; c++) {
            char *dst = r->sbuf + (size_t)c * r->stride;
            memset(dst, 0, r->stride);
            size_t n = sizes[c];
            if (n)
                memcpy(dst, ub + (p0 + prefix[c]) * es, n * es);
        }

        for (int phase = 0; phase < 2; phase++) {
            for (int step = 0; step < r->world - 1; step++) {
                int s, rr;
                if (phase == 0) {
                    s = (r->rank - step + r->world) % r->world;
                    rr = (r->rank - step - 1 + r->world) % r->world;
                } else {
                    s = (r->rank - step + 1 + r->world) % r->world;
                    rr = (r->rank - step + r->world) % r->world;
                }
                uint64_t seq = ++r->seq;
                int slot = (int)(seq % NSLOT);
                char *src = r->sbuf + (size_t)s * r->stride;
                char *dst_slot = r->rbuf + (size_t)slot * r->stride;

                *(uint64_t *)(src + marker_off) = seq;
                double tw0 = now_s();
                int st = post_write(r, r->right_in.addr + (size_t)slot * r->stride,
                                    r->right_in.rkey, src, total_msg, seq);
                double tw1 = now_s();
                if (st != USB4_RDMA_RING_OK) return st;
                st = wait_marker(dst_slot, marker_off, seq,
                                 phase ? "allgather-marker" : "reducescatter-marker");
                double tw2 = now_s();
                if (st != USB4_RDMA_RING_OK) return st;
                if (tw1 - tw0 > 0.001 || tw2 - tw1 > 0.001)
                    rlog(r->rank, r->verbose,
                         "STALL all_reduce phase=%d step=%d seq=%lu post=%.1fms marker=%.1fms\n",
                         phase, step, (unsigned long)seq,
                         (tw1 - tw0) * 1e3, (tw2 - tw1) * 1e3);

                char *local = r->sbuf + (size_t)rr * r->stride;
                if (phase == 0) {
                    if (reduce_elem(dtype, op, local, dst_slot, seg_elems) != 0) {
                        rerr(r->rank, "unsupported dtype/op combination\n");
                        return USB4_RDMA_RING_ERR_INVAL;
                    }
                } else {
                    memcpy(local, dst_slot, seg_bytes);
                }
            }
        }

        for (int c = 0; c < r->world; c++) {
            size_t n = sizes[c];
            if (n)
                memcpy(ub + (p0 + prefix[c]) * es, r->sbuf + (size_t)c * r->stride, n * es);
        }
        p0 += pass_elems;
    }
    return USB4_RDMA_RING_OK;
}

int usb4_rdma_ring_all_gather(struct usb4_rdma_ring *r, const void *sendbuf,
                              void *recvbuf, size_t count,
                              enum usb4_rdma_ring_dtype dtype) {
    if (!r) return USB4_RDMA_RING_ERR_INVAL;
    if (count == 0) return USB4_RDMA_RING_OK;
    if (!sendbuf || !recvbuf) return USB4_RDMA_RING_ERR_INVAL;
    size_t es = dtype_size(dtype);
    if (es == 0) return USB4_RDMA_RING_ERR_INVAL;
    if (count > SIZE_MAX / es) return USB4_RDMA_RING_ERR_INVAL;
    if (count > SIZE_MAX / ((size_t)r->world * es)) return USB4_RDMA_RING_ERR_INVAL;
    size_t chunk_elems = r->chunk_bytes / es;
    if (chunk_elems == 0) return USB4_RDMA_RING_ERR_INVAL;

    const unsigned char *send = (const unsigned char *)sendbuf;
    unsigned char *recv = (unsigned char *)recvbuf;

    for (size_t p0 = 0; p0 < count;) {
        size_t n = count - p0;
        if (n > chunk_elems) n = chunk_elems;
        size_t seg_bytes = n * es;
        if (seg_bytes > r->chunk_bytes || seg_bytes > r->stride - MARKER_SZ) {
            rerr(r->rank, "internal segment overflow in all_gather\n");
            return USB4_RDMA_RING_ERR_INVAL;
        }
        size_t marker_off = align_up(seg_bytes, 8);
        size_t total_msg = marker_off + MARKER_SZ;

        memset(r->sbuf, 0, r->sbuf_sz);
        memcpy(r->sbuf + (size_t)r->rank * r->stride, send + p0 * es, seg_bytes);

        int block = r->rank;
        for (int step = 0; step < r->world - 1; step++) {
            uint64_t seq = ++r->seq;
            int slot = (int)(seq % NSLOT);
            char *src = r->sbuf + (size_t)block * r->stride;
            char *dst_slot = r->rbuf + (size_t)slot * r->stride;

            *(uint64_t *)(src + marker_off) = seq;
            double tw0 = now_s();
            int st = post_write(r, r->right_in.addr + (size_t)slot * r->stride,
                                r->right_in.rkey, src, total_msg, seq);
            double tw1 = now_s();
            if (st != USB4_RDMA_RING_OK) return st;
            st = wait_marker(dst_slot, marker_off, seq, "allgather-marker");
            double tw2 = now_s();
            if (st != USB4_RDMA_RING_OK) return st;
            if (tw1 - tw0 > 0.001 || tw2 - tw1 > 0.001)
                rlog(r->rank, r->verbose,
                     "STALL all_gather step=%d seq=%lu post=%.1fms marker=%.1fms\n",
                     step, (unsigned long)seq, (tw1 - tw0) * 1e3, (tw2 - tw1) * 1e3);

            int got = (block - 1 + r->world) % r->world;
            memcpy(r->sbuf + (size_t)got * r->stride, dst_slot, seg_bytes);
            block = got;
        }

        for (int c = 0; c < r->world; c++)
            memcpy(recv + ((size_t)c * count + p0) * es,
                   r->sbuf + (size_t)c * r->stride, seg_bytes);
        p0 += n;
    }
    return USB4_RDMA_RING_OK;
}
