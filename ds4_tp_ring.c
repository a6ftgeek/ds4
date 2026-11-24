/* Ring collective transport for tensor parallelism. See ds4_tp_ring.h.
 *
 * The data plane is deliberately tiny: rank r only ever talks to its right
 * neighbour (r+1) % world and its left neighbour (r-1+world) % world, which is
 * the adjacency a switchless Thunderbolt/USB4 ring gives us. The all-reduce is
 * a ring all-gather followed by a local reduction of the world copies, so the
 * number of ring steps is world-1 regardless of element count.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_tp_ring.h"

#include <errno.h>
#include <arpa/inet.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <poll.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifdef USE_USB4_RDMA
/* The probe below needs the verbs types; the ring module itself keeps them
 * private behind its own header. Both are confined to this build mode. */
#include <infiniband/verbs.h>
#include "usb4_rdma_ring.h"
#endif

/* Bytes of payload handed to one send()/recv() in the TCP backend. */
#define DS4_TP_RING_IO_CHUNK (1u << 20)
/* Dummy collective rounds run from create() so the first real all-reduce does
 * not pay RDMA path bring-up (a one-off ~40 ms after the queue pairs reach
 * RTS). Two rounds cover the reduce-scatter and all-gather directions. */
#define DS4_TP_RING_WARMUP_ROUNDS 2u
#define DS4_TP_RING_WARMUP_ELEMS 256u
/* Disjoint TCP base for the USB4 library's handshake: it binds
 * base+100+rank, while this module's control ring binds port+100+rank. */
#define DS4_TP_RING_USB4_PORT_OFFSET 300

struct ds4_tp_ring {
    ds4_tp_ring_options opt;
    ds4_tp_ring_backend backend;
    int rank;
    int world;
    /* Ring control sockets: right_fd talks to the right neighbour,
     * left_fd is the accepted connection from the left neighbour. */
    int right_fd;
    int left_fd;
    int listen_fd;
    void *scratch;              /* world * count * esize all-gather staging */
    size_t scratch_bytes;
    size_t scratch_elems;
    int pinned_core;            /* -1 until the first collective */
    bool warned_conversion;
#ifdef USE_USB4_RDMA
    struct usb4_rdma_ring *usb4;
    struct usb4_rdma_ring_info usb4_info;
#endif
};

static void ring_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

/* ---------------------------------------------------------------- helpers */

static double ring_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Yield the pipeline while busy-waiting. macOS has no PAUSE; arm64 has YIELD. */
static void ring_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

static size_t ring_dtype_size(ds4_tp_ring_dtype dtype) {
    switch (dtype) {
    case DS4_TP_RING_DTYPE_U8:
    case DS4_TP_RING_DTYPE_I8: return 1;
    case DS4_TP_RING_DTYPE_U16:
    case DS4_TP_RING_DTYPE_I16:
    case DS4_TP_RING_DTYPE_F16:
    case DS4_TP_RING_DTYPE_BF16: return 2;
    case DS4_TP_RING_DTYPE_U32:
    case DS4_TP_RING_DTYPE_I32:
    case DS4_TP_RING_DTYPE_F32: return 4;
    case DS4_TP_RING_DTYPE_U64:
    case DS4_TP_RING_DTYPE_I64:
    case DS4_TP_RING_DTYPE_F64: return 8;
    }
    return 0;
}

static bool ring_dtype_supports(ds4_tp_ring_dtype dtype) {
    return ring_dtype_size(dtype) != 0;
}

/* acc[i] = acc[i] <op> src[i] over `count` elements of `dtype`.
 *
 * MAX/MIN are carried for completeness; the TP seam only ever issues SUM. */
static void ring_reduce_into(void *acc, const void *src, size_t count,
                             ds4_tp_ring_dtype dtype, ds4_tp_ring_op op) {
    size_t i;
    switch (dtype) {
    case DS4_TP_RING_DTYPE_F32: {
        float *a = acc;
        const float *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] += s[i];
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_F64: {
        double *a = acc;
        const double *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] += s[i];
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_F16:
    case DS4_TP_RING_DTYPE_BF16: {
        /* Only the order-preserving ops are valid on the raw bit pattern; the
         * SUM case is rejected by the caller so no wrong numbers escape. */
        uint16_t *a = acc;
        const uint16_t *s = src;
        for (i = 0; i < count; i++) {
            /* Encoding-preserving order: compare as unsigned for positive and
             * as reversed for negative values. */
            const uint16_t av = a[i], sv = s[i];
            const bool aneg = (av & 0x8000u) != 0, sneg = (sv & 0x8000u) != 0;
            int cmp;
            if (aneg != sneg) cmp = aneg ? -1 : 1;
            else cmp = aneg ? (av > sv ? -1 : av < sv ? 1 : 0)
                            : (av < sv ? -1 : av > sv ? 1 : 0);
            if (op == DS4_TP_RING_OP_MAX) { if (cmp < 0) a[i] = sv; }
            else if (op == DS4_TP_RING_OP_MIN) { if (cmp > 0) a[i] = sv; }
        }
        return;
    }
    case DS4_TP_RING_DTYPE_U8: {
        uint8_t *a = acc; const uint8_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] = (uint8_t)(a[i] + s[i]);
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_I8: {
        int8_t *a = acc; const int8_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] = (int8_t)(a[i] + s[i]);
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_U16: {
        uint16_t *a = acc; const uint16_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] = (uint16_t)(a[i] + s[i]);
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_I16: {
        int16_t *a = acc; const int16_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] = (int16_t)(a[i] + s[i]);
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_U32: {
        uint32_t *a = acc; const uint32_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] += s[i];
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_I32: {
        int32_t *a = acc; const int32_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] += s[i];
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_U64: {
        uint64_t *a = acc; const uint64_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] += s[i];
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    case DS4_TP_RING_DTYPE_I64: {
        int64_t *a = acc; const int64_t *s = src;
        for (i = 0; i < count; i++) {
            if (op == DS4_TP_RING_OP_SUM) a[i] += s[i];
            else if (op == DS4_TP_RING_OP_MAX) { if (s[i] > a[i]) a[i] = s[i]; }
            else if (s[i] < a[i]) a[i] = s[i];
        }
        return;
    }
    }
}

/* ------------------------------------------------------------ TCP backend */

static int ring_tcp_write(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int ring_tcp_read(int fd, void *buf, size_t len) {
    char *p = buf;
    while (len) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

/* Move `bytes` to the right neighbour while pulling the same amount from the
 * left one. Both directions must progress together: a blocking send() on a
 * full socket buffer while the peer is also sending would deadlock, because
 * neither side is draining its receive queue yet. */
static int ring_tcp_exchange(int out_fd, int in_fd, const void *sbuf, void *rbuf,
                             size_t bytes, char *err, size_t errlen) {
    const char *sp = sbuf;
    char *rp = rbuf;
    size_t so = 0, ro = 0;
    while (so < bytes || ro < bytes) {
        struct pollfd pfd[2];
        int out_slot = -1, in_slot = -1, n = 0;
        if (so < bytes) {
            pfd[n].fd = out_fd;
            pfd[n].events = POLLOUT;
            pfd[n].revents = 0;
            out_slot = n++;
        }
        if (ro < bytes) {
            pfd[n].fd = in_fd;
            pfd[n].events = POLLIN;
            pfd[n].revents = 0;
            in_slot = n++;
        }
        if (poll(pfd, (nfds_t)n, 30000) <= 0) {
            ring_set_err(err, errlen, "ring: timed out moving %zu of %zu bytes",
                         so < ro ? so : ro, bytes);
            return -1;
        }
        if (out_slot >= 0 && pfd[out_slot].revents) {
            ssize_t w = send(out_fd, sp + so, bytes - so, MSG_DONTWAIT);
            if (w > 0) so += (size_t)w;
            else if (w < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                     errno != EINTR) {
                ring_set_err(err, errlen, "ring: send failed: %s", strerror(errno));
                return -1;
            }
        }
        if (in_slot >= 0 && pfd[in_slot].revents) {
            ssize_t r = recv(in_fd, rp + ro, bytes - ro, MSG_DONTWAIT);
            if (r > 0) ro += (size_t)r;
            else if (r == 0) {
                ring_set_err(err, errlen, "ring: peer closed the ring");
                return -1;
            } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                ring_set_err(err, errlen, "ring: recv failed: %s", strerror(errno));
                return -1;
            }
        }
    }
    return 0;
}

static void ring_tcp_tune(int fd) {
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    (void)setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
}

/* One byte per rank, used to agree on the backend before any device is opened
 * and to keep the bring-up ordering collective. */
static int ring_tcp_exchange_flags(struct ds4_tp_ring *r, unsigned char mine,
                                   unsigned char *all, char *err,
                                   size_t errlen) {
    unsigned char right = mine;
    for (int step = 0; step < r->world - 1; step++) {
        unsigned char send_val = right, recv_val = 0;
        if (ring_tcp_write(r->right_fd, &send_val, 1) != 0 ||
            ring_tcp_read(r->left_fd, &recv_val, 1) != 0) {
            ring_set_err(err, errlen,
                         "ring: neighbour handshake failed at step %d", step);
            return -1;
        }
        const int from = (r->rank - step - 1 + r->world) % r->world;
        if (all) all[from] = recv_val;
        right = recv_val;
    }
    return 0;
}

/* TCP ring all-gather: after world-1 steps every rank holds the full `count`
 * elements contributed by each rank, rank r's copy at offset r*count. Only
 * point-to-point traffic with the two neighbours is used, which is what makes
 * this equivalent on a switchless ring. */
static int ring_tcp_gather_full(struct ds4_tp_ring *r, const void *sendbuf,
                                void *recvbuf, size_t count, size_t esize,
                                char *err, size_t errlen) {
    unsigned char *out = recvbuf;
    const size_t bytes = count * esize;
    memcpy(out + (size_t)r->rank * bytes, sendbuf, bytes);
    for (int step = 0; step < r->world - 1; step++) {
        const int s = (r->rank - step + r->world) % r->world;
        const int d = (r->rank - step - 1 + r->world) % r->world;
        if (ring_tcp_exchange(r->right_fd, r->left_fd, out + (size_t)s * bytes,
                              out + (size_t)d * bytes, bytes, err, errlen) != 0) {
            ring_set_err(err, errlen, "ring: all-gather failed at step %d", step);
            return -1;
        }
    }
    return 0;
}

/* --------------------------------------------------- scratch buffer growth */

static int ring_reserve_scratch(struct ds4_tp_ring *r, size_t elems,
                                size_t esize, char *err, size_t errlen) {
    if (elems <= r->scratch_elems && r->scratch) return 0;
    const size_t want_elems = elems + elems / 4u + 64u;
    if (want_elems > SIZE_MAX / esize) {
        ring_set_err(err, errlen, "ring: collective payload is too large");
        return -1;
    }
    const size_t bytes = want_elems * esize;
    void *p = realloc(r->scratch, bytes);
    if (!p) {
        ring_set_err(err, errlen, "ring: out of memory for a %zu-byte collective",
                     bytes);
        return -1;
    }
    r->scratch = p;
    r->scratch_bytes = bytes;
    r->scratch_elems = want_elems;
    return 0;
}

/* ------------------------------------------------------------- pinning */

static void ring_maybe_pin(struct ds4_tp_ring *r) {
    /* The USB4 backend pins its own threads from usb4_rdma_ring_params, and
     * pinning those again here would fight it. */
    if (r->backend == DS4_TP_RING_BACKEND_USB4) return;
    if (r->opt.cpu_core < 0 || r->pinned_core >= 0) return;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(r->opt.cpu_core, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0) {
        r->pinned_core = r->opt.cpu_core;
        fprintf(stderr, "ds4-tp: ring pinned to CPU core %d\n", r->pinned_core);
    } else {
        fprintf(stderr, "ds4-tp: ring could not pin to CPU core %d: %s\n",
                r->opt.cpu_core, strerror(errno));
        r->pinned_core = -2;    /* reported once, never retried */
    }
#else
    r->pinned_core = -2;
#endif
}

/* ------------------------------------------------------------ USB4 probe */

/* Local, non-blocking capability check used to decide the backend collectively.
 * True means "this rank could plausibly open a USB4 RDMA ring". */
static bool ring_probe_usb4(const ds4_tp_ring_options *opt) {
#ifdef USE_USB4_RDMA
    int num = 0;
    struct ibv_device **devs = ibv_get_device_list(&num);
    if (!devs) return false;
    bool ok = false;
    for (int i = 0; i < num && !ok; i++) {
        const char *name = ibv_get_device_name(devs[i]);
        if (opt->right_dev && opt->left_dev &&
            strcmp(opt->right_dev, name) != 0 && strcmp(opt->left_dev, name) != 0)
            continue;
        struct ibv_context *ctx = ibv_open_device(devs[i]);
        if (!ctx) continue;
        struct ibv_port_attr pa;
        if (ibv_query_port(ctx, 1, &pa) == 0 && pa.state == IBV_PORT_ACTIVE) {
            const int first = opt->gid_index >= 0 ? opt->gid_index : 0;
            const int end = opt->gid_index >= 0 ? first + 1 : (int)pa.gid_tbl_len;
            for (int g = first; g < end; g++) {
                union ibv_gid gid;
                if (ibv_query_gid(ctx, 1, g, &gid) != 0) continue;
                if (gid.raw[10] == 0xff && gid.raw[11] == 0xff) { ok = true; break; }
            }
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    return ok;
#else
    (void)opt;
    return false;
#endif
}

/* --------------------------------------------------------------- bring-up */

static int ring_listen(int port, char *err, size_t errlen) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        ring_set_err(err, errlen, "ring: socket: %s", strerror(errno));
        return -1;
    }
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0 ||
        listen(fd, 8) != 0) {
        ring_set_err(err, errlen, "ring: bind/listen on port %d: %s", port,
                     strerror(errno));
        close(fd);
        return -1;
    }
    ring_tcp_tune(fd);
    return fd;
}

static int ring_dial(const char *host, int port, int timeout_ms, char *err,
                     size_t errlen) {
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
        ring_set_err(err, errlen, "ring: control address '%s' is not IPv4", host);
        return -1;
    }
    const double deadline = ring_now_sec() + (double)timeout_ms / 1000.0;
    int spins = 0;
    for (;;) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            ring_tcp_tune(fd);
            if (connect(fd, (struct sockaddr *)&a, sizeof(a)) == 0) return fd;
            close(fd);
        }
        if (ring_now_sec() > deadline) {
            ring_set_err(err, errlen, "ring: could not connect to %s:%d", host,
                         port);
            return -1;
        }
        if ((++spins & 63) == 0) ring_cpu_relax();
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 20 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
}

static int ring_connect(struct ds4_tp_ring *r, char *err, size_t errlen) {
    const int world = r->world;
    const int right = (r->rank + 1) % world;
    const int base = r->opt.port;
    const int timeout = r->opt.connect_timeout_ms > 0 ? r->opt.connect_timeout_ms
                                                      : 60000;
    r->listen_fd = ring_listen(base + 100 + r->rank, err, errlen);
    if (r->listen_fd < 0) return -1;
    if (!r->opt.control_ip[right] || !r->opt.control_ip[right][0]) {
        ring_set_err(err, errlen, "ring: no control address for rank %d", right);
        return -1;
    }
    r->right_fd = ring_dial(r->opt.control_ip[right], base + 100 + right,
                            timeout, err, errlen);
    if (r->right_fd < 0) return -1;
    r->left_fd = accept(r->listen_fd, NULL, NULL);
    if (r->left_fd < 0) {
        ring_set_err(err, errlen, "ring: accept from left neighbour: %s",
                     strerror(errno));
        return -1;
    }
    ring_tcp_tune(r->left_fd);
    return 0;
}

static void ring_close_sockets(struct ds4_tp_ring *r) {
    if (r->right_fd >= 0) { close(r->right_fd); r->right_fd = -1; }
    if (r->left_fd >= 0) { close(r->left_fd); r->left_fd = -1; }
    if (r->listen_fd >= 0) { close(r->listen_fd); r->listen_fd = -1; }
}

/* Validate that the (rank, world) pair and the control addresses are sane. */
static int ring_validate(const ds4_tp_ring_options *opt, char *err,
                         size_t errlen) {
    if (!opt) {
        ring_set_err(err, errlen, "ring: no options");
        return -1;
    }
    if (opt->world < 2 || opt->world > DS4_TP_RING_MAX_WORLD) {
        ring_set_err(err, errlen, "ring: world size %d is outside 2..%d",
                     opt->world, DS4_TP_RING_MAX_WORLD);
        return -1;
    }
    if (opt->rank < 0 || opt->rank >= opt->world) {
        ring_set_err(err, errlen, "ring: rank %d is outside 0..%d", opt->rank,
                     opt->world - 1);
        return -1;
    }
    for (int i = 0; i < opt->world; i++) {
        if (!opt->control_ip[i] || !opt->control_ip[i][0]) {
            ring_set_err(err, errlen, "ring: control address for rank %d is missing", i);
            return -1;
        }
    }
    if ((opt->right_dev && !opt->left_dev) || (!opt->right_dev && opt->left_dev)) {
        /* world=2 has one peer, so one device override is enough: it is the
         * same HCA in both ring directions. Larger rings must name both. */
        if (opt->world != 2) {
            ring_set_err(err, errlen,
                         "ring: set both device overrides or neither");
            return -1;
        }
    }
    return 0;
}

int ds4_tp_ring_create(ds4_tp_ring **out, const ds4_tp_ring_options *opt,
                       char *err, size_t errlen) {
    if (out) *out = NULL;
    if (ring_validate(opt, err, errlen) != 0) return -1;

    /* At world=2 the right and left neighbour are the same peer, so the same
     * HCA plays both roles. Mirror a single override instead of letting the
     * two halves of the library pick different devices. */
    ds4_tp_ring_options normalized = *opt;
    if (normalized.world == 2) {
        if (normalized.right_dev && !normalized.left_dev)
            normalized.left_dev = normalized.right_dev;
        else if (normalized.left_dev && !normalized.right_dev)
            normalized.right_dev = normalized.left_dev;
    }

    ds4_tp_ring *r = calloc(1, sizeof(*r));
    if (!r) {
        ring_set_err(err, errlen, "ring: out of memory");
        return -1;
    }
    r->opt = normalized;
    r->rank = opt->rank;
    r->world = opt->world;
    r->right_fd = r->left_fd = r->listen_fd = -1;
    r->pinned_core = -1;
    r->backend = DS4_TP_RING_BACKEND_TCP;

    if (ring_connect(r, err, errlen) != 0) {
        ring_close_sockets(r);
        free(r);
        fprintf(stderr, "ds4-tp: ring unavailable: %s\n", err ? err : "unknown");
        return -1;
    }

    /* Decide the backend collectively: every rank must be able to open a USB4
     * ring, otherwise all of them take the TCP ring. This runs before any verbs
     * resource is opened, so a mismatch can never strand one rank in a
     * handshake the others never enter. */
    unsigned char *flags = calloc((size_t)r->world, 1);
    if (!flags) {
        ring_set_err(err, errlen, "ring: out of memory");
        ring_close_sockets(r);
        free(r);
        return -1;
    }
    const unsigned char mine = ring_probe_usb4(&r->opt) ? 1u : 0u;
    flags[r->rank] = mine;
    if (r->world > 1 &&
        ring_tcp_exchange_flags(r, mine, flags, err, errlen) != 0) {
        free(flags);
        ring_close_sockets(r);
        free(r);
        return -1;
    }
    bool usb4_ok = true;
    for (int i = 0; i < r->world; i++) if (!flags[i]) usb4_ok = false;
    free(flags);
#ifndef USE_USB4_RDMA
    /* Without the flag there is no verbs backend to fall back to, so the
     * decision is made but never acted on. */
    (void)usb4_ok;
#endif

#ifdef USE_USB4_RDMA
    if (usb4_ok) {
        struct usb4_rdma_ring_params p;
        memset(&p, 0, sizeof(p));
        p.rank = r->rank;
        p.world = r->world;
        for (int i = 0; i < r->world; i++) p.control_ip[i] = r->opt.control_ip[i];
        /* The library runs its own out-of-band handshake on
         * base+100+rank, so it needs a range this module's control ring is not
         * already holding. */
        p.port = r->opt.port + DS4_TP_RING_USB4_PORT_OFFSET;
        p.right_dev = r->opt.right_dev;
        p.left_dev = r->opt.left_dev;
        p.gid_index = r->opt.gid_index >= 0 ? r->opt.gid_index : -1;
        p.chunk_bytes = r->opt.chunk_bytes;
        p.connect_timeout_ms = r->opt.connect_timeout_ms;
        /* The library's own collective warmup absorbs the one-off path
         * bring-up; 0 keeps its default so every rank agrees. */
        p.warmup_iters = 0;
        p.pin_cpu = r->opt.cpu_core >= 0 ? 1 : 0;
        p.cpu_core = r->opt.cpu_core >= 0 ? r->opt.cpu_core : 0;
        p.verbose = r->opt.verbose ? 1 : 0;
        const int rc = usb4_rdma_ring_init(&r->usb4, &p, &r->usb4_info);
        if (rc != USB4_RDMA_RING_OK) {
            /* Every rank probed ready, so a failure here is a real fault on
             * this rank. Do not silently switch to TCP: that would leave the
             * peers driving a different transport. */
            ring_set_err(err, errlen, "ring: USB4 RDMA init failed: %s",
                         usb4_rdma_ring_strerror(rc));
            ring_close_sockets(r);
            free(r);
            return -1;
        }
        /* world=2 is degenerate: both ring directions must be the one HCA that
         * faces the only peer. A split here would mean discovery matched two
         * different XDomains and the data plane would silently use the wrong
         * link. */
        if (r->world == 2 &&
            strcmp(r->usb4_info.right_dev, r->usb4_info.left_dev) != 0) {
            ring_set_err(err, errlen,
                         "ring: world=2 resolved right=%s left=%s; both ring "
                         "directions must use the same HCA",
                         r->usb4_info.right_dev, r->usb4_info.left_dev);
            usb4_rdma_ring_destroy(r->usb4);
            r->usb4 = NULL;
            ring_close_sockets(r);
            free(r);
            return -1;
        }
        r->backend = DS4_TP_RING_BACKEND_USB4;
        if (r->opt.verbose) {
            fprintf(stderr,
                    "ds4-tp: ring via USB4 RDMA: right=%s(%s) left=%s(%s) "
                    "gid=%d/%d chunk=%zu B\n",
                    r->usb4_info.right_dev, r->usb4_info.right_peer,
                    r->usb4_info.left_dev, r->usb4_info.left_peer,
                    r->usb4_info.right_gid_index, r->usb4_info.left_gid_index,
                    r->usb4_info.chunk_bytes);
        }
    }
#endif
    if (r->backend != DS4_TP_RING_BACKEND_USB4) {
        fprintf(stderr,
                "ds4-tp: no usable USB4 RDMA device or IPv4-mapped GID on every "
                "rank; using the TCP ring\n");
    }

    /* Warm the path: the first RDMA write after RTS pays a one-off path
     * bring-up (~40 ms) and the first TCP round pays stream setup, so both ride
     * here instead of the first prompt token. Collective, and every rank is
     * already past the handshake. */
    unsigned char warm[DS4_TP_RING_WARMUP_ELEMS * 4];
    memset(warm, 0, sizeof(warm));
    for (unsigned i = 0; i < DS4_TP_RING_WARMUP_ROUNDS; i++) {
        if (ds4_tp_ring_all_reduce(r, warm, DS4_TP_RING_WARMUP_ELEMS,
                                  DS4_TP_RING_DTYPE_U32, DS4_TP_RING_OP_SUM,
                                  err, errlen) != 0) {
            ds4_tp_ring_destroy(r);
            return -1;
        }
    }

    if (opt->verbose) {
        fprintf(stderr, "ds4-tp: ring up: rank %d/%d backend=%s\n", r->rank,
                r->world, ds4_tp_ring_backend_name(r->backend));
    }
    if (out) *out = r;
    return 0;
}

void ds4_tp_ring_destroy(ds4_tp_ring *r) {
    if (!r) return;
#ifdef USE_USB4_RDMA
    if (r->usb4) usb4_rdma_ring_destroy(r->usb4);
#endif
    ring_close_sockets(r);
    free(r->scratch);
    free(r);
}

/* ---------------------------------------------------------- collectives */

int ds4_tp_ring_all_gather(ds4_tp_ring *r, const void *sendbuf, void *recvbuf,
                           size_t count, ds4_tp_ring_dtype dtype, char *err,
                           size_t errlen) {
    if (!r || !recvbuf) {
        ring_set_err(err, errlen, "ring: all-gather without a ring or buffer");
        return -1;
    }
    if (!ring_dtype_supports(dtype)) {
        ring_set_err(err, errlen, "ring: unsupported dtype %d", (int)dtype);
        return -1;
    }
    if (count == 0) return 0;
    ring_maybe_pin(r);

    if (r->backend == DS4_TP_RING_BACKEND_USB4) {
#ifdef USE_USB4_RDMA
        const int rc = usb4_rdma_ring_all_gather(r->usb4, sendbuf, recvbuf,
                                                 count,
                                                 (enum usb4_rdma_ring_dtype)dtype);
        if (rc != USB4_RDMA_RING_OK) {
            ring_set_err(err, errlen, "ring: USB4 all-gather failed: %s",
                         usb4_rdma_ring_strerror(rc));
            return -1;
        }
        return 0;
#else
        ring_set_err(err, errlen, "ring: built without USE_USB4_RDMA=1");
        return -1;
#endif
    }
    return ring_tcp_gather_full(r, sendbuf, recvbuf, count,
                                ring_dtype_size(dtype), err, errlen);
}

int ds4_tp_ring_all_reduce(ds4_tp_ring *r, void *buf, size_t count,
                           ds4_tp_ring_dtype dtype, ds4_tp_ring_op op, char *err,
                           size_t errlen) {
    if (!r || (!buf && count)) {
        ring_set_err(err, errlen, "ring: all-reduce without a ring or buffer");
        return -1;
    }
    if (!ring_dtype_supports(dtype)) {
        ring_set_err(err, errlen, "ring: unsupported dtype %d", (int)dtype);
        return -1;
    }
    if (op != DS4_TP_RING_OP_SUM && op != DS4_TP_RING_OP_MAX &&
        op != DS4_TP_RING_OP_MIN) {
        ring_set_err(err, errlen, "ring: unsupported op %d", (int)op);
        return -1;
    }
    if (count == 0) return 0;

    if (r->backend == DS4_TP_RING_BACKEND_USB4) {
#ifdef USE_USB4_RDMA
        const int rc = usb4_rdma_ring_all_reduce(r->usb4, buf, count,
                                                 (enum usb4_rdma_ring_dtype)dtype,
                                                 (enum usb4_rdma_ring_op)op);
        if (rc != USB4_RDMA_RING_OK) {
            ring_set_err(err, errlen, "ring: USB4 all-reduce failed: %s",
                         usb4_rdma_ring_strerror(rc));
            return -1;
        }
        return 0;
#else
        ring_set_err(err, errlen, "ring: built without USE_USB4_RDMA=1");
        return -1;
#endif
    }

    /* TCP ring: gather the world copies, then reduce them locally. Segment r of
     * the gather is rank r's whole contribution, so the element-wise sum needs
     * every segment -- not just the one this rank owns. */
    const size_t esize = ring_dtype_size(dtype);
    if ((dtype == DS4_TP_RING_DTYPE_F16 || dtype == DS4_TP_RING_DTYPE_BF16) &&
        op == DS4_TP_RING_OP_SUM) {
        ring_set_err(err, errlen,
                     "ring: half-precision all-reduce sum needs the USB4 backend");
        return -1;
    }
    if (count > SIZE_MAX / (esize * (size_t)r->world)) {
        ring_set_err(err, errlen, "ring: collective payload is too large");
        return -1;
    }
    if (ring_reserve_scratch(r, count * (size_t)r->world, esize, err, errlen) != 0)
        return -1;
    if (ring_tcp_gather_full(r, buf, r->scratch, count, esize, err, errlen) != 0)
        return -1;
    for (int peer = 0; peer < r->world; peer++) {
        if (peer == r->rank) continue;   /* buf already holds this rank's copy */
        const unsigned char *src =
            (const unsigned char *)r->scratch + (size_t)peer * count * esize;
        ring_reduce_into(buf, src, count, dtype, op);
    }
    return 0;
}

ds4_tp_ring_backend ds4_tp_ring_get_backend(const ds4_tp_ring *r) {
    return r ? r->backend : DS4_TP_RING_BACKEND_NONE;
}

const char *ds4_tp_ring_backend_name(ds4_tp_ring_backend backend) {
    switch (backend) {
    case DS4_TP_RING_BACKEND_USB4: return "usb4-rdma";
    case DS4_TP_RING_BACKEND_TCP: return "tcp-ring";
    case DS4_TP_RING_BACKEND_NONE: break;
    }
    return "none";
}

int ds4_tp_ring_rank(const ds4_tp_ring *r) { return r ? r->rank : -1; }
int ds4_tp_ring_world(const ds4_tp_ring *r) { return r ? r->world : 0; }
