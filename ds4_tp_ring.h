#ifndef DS4_TP_RING_H
#define DS4_TP_RING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Ring collective transport for tensor parallelism over a switchless
 * Thunderbolt/USB4 ring.
 *
 * Scope. Tensor parallelism in DS4 historically meant one leader and one
 * worker: every collective is a single pairwise exchange. This module supplies
 * the missing piece for N ranks -- an all-reduce/all-gather whose only
 * communication is rank r <-> rank (r+1) % world, which is exactly the
 * adjacency a switchless ring provides.
 *
 * Two backends, chosen at runtime in ds4_tp_ring_create():
 *   - USB4: one-sided IBV_WR_RDMA_WRITE through misc/usb4/usb4_rdma_ring.
 *     Compiled in only when the build sets USE_USB4_RDMA=1; that build is the
 *     only one that needs libibverbs headers or the library.
 *   - TCP: the same ring algorithm over the management network, so the
 *     collective contract is testable on a cluster with no RDMA devices and so
 *     a missing device degrades instead of failing.
 *
 * Contract, shared by both backends and by every caller:
 *   - `count` is an arbitrary runtime element count. It need not be a multiple
 *     of `world`, need not be a single token, and may be zero.
 *   - `buf` needs no particular alignment.
 *   - the reduction runs in the dtype the caller hands over; nothing is
 *     converted, widened or quantized on the way through.
 * All ranks must call the collectives in the same order with the same
 * count/dtype/op. A ds4_tp_ring is not thread-safe.
 */

#define DS4_TP_RING_MAX_WORLD 8

typedef enum {
    DS4_TP_RING_DTYPE_U8 = 0,
    DS4_TP_RING_DTYPE_I8,
    DS4_TP_RING_DTYPE_U16,
    DS4_TP_RING_DTYPE_I16,
    DS4_TP_RING_DTYPE_U32,
    DS4_TP_RING_DTYPE_I32,
    DS4_TP_RING_DTYPE_U64,
    DS4_TP_RING_DTYPE_I64,
    DS4_TP_RING_DTYPE_F16,
    DS4_TP_RING_DTYPE_BF16,
    DS4_TP_RING_DTYPE_F32,
    DS4_TP_RING_DTYPE_F64
} ds4_tp_ring_dtype;

typedef enum {
    DS4_TP_RING_OP_SUM = 0,
    DS4_TP_RING_OP_MAX,
    DS4_TP_RING_OP_MIN
} ds4_tp_ring_op;

typedef enum {
    DS4_TP_RING_BACKEND_NONE = 0,
    DS4_TP_RING_BACKEND_USB4 = 1,
    DS4_TP_RING_BACKEND_TCP = 2
} ds4_tp_ring_backend;

typedef struct {
    int rank;                   /* 0 .. world-1 */
    int world;                  /* 2 .. DS4_TP_RING_MAX_WORLD */
    /* Control-plane address per rank (index == rank), supplied by the caller.
     * Only the out-of-band handshake uses these; the data plane runs over the
     * ring links. Nothing here is compiled in. */
    const char *control_ip[DS4_TP_RING_MAX_WORLD];
    int port;                   /* rank r listens on port + 100 + r */
    /* USB4 device overrides; NULL selects runtime discovery. */
    const char *right_dev;
    const char *left_dev;
    int gid_index;              /* < 0 auto-selects the IPv4-mapped GID */
    size_t chunk_bytes;         /* per-rank segment size; 0 -> library default */
    int connect_timeout_ms;     /* 0 -> library default */
    /* Optional busy-spin CPU pinning. < 0 (the default) leaves scheduling to
     * the kernel; >= 0 pins the thread that first issues a collective and logs
     * the choice. */
    int cpu_core;
    bool verbose;
} ds4_tp_ring_options;

typedef struct ds4_tp_ring ds4_tp_ring;

/* Build the ring and run the warm-up collectives. Collectively blocking: all
 * ranks must call it, but no rank has to arrive at the same time. Returns 0 on
 * success, -1 with `err` set otherwise. A USB4 request falls back to TCP with a
 * one-line warning when no usable device or IPv4-mapped GID is present. */
int ds4_tp_ring_create(
        ds4_tp_ring **out,
        const ds4_tp_ring_options *opt,
        char *err,
        size_t errlen);
void ds4_tp_ring_destroy(ds4_tp_ring *ring);

/* Element-wise all-reduce over `count` elements of `dtype`, in place. */
int ds4_tp_ring_all_reduce(
        ds4_tp_ring *ring,
        void *buf,
        size_t count,
        ds4_tp_ring_dtype dtype,
        ds4_tp_ring_op op,
        char *err,
        size_t errlen);

/* recvbuf[rank*count + i] = sendbuf[i] from rank `rank`. `recvbuf` must hold
 * world*count elements. sendbuf may alias recvbuf only at offset 0. */
int ds4_tp_ring_all_gather(
        ds4_tp_ring *ring,
        const void *sendbuf,
        void *recvbuf,
        size_t count,
        ds4_tp_ring_dtype dtype,
        char *err,
        size_t errlen);

ds4_tp_ring_backend ds4_tp_ring_get_backend(const ds4_tp_ring *ring);
const char *ds4_tp_ring_backend_name(ds4_tp_ring_backend backend);
int ds4_tp_ring_rank(const ds4_tp_ring *ring);
int ds4_tp_ring_world(const ds4_tp_ring *ring);

#endif
