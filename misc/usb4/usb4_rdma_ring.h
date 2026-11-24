/*
 * usb4_rdma_ring.h — switchless-ring collectives over USB4 RDMA (libibverbs),
 *                    with runtime HCA auto-discovery.
 *
 * Design: one-sided IBV_WR_RDMA_WRITE only, payload followed by a trailing
 * 64-bit sequence marker, receiver busy-polls the marker. No receive WQEs are
 * posted, so RNR stalls are structurally impossible.
 *
 * The library hardcodes no device names, IP addresses or node names: every
 * cluster detail is supplied through struct usb4_rdma_ring_params. If the two
 * device names are left NULL (or "auto"), the library discovers at runtime
 * which local HCA faces the right neighbour and which faces the left one
 * through the Thunderbolt sysfs topology, and auto-selects the local RoCE v2
 * IPv4-mapped GID index.
 *
 * Arbitrary sizes: count is any runtime element count. The implementation
 * splits the payload into balanced per-rank segments (remainder distributed to
 * the lowest ranks), pads only internally, and runs as many ring passes as
 * needed to cover the payload with the configured per-rank chunk size. The
 * trailing sequence number is strictly monotonic across all passes, segments
 * and calls.
 *
 * All ranks must call the collective functions in the same order with the same
 * count/dtype/op. A ring object is not thread-safe.
 *
 * NOTE (measured on this cluster): the USB4 link may take up to one QP ACK
 * timeout (~67 ms) to deliver the first write after the ring has been idle for
 * several milliseconds. Back-to-back collectives show no such stalls; callers
 * that interleave long compute gaps should expect an occasional one-off
 * wake-up penalty on the first collective after the gap.
 */
#ifndef USB4_RDMA_RING_H
#define USB4_RDMA_RING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define USB4_RDMA_RING_MAX_WORLD 8
#define USB4_RDMA_RING_DEV_NAME_MAX 64
#define USB4_RDMA_RING_PEER_NAME_MAX 128
#define USB4_RDMA_RING_DEFAULT_CHUNK_BYTES (256u * 1024u)

/* Element types understood by all_reduce (SUM/MAX/MIN) and all_gather. */
enum usb4_rdma_ring_dtype {
    USB4_RDMA_RING_DTYPE_U8 = 0,
    USB4_RDMA_RING_DTYPE_I8,
    USB4_RDMA_RING_DTYPE_U16,
    USB4_RDMA_RING_DTYPE_I16,
    USB4_RDMA_RING_DTYPE_U32,
    USB4_RDMA_RING_DTYPE_I32,
    USB4_RDMA_RING_DTYPE_U64,
    USB4_RDMA_RING_DTYPE_I64,
    USB4_RDMA_RING_DTYPE_F16,      /* IEEE binary16 */
    USB4_RDMA_RING_DTYPE_BF16,     /* bfloat16 */
    USB4_RDMA_RING_DTYPE_F32,
    USB4_RDMA_RING_DTYPE_F64
};

enum usb4_rdma_ring_op {
    USB4_RDMA_RING_OP_SUM = 0,
    USB4_RDMA_RING_OP_MAX,
    USB4_RDMA_RING_OP_MIN
};

/* Status codes (negative). */
enum usb4_rdma_ring_status {
    USB4_RDMA_RING_OK = 0,
    USB4_RDMA_RING_ERR_INVAL = -1,      /* bad parameter / unsupported dtype+op */
    USB4_RDMA_RING_ERR_NO_DEVICE = -2,  /* no USB4 RDMA HCA present at all */
    USB4_RDMA_RING_ERR_DISCOVERY = -3,  /* HCAs present, peer mapping unresolved */
    USB4_RDMA_RING_ERR_SYS = -4,        /* verbs / socket / memory failure */
    USB4_RDMA_RING_ERR_TIMEOUT = -5,    /* handshake or completion timed out */
    USB4_RDMA_RING_ERR_CORRUPT = -6     /* unexpected completion / marker state */
};

struct usb4_rdma_ring_params {
    int rank;                       /* 0 .. world-1 */
    int world;                      /* 2 .. USB4_RDMA_RING_MAX_WORLD */
    /* Management/control-plane IPv4 address of every rank (index == rank).
     * Required; used for the out-of-band TCP handshake and, unless peer_name is
     * given, reverse-resolved to the rank's Thunderbolt device_name. */
    const char *control_ip[USB4_RDMA_RING_MAX_WORLD];
    /* Optional Thunderbolt XDomain device_name per rank (index == rank).
     * NULL entry -> reverse DNS of control_ip[rank]. No names are hardcoded. */
    const char *peer_name[USB4_RDMA_RING_MAX_WORLD];
    int port;                       /* base TCP port; rank r listens on port+100+r */
    /* Device overrides. NULL or "auto" -> runtime discovery. Set both or
     * neither; a single override is rejected. */
    const char *right_dev;
    const char *left_dev;
    /* Local RoCE v2 GID index. <0 -> auto-select the first IPv4-mapped GID. */
    int gid_index;
    /* Per-rank internal chunk size in bytes; 0 -> 256 KiB. This bounds the MR
     * footprint (memory ~ world*chunk + NSLOT*chunk) and sets how many ring
     * passes a large call takes. It is fixed for the ring's lifetime because
     * peers exchange MR keys at init time. */
    size_t chunk_bytes;
    /* Dummy collective iterations in init() to absorb the one-off ~40 ms path
     * bring-up. 0 -> default 2; <0 disables. MUST match across all ranks. */
    int warmup_iters;
    /* Optional CPU pinning, disabled by default. pin_cpu=1 pins to cpu_core. */
    int pin_cpu;
    int cpu_core;
    int connect_timeout_ms;         /* 0 -> 60000 */
    int verbose;                    /* 0/1 diagnostic logging to stderr */
};

struct usb4_rdma_ring_info {
    char right_dev[USB4_RDMA_RING_DEV_NAME_MAX];
    char left_dev[USB4_RDMA_RING_DEV_NAME_MAX];
    char right_peer[USB4_RDMA_RING_PEER_NAME_MAX];
    char left_peer[USB4_RDMA_RING_PEER_NAME_MAX];
    int right_gid_index;
    int left_gid_index;
    uint8_t right_gid[16];
    uint8_t left_gid[16];
    int discovered;                 /* 1 = names came from sysfs discovery */
    size_t chunk_bytes;             /* effective internal chunk size */
};

struct usb4_rdma_ring;              /* opaque */

/* Create a ring: resolve HCAs (discovery or overrides), open verbs resources,
 * resolve GIDs, perform the out-of-band TCP exchange and bring both QPs to RTS.
 * IMPORTANT: init() runs a small collective warmup by default, so EVERY rank
 * must call it (it is not a local-only call). Returns USB4_RDMA_RING_OK on
 * success. On failure *ring is NULL and the returned code explains why;
 * USB4_RDMA_RING_ERR_NO_DEVICE means the caller should fall back to another
 * transport. Never blocks indefinitely. */
int usb4_rdma_ring_init(struct usb4_rdma_ring **ring,
                        const struct usb4_rdma_ring_params *params,
                        struct usb4_rdma_ring_info *info);

/* In-place element-wise all-reduce over `count` elements of `dtype`.
 * `buf` may be arbitrarily aligned; count is arbitrary (not necessarily a
 * multiple of world), including 0 (no-op success) and 1. */
int usb4_rdma_ring_all_reduce(struct usb4_rdma_ring *ring, void *buf,
                              size_t count, enum usb4_rdma_ring_dtype dtype,
                              enum usb4_rdma_ring_op op);

/* Concatenate each rank's `count`-element send buffer into
 * recvbuf[rank*count + i] (world*count elements total). */
int usb4_rdma_ring_all_gather(struct usb4_rdma_ring *ring, const void *sendbuf,
                              void *recvbuf, size_t count,
                              enum usb4_rdma_ring_dtype dtype);

/* Release all resources. NULL is a no-op. */
int usb4_rdma_ring_destroy(struct usb4_rdma_ring *ring);

/* Pin the current thread to `core` after init. core < 0 restores the affinity
 * mask that was in effect at init(). Returns 0 or a status code. */
int usb4_rdma_ring_set_cpu(struct usb4_rdma_ring *ring, int core);

/* Human-readable status string. */
const char *usb4_rdma_ring_strerror(int status);

#ifdef __cplusplus
}
#endif

#endif /* USB4_RDMA_RING_H */
