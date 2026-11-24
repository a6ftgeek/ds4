#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm.h"
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#define FULL_WARP_MASK 0xFFFFFFFFu
#define MASK_T uint32_t
#define DS4_GPU_BACKEND_NAME "CUDA"
#define DS4_GPU_LOG_PREFIX "ds4: CUDA "
#define DS4_GPU_BLAS_NAME "cuBLAS"
#endif

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"
#include "ds4_image.h"

static thread_local bool g_dspark_verify_mode;

extern "C" void ds4_gpu_set_dspark_verify_mode(bool enabled) {
    g_dspark_verify_mode = enabled;
}

extern "C" int ds4_mmq_init(int device);
extern "C" int ds4_mmq_iq2_xxs_moe_pair(
    const void *W_a, const void *W_b, const float *X_f32,
    const int32_t *ids, float *out_a, float *out_b,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    cudaStream_t stream);
extern "C" int ds4_mmq_q8_0_dense_vec(
    const void *W, const float *X_f32, float *out_f32,
    int M, int N, int K, cudaStream_t stream);
extern "C" int ds4_mmq_q2_K_moe_down_sum6_vec(
    const void *W, const float *X, const int32_t *ids, float *out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    cudaStream_t stream);
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

/* Network tensor parallelism: a bound rank owns one contiguous half of every
 * layer's routed experts (weights_model_map_sharded_spans in ds4.c maps exactly
 * that range).  The routed-MoE launcher filters the router selection to the
 * owned range and rebases the three expert-table offsets onto it, so a
 * non-owned expert's bytes are never read.  This mirrors ds4_cuda.cu's
 * g_network_tp_rank gate; rank -1 means "no network TP bound". */
int g_network_tp_rank = -1;
bool g_network_tp_sharding_suspended = false;
static int32_t *g_network_tp_pairs_ids;
static float *g_network_tp_pairs_weights;
static uint64_t g_network_tp_pairs_bytes;

static void ds4_rocm_tp_pairs_release(void) {
    if (g_network_tp_pairs_ids) (void)cudaFree(g_network_tp_pairs_ids);
    g_network_tp_pairs_ids = NULL;
    g_network_tp_pairs_weights = NULL;
    g_network_tp_pairs_bytes = 0;
}

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

typedef struct {
    uint8_t e;
    uint8_t qs[16];
} cuda_block_mxfp4;

static_assert(sizeof(cuda_block_mxfp4) == 17, "cuda_block_mxfp4 must match the GGUF MXFP4 block layout");

/* Twice the MXFP4 values so each 32-value sub-block can use signed-int8
 * dp4a; the factor of 1/2 is folded into the sub-block scale. */
__device__ __constant__ static const int8_t cuda_mxfp4_values_x2[16] = {
     0,  1,  2,  3,  4,  6,  8,  12,
     0, -1, -2, -3, -4, -6, -8, -12,
};

#include "ds4_iq2_tables_cuda.inc"

#include "rocm/ds4_rocm_runtime.cuh"

#include "rocm/ds4_rocm_common.cuh"

extern "C" int ds4_gpu_dspark_gfx1151_fast_path(void) {
    return ds4_rocm_is_gfx1151();
}

#include "rocm/ds4_rocm_q8.cuh"

#include "rocm/ds4_rocm_norm_rope.cuh"

#include "rocm/ds4_rocm_fp8_kv.cuh"

#include "rocm/ds4_rocm_attention.cuh"

#include "rocm/ds4_rocm_hc.cuh"

#include "rocm/ds4_rocm_output.cuh"

#include "rocm/ds4_rocm_indexer.cuh"

#include "rocm/ds4_rocm_embedding_launch.cuh"

#include "rocm/ds4_rocm_matmul.cuh"

#include "rocm/ds4_rocm_fp8_kv_launch.cuh"

#include "rocm/ds4_rocm_compressor.cuh"

#include "rocm/ds4_rocm_attention_launch.cuh"

#include "rocm/ds4_rocm_shared_expert.cuh"

#include "rocm/ds4_rocm_misc_launch.cuh"
#include "rocm/ds4_rocm_router.cuh"

#include "rocm/ds4_rocm_moe.cuh"

#include "rocm/ds4_rocm_moe_launch.cuh"

#include "rocm/ds4_rocm_glm.cuh"

#include "rocm/ds4_rocm_hc_output_launch.cuh"

#include "rocm/ds4_rocm_current_api_compat.cuh"

#include "ds4_glm53_vision_gpu.cuh"
#include "ds4_deepseek4_vision_gpu.cuh"
#include "rocm/ds4_rocm_deepseek4_vision.cuh"
#include "ds4_deepseek41_cuda.cuh"

/* Network tensor parallelism: the gate service.  This mirrors the CUDA
 * implementation in ds4_cuda.cu; ROCm orders every launch on the default
 * stream, so a device synchronize is the equivalent arrival barrier before the
 * host hands a gate slot to the transport. */
static struct {
    ds4_gpu_tp_exchange_fn row;
    ds4_gpu_tp_batch_exchange_fn batch;
    ds4_gpu_tp_big_exchange_fn big;
    void *ud, *staging;
    uint64_t vec_bytes, staging_bytes, row_seq, batch_seq;
    bool failed;
} g_rocm_tp;

extern "C" int ds4_gpu_tp_init(uint32_t rank, ds4_gpu_tensor *slab,
        uint64_t gpu_flags_off, uint64_t out_off, uint64_t vec_bytes,
        ds4_gpu_tp_exchange_fn fn, void *ud) {
    if (g_rocm_tp.row || rank > 1 || !slab || !fn || !vec_bytes ||
        out_off > slab->bytes || vec_bytes > slab->bytes - out_off ||
        gpu_flags_off >= slab->bytes) return 0;
    g_rocm_tp.row = fn;
    g_rocm_tp.ud = ud;
    g_rocm_tp.vec_bytes = vec_bytes;
    g_network_tp_rank = (int)rank;
    return 1;
}

extern "C" void ds4_gpu_tp_shutdown(void) {
    if (g_rocm_tp.row) (void)cudaDeviceSynchronize();
    g_network_tp_rank = -1;
    g_network_tp_sharding_suspended = false;
    ds4_rocm_tp_pairs_release();
    if (g_rocm_tp.staging) (void)cudaFreeHost(g_rocm_tp.staging);
    g_rocm_tp = {};
}

extern "C" int ds4_gpu_tp_failed(void) { return g_rocm_tp.failed ? 1 : 0; }

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    if (!g_rocm_tp.row || g_rocm_tp.failed) return 0;
    /* Do not leave the graph running ahead of a remote process. */
    const int ok = cuda_ok(cudaDeviceSynchronize(), "TP row arrival") &&
        g_rocm_tp.row(g_rocm_tp.ud, layer, gate, ++g_rocm_tp.row_seq);
    if (!ok) g_rocm_tp.failed = true;
    return ok;
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    g_rocm_tp.batch = fn;
}

extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    g_rocm_tp.big = fn;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    if (!g_rocm_tp.row || !g_rocm_tp.batch || g_rocm_tp.failed ||
        !rows || rows > 8) return 0;
    const int ok = cuda_ok(cudaDeviceSynchronize(), "TP batch arrival") &&
        g_rocm_tp.batch(g_rocm_tp.ud, layer, rows, ++g_rocm_tp.batch_seq);
    if (!ok) g_rocm_tp.failed = true;
    return ok;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
        const ds4_gpu_tensor *out_t, ds4_gpu_tensor *in_t, uint64_t bytes) {
    if (!g_rocm_tp.row || !g_rocm_tp.big || g_rocm_tp.failed ||
        !out_t || !in_t || !rows ||
        bytes != (uint64_t)rows * g_rocm_tp.vec_bytes ||
        bytes > SIZE_MAX / 2 || bytes > out_t->bytes ||
        bytes > in_t->bytes) return 0;
    if (bytes > g_rocm_tp.staging_bytes) {
        if (g_rocm_tp.staging) (void)cudaFreeHost(g_rocm_tp.staging);
        g_rocm_tp.staging = NULL;
        g_rocm_tp.staging_bytes = 0;
        if (!cuda_ok(cudaMallocHost(&g_rocm_tp.staging, (size_t)bytes * 2),
                     "TP bulk staging")) {
            g_rocm_tp.failed = true;
            return 0;
        }
        g_rocm_tp.staging_bytes = bytes;
    }
    void *peer = (char *)g_rocm_tp.staging + g_rocm_tp.staging_bytes;
    const int ok = cuda_ok(cudaDeviceSynchronize(), "TP bulk arrival") &&
        ds4_gpu_tensor_read(out_t, 0, g_rocm_tp.staging, bytes) &&
        g_rocm_tp.big(g_rocm_tp.ud, layer, ++g_rocm_tp.batch_seq,
                      g_rocm_tp.staging, peer, bytes) &&
        ds4_gpu_tensor_write(in_t, 0, peer, bytes);
    if (!ok) g_rocm_tp.failed = true;
    return ok;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    /* The DSpark drafter runs single-rank while this is set: keep the raw
     * (unfiltered) expert tables reachable for the leader's own draft pass. */
    g_network_tp_sharding_suspended = suspend != 0;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)full_in_dim; (void)k_off; (void)k_cnt; (void)out_dim; (void)x;
    (void)x_elem_off;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    (void)out; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups_total; (void)group0; (void)group_cnt; (void)out_dim;
    (void)heads;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add; (void)residual_hc;
    (void)post; (void)comb; (void)n_embd; (void)n_hc;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}
