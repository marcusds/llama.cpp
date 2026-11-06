// MUL_MAT_ID for NVFP4 weights with the cuBLASLt grouped FP4 GEMM (W4A4, VEC16_UE4M3 block scales).
// Activations are quantized to NVFP4 with a per-row fp32 scale applied to the output, alpha is 1.
#include "mmid-nvfp4-lt.cuh"

// needs the cuBLASLt grouped FP4 API, only tried with CUDA 13.3
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(CUDART_VERSION) && CUDART_VERSION >= 13030

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <vector>

#define LT_CHECK(x) do { const cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { GGML_ABORT("cublasLt error %d in %s", (int) s_, #x); } } while (0)

struct nvfp4_lt_pair {
    int64_t src_off; // byte offset of the src1 row
    int64_t dst_off; // byte offset of the dst row
    int64_t sc_base; // byte offset of the expert's activation scales
    int32_t row;     // row inside the expert
    int32_t pad;
};

// scale tile layout: 128 rows x 4 scale columns per 512 bytes, tiles in row-tile major order
static __device__ __forceinline__ int64_t nvfp4_lt_scale_off(const int row, const int col, const int ncol_tiles) {
    return ((int64_t) (row / 128) * ncol_tiles + col / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + col % 4;
}

// 8 ggml bytes (value j in the low, j+8 in the high nibble of byte j) -> cuBLAS order (2i low, 2i+1 high)
static __device__ __forceinline__ uint2 nvfp4_lt_shuffle(const uint32_t a, const uint32_t b) {
    const uint32_t la = a & 0x0F0F0F0Fu, lb = b & 0x0F0F0F0Fu;
    const uint32_t ha = (a >> 4) & 0x0F0F0F0Fu, hb = (b >> 4) & 0x0F0F0F0Fu;
    // the even bytes of x | (x >> 4) hold the nibble pairs
    return make_uint2(__byte_perm(la | (la >> 4), lb | (lb >> 4), 0x6420), __byte_perm(ha | (ha >> 4), hb | (hb >> 4), 0x6420));
}

// ggml block: 4 scales then 32 bytes of values; a CUDA block does 128 rows x 4 k-blocks,
// the scales go through shared memory so each scale tile is written as 512 contiguous bytes
static __global__ void __launch_bounds__(512) k_nvfp4_lt_repack_weights(
        const char * __restrict__ src, const int32_t * __restrict__ counts, uint8_t * __restrict__ dst_q, uint8_t * __restrict__ dst_s,
        const int64_t N, const int64_t K, const size_t nb01, const size_t nb02) {
    const int64_t a = blockIdx.z; // expert, empty experts are skipped
    if (counts[a] == 0) {
        return;
    }
    const int nkb = K / QK_NVFP4;
    const int kbl = threadIdx.x % 4;
    const int r   = threadIdx.x / 4;
    const int kb  = blockIdx.x * 4 + kbl;
    const int64_t n = (int64_t) blockIdx.y * 128 + r;

    __shared__ uint32_t sc[4][128];
    if (kb < nkb) {
        const uint32_t * p = (const uint32_t *) (src + a * nb02 + n * nb01 + (int64_t) kb * sizeof(block_nvfp4));
        uint32_t w[9];
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            w[j] = __ldg(p + j);
        }
        const uint2 o0 = nvfp4_lt_shuffle(w[1], w[2]);
        const uint2 o1 = nvfp4_lt_shuffle(w[3], w[4]);
        const uint2 o2 = nvfp4_lt_shuffle(w[5], w[6]);
        const uint2 o3 = nvfp4_lt_shuffle(w[7], w[8]);
        uint4 * q = (uint4 *) (dst_q + (a * N + n) * (K / 2) + (int64_t) kb * (QK_NVFP4 / 2));
        q[0] = make_uint4(o0.x, o0.y, o1.x, o1.y);
        q[1] = make_uint4(o2.x, o2.y, o3.x, o3.y);
        // position of row r inside its 128 row scale tile, see nvfp4_lt_scale_off
        sc[kbl][(r % 32) * 4 + r / 32] = w[0];
    }
    __syncthreads();
    const int c = threadIdx.x / 128;
    if (blockIdx.x * 4 + c < nkb) {
        uint32_t * s = (uint32_t *) (dst_s + a * N * (K / 16) + ((int64_t) blockIdx.y * nkb + blockIdx.x * 4 + c) * 512);
        s[threadIdx.x % 128] = sc[c][threadIdx.x % 128];
    }
}

static __device__ __forceinline__ uint8_t nvfp4_lt_e2m1(const float x) {
    return ggml_cuda_float_to_fp4_e2m1(x, 1.0f);
}

// one block per (expert sorted) row, writes packed e2m1, swizzled ue4m3 scales and the row scale
static __global__ void k_nvfp4_lt_quantize_act(
        const char * __restrict__ src, const nvfp4_lt_pair * __restrict__ pairs, uint8_t * __restrict__ dst_q, uint8_t * __restrict__ dst_s,
        float * __restrict__ row_scale, const int64_t K) {
    const nvfp4_lt_pair pr = pairs[blockIdx.x];
    if (pr.row < 0) {
        return; // invalid expert id, no row was placed
    }
    const float * x = (const float *) (src + pr.src_off);
    const int nsb = K / 16;

    float amax = 0.0f;
    for (int sb = threadIdx.x; sb < nsb; sb += blockDim.x) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            amax = fmaxf(amax, fabsf(x[sb*16 + j]));
        }
    }
    amax = warp_reduce_max<WARP_SIZE>(amax);
    __shared__ float smax[32];
    if (threadIdx.x % WARP_SIZE == 0) {
        smax[threadIdx.x / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = 0.0f;
    for (int w = 0; w < blockDim.x / WARP_SIZE; ++w) {
        amax = fmaxf(amax, smax[w]);
    }

    // sub-block scale amax_sub/6/g is at most 448, the largest finite ue4m3
    const float g     = amax / (6.0f * 448.0f);
    const float g_inv = g > 0.0f ? 1.0f / g : 0.0f;
    if (threadIdx.x == 0) {
        row_scale[blockIdx.x] = g;
    }

    for (int sb = threadIdx.x; sb < nsb; sb += blockDim.x) {
        float v[16];
        float amax_sb = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = x[sb*16 + j] * g_inv;
            amax_sb = fmaxf(amax_sb, fabsf(v[j]));
        }
        const __nv_fp8_e4m3 sc(amax_sb / 6.0f);
        const float sc_f   = (float) sc;
        const float sc_inv = sc_f > 0.0f ? 1.0f / sc_f : 0.0f;

        uint64_t o = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o |= (uint64_t) (nvfp4_lt_e2m1(v[2*j] * sc_inv) | (nvfp4_lt_e2m1(v[2*j + 1] * sc_inv) << 4)) << (8*j);
        }
        *(uint2 *) (dst_q + (int64_t) blockIdx.x * (K / 2) + sb * 8) = make_uint2((uint32_t) o, (uint32_t) (o >> 32));
        dst_s[pr.sc_base + nvfp4_lt_scale_off(pr.row, sb, (int) (K / 64))] = sc.__x;
    }
}

static __global__ void k_nvfp4_lt_scatter(
        const nv_bfloat16 * __restrict__ src, const nvfp4_lt_pair * __restrict__ pairs, const float * __restrict__ row_scale,
        char * __restrict__ dst, const int64_t N) {
    if (pairs[blockIdx.x].row < 0) {
        return;
    }
    const float g = row_scale[blockIdx.x];
    float * d = (float *) (dst + pairs[blockIdx.x].dst_off);
    for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
        d[n] = __bfloat162float(src[(int64_t) blockIdx.x * N + n]) * g;
    }
}

// device side routing: count tokens per expert, prefix sums + cuBLASLt group arrays, then place the pairs sorted by expert
static __global__ void k_nvfp4_lt_count(const char * __restrict__ ids, int32_t * __restrict__ counts,
        const int64_t n_used, const size_t nb0, const size_t nb1, const int64_t n_pairs, const int32_t E) {
    const int64_t p = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) {
        return;
    }
    const int32_t e = *(const int32_t *) (ids + (p / n_used) * nb1 + (p % n_used) * nb0);
    if (e >= 0 && e < E) {
        atomicAdd(counts + e, 1);
    }
}

#define NVFP4_LT_SETUP_THREADS 1024

// one block; group g is expert g. dims: 9 int64 arrays (A,B,D rows/cols/ld), then pointer arrays A,B,D,SA,SB
static __global__ void k_nvfp4_lt_setup(const int32_t * __restrict__ counts, int32_t * __restrict__ row_off, int64_t * __restrict__ sc_off,
        int64_t * __restrict__ meta, const int32_t E, const int64_t N, const int64_t K,
        uint8_t * w_q, uint8_t * w_s, uint8_t * x_q, uint8_t * x_s, nv_bfloat16 * y) {
    typedef cub::BlockScan<int32_t, NVFP4_LT_SETUP_THREADS> scan_t;
    __shared__ typename scan_t::TempStorage tmp;
    __shared__ int32_t carry[2];
    const int64_t kb16 = K / 16;
    int64_t * dims = meta;
    void ** ptrs = (void **) (meta + 9 * (int64_t) E);
    if (threadIdx.x == 0) {
        carry[0] = 0;
        carry[1] = 0;
    }
    __syncthreads();
    for (int32_t base = 0; base < E; base += NVFP4_LT_SETUP_THREADS) {
        const int32_t e = base + threadIdx.x;
        const int32_t M = e < E ? counts[e] : 0;
        const int32_t Mp = (M + 127) / 128 * 128;
        int32_t r, rp, t0, t1;
        scan_t(tmp).ExclusiveSum(M, r, t0);
        __syncthreads();
        scan_t(tmp).ExclusiveSum(Mp, rp, t1);
        __syncthreads();
        r += carry[0];
        rp += carry[1];
        if (e < E) {
            row_off[e] = r;
            sc_off[e]  = (int64_t) rp * kb16;
            const int64_t d[9] = { K, N, K,   K, M, K,   N, M, N };
            for (int i = 0; i < 9; ++i) {
                dims[i * (int64_t) E + e] = d[i];
            }
            ptrs[0 * (int64_t) E + e] = w_q + (int64_t) e * N * K / 2;
            ptrs[1 * (int64_t) E + e] = x_q + (int64_t) r * K / 2;
            ptrs[2 * (int64_t) E + e] = y   + (int64_t) r * N;
            ptrs[3 * (int64_t) E + e] = w_s + (int64_t) e * N * kb16;
            ptrs[4 * (int64_t) E + e] = x_s + (int64_t) rp * kb16;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            carry[0] += t0;
            carry[1] += t1;
        }
        __syncthreads();
    }
}

static __global__ void k_nvfp4_lt_place(const char * __restrict__ ids, const int32_t * __restrict__ row_off, const int64_t * __restrict__ sc_off,
        int32_t * __restrict__ cursor, nvfp4_lt_pair * __restrict__ pairs, const int64_t n_used, const size_t nb0, const size_t nb1,
        const int64_t n_pairs, const int32_t E, const int64_t src_nb1, const int64_t src_nb2, const int64_t src_ne1, const int64_t dst_nb1, const int64_t dst_nb2) {
    const int64_t p = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) {
        return;
    }
    const int64_t t = p / n_used, u = p % n_used;
    const int32_t e = *(const int32_t *) (ids + t * nb1 + u * nb0);
    if (e < 0 || e >= E) {
        return;
    }
    const int32_t r = atomicAdd(cursor + e, 1); // order inside an expert does not change the per row results
    nvfp4_lt_pair pr;
    pr.src_off = t * src_nb2 + (u % src_ne1) * src_nb1;
    pr.dst_off = u * dst_nb1 + t * dst_nb2;
    pr.sc_base = sc_off[e];
    pr.row     = r;
    pr.pad     = 0;
    pairs[row_off[e] + r] = pr;
}

bool ggml_cuda_mul_mat_id_nvfp4_lt_applicable(const ggml_tensor * dst, const int cc) {
    const char * env = getenv("GGML_CUDA_NVFP4_CUBLASLT");
    const bool enabled = env != nullptr && atoi(env) != 0;
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // above 16 tokens: smaller batches stay on MMVQ, from 17 tokens on this beats MMQ (17-32 streams, 30-250 token prompts)
    return enabled && cc >= 1000 && cc < 1100 &&
        src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_get_op_params_i32(dst, 3) != GGML_PREC_Q8 && // Q8 activations requested
        src1->ne[2] > 16 && src0->ne[0] % 64 == 0 && src0->ne[1] % 128 == 0 && src0->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float);
}

bool ggml_cuda_mul_mat_id_nvfp4_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t K = src0->ne[0], N = src0->ne[1], E = src0->ne[2];
    const int64_t n_tok = src1->ne[2], n_used = ids->ne[0];
    const int64_t n_pairs = n_tok * n_used;
    cudaStream_t stream = ctx.stream();

    static cublasLtHandle_t handles[GGML_CUDA_MAX_DEVICES] = {};
    cublasLtHandle_t lt = handles[ctx.device];
    if (lt == nullptr) {
        LT_CHECK(cublasLtCreate(&lt));
        handles[ctx.device] = lt;
    }

    const int64_t kb16 = K / 16;
    const int32_t Ei = (int32_t) E;
    // upper bound of the padded (128 rows per expert) activation scale rows
    const int64_t sc_rows_max = n_pairs + 127 * E;
    const size_t  sc_bytes    = sc_rows_max * kb16;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<uint8_t> w_q(pool, E * N * K / 2);
    ggml_cuda_pool_alloc<uint8_t> w_s(pool, E * N * kb16);
    ggml_cuda_pool_alloc<uint8_t> x_q(pool, n_pairs * K / 2);
    ggml_cuda_pool_alloc<uint8_t> x_s(pool, sc_bytes);
    ggml_cuda_pool_alloc<nv_bfloat16> y(pool, n_pairs * N);
    ggml_cuda_pool_alloc<float> row_scale(pool, n_pairs);
    ggml_cuda_pool_alloc<nvfp4_lt_pair> pairs_d(pool, n_pairs);
    ggml_cuda_pool_alloc<int32_t> counts_d(pool, 3 * E); // counts, cursor, row_off
    ggml_cuda_pool_alloc<int64_t> sc_off_d(pool, E);
    ggml_cuda_pool_alloc<int64_t> meta_d(pool, 14 * E);
    ggml_cuda_pool_alloc<char> work(pool, (size_t(64) << 20) + 256);
    const int64_t G = E;
    int32_t * counts = counts_d.ptr, * cursor = counts_d.ptr + E, * row_off = counts_d.ptr + 2 * E;

    // no host sync: the expert ids are routed on the device
    CUDA_CHECK(cudaMemsetAsync(counts_d.ptr, 0, 2 * E * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(pairs_d.ptr, 0xFF, n_pairs * sizeof(nvfp4_lt_pair), stream));
    CUDA_CHECK(cudaMemsetAsync(x_s.ptr, 0, sc_bytes, stream));
    const int nbp = (int) ((n_pairs + 255) / 256);
    k_nvfp4_lt_count<<<nbp, 256, 0, stream>>>((const char *) ids->data, counts, n_used, ids->nb[0], ids->nb[1], n_pairs, Ei);
    k_nvfp4_lt_setup<<<1, NVFP4_LT_SETUP_THREADS, 0, stream>>>(counts, row_off, sc_off_d.ptr, meta_d.ptr, Ei, N, K,
        w_q.ptr, w_s.ptr, x_q.ptr, x_s.ptr, y.ptr);
    k_nvfp4_lt_place<<<nbp, 256, 0, stream>>>((const char *) ids->data, row_off, sc_off_d.ptr, cursor, pairs_d.ptr, n_used, ids->nb[0], ids->nb[1],
        n_pairs, Ei, src1->nb[1], src1->nb[2], src1->ne[1], dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    {
        const int64_t nkb = K / QK_NVFP4;
        k_nvfp4_lt_repack_weights<<<dim3((nkb + 3) / 4, N / 128, E), 512, 0, stream>>>(
            (const char *) src0->data, counts, w_q.ptr, w_s.ptr, N, K, src0->nb[1], src0->nb[2]);
        CUDA_CHECK(cudaGetLastError());
    }
    k_nvfp4_lt_quantize_act<<<n_pairs, 128, 0, stream>>>((const char *) src1->data, pairs_d.ptr, x_q.ptr, x_s.ptr, row_scale.ptr, K);
    CUDA_CHECK(cudaGetLastError());

    const int64_t * dev = meta_d.ptr;
    void ** pa = (void **) (dev + 9*G);
    void ** pb = pa + G, ** pd = pb + G, ** psa = pd + G, ** psb = psa + G;

    cublasLtMatmulDesc_t op;
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t t_op = CUBLAS_OP_T, n_op = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t_op, sizeof(t_op)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n_op, sizeof(n_op)));
    const int sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
    // the scale pointers must be set before the heuristic call
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &psa, sizeof(psa)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &psb, sizeof(psb)));

    cublasLtMatrixLayout_t lay[3];
    const cudaDataType types[3] = { CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16BF };
    const int w64 = CUBLASLT_INTEGER_WIDTH_64;
    for (int i = 0; i < 3; ++i) {
        LT_CHECK(cublasLtGroupedMatrixLayoutCreate(&lay[i], types[i], G, (void *) (dev + (3*i + 0)*G), (void *) (dev + (3*i + 1)*G), (void *) (dev + (3*i + 2)*G)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(lay[i], CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH, &w64, sizeof(w64)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(lay[i], CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH, &w64, sizeof(w64)));
    }

    cublasLtMatmulPreference_t pref;
    LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const size_t ws = size_t(64) << 20;
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t hr;
    int nr = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, lay[0], lay[1], lay[2], lay[2], pref, 1, &hr, &nr));
    GGML_ASSERT(nr > 0);

    const float alpha = 1.0f, beta = 0.0f;
    LT_CHECK(cublasLtMatmul(lt, op, &alpha, pa, lay[0], pb, lay[1], &beta, pd, lay[2], pd, lay[2], &hr.algo, (void *) (((uintptr_t) work.ptr + 255) & ~(uintptr_t) 255), ws, stream));

    k_nvfp4_lt_scatter<<<n_pairs, 256, 0, stream>>>(y.ptr, pairs_d.ptr, row_scale.ptr, (char *) dst->data, N);
    CUDA_CHECK(cudaGetLastError());

    // free the cublasLt objects only after the stream is done with the host side arguments
    cublasLtMatmulPreferenceDestroy(pref);
    for (int i = 0; i < 3; ++i) {
        cublasLtMatrixLayoutDestroy(lay[i]);
    }
    cublasLtMatmulDescDestroy(op);
    return true;
}

#else

bool ggml_cuda_mul_mat_id_nvfp4_lt_applicable(const ggml_tensor *, int) { return false; }
bool ggml_cuda_mul_mat_id_nvfp4_lt(ggml_backend_cuda_context &, ggml_tensor *) { return false; }

#endif // cuBLASLt grouped FP4 available
