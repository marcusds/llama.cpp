#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tc05.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
#define FATTN_TC05_HOST
#endif

#ifdef FATTN_TC05_HOST

#include <atomic>
#include <cuda.h>
#include <cudaTypedefs.h>

// Sparse flash attention with tcgen05 (datacenter Blackwell, sm_100a/sm_103a).
// One CTA per query: the 64 Q heads are the MMA M dimension, each query gathers its own K rows (V == K).
// Warps 0-3: Q load, softmax, epilogue; warp 4: TMEM alloc + MMA issue; warps 5-12: K row gather (TMA gather4)
// and the mask values of each block. K uses 3 stages of 64 keys, P is double buffered. Q is staged in smem
// (bulk copies) and O is written through smem (bulk stores) while the K stages are idle.
// All MMAs are weight-stationary (.ws) M = 64: TMEM lanes 0-63 and 64-127 compute the two N halves.
// TMEM columns: O (64 x 512 f32) 0-255, Q (f16) 256-383, S (2 buffers) 384-511.
//   Q: lane l holds row l%64, dims (l/64)*256 + [0, 256), so QK^T with N = 128 gives in lanes 0-63 the
//   dot products over dims 0-255 and in lanes 64-127 those over dims 256-511 (B = the same keys twice,
//   K tile 64-dim chunks stored in the order 0,4,1,5,2,6,3,7). The softmax adds the two partials.
//   O half h (dims h*256 + [0, 256)) at cols h*128: lane l holds dims h*256 + (l/64)*128 + [0, 128) of row l%64.

#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ >= 1000 && __CUDA_ARCH_SPECIFIC__ < 1100
#define FATTN_TC05_AVAILABLE
#elif defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100
#define FATTN_TC05_AVAILABLE
#endif

#define FATTN_TC05_D        512
#define FATTN_TC05_H        64
#define FATTN_TC05_BK       64   // keys per block
#define FATTN_TC05_NS       3    // K stages
#define FATTN_TC05_THREADS  416

static constexpr size_t fattn_tc05_smem_k = FATTN_TC05_BK*FATTN_TC05_D*sizeof(half);
static constexpr size_t fattn_tc05_smem_p = FATTN_TC05_H*FATTN_TC05_BK*sizeof(half);
static constexpr size_t fattn_tc05_smem_x = FATTN_TC05_H*FATTN_TC05_BK*sizeof(float);
static constexpr size_t fattn_tc05_smem   = 1024 + FATTN_TC05_NS*fattn_tc05_smem_k + 2*fattn_tc05_smem_p + fattn_tc05_smem_x;

#ifdef FATTN_TC05_AVAILABLE

static __device__ __forceinline__ uint32_t tc05_smem(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

// SW128 smem descriptor, 8-row groups 1024 B apart
static __device__ __forceinline__ uint64_t tc05_desc(const uint32_t addr, const uint32_t lbo) {
    return (uint64_t) ((addr >> 4) & 0x3FFF) | ((uint64_t) ((lbo >> 4) & 0x3FFF) << 16) |
        ((uint64_t) (1024 >> 4) << 32) | ((uint64_t) 1 << 46) | ((uint64_t) 2 << 61);
}

// f16 x f16 -> f32, M = 64
static __host__ __device__ constexpr uint32_t tc05_idesc(const int N, const int b_mn) {
    return (1u << 4) | ((uint32_t) b_mn << 16) | ((uint32_t) (N >> 3) << 17) | ((uint32_t) (64 >> 4) << 24);
}

// A and B in smem
static __device__ __forceinline__ void tc05_mma_ss(const uint32_t d, const uint64_t a, const uint64_t b, const uint32_t idesc, const uint32_t acc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc));
}

// A in TMEM
static __device__ __forceinline__ void tc05_mma_ts(const uint32_t d, const uint32_t a, const uint64_t b, const uint32_t idesc, const uint32_t acc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n"
                 :: "r"(d), "r"(a), "l"(b), "r"(idesc), "r"(acc));
}

static __device__ __forceinline__ void tc05_commit(uint64_t * bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(tc05_smem(bar)) : "memory");
}

static __device__ __forceinline__ void tc05_bar_init(uint64_t * bar, const int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(tc05_smem(bar)), "r"(n));
}

static __device__ __forceinline__ void tc05_bar_wait(uint64_t * bar, const uint32_t parity) {
    asm volatile("{\n\t.reg .pred p;\n\tWAIT_%=:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra WAIT_%=;\n\t}\n" :: "r"(tc05_smem(bar)), "r"(parity) : "memory");
}

static __device__ __forceinline__ void tc05_bar_arrive(uint64_t * bar) {
    asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared::cta.b64 st, [%0];\n\t}\n" :: "r"(tc05_smem(bar)) : "memory");
}

static __device__ __forceinline__ void tc05_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
static __device__ __forceinline__ void tc05_fence_after()  { asm volatile("tcgen05.fence::after_thread_sync;"  ::: "memory"); }
static __device__ __forceinline__ void tc05_fence_proxy()  { asm volatile("fence.proxy.async.shared::cta;"     ::: "memory"); }
static __device__ __forceinline__ void tc05_sync_softmax() { asm volatile("bar.sync 1, 128;"                   ::: "memory"); }

// 32 lanes x 32 consecutive columns, thread t <-> lane (warp quadrant base) + t
static __device__ __forceinline__ void tc05_ld32(const uint32_t addr, uint32_t (&r)[32]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
          "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
          "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
        : "r"(addr));
}

static __device__ __forceinline__ void tc05_st32(const uint32_t addr, const uint32_t (&r)[32]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
        "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
        :: "r"(addr),
           "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
           "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
           "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
           "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]) : "memory");
}

static __device__ __forceinline__ void tc05_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
static __device__ __forceinline__ void tc05_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

#endif // FATTN_TC05_AVAILABLE

__launch_bounds__(FATTN_TC05_THREADS, 1)
static __global__ void flash_attn_ext_tc05(
        const __grid_constant__ CUtensorMap tmK, const char * __restrict__ Q, const half * __restrict__ mask,
        const float * __restrict__ sinks, const int * __restrict__ indices, const int * __restrict__ counts,
        float * __restrict__ dst, const float scale, const int n_q, const int n_kv_max,
        const int64_t nb01, const int64_t nb02, const int64_t s31) {
#ifdef FATTN_TC05_AVAILABLE
    constexpr int D  = FATTN_TC05_D;
    constexpr int H  = FATTN_TC05_H;
    constexpr int BK = FATTN_TC05_BK;
    constexpr int NS = FATTN_TC05_NS;
    constexpr float log2e = 1.4426950408889634f;
    constexpr uint32_t tO = 0, tQ = 256, tS = 384;

    extern __shared__ uint8_t smem_raw[];
    uint8_t * smem = (uint8_t *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    uint8_t * sK    = smem;
    uint8_t * sP    = sK + NS*fattn_tc05_smem_k;
    float   * sX    = (float *) (sP + 2*fattn_tc05_smem_p);

    __shared__ uint64_t bar_q, bar_qld, bar_pfull[2], bar_odone[2];
    __shared__ float sMaskSt[NS][BK];
    __shared__ uint64_t bar_kfull[NS], bar_kempty[NS], bar_sfull[2];
    __shared__ uint32_t tmem_base;
    __shared__ float sRed[2][H];
    __shared__ float sSum[2][H];

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int iq   = blockIdx.x;

    const int count = counts[iq];
    const int nblk  = (count + BK - 1) / BK;
    const int * idx_g = indices + int64_t(iq)*n_kv_max;
    const half * mask_q = mask + int64_t(iq)*s31;

    if (warp == 4) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(tc05_smem(&tmem_base)));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (tid == 0) {
        tc05_bar_init(&bar_q,     128);
        tc05_bar_init(&bar_qld,   1);
        tc05_bar_init(&bar_pfull[0], 128);
        tc05_bar_init(&bar_pfull[1], 128);
        tc05_bar_init(&bar_odone[0], 1);
        tc05_bar_init(&bar_odone[1], 1);
        for (int b = 0; b < NS; ++b) {
            tc05_bar_init(&bar_kfull[b], 8);
            tc05_bar_init(&bar_kempty[b], 1);
        }
        tc05_bar_init(&bar_sfull[0], 1);
        tc05_bar_init(&bar_sfull[1], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    tc05_fence_before();
    __syncthreads();
    tc05_fence_after();
    const uint32_t tm = tmem_base;

    // Q rows (64 x 2 KB f32) into the K stage area, row stride QS bytes keeps the row reads bank conflict free
    constexpr int QS = D*sizeof(float) + 16;
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(tc05_smem(&bar_qld)), "r"(H*D*(int) sizeof(float)) : "memory");
        const char * Q_q = Q + int64_t(iq)*nb01;
        for (int r = 0; r < H; ++r) {
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :: "r"(tc05_smem(sK + r*QS)), "l"(Q_q + r*nb02), "r"(D*(int) sizeof(float)), "r"(tc05_smem(&bar_qld)) : "memory");
        }
    }
    if (warp >= 5) {
        // K gather: 16 groups of 4 rows, 8 lanes per group (one 64-dim chunk each, lanes 8-15 and 24-31 idle);
        // dim chunk c of the tile goes to position (c%4)*2 + c/4; lanes c < 4 also provide the mask values
        const int  lane = tid % 32;
        const int  c    = lane % 8;
        const int  pos  = (c % 4)*2 + c/4;
        const int  r    = 4*((warp - 5)*2 + lane / 16);
        const bool act  = lane % 16 < 8;
        auto load_idx = [&](const int j, int (&ix)[4]) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int p = j*BK + r + t;
                ix[t] = p < count ? idx_g[p] : -1;
            }
        };
        auto load_mask = [&](const int (&ix)[4]) {
            const int it = ix[c % 4];
            return act && c < 4 && it >= 0 ? __half2float(mask_q[it])*log2e : -INFINITY;
        };
        int ix[4];
        load_idx(0, ix);
        float mv = load_mask(ix);
        tc05_bar_wait(&bar_q, 0); // the Q staging area has been read
        for (int j = 0; j < nblk; ++j) {
            const int st = j % NS;
            int nx[4];
            if (j + 1 < nblk) {
                load_idx(j + 1, nx);
            }
            if (j >= NS) {
                tc05_bar_wait(&bar_kempty[st], (j/NS - 1) & 1);
            }
            if (act && c < 4) {
                sMaskSt[st][r + c] = mv;
            }
            __syncwarp();
            if (lane == 0) {
                if (warp == 5) {
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                        :: "r"(tc05_smem(&bar_kfull[st])), "r"((int) fattn_tc05_smem_k) : "memory");
                } else {
                    tc05_bar_arrive(&bar_kfull[st]);
                }
            }
            if (act) {
                const uint32_t dsts = tc05_smem(sK + st*fattn_tc05_smem_k) + pos*BK*128 + r*128;
                asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                    :: "r"(dsts), "l"(&tmK), "r"(c*64), "r"(max(ix[0], 0)), "r"(max(ix[1], 0)), "r"(max(ix[2], 0)), "r"(max(ix[3], 0)),
                       "r"(tc05_smem(&bar_kfull[st])) : "memory");
            }
            if (j + 1 < nblk) {
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    ix[t] = nx[t];
                }
                mv = load_mask(ix);
            }
        }
    } else if (warp == 4) {
        if (tid % 32 == 0 && nblk > 0) {
            constexpr uint32_t idS = tc05_idesc(128, 0);
            constexpr uint32_t idO = tc05_idesc(256, 1);

            auto issue_pv = [&](const int i) {
                tc05_bar_wait(&bar_pfull[i % 2], (i/2) & 1);
                tc05_fence_proxy();
                tc05_fence_after();
                const uint32_t aP = tc05_smem(sP + (i % 2)*fattn_tc05_smem_p);
                const uint32_t aV = tc05_smem(sK + (i % NS)*fattn_tc05_smem_k);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int kk = 0; kk < BK/16; ++kk) {
                        tc05_mma_ss(tm + tO + h*128, tc05_desc(aP + kk*32, 16), tc05_desc(aV + h*BK*128 + kk*16*128, 2*BK*128),
                            idO, i > 0 || kk > 0);
                    }
                }
                tc05_commit(&bar_odone[i % 2]);
                tc05_commit(&bar_kempty[i % NS]);
            };

            tc05_bar_wait(&bar_q, 0);
            tc05_fence_after();
            for (int j = 0; j < nblk; ++j) {
                tc05_bar_wait(&bar_kfull[j % NS], (j/NS) & 1);
                tc05_fence_proxy();
                tc05_fence_after();
                const uint32_t aK = tc05_smem(sK + (j % NS)*fattn_tc05_smem_k);
#pragma unroll
                for (int kk = 0; kk < D/32; ++kk) {
                    tc05_mma_ts(tm + tS + (j % 2)*BK, tm + tQ + kk*8, tc05_desc(aK + 2*(kk/4)*BK*128 + (kk%4)*32, 16), idS, kk > 0);
                }
                tc05_commit(&bar_sfull[j % 2]);
                if (j > 0) {
                    issue_pv(j - 1);
                }
            }
            issue_pv(nblk - 1);
        }
        __syncwarp();
    } else {
        // softmax warps: thread t <-> TMEM lane t, row t%64, part t/64 (dims or keys half)
        const int row  = tid % 64;
        const int part = tid / 64;
        const uint32_t lane_addr = tm + ((uint32_t) (32*warp) << 16);

        // Q row, dims part*256 + [0, 256), f32 -> packed f16 pairs in TMEM
        {
            tc05_bar_wait(&bar_qld, 0);
            const float4 * Q_r = (const float4 *) (sK + row*QS + part*256*sizeof(float));
#pragma unroll 1
            for (int c0 = 0; c0 < 128; c0 += 64) {
                float4 v[32];
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    v[i] = Q_r[c0/2 + i];
                }
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    uint32_t r[32];
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float4 f = v[16*k + i];
                        const half2 a = __floats2half2_rn(f.x, f.y);
                        const half2 b = __floats2half2_rn(f.z, f.w);
                        r[2*i + 0] = *(const uint32_t *) &a;
                        r[2*i + 1] = *(const uint32_t *) &b;
                    }
                    tc05_st32(lane_addr + tQ + c0 + 32*k, r);
                }
            }
            tc05_wait_st();
            tc05_fence_proxy();
            tc05_fence_before();
            tc05_bar_arrive(&bar_q);
        }

        const float scale_l2 = scale*log2e;
        float m = -1e30f; // same in both threads of a row
        float l = 0.0f;   // partial sum over this thread's keys

        for (int j = 0; j < nblk; ++j) {
            tc05_bar_wait(&bar_sfull[j % 2], (j/2) & 1);
            tc05_fence_after();
            // keep keys part*32 + [0, 32), hand the other half to the partner thread
            uint32_t mine[32], other[32];
            tc05_ld32(lane_addr + tS + (j % 2)*BK + part*32,       mine);
            tc05_ld32(lane_addr + tS + (j % 2)*BK + (1 - part)*32, other);
            tc05_wait_ld();
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                sX[((1 - part)*32 + i)*H + row] = __uint_as_float(other[i]);
            }
            tc05_bar_wait(&bar_kfull[j % NS], (j/NS) & 1); // mask values of this stage
            tc05_sync_softmax();
            const float * mrow = sMaskSt[j % NS] + part*32;
            float s[32];
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                s[i] = (__uint_as_float(mine[i]) + sX[(part*32 + i)*H + row])*scale_l2 + mrow[i];
                mx = fmaxf(mx, s[i]);
            }
            sRed[part][row] = mx;
            tc05_sync_softmax();
            mx = fmaxf(mx, sRed[1 - part][row]);

            const float m_new = fmaxf(m, mx);
            // rescale only when the max grows by more than 2^8, P stays <= 256 in f16
            const bool need = m_new > m + 8.0f;
            if (j > 0 && __any_sync(0xFFFFFFFF, need)) {
                // PV(j-1) has to finish before O is rescaled
                tc05_bar_wait(&bar_odone[(j - 1) % 2], ((j - 1)/2) & 1);
                tc05_fence_after();
                {
                    const float alpha = need ? exp2f(m - m_new) : 1.0f;
#pragma unroll 1
                    for (int c = 0; c < 256; c += 32) {
                        uint32_t o[32];
                        tc05_ld32(lane_addr + tO + c, o);
                        tc05_wait_ld();
#pragma unroll
                        for (int i = 0; i < 32; ++i) {
                            o[i] = __float_as_uint(__uint_as_float(o[i])*alpha);
                        }
                        tc05_st32(lane_addr + tO + c, o);
                    }
                    tc05_wait_st();
                    l *= alpha;
                }
            }
            if (need) {
                m = m_new;
            }

            float sum = 0.0f;
            uint32_t p[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float p0 = exp2f(s[2*i + 0] - m);
                const float p1 = exp2f(s[2*i + 1] - m);
                sum += p0 + p1;
                const half2 h = __floats2half2_rn(p0, p1);
                p[i] = *(const uint32_t *) &h;
            }
            l += sum;

            if (j >= 2) {
                // P buffer of PV(j-2)
                tc05_bar_wait(&bar_odone[j % 2], ((j - 2)/2) & 1);
            }
            uint8_t * sPr = sP + (j % 2)*fattn_tc05_smem_p + row*128;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int uu = part*4 + t;
                *(uint4 *) (sPr + ((uu ^ (row % 8))*16)) = make_uint4(p[4*t + 0], p[4*t + 1], p[4*t + 2], p[4*t + 3]);
            }
            tc05_fence_proxy();
            tc05_fence_before();
            tc05_bar_arrive(&bar_pfull[j % 2]);
        }

        // epilogue
        sSum[part][row] = l;
        tc05_sync_softmax();
        float lt = sSum[0][row] + sSum[1][row];
        if (sinks) {
            lt += exp2f(sinks[row]*log2e - m);
        }
        const float inv = nblk > 0 && lt > 0.0f ? 1.0f/lt : 0.0f;
        if (nblk > 0) {
            tc05_bar_wait(&bar_odone[(nblk - 1) % 2], ((nblk - 1)/2) & 1);
            tc05_fence_after();
        }
        // all MMAs are done, stage O in the K area and write whole rows with bulk copies
        float * srow = (float *) (sK + row*QS) + part*128;
#pragma unroll 1
        for (int c = 0; c < 256; c += 32) {
            uint32_t o[32];
            if (nblk > 0) {
                tc05_ld32(lane_addr + tO + c, o);
                tc05_wait_ld();
            }
            float * d = srow + (c/128)*256 + c % 128;
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                float4 v;
                v.x = nblk > 0 ? __uint_as_float(o[i + 0])*inv : 0.0f;
                v.y = nblk > 0 ? __uint_as_float(o[i + 1])*inv : 0.0f;
                v.z = nblk > 0 ? __uint_as_float(o[i + 2])*inv : 0.0f;
                v.w = nblk > 0 ? __uint_as_float(o[i + 3])*inv : 0.0f;
                *(float4 *) (d + i) = v;
            }
        }
        tc05_fence_proxy();
        tc05_sync_softmax();
        if (tid < H) {
            float * dst_row = dst + (int64_t(iq)*H + tid)*D;
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                :: "l"(dst_row), "r"(tc05_smem(sK + tid*QS)), "r"(D*(int) sizeof(float)) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        }
    }

    tc05_fence_before();
    __syncthreads();
    if (warp == 4) {
        tc05_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
    }
#else
    GGML_UNUSED_VARS(tmK, Q, mask, sinks, indices, counts, dst, scale, n_q, n_kv_max, nb01, nb02, s31);
    NO_DEVICE_CODE;
#endif // FATTN_TC05_AVAILABLE
}

static bool fattn_tc05_compiled_for(const int device) {
    // the real kernel declares static smem (barriers), the stub for other archs does not
    static std::atomic<int> state[GGML_CUDA_MAX_DEVICES] = {};
    int s = state[device].load();
    if (s == 0) {
        ggml_cuda_set_device(device);
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, flash_attn_ext_tc05));
        s = attr.sharedSizeBytes > 0 ? 1 : 2;
        state[device].store(s);
    }
    return s == 1;
}

bool ggml_cuda_flash_attn_ext_tc05_usable(const int device, const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < 1000 || cc >= 1100) {
        return false;
    }
    static const bool disabled = [] {
        const char * e = getenv("GGML_CUDA_FA_TC05");
        return e != nullptr && atoi(e) == 0;
    }();
    if (disabled) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    const bool ok =
        Q->type == GGML_TYPE_F32 && Q->ne[0] == FATTN_TC05_D && Q->ne[2] == FATTN_TC05_H && Q->ne[1] > 1 && Q->nb[0] == sizeof(float) &&
        Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && (uintptr_t) Q->data % 16 == 0 && Q->ne[3] == 1 &&
        K->type == GGML_TYPE_F16 && K->ne[0] == FATTN_TC05_D && K->ne[2] == 1 && K->ne[3] == 1 && K->nb[0] == sizeof(half) &&
        K->nb[1] % 16 == 0 && (uintptr_t) K->data % 16 == 0 &&
        V->type == GGML_TYPE_F16 && V->ne[0] == FATTN_TC05_D && V->data == K->data && V->nb[1] == K->nb[1] &&
        mask != nullptr && mask->type == GGML_TYPE_F16 && mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        mask->ne[3] == 1 &&
        (sinks == nullptr || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] == FATTN_TC05_H)) &&
        dst->src[5] == nullptr && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        n_kv_max > 0 && K->ne[1] >= 2LL*n_kv_max;
    return ok && fattn_tc05_compiled_for(device);
}

void ggml_cuda_flash_attn_ext_tc05(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    cudaStream_t stream = ctx.stream();

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    const int n_q = Q->ne[1];

    const size_t n_lists = n_q;
    ggml_cuda_pool_alloc<int> idx(ctx.pool(), size_t(n_kv_max)*n_lists + n_lists);
    ggml_cuda_flash_attn_ext_compact_mask(mask, idx.ptr, idx.ptr + size_t(n_kv_max)*n_lists, n_q, 1, n_kv_max, stream);

    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {};
    if (!smem_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(flash_attn_ext_tc05, cudaFuncAttributeMaxDynamicSharedMemorySize, fattn_tc05_smem));
        smem_set[ctx.device] = true;
    }

    // K rows as a 2D tensor: 512 f16 x K->ne[1] rows, read 4 rows x 64 values at a time, 128 B swizzle
    static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    if (encode == nullptr) {
        cudaDriverEntryPointQueryResult qres;
        void * fn = nullptr;
        CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qres));
        GGML_ASSERT(qres == cudaDriverEntryPointSuccess && fn != nullptr);
        encode = (PFN_cuTensorMapEncodeTiled_v12000) fn;
    }
    CUtensorMap tmK;
    const cuuint64_t gdim[2]    = { (cuuint64_t) K->ne[0], (cuuint64_t) K->ne[1] };
    const cuuint64_t gstride[1] = { (cuuint64_t) K->nb[1] };
    const cuuint32_t box[2]     = { 64, 1 };
    const cuuint32_t estride[2] = { 1, 1 };
    const CUresult res = encode(&tmK, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, K->data, gdim, gstride, box, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    GGML_ASSERT(res == CUDA_SUCCESS);

    const dim3 grid(n_q, 1, 1);
    flash_attn_ext_tc05<<<grid, FATTN_TC05_THREADS, fattn_tc05_smem, stream>>>(
        tmK, (const char *) Q->data, (const half *) mask->data,
        sinks ? (const float *) sinks->data : nullptr, idx.ptr, idx.ptr + size_t(n_kv_max)*n_lists,
        (float *) dst->data, scale, n_q, n_kv_max, Q->nb[1], Q->nb[2], mask->nb[1]/sizeof(half));
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_flash_attn_ext_tc05_usable(const int device, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(device, dst);
    return false;
}

void ggml_cuda_flash_attn_ext_tc05(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("tcgen05 flash attention is not available");
}

#endif // FATTN_TC05_HOST
