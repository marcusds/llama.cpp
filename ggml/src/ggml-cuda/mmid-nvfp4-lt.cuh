#include "common.cuh"

// true when GGML_CUDA_NVFP4_CUBLASLT=1 and dst is a large NVFP4 MUL_MAT_ID on cc 10.x
bool ggml_cuda_mul_mat_id_nvfp4_lt_applicable(const ggml_tensor * dst, int cc);

// optional output stage: out[i] = glu(gate[i], y[i]*expert_scale[e]) (without gate: y[i]*expert_scale[e]),
// out and gate have the layout of dst. With weights (router weights, [n_used, n_tokens] contiguous) the
// output is instead the weighted sum over the used experts, out[t] = sum_u weights[t,u]*expert_scale[e]*y[t,u],
// contiguous [ne0, n_tokens]
struct ggml_cuda_nvfp4_lt_epilogue {
    const float * expert_scale = nullptr;
    const float * gate         = nullptr;
    const float * weights      = nullptr;
    ggml_glu_op   glu_op       = GGML_GLU_OP_SWIGLU;
    float         glu_limit    = 0.0f;
    void        * out          = nullptr;
};

// returns false when the op cannot be done here (the caller then uses the default path)
bool ggml_cuda_mul_mat_id_nvfp4_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_nvfp4_lt_epilogue * epi = nullptr);

// call at the start of each graph compute: drops the cached activations unless the graph continues the previous one
void ggml_cuda_mul_mat_id_nvfp4_lt_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);
void ggml_cuda_mul_mat_id_nvfp4_lt_free(ggml_backend_cuda_context & ctx);

// NVFP4 split layout (weights in the CUDA NVFP4 split buffer type, and the device copies of host weights):
// the 32 value bytes of all blocks in block order, then the 4 scale bytes of all blocks, in block order or
// (nkb != 0, blocks per row) in the cuBLAS scale tile order (see nvfp4_split_scale_idx); same size as ggml.
// Converts nblocks ggml blocks from src (device) to dst.
void ggml_cuda_nvfp4_split_convert(const void * src, void * dst, int64_t nblocks, int64_t nkb, cudaStream_t stream);
// blocks [b0, b0 + n) of a tensor of nb blocks: src points at block b0 in the ggml layout (device memory or
// host memory the device can read), dst is the start of the split tensor
void ggml_cuda_nvfp4_split_convert_range(const void * src, void * dst, int64_t b0, int64_t n, int64_t nb, int64_t nkb, cudaStream_t stream);
// the inverse: nblocks blocks in the split layout from src (device) to the ggml layout in dst
void ggml_cuda_nvfp4_split_unconvert(const void * src, void * dst, int64_t nblocks, int64_t nkb, cudaStream_t stream);
