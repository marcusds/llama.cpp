#include "common.cuh"

// true when GGML_CUDA_NVFP4_CUBLASLT=1 and dst is a large NVFP4 MUL_MAT_ID on cc 10.x
bool ggml_cuda_mul_mat_id_nvfp4_lt_applicable(const ggml_tensor * dst, int cc);

// optional output stage: out[i] = glu(gate[i], y[i]*expert_scale[e]) (without gate: y[i]*expert_scale[e]),
// out and gate have the layout of dst
struct ggml_cuda_nvfp4_lt_epilogue {
    const float * expert_scale = nullptr;
    const float * gate         = nullptr;
    ggml_glu_op   glu_op       = GGML_GLU_OP_SWIGLU;
    float         glu_limit    = 0.0f;
    void        * out          = nullptr;
};

// returns false when the op cannot be done here (the caller then uses the default path)
bool ggml_cuda_mul_mat_id_nvfp4_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_nvfp4_lt_epilogue * epi = nullptr);

// the cached activations of the previous call must not be reused in a new graph
void ggml_cuda_mul_mat_id_nvfp4_lt_invalidate(ggml_backend_cuda_context & ctx);
void ggml_cuda_mul_mat_id_nvfp4_lt_free(ggml_backend_cuda_context & ctx);
