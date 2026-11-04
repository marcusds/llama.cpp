#include "common.cuh"

// true when GGML_CUDA_NVFP4_CUBLASLT=1 and dst is a large NVFP4 MUL_MAT_ID on cc 10.x
bool ggml_cuda_mul_mat_id_nvfp4_lt_applicable(const ggml_tensor * dst, int cc);

// returns false when the op cannot be done here (the caller then uses the default path)
bool ggml_cuda_mul_mat_id_nvfp4_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
