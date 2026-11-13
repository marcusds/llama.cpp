#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
// add: a fused ADD node that computes src[0] (see ggml_cuda_dsv4_hc_post_can_add)
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr, const ggml_tensor * add = nullptr);
// true if add (src[0] of dst, no broadcast) can be fused into the hc_post dst
bool ggml_cuda_dsv4_hc_post_can_add(const ggml_tensor * add, const ggml_tensor * dst);
