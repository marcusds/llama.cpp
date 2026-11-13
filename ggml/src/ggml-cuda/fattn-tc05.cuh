#pragma once

#include "common.cuh"

// Sparse FA for DKQ = DV = 512, 64 Q heads per K/V head (DeepSeek-V4 prefill) with tcgen05 MMAs (cc 10.x)
bool ggml_cuda_flash_attn_ext_tc05_usable(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_tc05(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
