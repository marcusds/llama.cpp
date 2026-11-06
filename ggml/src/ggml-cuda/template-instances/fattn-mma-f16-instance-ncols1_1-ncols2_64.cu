// DeepSeek-V4 prefill: 64 heads of one query per tile (MQA, sparse gather)

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(512, 512, 1, 64);
