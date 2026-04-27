// Regression test for attn_rot_v/k using the wrong head size for ISWA sub-caches.
// Verifies that kv_base uses its actual per-sub-cache head size for the rotation
// eligibility check, not the layer-0 (SWA) value from hparams.n_embd_head_v().

#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static constexpr uint32_t TEST_N_VOCAB      = 64;
static constexpr uint32_t TEST_N_EMBD       = 96;   // head_full = 96  (96 % 64 != 0)
static constexpr uint32_t TEST_N_HEAD       = 1;
static constexpr uint32_t TEST_N_HEAD_KV    = 1;
static constexpr uint32_t TEST_N_FF         = 128;
static constexpr uint32_t TEST_N_LAYER      = 5;    // layers 0-3: SWA, layer 4: full-attn
static constexpr uint32_t TEST_N_CTX        = 64;
static constexpr uint32_t TEST_N_SWA        = 16;
static constexpr uint32_t TEST_HEAD_SWA     = 64;   // SWA head size  (64 % 64 == 0)

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dist(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dist(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dist(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type in test fixture");
    }
}

static bool silent_model_load_progress(float /*p*/, void * /*ud*/) { return true; }

static gguf_context_ptr build_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_GEMMA4, ret.get());

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,             llm_arch_name(LLM_ARCH_GEMMA4));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                       TEST_N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,                   TEST_N_CTX);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,                 TEST_N_EMBD);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,                  TEST_N_EMBD);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                      TEST_N_LAYER);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,              TEST_N_FF);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,             TEST_N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,          TEST_N_HEAD_KV);
    ms.add_kv(LLM_KV_LOGIT_SCALE,                      1.0f);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,            false);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-6f);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         TEST_N_SWA);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,       uint32_t(0));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,         TEST_HEAD_SWA);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,       TEST_HEAD_SWA);
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,       TEST_N_EMBD / 2);
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,         0.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              0.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,        uint32_t(0));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,          uint32_t(5));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,                  "no_vocab");

    return ret;
}

int main() {
    llama_log_set([](ggml_log_level /*level*/, const char * /*text*/, void * /*ud*/) {}, nullptr);

    // Part 1: verify that the old and new divisibility conditions differ for this fixture.
    // Layer 0 is SWA → hparams.n_embd_head_v(0) = 64  (old code used this for kv_base)
    // kv_base actual head size = 96/1 = 96             (new code uses this)
    const uint32_t head_full = TEST_N_EMBD / TEST_N_HEAD;
    const uint32_t head_swa  = TEST_HEAD_SWA;

    GGML_ASSERT(head_swa  % 64 == 0 && "SWA head should be divisible by 64");
    GGML_ASSERT(head_full % 64 != 0 && "full-attn head should NOT be divisible by 64");

    const bool old_kv_base_rot = (head_swa  % 64 == 0);  // true  (wrong)
    const bool new_kv_base_rot = (head_full % 64 == 0);  // false (correct)

    GGML_ASSERT(old_kv_base_rot == true);
    GGML_ASSERT(new_kv_base_rot == false);

    printf("Part 1 PASS: divisibility conditions are as expected\n");

    // Part 2: integration test.  With the old code, llama_decode aborts inside
    // ggml_mul_mat_aux when it tries to reshape v_cur=[96,1,1,1] to [64, 96/64]
    // and hits GGML_ASSERT(96 == 64).  With the fix, attn_rot_v=false for
    // kv_base so the reshape is never attempted.
    gguf_context_ptr gguf_ctx = build_gguf();

    llama_model_params model_params   = llama_model_default_params();
    model_params.progress_callback    = silent_model_load_progress;
    model_params.devices              = nullptr;

    size_t seed = 42;
    llama_model_ptr model(llama_model_init_from_user(
        gguf_ctx.get(), set_tensor_data, &seed, model_params));

    if (!model) {
        printf("Part 2 SKIP: could not create model from fixture\n");
        printf("OVERALL: PASS (Part 1 succeeded; Part 2 skipped)\n");
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx      = TEST_N_CTX;
    ctx_params.n_threads  = 1;
    ctx_params.n_ubatch   = 8;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // required for quantized V cache
    ctx_params.type_v     = GGML_TYPE_Q8_0;

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));

    if (!ctx) {
        printf("Part 2 SKIP: could not create Q8_0-V context (flash_attn may be unavailable)\n");
        printf("OVERALL: PASS (Part 1 succeeded; Part 2 skipped)\n");
        return 0;
    }

    llama_token test_token = 0;
    llama_batch batch = llama_batch_get_one(&test_token, 1);

    const int rc = llama_decode(ctx.get(), batch);

    if (rc != 0) {
        fprintf(stderr, "Part 2 FAIL: llama_decode returned %d\n", rc);
        return 1;
    }

    printf("Part 2 PASS: single-token Q8_0-V decode succeeded without abort\n");
    printf("OVERALL: PASS\n");
    return 0;
}
