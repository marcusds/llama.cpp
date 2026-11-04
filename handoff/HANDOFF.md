# Handoff: DeepSeek-V4.1-Flash NVFP4 GGUF + llama-swap / llama.cpp update

Written 2026-11-03 on the GB300 station (`/opt/llm`). Everything here is on the fork `marcusds/llama.cpp`. Nothing has been opened as a PR upstream.

## 1. What was asked and where it stands

| Goal | Status |
|---|---|
| Update llama-swap | Done: v262 -> v269 (`bin/llama-swap`, backup `bin/llama-swap.bak-20261103-v262`) |
| Update llama.cpp | Done: llama-server rebuilt from local branch `station-20261103` (master `2cbca389f` + the 9-commit grammar PR #30193 series, force-pushed upstream on 2026-11-02). Installed binary reports `0.8.0-dev (build 12072, commit 64c1c1211)`. Backups: `bin/llama-server.bak-20261103`, `lib.bak-20261103` |
| llama-swap shows/starts/stops `deepseek-v4.1-flash` (container `dsv41-vllm-a`) | Done and verified (see 2) |
| GGUF of `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` | Done, loads and answers correctly; speed tuning inconclusive (see 5) |
| Draft PR for the converter change | NOT opened. Branch is on the fork. Rules in 8 |

## 2. llama-swap and the DeepSeek container

Cause of the bad state: the fragment was proxy-only with `cmd: sleep infinity` and `ttl: 600`, so llama-swap never knew about the container.

Fix (files in `handoff/llama-swap/`, installed in `/opt/llm`):
- `etc/conf.d/deepseek-v4.1-flash.yaml`: `cmd: /opt/llm/vllm-a/llama-swap-dsv41.sh`, `cmdStop: docker stop --time 60 dsv41-vllm-a`, `unloadTimeout: 90`, `ttl: 0`, `checkEndpoint: /health`. The old fragment is `deepseek-v4.1-flash.yaml.before` (also `etc/bak-20261103-deepseek.yaml`).
- `vllm-a/llama-swap-dsv41.sh`: `docker start` the existing container, then `exec docker logs -f`. The process lives exactly as long as the container.
- The container is still CREATED by `vllm-a/serve-dsv41-uva.sh`; only start/stop moved to llama-swap.

Verified: `/unload?model=deepseek-v4.1-flash` ran `docker stop` (container exit code 137: the 60 s grace expired because it was still loading). Later a request through :9292 started the container (11:29 UTC) and llama-swap showed it `ready` once `/health` returned 200.

Gotchas:
- Cold start is about 24 min (weights ~13.5 min, draft model ~2.3 min, compile + graph capture ~3.7 min). It pins ~440 GiB of host RAM, which is the OOM source on this box.
- `/unload?model=X` unloaded EVERY model (the matrix residents too). Re-warm `nemotron-nano-4b` and `nomic-embed-text-v1.5` afterwards.
- A llama-swap restart or `systemctl stop` now also `docker stop`s DeepSeek if it is loaded.
- If 60 s is too short for a fully loaded engine, raise the `cmdStop` grace to ~120 s and `unloadTimeout` above it.
- The container may be running right now (someone started it); it competes with any GPU benchmark.

## 3. The GGUF

`/data/llm/models/DeepSeek-V4.1-Flash-1017-NVFP4.gguf`, 427,057,133,344 bytes, 1283 tensors. Source checkpoint (400 GB): `/data/llm/models/deepseek-v41-flash-1017-nvfp4` (hf download of the repo). It is NOT in git.

Layout: routed experts NVFP4 (bit-exact repack, 120 tensors `{384, 2304|5120, 5120|2304}`) with `[384]` F32 `.scale` (weight_scale_2) and `.input_scale`; Engram tables kept as MXFP4 (about 98 GB); FP8 linears dequantized to BF16 (lossless); the rest BF16/F32.

Regenerate (about 15 min on this box):
```
cd src/llama.cpp-upstream   # branch with the converter changes
/opt/llm/vllm/.venv/bin/python convert_hf_to_gguf.py --outtype bf16 \
  --outfile /data/llm/models/DeepSeek-V4.1-Flash-1017-NVFP4.gguf \
  /data/llm/models/deepseek-v41-flash-1017-nvfp4
```
Do not use `python -I` (it hides the repo's `conversion` package). `--dry-run` checks tensor preparation only, not the packed bytes.

Fidelity notes (from the model card): the NVFP4 repo converted MXFP4 experts to NVFP4 losslessly, but converted the Engram tables from FP8 to MXFP4, which is lossy. So this GGUF is lossier than the original checkpoint in the Engram only. Not compared against vLLM output yet.

## 4. Code (branches on the fork)

| Branch | Contents |
|---|---|
| `dsv41-nvfp4-convert` | Clean PR candidate on top of upstream master `ca8daad19`: NVFP4 experts + MXFP4 Engram in `conversion/deepseek41.py`; per-expert scale tensors passed to `build_moe_ffn` in `src/models/deepseek41.cpp`; bundled V4.1 chat template embedded by default (`models/templates/deepseek-ai-DeepSeek-V4.1-Flash.jinja`). Two commits. |
| `station-20261103` | What the live llama-server is built from: master `2cbca389f` + the 9 grammar commits + the same changes. Rebase onto new master and keep the grammar PR until it merges. |
| `dsv41-nvfp4-handoff` | This branch: the PR branch plus `handoff/`. Do not send this one upstream. |

Why the runtime change matters: the `.scale` tensors are loaded as optional, but the DeepSeek-V4.1 graph did not pass them to `build_moe_ffn`. Without it the GGUF loads and silently produces garbage. It is a no-op for the MXFP4 checkpoint (tensors absent).

Chat template: the NVFP4 repo ships no `chat_template`, so llama.cpp fell back to ChatML and the model wrote a literal `<|im_end|>`. The bundled template is the model's own port of `encoding/encoding.py`, edited because llama.cpp's jinja engine rejects (a) `dict(a, **{k: v})` kwargs unpacking and (b) chained comparisons `1 <= x <= 100`. Verified with `handoff/scripts/render.cpp` (links libllama-common; same code path as the server): NVIDIA's `encoding/tests` cases 1-4 are byte-exact, reasoning-effort mapping checked for max/high/low/1/42/100 and invalid values raise. Case 5 (images) not testable (text-only GGUF). NOT yet tested: an actual generation with the template loaded. Existing GGUFs have no template; serve with `--jinja --chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1-Flash.jinja`.

## 5. Benchmarks (read the caveats first)

Scripts: `handoff/scripts/bench.py` (streaming, 3x192-token decode, ~3.7K-token prefill, 4 concurrent) and `sweep4.sh` (waits for an idle GPU, runs each variant cold then warm, flags CONTAMINATED). Raw output in `handoff/results/`. Test server: standalone `llama-server` on :9494, ctx 32768, `--n-gpu-layers 99`, `--jinja`, default ChatML template.

Caveats:
- This is a shared box (k3s, clickhouse, other people's llama-server and vLLM jobs). The same flags gave ~13 tok/s decode in the first sweep and ~9 tok/s in the clean sweep. Treat absolute numbers as +/- 40%.
- The "warm" `long-prefill` figures (7,000-22,000 tok/s) are INVALID: the same prompt was resent and hit the prompt cache. Only the cold figure is a prefill measurement.
- `mmap-base` failed to start (CUDA OOM while another job held the GPU). `none-ub2048` and `mlock` are flagged CONTAMINATED.

Cold results, 3,743-token prompt (decode = median of 3 short generations, tok/s):

| Variant (all `--n-cpu-moe 15` unless noted) | VRAM | Prefill TTFT | Decode |
|---|---|---|---|
| default mmap, ub 1024 / 2048 / 4096 | ~201-204 GB | 257 / 262 / 252 s (~14-15 tok/s) | 8.9 / 8.7 / 9.2 |
| `--load-mode none` (ub 512 default) | 201 GB | 410 s in the clean sweep, 7 s in the first sweep (unexplained) | 9.0 |
| `--load-mode none -ub 2048` | 202 GB | 4.1 s (~915 tok/s) | 9.1 |
| `--load-mode mmap+mlock` | 252 GB (suspect) | 272 s | 9.0 |
| `--load-mode mlock` | 220 GB | 7.7 s | 5.0; swap activity, 4-stream throughput collapsed to 0.2 tok/s |
| `-t 36` | 201 GB | 270 s | 4.7 |
| `-t 48` | 215 GB | 604 s | 1.6 cold, 7.9 warm |
| first sweep: `--n-cpu-moe` 11 / 15 / 20 | 230 / 206 / 165 GB | 196 / 293 / 285 s | 13.6 / 13.0 / 13.5 |

What I think this shows (moderate confidence):
- Decode is about the same across every flag combination tried; where experts live moves it by under 5%. Default thread count (72) beat 36 and 48.
- Concurrency does not help: 4 streams total about the same tok/s as one.
- Prefill is the lever. With the default mmap load, prefill is ~14 tok/s regardless of ubatch. With `--load-mode none` plus `-b 2048 -ub 2048`, a 3.7K prompt took 4 s. Hypothesis (unconfirmed): with mmap'd weights the scheduler does not offload the host-resident expert matmuls to the GPU for large batches, so the CPU does prefill; with a normal allocation it does offload. One `none` run at default ubatch took 410 s, so this is not settled.
- Avoid `mlock`: it pinned ~110 GB, free RAM fell to ~14 GB and swap was used. This is the same failure class as the vLLM OOM.
- Best known config to start from: `--n-cpu-moe 15 --load-mode none -b 2048 -ub 2048`.

Memory: page cache holds most of the GGUF (up to ~386 GB); host weights resident ~110 GB in `none` mode. vLLM, for comparison, pins ~440 GiB.

## 6. Not done / next steps

1. Re-run the prefill comparison on an idle machine with a DIFFERENT prompt per run (avoid the prompt cache): `mmap` vs `none` vs `none -ub 2048`, several repeats, to settle the 410 s vs 7 s discrepancy. Use `sweep4.sh` (edit the variant list).
2. Generation test with the bundled chat template (confirm the stray `<|im_end|>` is gone, check `reasoning_content` splitting and tool calls).
3. Fair comparison against vLLM. The running vLLM uses DSpark (5 draft tokens); the GGUF has no draft model. Needs the container up and exclusive use of the GPU.
4. Output quality check against vLLM (the Engram is 4-bit in this checkpoint).
5. NVIDIA's own vLLM recipe for the NVFP4 repo (`--tokenizer-mode deepseek_v41`, `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`, `--kv-cache-dtype fp8`, `--engram-config '{"cpu_offload":true}'`, `--reasoning-parser deepseek_v41`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`, TP4 in their setup) needs a vLLM build with unmerged PR vllm-project/vllm#61530; our image `vllm/vllm-openai:deepseekv41-flash-0909` probably lacks it (unchecked). Possible upside: FlashInfer backend may avoid the sm_103 FlashMLA sparse-prefill crash with DSpark on tool calls; NVFP4 halves the pinned Engram RAM.
6. Wire the GGUF into llama-swap (a new `conf.d` fragment): remember every reload restarts all models. Model residency: do not co-load with the vLLM container.
7. Rebuild/install: the live `bin/llama-server` already includes the NVFP4 scale change only if you reinstall from `src/llama.cpp-upstream/build` (the test server ran from `build/bin`; `/opt/llm/lib` and `bin/llama-server` were installed BEFORE that change). Verify with `bin/llama-server --version` and re-run the cutover steps in `AGENTS.md` if you install.

## 7. Environment notes

- `/opt/llm` is not a git repo; configs, wrappers and backups live only on the box (copies in `handoff/llama-swap/`).
- `gh` is logged in as `marcusds`. The `fork` remote is HTTPS with no credentials here; push over SSH: `git push git@github.com:marcusds/llama.cpp.git <branch>`.
- A raised memlock is needed only for `mlock` modes: `sudo -n prlimit --pid <pid> --memlock=unlimited:unlimited` (the shell limit is 8 MB).
- Another job (`~/workspace/bench-pr30193/run-bench.sh`, llama-server on :18080) ran on the same GPU during the night; check for such jobs before benchmarking.

## 8. Upstream contribution rules (`src/llama.cpp-upstream/AGENTS.md`)

For ggml-org/llama.cpp: the contributor writes the PR description and any reviewer replies; an AI must not run `gh pr create`; commit messages are better written by the contributor (you OK'd drafts here); the contributor must understand and be able to defend every line; discuss feature work in an issue first; disclose meaningful AI help per the PR template (this conflicts with the global no-attribution preference: decide before opening). Code comments: short, ASCII only. No existing upstream PR for DeepSeek-V4.1 NVFP4 conversion was found (searched 2026-11-03); the architecture landed in #31086. Rebase `dsv41-nvfp4-convert` onto current master first (master moves daily).

## 9. Machines and copies (added 2026-11-04)

| Host | RAM | GPU | Disk free | Role |
|---|---|---|---|---|
| this station (`/opt/llm`, GB300) | 482 GiB (~494 GB) | GB300, 256,703 MiB (~251 GiB) | `/data` 6.6 TB | where everything was built and benchmarked; the GGUF was made here |
| `h-station` (`4u2g-0415`) | 251 GiB | RTX A6000 | 1.7 TB on `/` | has copies only; the GPU is far too small to run this model |
| `m-station-1` (`pmgb300ws-0160`, GB300) | 482 GiB | GB300, 256,703 MiB | 3.2 TB on `/` | has copies; same hardware as this station |

Copies on both `h-station` and `m-station-1`:
- `/data/llm/models/DeepSeek-V4.1-Flash-1017-NVFP4.gguf`, 427,057,133,344 bytes, verified with a SHA-256 per 16 chunks against the original, GGUF header present.
- `/opt/llm/HANDOFF.md`: an earlier version of this doc (before this section). Use the git branch for the current one.

How it was copied: one `rsync` stream got only ~25 MB/s on `h-station` (per-connection limit); 16 parallel `dd | ssh dd` streams got ~400 MB/s there and ~1.5 GB/s to `m-station-1`. `handoff/scripts/ptransfer2.sh <host>` does the chunked copy with retries and per-chunk hash verification, and renames `.part` to the final name only if every chunk matches. Notes: the destination sshd drops simultaneous logins (stagger them, the script retries); a killed rsync leaves its partial data under the FINAL filename (delete it); never `pkill -f` a pattern that appears in your own command line.

`m-station-1` setup state:
- `/data/llm/models` and `/opt/llm` were created by hand (sudo, owned by `local-mschwab`, mode 2775; there is no `llm` group there). Nothing else from `/opt/llm` exists there: no binaries, no source checkouts, no llama-swap.
- Its NVIDIA driver (595.99.02, open) did not respond until the machine was rebooted. After the reboot `nvidia-smi` shows the GB300 with 255 GB free. The kernel log still has a boot-time `RmInitAdapter failed` / "GPU memory zone movable auto onlining failed" at ~52 s that recovered, and the box reports 8 NUMA nodes (0,2-8) where the original station reports 9. Run a real CUDA test before relying on it for this model.
- To use the GGUF there: clone `dsv41-nvfp4-handoff` (or `station-20261103`), build llama.cpp for CUDA arch 103 (flags in section 7 of the build memory: Release, shared libs, `-DCMAKE_CUDA_ARCHITECTURES=103`), then follow section 5 for serve flags. The source checkpoint (`hf download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`, 400 GB) is only needed to re-convert.

Correction to earlier chat statements: GPU memory on the GB300 is 256,703 MiB (~251 GiB), not 288 GB.
