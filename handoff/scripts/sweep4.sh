#!/usr/bin/env bash
# Clean re-run of all GGUF tuning experiments on :9494. Waits for the GPU to be free of other jobs,
# re-checks before every variant, and flags a variant CONTAMINATED if another GPU job appeared during it.
S=/tmp/claude-2583/-opt-llm/6208515b-34fc-4614-9a6c-09726d5719d6/scratchpad
SRC=/opt/llm/src/llama.cpp-upstream
OUT=$S/sweep4.out
GGUF=/data/llm/models/DeepSeek-V4.1-Flash-1017-NVFP4.gguf
: > $OUT

others() {  # number of GPU compute processes that are not my :9494 server, plus other llama-servers / bench scripts
  local mine; mine=$(pgrep -f 'llama-server --host 127.0.0.1 --port 9494' | tr '\n' ' ')
  local n=0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    case " $mine " in *" $p "*) ;; *) n=$((n+1));; esac
  done
  n=$((n + $(pgrep -af 'llama-server|run-bench' | grep -v -E 'port 9494|sweep4|pgrep' | wc -l)))
  echo $n
}
wait_idle() {
  local waited=0
  while [ "$(others)" -gt 0 ]; do
    [ $waited = 0 ] && echo "waiting for other GPU jobs to finish ($(date -u +%T))" >> $OUT
    waited=1; sleep 30
  done
  [ $waited = 1 ] && echo "idle again, continuing ($(date -u +%T))" >> $OUT
  return 0
}

variants=(
 "none|--n-cpu-moe 15 --load-mode none"
 "mmap-base|--n-cpu-moe 15"
 "mmap-ub1024|--n-cpu-moe 15 -b 2048 -ub 1024"
 "mmap-ub2048|--n-cpu-moe 15 -b 2048 -ub 2048"
 "mmap-ub4096|--n-cpu-moe 15 -b 4096 -ub 4096"
 "none-ub2048|--n-cpu-moe 15 --load-mode none -b 2048 -ub 2048"
 "mmap+mlock|--n-cpu-moe 15 --load-mode mmap+mlock"
 "mlock|--n-cpu-moe 15 --load-mode mlock"
 "mmap-t36|--n-cpu-moe 15 -t 36 -tb 36"
 "mmap-t48|--n-cpu-moe 15 -t 48 -tb 48"
)
for v in "${variants[@]}"; do
  tag=${v%%|*}; flags=${v#*|}
  wait_idle
  pkill -f 'llama-server --host 127.0.0.1 --port 9494' 2>/dev/null; sleep 5
  t0=$(date +%s)
  LD_LIBRARY_PATH=$SRC/build/bin nohup $SRC/build/bin/llama-server --host 127.0.0.1 --port 9494 -m $GGUF \
    --n-gpu-layers 99 $flags --ctx-size 32768 --no-webui --jinja > $S/srv4-$tag.log 2>&1 &
  pid=$!
  sleep 1
  case "$flags" in *lock*) sudo -n prlimit --pid $pid --memlock=unlimited:unlimited ;; esac
  ok=0
  for i in $(seq 1 240); do
    if curl -s -m3 localhost:9494/health | grep -q ok; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  if [ $ok = 0 ]; then echo "$tag FAILED to start: $(tail -2 $S/srv4-$tag.log | cut -c1-200)" >> $OUT; continue; fi
  echo "$tag load=$(( $(date +%s) - t0 ))s vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) free_ram=$(free -g | awk 'NR==2{print $4"G/avail "$7"G"}') mlocked_kB=$(grep VmLck /proc/$pid/status | awk '{print $2}')" >> $OUT
  contam=0
  python3 -I $S/bench.py http://127.0.0.1:9494 deepseek-v4.1-flash $tag-cold >> $OUT 2>&1
  [ "$(others)" -gt 0 ] && contam=1
  python3 -I $S/bench.py http://127.0.0.1:9494 deepseek-v4.1-flash $tag-warm >> $OUT 2>&1
  [ "$(others)" -gt 0 ] && contam=1
  echo "$tag top_cpu: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,5p | tr -s ' ' | tr '\n' ';')" >> $OUT
  echo "$tag vram_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) swap_so=$(vmstat 1 2 | tail -1 | awk '{print $8}') CONTAMINATED=$contam" >> $OUT
done
pkill -f 'llama-server --host 127.0.0.1 --port 9494'
echo DONE >> $OUT
