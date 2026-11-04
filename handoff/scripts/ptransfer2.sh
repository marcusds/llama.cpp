#!/usr/bin/env bash
# usage: ptransfer2.sh <host>   chunked parallel copy of the GGUF with retries + per-chunk sha256 verification
set -u
H=${1:?host}
F=/data/llm/models/DeepSeek-V4.1-Flash-1017-NVFP4.gguf
D=$F
N=16; BS=$((16*1024*1024))
SIZE=$(stat -c %s $F)
TOTAL=$(( (SIZE + BS - 1) / BS )); PER=$(( (TOTAL + N - 1) / N ))
SSH="ssh -o BatchMode=yes -c aes128-gcm@openssh.com"
ST=/opt/llm/var/log/gguf-transfer-$H.status; rm -f $ST
range() { s=$(( $1*PER )); c=$PER; [ $((s+c)) -gt $TOTAL ] && c=$((TOTAL-s)); echo "$s $c"; }
$SSH $H "truncate -s $SIZE $D.part" || { echo FAILED-prealloc; echo 1 > $ST; exit 1; }
echo "start $(date -u +%T) host=$H size=$SIZE blocks=$TOTAL per_worker=$PER"
copy_chunk() { i=$1; read s c < <(range $i); [ $c -le 0 ] && return 0
  for try in 1 2 3 4 5; do
    dd if=$F bs=$BS skip=$s count=$c iflag=fullblock status=none | $SSH $H "dd of=$D.part bs=$BS seek=$s conv=notrunc iflag=fullblock status=none" && { echo "chunk $i copied $(date -u +%T)"; return 0; }
    echo "chunk $i retry $try"; sleep $((try*3))
  done; echo "chunk $i FAILED"; return 1; }
for i in $(seq 0 $((N-1))); do copy_chunk $i & sleep 1; done
wait
echo "copy done $(date -u +%T); verifying chunk hashes"
hash_remote() { i=$1; read s c < <(range $i)
  for try in 1 2 3 4 5; do r=$($SSH $H "dd if=$D.part bs=$BS skip=$s count=$c iflag=fullblock status=none | sha256sum | cut -c1-64") && [ ${#r} = 64 ] && { echo "$r" > /tmp/gh-r-$H-$i; return 0; }; sleep $((try*3)); done; echo "NOHASH" > /tmp/gh-r-$H-$i; }
hash_local() { i=$1; read s c < <(range $i); dd if=$F bs=$BS skip=$s count=$c iflag=fullblock status=none | sha256sum | cut -c1-64 > /tmp/gh-l-$H-$i; }
for i in $(seq 0 $((N-1))); do hash_local $i & hash_remote $i & sleep 1; done
wait
bad=0
for i in $(seq 0 $((N-1))); do read s c < <(range $i); [ $c -le 0 ] && continue
  [ "$(cat /tmp/gh-l-$H-$i)" = "$(cat /tmp/gh-r-$H-$i)" ] || { echo "chunk $i HASH MISMATCH"; bad=1; }; done
rm -f /tmp/gh-l-$H-* /tmp/gh-r-$H-*
if [ $bad = 0 ]; then $SSH $H "mv $D.part $D && stat -c '%s %n' $D" && echo "VERIFIED-AND-RENAMED $(date -u +%T)"; echo 0 > $ST
else echo "VERIFY-FAILED (left $D.part)"; echo 2 > $ST; fi
