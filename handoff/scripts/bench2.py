import sys, json, time, urllib.request, threading, uuid
url, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = ("Transformers process sequences with self-attention, mixing information across positions. "
        "A mixture-of-experts layer routes each token to a few expert feed-forward networks. ") * 120
SHORT = "Explain in about 150 words how speculative decoding speeds up LLM inference."
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def run(prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, **EXTRA}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"content-type": "application/json"})
    t0 = time.time(); first = None; usage = None; n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                    n += 1
                    if first is None:
                        first = time.time()
    t1 = time.time()
    return dict(ttft=(first or t1) - t0, decode_s=t1 - (first or t1), chunks=n,
                ptok=(usage or {}).get("prompt_tokens"), ctok=(usage or {}).get("completion_tokens"))


def dec(r):
    ct = r["ctok"] or r["chunks"]
    return (ct - 1) / r["decode_s"] if r["decode_s"] > 0 and ct > 1 else float("nan")


run(SHORT, 16)  # warmup
d = []
for i in range(3):
    r = run(SHORT + " (run %s)" % uuid.uuid4().hex[:6], 192)
    d.append(dec(r))
    print(f"{tag} short-gen#{i}: ttft={r['ttft']:.2f}s gen={r['ctok']} tok decode={dec(r):.1f} tok/s", flush=True)
print(f"{tag} SINGLE-STREAM decode median={sorted(d)[1]:.1f} tok/s")
for i in range(3):
    r = run("[id %s] " % uuid.uuid4().hex + BASE, 32)  # unique prefix defeats the prefix cache
    print(f"{tag} prefill#{i}: prompt={r['ptok']} ttft={r['ttft']:.2f}s prefill~{r['ptok'] / r['ttft']:.0f} tok/s", flush=True)
for n in (4, 8):
    res = []
    ths = [threading.Thread(target=lambda: res.append(run(SHORT + " (c %s)" % uuid.uuid4().hex[:6], 128))) for _ in range(n)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; w = time.time() - t0
    tot = sum((r["ctok"] or r["chunks"]) for r in res)
    per = sorted(dec(r) for r in res)
    print(f"{tag} {n}-concurrent: {tot} tok in {w:.1f}s = {tot / w:.1f} tok/s aggregate, per-stream decode min/median={per[0]:.1f}/{per[len(per)//2]:.1f}", flush=True)
