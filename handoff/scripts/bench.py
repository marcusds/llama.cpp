import sys, json, time, urllib.request, threading
url, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = ("Transformers process sequences with self-attention, mixing information across positions. "
        "A mixture-of-experts layer routes each token to a few expert feed-forward networks. ") * 120
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


def show(name, r):
    ct = r["ctok"] or r["chunks"]
    dec = (ct - 1) / r["decode_s"] if r["decode_s"] > 0 and ct > 1 else float("nan")
    pre = r["ptok"] / r["ttft"] if r["ptok"] else float("nan")
    print(f"{tag} {name}: prompt={r['ptok']} ttft={r['ttft']:.2f}s prefill~{pre:.0f} tok/s | gen={ct} tok decode={dec:.1f} tok/s", flush=True)
    return dec


short = "Explain in about 150 words how speculative decoding speeds up LLM inference."
run(short, 16)  # warmup
d = [show("short-gen#%d" % i, run(short, 192)) for i in range(3)]
show("long-prefill", run(BASE, 32))
res = []
ths = [threading.Thread(target=lambda: res.append(run(short, 128))) for _ in range(4)]
t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; w = time.time() - t0
tot = sum((r["ctok"] or r["chunks"]) for r in res)
print(f"{tag} 4-concurrent: {tot} tok in {w:.1f}s = {tot / w:.1f} tok/s aggregate", flush=True)
print(f"{tag} SUMMARY single-stream decode median={sorted(d)[1]:.1f} tok/s")
