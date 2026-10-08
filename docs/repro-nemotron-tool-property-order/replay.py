#!/usr/bin/env python3
"""Replay a chat-completions request N times and score the tool-call arguments.

usage: replay.py REQUEST.json [N=32] [BASE_URL=http://localhost:8080]

"corrupt":    a JSON key contains a quote, backslash, colon, brace or edge whitespace
              (e.g. "expr\\": ").
"incomplete": the proposed_units column is missing "expr" or "dtype" (the grammar
              forced the model off the key it wanted, so the field was silently dropped).
"complete":   neither of the above.
"""
import collections, json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
base = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8080"
req = json.load(open(path))


def keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.append(k)
            keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            keys(v, acc)
    return acc


def score(args):
    try:
        o = json.loads(args)
    except ValueError:
        return "invalid-json"
    if any(re.search(r'[\\":{}]|^\s|\s$', k) for k in keys(o, [])):
        return "corrupt"
    o = o.get("job_request", o)
    cols = o.get("spec", {}).get("config", {}).get("columns", [])
    pu = [c for c in cols if c.get("name") == "proposed_units"]
    return "complete" if pu and "expr" in pu[0] and "dtype" in pu[0] else "incomplete"


def one(_):
    r = urllib.request.Request(base + "/v1/chat/completions", json.dumps(req).encode(),
                               {"Content-Type": "application/json"})
    try:
        msg = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]
    except Exception as e:  # HTTP 500 on unparseable tool calls, etc.
        return "error: %s" % e
    tc = msg.get("tool_calls")
    return score(tc[0]["function"]["arguments"]) if tc else "no-tool-call"


with ThreadPoolExecutor(4) as ex:
    print(dict(collections.Counter(ex.map(one, range(n)))))
