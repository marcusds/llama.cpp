# Nemotron-3 Super: corrupted / dropped tool-call arguments

Tool-call arguments returned by `/v1/chat/completions` are intermittently corrupted
(`"expr\": ": 0`, `"prompt\": \"...`) or silently missing fields, while the same prompt
sampled through raw `/completion` is clean.

## Observed cause (inferred from experiments, grammar code not traced)

For a non-string tool parameter the chat parser builds a schema-derived grammar
(`common/chat-auto-parser-generator.cpp`, `p.schema(p.json(), ..., param_schema, false)`),
and that grammar enforces the schema's **property order**. If the model wants to emit keys
in a different order than the schema lists them, the grammar masks the key it wants and
sampling drifts into garbage keys (with `additionalProperties: true`) or the field is
dropped (with `false`).

In `request-original-order.json` the column item properties are
`name, column_type, drop, expr, dtype, ..., prompt, model_alias`. Unconstrained,
Nemotron-3 Super writes `dtype` before `expr` (56 vs 2 samples) and writes
`model_alias`/`prompt` in either order (11 vs 13).

## Results (Nemotron-3-Super-120B-A12B UD-Q6_K, temp 1.0, tool_choice required, 32 samples each)

| request / variation                                    | complete | notes |
|--------------------------------------------------------|----------|-------|
| original order                                         | 1/32     | 30 defective (garbage keys or missing expr/dtype) |
| natural order (`dtype`,`expr`,`model_alias`,`prompt`)  | 28/32    | `request-natural-order.json` |
| natural order, temp 0                                  | 32/32    | |
| natural order, `prompt` before `model_alias`           | 25/32    | residual order sensitivity |
| natural order, `tool_choice: auto`                     | 24/32    | lazy grammar, same effect |
| original order, `additionalProperties: false`          | 2/32     | hides garbage keys but drops the fields |
| raw `/completion` (no grammar), same prompt            | 31/32 valid, 0 corrupt keys | 1 invalid sample is an unrelated trailing-bracket error |

Flat `columns` array schema and a bare `{"type":"object"}` parameter: 0/16 corrupt.

## Run

    python3 replay.py request-original-order.json 32 http://localhost:8080
    python3 replay.py request-natural-order.json  32 http://localhost:8080

Edit `model` in the request to match your server's model id.

## Open questions

- Whether property order should be enforced at all during sampling (JSON objects are unordered).
- The 1 invalid raw sample was not investigated further.
