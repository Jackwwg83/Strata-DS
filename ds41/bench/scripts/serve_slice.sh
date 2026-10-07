#!/bin/bash
# The first vertical slice: serve/server.py with the deepseek_v41 format and ds41_serve on the real model.
# Real requests through the OpenAI and Anthropic APIs (plain, streamed, a tool call, the tool result turn); the raw
# replies and the engine's request lines go to $OUT. Usage: bash serve_slice.sh REPO BUILD PACK PROFILE OUT
set -u
R=${1:?repo}; B=${2:?build dir}; P=${3:?pack}; PROF=${4:?expert profile}; OUT=${5:?out dir}
PORT=${PORT:-18000}
mkdir -p $OUT
cat > $OUT/config.json <<EOF
{"exe": "$B/ds41_serve",
 "args": ["--pack", "$P", "--max-context", "32768", "--expert-profile", "$PROF", "--threads", "8"],
 "cwd": "$R", "tokenizer": "$P", "format": "deepseek_v41", "model_name": "deepseek-v4.1-flash",
 "log": "$OUT/engine.log", "port": $PORT}
EOF
cd $R
python3 serve/server.py --engine strata --config $OUT/config.json --port $PORT > $OUT/server.out 2>&1 &
SERVER=$!
cleanup() { kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; }
trap cleanup EXIT
t0=$(date +%s)
until curl -sf localhost:$PORT/health > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "the server ended"; tail -20 $OUT/server.out; exit 1; }
  sleep 5
done
echo "server up after $(( $(date +%s) - t0 )) s"
H='Content-Type: application/json'
TOOL='{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
       "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}'

echo "== 1. OpenAI, thinking on, not streamed"
curl -s localhost:$PORT/v1/chat/completions -H "$H" -d '{"model": "deepseek-v4.1-flash", "max_tokens": 400,
  "messages": [{"role": "user", "content": "Explain in two sentences what a mixture-of-experts model is."}]}' \
  | tee $OUT/1_openai.json | python3 -m json.tool | head -40

echo "== 2. OpenAI, streamed, thinking off, a tool"
curl -sN localhost:$PORT/v1/chat/completions -H "$H" -d '{"model": "deepseek-v4.1-flash", "max_tokens": 300,
  "stream": true, "reasoning_effort": "none", "tools": ['"$TOOL"'],
  "messages": [{"role": "user", "content": "What is the weather in Beijing right now?"}]}' | tee $OUT/2_openai_stream.sse | tail -8

echo "== 3. Anthropic, a tool"
curl -s localhost:$PORT/v1/messages -H "$H" -H "anthropic-version: 2023-06-01" -d '{"model": "deepseek-v4.1-flash",
  "max_tokens": 600, "tools": [{"name": "get_weather", "description": "Current weather of a city",
  "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
  "messages": [{"role": "user", "content": "北京现在天气怎么样？"}]}' | tee $OUT/3_anthropic.json | python3 -m json.tool | head -60

echo "== 4. Anthropic, the tool result turn (the session should be reused)"
python3 - $OUT/3_anthropic.json > $OUT/4_request.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
use = next(b for b in r["content"] if b["type"] == "tool_use")
print(json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 600,
    "tools": [{"name": "get_weather", "description": "Current weather of a city",
               "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
    "messages": [{"role": "user", "content": "北京现在天气怎么样？"},
                 {"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"],
                                               "content": "Sunny, 23 C, light wind from the north."}]}]}))
PY
curl -s localhost:$PORT/v1/messages -H "$H" -H "anthropic-version: 2023-06-01" -d @$OUT/4_request.json \
  | tee $OUT/4_anthropic.json | python3 -m json.tool | head -40

echo "== the engine's request lines"
grep "ds41_serve: prompt" $OUT/engine.log
