#!/bin/bash
# DeepSeek V4.1 Flash release acceptance on a 128 GB PC: the container held at 119.9 GiB, a fresh checkout installed
# with ./setup.sh --family deepseek (the model files already on disk are linked, not downloaded again), the run
# script started, then real requests: a three-turn chat with thinking (turns 2 and 3 should go back to the snapshot
# before the last prompt token), an OpenAI streamed request with a tool, and an Anthropic tool call with its result.
#   bash accept_128gb.sh REPO_TGZ MODEL_DIR OUT
set -u
TGZ=${1:?repo archive}; MODEL=${2:?model dir}; OUT=${3:?out dir}
W=/workspace; REL=$W/strata-release; PORT=8080
mkdir -p $OUT
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
GB=$(python3 -c "print(($LIMIT - int(119.9 * 2**30)) / 2**30)")
python3 -c "import numpy as np, time; a = np.ones(int($GB * 2**30 / 8)); print('balloon', $GB, 'GiB', flush=True); time.sleep(1e9)" &
BAL=$!
SERVER=
cleanup() {
  [ -n "$SERVER" ] && kill $SERVER 2>/dev/null
  pkill -f "[d]s41_serve --serve" 2>/dev/null
  kill $BAL 2>/dev/null; wait $BAL 2>/dev/null; echo "balloon stopped"
}
trap cleanup EXIT
sleep 60
free -g | head -2

echo "== install $(date +%T)"
rm -rf $REL && mkdir -p $REL && tar xzf $TGZ -C $REL
mkdir -p $W/models/deepseek-sage-1.59bpw
for f in $MODEL/*.json $MODEL/tokenizer* $MODEL/model-*.safetensors; do ln -f $f $W/models/deepseek-sage-1.59bpw/; done
t0=$(date +%s)
(cd $REL && ./setup.sh --yes --family deepseek --context 32768 --no-browser --no-start --models-dir $W/models \
   --data-dir $W/sdata > $OUT/setup.log 2>&1)
echo "setup exit $? after $(( $(date +%s) - t0 )) s"
cp $REL/strata-deepseek-sage-1.59bpw.json $OUT/config.json

echo "== start $(date +%T)"
t0=$(date +%s)
(cd $REL && ./run-deepseek-sage-1.59bpw.sh > $OUT/server.out 2>&1) &
SERVER=$!
until curl -sf localhost:$PORT/health > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "the server ended"; tail -20 $OUT/server.out; exit 1; }
  sleep 5
done
echo "server up after $(( $(date +%s) - t0 )) s"
H='Content-Type: application/json'

echo "== a three-turn chat with thinking"
python3 - $PORT $OUT <<'PY'
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
def ask(messages):
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", method="POST",
        data=json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 600, "messages": messages}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r, time.time() - t0
msgs = [{"role": "user", "content": "I have 3 apples and buy 5 more, then eat 2. How many are left? Answer briefly."}]
for turn, follow in enumerate(["Now double that number and add 7.", "Is the result a prime number? One sentence."], 1):
    r, s = ask(msgs)
    m = r["choices"][0]["message"]
    t = r.get("timings", {})
    print(f"turn {turn}: {s:.1f} s, prompt {t.get('prompt_n')} ({t.get('prompt_ms')} ms), "
          f"{t.get('predicted_per_second')} tok/s, answer: {m['content'].strip()[:120]!r}", flush=True)
    json.dump(r, open(f"{out}/chat_turn{turn}.json", "w"), ensure_ascii=False, indent=1)
    msgs += [{"role": "assistant", "content": m["content"], "reasoning_content": m.get("reasoning_content")},
             {"role": "user", "content": follow}]
r, s = ask(msgs)
m = r["choices"][0]["message"]
t = r.get("timings", {})
print(f"turn 3: {s:.1f} s, prompt {t.get('prompt_n')} ({t.get('prompt_ms')} ms), {t.get('predicted_per_second')} "
      f"tok/s, answer: {m['content'].strip()[:160]!r}", flush=True)
json.dump(r, open(f"{out}/chat_turn3.json", "w"), ensure_ascii=False, indent=1)
PY

TOOL='{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
       "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}'
echo "== OpenAI streamed, a tool"
curl -sN localhost:$PORT/v1/chat/completions -H "$H" -d '{"model": "deepseek-v4.1-flash", "max_tokens": 300,
  "stream": true, "tools": ['"$TOOL"'], "messages": [{"role": "user", "content": "What is the weather in Paris?"}]}' \
  > $OUT/openai_tool.sse
grep -o '"name": "[a-z_]*"\|"arguments": "[^"]*"\|"finish_reason": "[a-z_]*"' $OUT/openai_tool.sse | tr '\n' ' '; echo

echo "== Anthropic, a tool call and its result"
ATOOL='{"name": "get_weather", "description": "Current weather of a city", "input_schema": {"type": "object",
        "properties": {"city": {"type": "string"}}, "required": ["city"]}}'
curl -s localhost:$PORT/v1/messages -H "$H" -H "anthropic-version: 2023-06-01" -d '{"model": "deepseek-v4.1-flash",
  "max_tokens": 600, "tools": ['"$ATOOL"'], "messages": [{"role": "user", "content": "东京现在天气怎么样？"}]}' > $OUT/anthropic_1.json
python3 - $OUT <<'PY' > $OUT/anthropic_2_request.json
import json, sys
r = json.load(open(sys.argv[1] + "/anthropic_1.json"))
use = next(b for b in r["content"] if b["type"] == "tool_use")
print(json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 600,
    "tools": [{"name": "get_weather", "description": "Current weather of a city", "input_schema": {"type": "object",
               "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
    "messages": [{"role": "user", "content": "东京现在天气怎么样？"}, {"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"],
                                               "content": "Cloudy, 18 C, light rain expected in the evening."}]}]}))
PY
curl -s localhost:$PORT/v1/messages -H "$H" -H "anthropic-version: 2023-06-01" -d @$OUT/anthropic_2_request.json \
  > $OUT/anthropic_2.json
python3 - $OUT <<'PY'
import json, sys
for f in ("anthropic_1.json", "anthropic_2.json"):
    r = json.load(open(sys.argv[1] + "/" + f))
    print(f, r.get("stop_reason"), [(b["type"], b.get("input") or (b.get("text") or "")[:80]) for b in r["content"]])
PY

echo "== the engine's request lines"
cp $REL/strata-deepseek-sage-1.59bpw.log $OUT/engine.log
grep "ds41_serve: prompt\|RAM tier\|VRAM expert slots" $OUT/engine.log
echo "== ALL DONE $(date +%T)"
