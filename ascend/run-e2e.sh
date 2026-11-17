#!/bin/sh
# ascend/run-e2e.sh - the whole stack: serve/server.py -> the Ascend engine -> the real checkpoint.
#
# This is the one check that cannot run in CI: it needs the 578 GB checkpoint and an Ascend card.  It
# writes a serve config pointing at ascend/strata-ascend, starts serve/server.py exactly as an operator
# would, waits for /health, sends one chat completion over HTTP, and prints the answer, the engine's log
# and the server's log.
#
#   ascend/setup_tokenizer.sh /mnt/weights/DeepSeek-V4-Flash-0731-w8a8 .ascend-tokenizer
#   ascend/run-e2e.sh
#   MAYA_ASCEND_MODEL=/mnt/weights/... PROMPT="1, 2, 3, 4," MAX_TOKENS=4 ascend/run-e2e.sh
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
model=${MAYA_ASCEND_MODEL:-/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8}
tokdir=${MAYA_ASCEND_TOKENIZER_DIR:-$root/.ascend-tokenizer}
port=${MAYA_ASCEND_PORT:-8199}
prompt=${PROMPT:-The quick brown fox jumps over the lazy}
max_tokens=${MAX_TOKENS:-8}
context=${CONTEXT:-2048}
cfg=$(mktemp)
engine_log=$(mktemp)
server_log=$(mktemp)

cat > "$cfg" <<JSON
{
  "exe": "$root/ascend/strata-ascend",
  "args": ["--model", "$model", "--max-context", "$context"],
  "tokenizer": "$tokdir",
  "model_name": "DeepSeek-V4.1-Flash",
  "log": "$engine_log"
}
JSON

if [ ! -f "$tokdir/vocab.json" ]; then
    echo "no tokenizer directory at $tokdir: run ascend/setup_tokenizer.sh first" >&2
    exit 2
fi

python3 -m serve.server --engine strata --config "$cfg" --port "$port" --host 127.0.0.1 \
    > "$server_log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 90); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" > /dev/null 2>&1; then ready=1; break; fi
    sleep 3
done
if [ "$ready" -ne 1 ]; then
    echo "the server did not become healthy" >&2
    tail -30 "$server_log" >&2
    tail -30 "$engine_log" >&2
    exit 1
fi
echo "=== health ==="
curl -s -m 5 "http://127.0.0.1:$port/health" | head -c 400
echo

body=$(PROMPT_TEXT="$prompt" MAX_TOKENS="$max_tokens" python3 -c '
import json, os
print(json.dumps({"model": "DeepSeek-V4.1-Flash",
                  "messages": [{"role": "user", "content": os.environ["PROMPT_TEXT"]}],
                  "max_tokens": int(os.environ["MAX_TOKENS"]), "temperature": 0}))')
echo "=== /v1/chat/completions ==="
curl -s -m 1800 "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$body" | head -c 3000
echo
echo "=== engine log ==="
tail -12 "$engine_log"
echo "=== server log ==="
tail -12 "$server_log"
kill $srv 2>/dev/null || true
sleep 2
