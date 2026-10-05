#!/usr/bin/env bash
# Throughput presets on vLLM's own benchmark (`vllm bench serve`), run inside the server container so the host
# needs nothing. Random token prompts, fixed output length, completions endpoint (no chat template, so no thinking).
#   tools/bench.sh            decode at 1/4/8 concurrent streams, prefill at 8k/32k/128k
#   tools/bench.sh decode     only decode      tools/bench.sh prefill     only prefill
set -euo pipefail
cd "$(dirname "$0")/.."
setting() { [[ -f .env ]] && sed -n "s/^$1=//p" .env | tail -1 || true; }   # the overrides compose reads
port=$(setting PORT) name=$(setting SERVED_NAME) model=$(setting MODEL_ID) rev=$(setting MODEL_REVISION)
model=${model:-Aleph-Alpha/Kolibri-1} rev=${rev:-e52eb4627d11516b0c01de49210ab5a4e4061444}
# The server runs offline, so point the benchmark's tokenizer at the cached snapshot rather than the Hub name.
snapshot="/root/.cache/huggingface/hub/models--${model//\//--}/snapshots/$rev"

bench() {  # bench <label> <input tokens> <output tokens> <concurrency> <prompts>
  printf '\n== %s\n' "$1"
  docker compose exec kolibri vllm bench serve --backend vllm --base-url "http://127.0.0.1:${port:-8888}" \
    --model "$snapshot" --served-model-name "${name:-Kolibri-1}" \
    --dataset-name random --random-input-len "$2" --random-output-len "$3" --ignore-eos \
    --max-concurrency "$4" --num-prompts "$5" --seed "$RANDOM" 2>&1 |
    grep -E '^(Failed requests|Output token throughput|Median TTFT|Median TPOT)'
}

if [[ "${1:-all}" != prefill ]]; then
  for c in 1 4 8; do bench "decode, $c stream(s), 512 in / 512 out" 512 512 "$c" $((c * 3)); done
fi
if [[ "${1:-all}" != decode ]]; then
  for n in 8192 32768 131072; do bench "prefill, $n tokens" "$n" 1 1 2; done
fi
