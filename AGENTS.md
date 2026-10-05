# AGENTS.md

Serve Aleph Alpha's Kolibri-1 (FP8, 78B MoE / 3.5B active) on one DGX Spark with vLLM via Docker Compose.

## Layout
- `compose.yaml`: `download` (one-shot `hf download` of the pinned revision) -> `kolibri` (vLLM server, healthcheck).
  All knobs are `${VAR:-default}`; `.env.example` documents them.
- `Dockerfile`: `vllm/vllm-openai:v$VLLM_VERSION` + `pip install --no-deps aleph-alpha-inference`.
- `tools/check.py`: functional checks (+ `--needle N`). `tools/bench.sh`: `vllm bench serve` presets in the container.

## Spark
- `ssh zenjoy@10.0.0.158`, checkout at `~/Development/Kolibri-1-DGX-Spark`.
- GB10, sm_121, aarch64, 128 GB unified (121.6 GiB visible), driver 580, CUDA 13, Compose v5.
- Other model repos in `~/Development` default to port 8888 too: stop them first.

## Verify
- `docker compose config --quiet`, `shellcheck tools/bench.sh`
- `docker compose up -d --wait`, then `tools/check.py` (all ok) and `tools/bench.sh`.

## Findings (2026-10-05)
- Cold start ~10 min, CPU-bound weight loading (115,200 per-expert tensors); NVMe does 5.3 GB/s.
- Decode 48 tok/s, ~68% of the ~71 tok/s roofline. No gain from a GB10 fused-MoE config (kernel already
  bandwidth-bound) or `--linear-backend b12x` (needs `pip install b12x==1.2.6`; 48.1 vs 47.9 tok/s).
- `GPU_UTIL=0.80` + any other GPU job -> swap. Keep 0.75.
- Setting a server-wide `reasoning_effort` default makes the template ignore a request's `enable_thinking: false`.
- "incorrect regex pattern / fix_mistral_regex" warning is a false positive (no `transformers_version` in config).

## Constraints
- aleph-alpha-inference supports exactly one vLLM minor (1.x -> 0.29). Bump both together.
- Aleph Alpha's GHCR image is amd64-only; that is why we build our own.
- DeepGEMM is disabled by the plugin (fp32 block scales); do not enable `VLLM_USE_DEEP_GEMM_E8M0`.
- Credits: write our own code. Other Spark recipes (MiaAI-Lab, sudoingX) get inspiration credit; never copy their
  scripts.
