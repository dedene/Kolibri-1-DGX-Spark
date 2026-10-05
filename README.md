# Kolibri-1 on a single DGX Spark (vLLM, Docker Compose)

Serve Aleph Alpha's [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) on one NVIDIA DGX Spark with vLLM, using the official FP8 checkpoint as published. One command builds the image, downloads the weights and starts an OpenAI-compatible API.

> **Experimental, work in progress.** Kolibri-1 came out on 3 October 2026 and this repository follows it closely. The default stack is vLLM 0.31.0, which is newer than the 0.29.x that Aleph Alpha's plugin officially supports. It passes `tools/check.py` and benchmarks the same as 0.29, but it is an untested combination as far as Aleph Alpha is concerned. Set `VLLM_VERSION=0.29.0` for the supported setup. Expect defaults and numbers to change.

Kolibri-1 is a 78B mixture-of-experts model (3.5B active per token) focused on German and English, with a reasoning mode and tool calling. The model card lists 2x H100 as the minimum. On a Spark it fits in one box: the FP8 weights take 73.4 GiB of the 128 GB unified memory, and the KV cache stays small because only 10 of the 50 layers attend over the full context (the other 40 use a 513-token sliding window). A 262k-token sequence needs about 2.6 GiB of fp8 KV.

**Hardware:** DGX Spark (GB10, sm_121, aarch64), driver 580, CUDA 13, Docker with the NVIDIA Container Toolkit and Compose v2. About 85 GB free disk.

## Quickstart

```bash
git clone https://github.com/dedene/Kolibri-1-DGX-Spark.git && cd Kolibri-1-DGX-Spark
docker compose up -d --wait     # build, download 78.8 GB (first run only), load, wait until healthy
tools/check.py                  # does the plugin behave? reasoning switch, tool calls, streaming
docker compose down
```

`up --wait` returns once `/health` answers. A cold start spends about 10 minutes loading weights; `docker compose logs -f kolibri` shows progress.

The API is at `http://<spark-ip>:8888/v1`, model `Kolibri-1`:

```bash
curl http://<spark-ip>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Kolibri-1",
  "messages": [{"role": "user", "content": "Erkläre kurz, was ein Mixture-of-Experts-Modell ist."}],
  "chat_template_kwargs": {"reasoning_effort": "medium"}
}'
```

Thinking is on by default at effort `high`. Per request, set `chat_template_kwargs.reasoning_effort` to `none`, `low`, `medium` or `high` (or `enable_thinking: false`). The reasoning comes back in `message.reasoning`, the answer in `message.content`. Sampling defaults come from the checkpoint's `generation_config.json` (temperature 1.0, top_p 0.97, top_k 128), as Aleph Alpha recommends.

## How it is put together

- **`Dockerfile`**: `vllm/vllm-openai:v0.31.0` plus Aleph Alpha's vLLM plugin [`aleph-alpha-inference`](https://github.com/Aleph-Alpha/aleph-alpha-inference) 1.0.0, which registers the `Kolibri1ForCausalLM` architecture and the `kolibri1` reasoning and tool parsers. Aleph Alpha publishes an image of their own, but only for amd64; this is the same recipe on the arm64 vLLM image. The plugin is installed with `--no-deps`, because its pin would otherwise pull vLLM back to 0.29. The build fails if the architecture does not register.
- **`compose.yaml`**: a one-shot `download` service fetches the pinned checkpoint revision into your normal Hugging Face cache (resumable, and a no-op once complete). The `kolibri` service starts after it, serves offline from that cache, keeps torch.compile and Triton caches in `~/.cache/kolibri-1-dgx-spark` so restarts skip compilation, and has a healthcheck on `/health`.
- Nothing is installed on the host.

## Configuration

Copy `.env.example` to `.env` and uncomment what you want to change, then `docker compose up -d --wait` again (Compose recreates the container when settings change).

| Variable | Default | |
|---|---|---|
| `CONTEXT` | 262144 | Tokens per sequence. `1048576` works as is. |
| `MAX_SEQS` | 8 | Concurrent sequences |
| `GPU_UTIL` | 0.75 | Share of unified memory vLLM claims. See the notes before raising it. |
| `BATCHED_TOKENS` | 8192 | Chunked-prefill size |
| `KV_DTYPE` | fp8 | What Aleph Alpha evaluated with |
| `REASONING_EFFORT` | (template default: high) | Server-wide default. Once set, clients override it with `reasoning_effort`; `enable_thinking: false` alone no longer does. |
| `PORT`, `HOST`, `SERVED_NAME` | 8888, 0.0.0.0, Kolibri-1 | |
| `VLLM_VERSION` | 0.31.0 (experimental) | `0.29.0` is what the plugin officially supports. Rebuilds the image under a new tag. |

For any other vLLM flag, add it to the `command:` list in `compose.yaml`.

## Numbers

One DGX Spark, vLLM 0.31.0, FP8 weights, fp8 KV, CUDA graphs on, measured 5 October 2026.

| `tools/bench.sh` (`vllm bench serve`, random prompts) | Throughput | Median TTFT | Median time per token |
|---|---|---|---|
| Decode, 1 stream, 512 in / 512 out | **47 tok/s** | 0.27 s | 20.9 ms (48 tok/s) |
| Decode, 4 streams | 121 tok/s total | 0.56 s | 32 ms (31 tok/s per stream) |
| Decode, 8 streams | 161 tok/s total | 0.80 s | 48 ms (21 tok/s per stream) |
| Prefill, 8k tokens | ~6,200 tok/s | 1.3 s | |
| Prefill, 32k tokens | ~5,000 tok/s | 6.5 s | |
| Prefill, 128k tokens | ~2,800 tok/s | 47.0 s | |

Cold start (`up --wait` with the weights already downloaded): about 10 minutes, almost all of it weight loading. The KV pool at default settings holds 1.29M tokens.

The vLLM versions compared on the same box:

| | 0.29.0 (supported) | 0.30.0 | 0.31.0 (default) |
|---|---|---|---|
| Decode, 1 / 4 / 8 streams (tok/s) | 46.9 / 121 / 160 | 46.8 / 121 / 161 | 46.9 / 121 / 161 |
| TTFT at 8k / 32k / 128k (s) | 1.30 / 6.46 / 46.6 | 1.32 / 6.47 / 46.5 | 1.32 / 6.50 / 47.0 |
| Weight loading (s) | 562 | 598 | 537 |
| KV pool (tokens) | 1.39M | 1.37M | 1.29M |

Speed is the same across all three. 0.31 loads a little faster and sizes the KV pool slightly more conservatively, because it now counts MoE memory when profiling.

Long context: `tools/check.py --needle` hides a key in synthetic logs and asks for it back. It found the key at 167k tokens on the default 262k setup (68 s), and at about 350k tokens with `CONTEXT=1048576` (248 s), past the native window.

A decode step reads about 3.85 GB: 1.7 GB of attention projections, 1.4 GB of routed and shared experts in FP8, 0.66 GB of bf16 LM head. At the Spark's 273 GB/s that caps decode near 71 tok/s, so 48 tok/s is about 68% of the roofline.

Benchmark it yourself with `tools/bench.sh`, which runs vLLM's own `vllm bench serve` inside the container (random prompts, fixed output length) for decode at 1/4/8 streams and prefill at 8k/32k/128k.

## Notes

- **Keep `GPU_UTIL` at 0.75.** At 0.80 the server leaves about 15 GiB for everything else, and a second GPU job pushed the box into swap. 0.75 still gives a 1.29M-token KV pool, almost five full 262k contexts.
- **Loading is CPU-bound.** The NVMe reads 5.3 GB/s, but 115,200 of the checkpoint's 116,303 tensors are per-expert weights and scales, and vLLM's loader processes each one in Python. Repacking the experts would fix it, at the cost of no longer serving the official files. `--safetensors-load-strategy prefetch` cut loading from 568 to 501 s but triggers a RAM warning from vLLM, so it is not enabled.
- **Things that did not make it faster.** vLLM ships no GB10 fused-MoE config for Kolibri's expert shape (E=384, N=512, FP8 128x128 blocks) and logs "Using default MoE config", but benchmarking the kernel showed it already bandwidth-bound at decode batch sizes. `--linear-backend b12x` (native SM12x FP8 kernels) measured 48.1 tok/s against 47.9 for the default CUTLASS path. The SM12x attention backend (`--attention-backend B12X --block-size 128`, with `pip install b12x==1.3.0`) passes `tools/check.py` but matched FlashInfer everywhere: 46.9 tok/s decode, and 47.9 s against 47.0 s to first token at 128k.
- **A harmless tokenizer warning.** The log says the tokenizer has "an incorrect regex pattern" and suggests `fix_mistral_regex=True`. Transformers 5 prints this for any local tokenizer whose `config.json` lacks a `transformers_version` field, which Kolibri's does. It only warns.
- **Port 8888** is a common default for Spark model servers. If another one is running, stop it or set `PORT`.

## Credits

- [Aleph Alpha](https://aleph-alpha.com/) for Kolibri-1 and the `aleph-alpha-inference` plugin (Apache-2.0). The Dockerfile follows the one in their plugin repository.
- The [vLLM](https://github.com/vllm-project/vllm) project, including `vllm bench serve`.
- Inspired by the single-Spark recipes from [MiaAI-Lab](https://github.com/MiaAI-Lab) and [sudoingX](https://github.com/sudoingX/dgx-spark-ling), which showed how much a well-documented one-command setup helps on this box.

## License

MIT, see [LICENSE](LICENSE). The model weights are not part of this repository and are licensed by Aleph Alpha under Apache-2.0.
