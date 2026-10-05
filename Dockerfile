# Kolibri-1 on DGX Spark: the official vLLM image plus Aleph Alpha's vLLM plugin.
# Aleph Alpha's own image (ghcr.io/aleph-alpha/aleph-alpha-inference) is amd64-only; vllm/vllm-openai ships arm64,
# so this is their Dockerfile rebuilt for aarch64 with the plugin from PyPI.
# Default 0.31.0 is newer than the plugin's official pin (0.29.x): experimental, but checked and benchmarked.
ARG VLLM_VERSION=0.31.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

ARG AA_INFERENCE_VERSION=1.0.0
# --no-deps: keep the image's vllm, torch and transformers; the plugin's own pin would pull in vLLM 0.29.
RUN pip install --no-cache-dir --no-deps "aleph-alpha-inference==${AA_INFERENCE_VERSION}" \
 && python3 -c "import aleph_alpha_inference as a; a.register(); \
from vllm.model_executor.models.registry import ModelRegistry as R; \
assert 'Kolibri1ForCausalLM' in R.get_supported_archs(), 'Kolibri1ForCausalLM not registered'"

LABEL org.opencontainers.image.title="kolibri-1-dgx-spark" \
      kolibri.aa-inference="${AA_INFERENCE_VERSION}"
