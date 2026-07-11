# SmolVLM2 500M video baseline

This port runs the pinned `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
model through the committed `llama.cpp-et` runtime. It is a functional
baseline, not a fully optimized vision implementation.

## Model identity

- Hugging Face model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Revision: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- Quantization: Q8_0 model and Q8_0 multimodal projector
- Runtime revision: `cc4049d86b14e4ef72f827f3bb767b577f18fbcd`
- Full hashes and public fixtures: `.github/ci/reference/smolvlm2_500m_video.json`

The canonical GGUF files are pinned from
`ggml-org/SmolVLM2-500M-Video-Instruct-GGUF`. They can be reproduced from the
original Hugging Face repository with the converter in the committed runtime:

```bash
python3 -m pip install huggingface_hub
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --local-dir local-artifacts/hf/SmolVLM2-500M-Video-Instruct

python3 ported_models/llama_cpp_et/src/llama.cpp-et/convert_hf_to_gguf.py \
  local-artifacts/hf/SmolVLM2-500M-Video-Instruct \
  --outfile local-artifacts/models/smolvlm2_500m_video/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --outtype q8_0

python3 ported_models/llama_cpp_et/src/llama.cpp-et/convert_hf_to_gguf.py \
  local-artifacts/hf/SmolVLM2-500M-Video-Instruct \
  --outfile local-artifacts/models/smolvlm2_500m_video/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --outtype q8_0 --mmproj
```

## Correctness

The board runner downloads two fixed COCO images and executes four cases: cat,
giraffes, cat then giraffes, and giraffes then cat. For every case it first
generates a CPU answer from the exact pinned GGUF files, then requires the
ET-SoC1 answer to match it and the fixed accepted-answer set. Reversing the
images must change the answer to the animal in the new second image.

The log must also report the vision encoder on ET and all language-model layers
offloaded. The ET backend implements the vision graph's 2D `IM2COL`, LayerNorm
(`NORM`), and GELU (`UNARY`) operations. CI rejects the run if the CLIP graph
reports any CPU fallback operation.

## Performance

The leaderboard uses video real-time factor at one frame per second:

```text
sum(board request wall time) / sum(input frame duration)
```

Lower is better. Request timing includes image decoding and preprocessing,
vision encoding, projection, prompt evaluation, and answer decoding. Downloads,
build time, model loading, and the CPU reference pass are excluded.

Run it on an ET-SoC1 board host with:

```bash
BENCHMARK_DEVICE=soc1sim BOARD_BENCHMARK=1 \
  .github/ci/scripts/run_model_benchmark.sh smolvlm2_500m_video
```
