# SAE & CLT Download Sources

Reference for building a general-purpose SAE/CLT download tool.

## Major Weight Releases

| Source | Models | HuggingFace Repo | Format | Features |
|---|---|---|---|---|
| GemmaScope v1 | Gemma 2 2B, 9B | `google/gemma-scope-{size}-pt-{component}` | `.npz` + `cfg.json` | res/mlp/attn/transcoders |
| GemmaScope v2 | Gemma 3 270M–27B | `google/gemma-scope-2-{size}-{pt/it}` | `.npz`/`.safetensors` + `cfg.json` | + CLTs for 270M, 1B |
| EleutherAI | Llama 3/3.1 8B | `EleutherAI/sae-llama-3.1-8b-{32/64}x` | `.safetensors` | All layers, MultiTopK |
| Goodfire | Llama 3.x, DeepSeek R1 | `Goodfire/DeepSeek-R1-SAE-l37` etc. | `.pt` | Feature labels as CSV/SQL |
| LlamaScope | Llama 3.1 8B | `fnlp/Llama-Scope` | Custom | 256 SAEs, every layer+sublayer |
| OpenAI | GPT-2 Small, GPT-4 | `openai/sparse_autoencoder` (GitHub) | `.pt` | 16M-latent GPT-4 SAE |
| Apollo e2e | GPT-2 | `apollo-research/e2e-saes-gpt2` | `.safetensors` | KL-divergence trained |
| jbloom | GPT-2, Gemma 2B | `jbloom/GPT2-Small-SAEs-Reformatted` | `.safetensors` | SAELens format |

## CLT (Cross-Layer Transcoder) Sources

- `mntss/gemma-scope-transcoders` — Gemma 2 2B, 426K and 2.5M features
- `google/gemma-scope-2-{size}` — CLTs for Gemma 3 270M, 1B (in same repos)
- `EleutherAI/clt-training` — training code, no major public weight releases yet
- `decoderesearch/circuit-tracer` — main CLT consumer library

## Loading Libraries

- **SAELens** (`pip install sae-lens`): `SAE.from_pretrained(release=..., sae_id=...)`. Central registry at `pretrained_saes.yaml`. HF filter: `huggingface.co/models?library=saelens`
- **EleutherAI/sparsify**: `Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", layer=10)`
- **circuit-tracer**: `ReplacementModel.from_pretrained(transcoders="mntss/gemma-scope-transcoders")` for CLTs
- **Neuronpedia** (`pip install neuronpedia`): Feature metadata/explanations only, not weight tensors. API: `GET neuronpedia.org/api/feature/{model}/{source}/{idx}`

## Design Notes for Download Tool

- `huggingface_hub.hf_hub_download` works for everything on HF
- Three format loaders needed: GemmaScope `.npz`, SAELens `.safetensors` + `cfg.json`, generic `.pt`
- SAELens `pretrained_saes.yaml` is the most complete single-file registry
- Neuronpedia is metadata enrichment (feature explanations, activation stats), not a weight source
