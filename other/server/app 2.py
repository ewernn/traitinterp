"""
FastAPI model server - wraps existing utils/generation functions.

Usage:
    # Start with Kimi K2 (loads + fuses MoE, ~25 min)
    source .env && python -u other/server/app.py --port 8765 --model moonshotai/Kimi-K2-Thinking

    # Then from another terminal (or Claude Code):
    curl localhost:8765/health
    curl -X POST localhost:8765/eval/steering -H 'Content-Type: application/json' -d '{
        "experiment": "mats-mental-state-circuits",
        "traits": ["mental_state/anxiety", "mental_state/guilt"],
        "model_variant": "kimi_k2",
        "extraction_variant": "kimi_k2_base",
        "layers": [15, 20, 25, 30],
        "subset": 0,
        "max_new_tokens": 32
    }'

Endpoints:
    GET  /health              - Server status
    POST /model/load          - Load model by name
    POST /generate            - Text generation
    POST /generate/with-capture   - Generation with activation capture
    POST /generate/with-steering  - Generation with steering vectors
    POST /eval/steering       - Run full steering eval (async, returns task_id)
    POST /capture             - Capture raw activations (async, returns task_id)
    GET  /eval/status/{task_id}   - Check eval/capture task status
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import asyncio
import argparse
import uuid
import time
import traceback

from utils.model import load_model_with_lora as _load_model
from utils.generation import generate_batch, generate_with_capture
from utils.vectors import MIN_COHERENCE
from other.server.serialization import serialize_capture_result, deserialize_tensor

app = FastAPI(title="Model Server", description="Persistent model loading for trait extraction")

# ============ Model State ============
_model = None
_tokenizer = None
_model_name = None

# ============ Eval Task State ============
_eval_tasks = {}  # task_id -> {status, result, error, started, finished}


def get_model():
    """Get loaded model or raise error."""
    if _model is None:
        raise HTTPException(400, "No model loaded. POST /model/load first.")
    return _model, _tokenizer


# ============ Request/Response Schemas ============
class GenerateRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: int = 256
    temperature: float = 0.0


class CaptureRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: int = 50
    temperature: float = 0.7
    n_layers: Optional[int] = None
    capture_mlp: bool = False


class SteeringRequest(BaseModel):
    prompts: List[str]
    vectors: Dict[str, str]  # layer (str) -> serialized vector
    coefficients: Dict[str, float]  # layer (str) -> coefficient
    max_new_tokens: int = 256
    component: str = "residual"


class SteeringEvalRequest(BaseModel):
    experiment: str
    traits: List[str]  # ["mental_state/anxiety", "alignment/deception"]
    model_variant: str
    extraction_variant: str
    # Note: CLI default is "30%-60%" (dynamic), server requires explicit layers
    layers: List[int] = [15, 20, 25, 30]
    subset: int = 5
    max_new_tokens: int = 64
    save_responses: str = "best"  # "all", "best", "none"
    method: str = "probe"
    component: str = "residual"
    position: str = "response[:5]"
    prompt_set: str = "steering"
    direction: str = "positive"
    min_coherence: float = MIN_COHERENCE
    search_steps: int = 5
    up_mult: float = 1.3
    down_mult: float = 0.85
    start_mult: float = 0.7
    momentum: float = 0.1
    force: bool = False


# ============ Endpoints ============
@app.get("/health")
def health():
    """Check server status and loaded model."""
    return {
        "status": "ok",
        "model": _model_name,
        "loaded": _model is not None,
        "active_evals": sum(1 for t in _eval_tasks.values() if t["status"] == "running"),
    }


@app.post("/model/load")
def load_model_endpoint(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Load a model by name. Unloads previous model if different."""
    global _model, _tokenizer, _model_name

    if _model_name == model_name:
        return {"status": "already_loaded", "model": model_name}

    # Unload previous model
    if _model is not None:
        del _model, _tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load new model
    _model, _tokenizer = _load_model(
        model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    _model_name = model_name

    return {"status": "loaded", "model": model_name}


@app.post("/model/save")
def save_model_endpoint():
    """Save loaded model to cache for fast reload (skips from_pretrained)."""
    from utils.moe import save_model_cache
    model, tokenizer = get_model()
    cache_dir = save_model_cache(model, tokenizer, _model_name)
    return {"status": "saved", "cache_dir": str(cache_dir)}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate text from prompts."""
    model, tokenizer = get_model()
    responses = generate_batch(
        model, tokenizer, req.prompts,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return {"responses": responses}


@app.post("/generate/with-capture")
def generate_capture(req: CaptureRequest):
    """Generate text and capture activations."""
    model, tokenizer = get_model()
    results = generate_with_capture(
        model, tokenizer, req.prompts,
        n_layers=req.n_layers,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        capture_mlp=req.capture_mlp,
        show_progress=False,
    )
    return {"results": [serialize_capture_result(r) for r in results]}


@app.post("/generate/with-steering")
def generate_steering(req: SteeringRequest):
    """Generate text with steering vectors applied."""
    from core import MultiLayerSteeringHook

    model, tokenizer = get_model()

    # Deserialize vectors
    vectors = {int(l): deserialize_tensor(v) for l, v in req.vectors.items()}

    # Build steering configs: [(layer, vector, coef), ...]
    configs = [
        (layer, vectors[layer].to(model.device), req.coefficients[str(layer)])
        for layer in sorted(vectors.keys())
    ]

    with MultiLayerSteeringHook(model, configs, component=req.component):
        responses = generate_batch(
            model, tokenizer, req.prompts,
            max_new_tokens=req.max_new_tokens,
        )

    return {"responses": responses}


# ============ Steering Eval ============
async def _run_steering_eval(task_id: str, req: SteeringEvalRequest):
    """Run steering eval in background using the server's loaded model."""
    task = _eval_tasks[task_id]
    try:
        from utils.backends import LocalBackend
        from utils.judge import TraitJudge
        from steering.steering_evaluate import _run_main

        model, tokenizer = get_model()
        backend = LocalBackend.from_model(model, tokenizer)
        judge = TraitJudge()

        parsed_traits = [(req.experiment, t) for t in req.traits]
        layers_str = ",".join(str(l) for l in req.layers)

        eval_args = argparse.Namespace(
            experiment=req.experiment,
            no_batch=False,
            ablation=None,
            load_in_8bit=False,
            load_in_4bit=False,
            bnb_4bit_quant_type="nf4",
            layers=layers_str,
            coefficients=None,
            search_steps=req.search_steps,
            up_mult=req.up_mult,
            down_mult=req.down_mult,
            start_mult=req.start_mult,
            momentum=req.momentum,
            max_new_tokens=req.max_new_tokens,
            save_responses=req.save_responses,
            min_coherence=req.min_coherence,
            no_relevance_check=False,
            no_custom_prompt=False,
            eval_prompt_from=None,
            trait_judge=None,
            method=req.method,
            component=req.component,
            position=req.position,
            prompt_set=req.prompt_set,
            subset=req.subset,
            judge="openai",
            vector_experiment=None,
            extraction_variant=req.extraction_variant,
        )

        task["status"] = "running"
        task["traits"] = req.traits

        await _run_main(
            args=eval_args,
            parsed_traits=parsed_traits,
            model_variant=req.model_variant,
            model_name=_model_name,
            lora=None,
            layers_arg=layers_str,
            coefficients=None,
            direction=req.direction,
            force=req.force,
            backend=backend,
            judge=judge,
        )

        task["status"] = "completed"
        task["finished"] = time.time()

    except Exception as e:
        task["status"] = "failed"
        task["error"] = traceback.format_exc()
        task["finished"] = time.time()


@app.post("/eval/steering")
async def start_steering_eval(req: SteeringEvalRequest):
    """Start a steering eval as a background task. Returns task_id to poll."""
    if _model is None:
        raise HTTPException(400, "No model loaded. POST /model/load first.")

    # Check no other eval is running (model can only do one at a time)
    running = [tid for tid, t in _eval_tasks.items() if t["status"] == "running"]
    if running:
        raise HTTPException(409, f"Eval already running: {running[0]}")

    task_id = str(uuid.uuid4())[:8]
    _eval_tasks[task_id] = {
        "status": "starting",
        "started": time.time(),
        "finished": None,
        "error": None,
        "traits": req.traits,
    }

    asyncio.create_task(_run_steering_eval(task_id, req))

    return {
        "task_id": task_id,
        "status": "starting",
        "traits": req.traits,
        "layers": req.layers,
    }


@app.get("/eval/status/{task_id}")
def eval_status(task_id: str):
    """Check status of a steering eval task."""
    if task_id not in _eval_tasks:
        raise HTTPException(404, f"Task {task_id} not found")

    task = _eval_tasks[task_id]
    result = {
        "task_id": task_id,
        "status": task["status"],
        "traits": task.get("traits"),
        "elapsed": round(time.time() - task["started"], 1),
    }
    if task["finished"]:
        result["duration"] = round(task["finished"] - task["started"], 1)
    if task["error"]:
        result["error"] = task["error"]
    return result


# ============ Activation Capture ============
class CaptureActivationsRequest(BaseModel):
    experiment: str
    prompt_set: str
    model_variant: Optional[str] = None
    components: str = "residual"
    layers: Optional[str] = None
    response_only: bool = False
    responses_from: Optional[str] = None
    skip_existing: bool = False
    output_suffix: Optional[str] = None
    prompt_ids: Optional[List[str]] = None


async def _run_capture(task_id: str, req: CaptureActivationsRequest):
    """Run activation capture in background using the server's loaded model."""
    task = _eval_tasks[task_id]
    try:
        from inference.capture_activations import capture_raw_activations

        model, tokenizer = get_model()
        task["status"] = "running"

        n = await asyncio.to_thread(
            capture_raw_activations,
            experiment=req.experiment,
            prompt_set=req.prompt_set,
            model_variant=req.model_variant,
            components=req.components,
            layers=req.layers,
            response_only=req.response_only,
            responses_from=req.responses_from,
            skip_existing=req.skip_existing,
            output_suffix=req.output_suffix,
            prompt_ids=req.prompt_ids,
            model=model,
            tokenizer=tokenizer,
        )

        task["status"] = "completed"
        task["n_captured"] = n
        task["finished"] = time.time()

    except Exception as e:
        task["status"] = "failed"
        task["error"] = traceback.format_exc()
        task["finished"] = time.time()


@app.post("/capture")
async def start_capture(req: CaptureActivationsRequest):
    """Start activation capture as a background task. Returns task_id to poll."""
    if _model is None:
        raise HTTPException(400, "No model loaded.")

    running = [tid for tid, t in _eval_tasks.items() if t["status"] == "running"]
    if running:
        raise HTTPException(409, f"Task already running: {running[0]}")

    task_id = str(uuid.uuid4())[:8]
    _eval_tasks[task_id] = {
        "status": "starting",
        "started": time.time(),
        "finished": None,
        "error": None,
        "prompt_set": req.prompt_set,
        "prompt_ids": req.prompt_ids,
    }

    asyncio.create_task(_run_capture(task_id, req))

    return {
        "task_id": task_id,
        "status": "starting",
        "prompt_set": req.prompt_set,
        "layers": req.layers,
        "n_prompt_ids": len(req.prompt_ids) if req.prompt_ids else "all",
    }


# ============ Debug / Profiling ============
class MemoryProfileRequest(BaseModel):
    prompt: str = "Hello, how are you today?"
    batch_sizes: List[int] = [1, 2, 4]
    max_new_tokens: int = 8


@app.post("/debug/memory-profile")
def memory_profile(req: MemoryProfileRequest):
    """Profile actual GPU memory during generation at different batch sizes.

    Returns per-GPU memory at each stage: baseline, post-prefill, during MoE,
    post-generation. Also captures per-layer MoE dequant memory deltas.
    """
    import utils.moe as moe_utils
    from utils.generation import generate_batch
    from utils.vram import get_free_vram_gb, calculate_max_batch_size

    model, tokenizer = get_model()
    n_gpus = torch.cuda.device_count()
    results = {}

    for batch_size in req.batch_sizes:
        prompts = [req.prompt] * batch_size

        # 1. Reset peak stats
        for i in range(n_gpus):
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.synchronize(i)

        # Baseline: after model loaded, before generation
        baseline = {}
        for i in range(n_gpus):
            baseline[f"gpu{i}"] = {
                "allocated_mb": round(torch.cuda.memory_allocated(i) / 1e6, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(i) / 1e6, 1),
                "free_mb": round(torch.cuda.mem_get_info(i)[0] / 1e6, 1),
            }

        # 2. Enable MoE profiling
        moe_utils._moe_profile = []

        # 3. Run generation
        try:
            responses = generate_batch(
                model, tokenizer, prompts,
                max_new_tokens=req.max_new_tokens,
                temperature=0.0,
            )
            error = None
        except torch.cuda.OutOfMemoryError as e:
            responses = None
            error = str(e)
        finally:
            moe_snapshots = moe_utils._moe_profile
            moe_utils._moe_profile = None

        # 4. Post-generation stats
        post = {}
        for i in range(n_gpus):
            torch.cuda.synchronize(i)
            post[f"gpu{i}"] = {
                "allocated_mb": round(torch.cuda.memory_allocated(i) / 1e6, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(i) / 1e6, 1),
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated(i) / 1e6, 1),
                "free_mb": round(torch.cuda.mem_get_info(i)[0] / 1e6, 1),
            }

        # 5. Summarize MoE snapshots: group by GPU, show deltas
        moe_summary = {}
        if moe_snapshots:
            by_gpu = {}
            for snap in moe_snapshots:
                gpu = snap['gpu']
                if gpu not in by_gpu:
                    by_gpu[gpu] = []
                by_gpu[gpu].append(snap)
            for gpu, snaps in sorted(by_gpu.items()):
                # Show first layer's full trace + summary across all layers
                first_entry = next(i for i, s in enumerate(snaps) if s['label'] == 'moe_entry')
                first_exit = next(i for i, s in enumerate(snaps) if s['label'] == 'moe_exit')
                first_layer = snaps[first_entry:first_exit + 1]
                # Count layers
                n_layers = sum(1 for s in snaps if s['label'] == 'moe_entry')
                # Peak across all layers
                peak_alloc = max(s['allocated_mb'] for s in snaps)
                min_alloc = min(s['allocated_mb'] for s in snaps if s['label'] == 'moe_entry')
                moe_summary[f"gpu{gpu}"] = {
                    "n_moe_layers": n_layers,
                    "peak_allocated_mb": round(peak_alloc, 1),
                    "baseline_allocated_mb": round(min_alloc, 1),
                    "moe_delta_mb": round(peak_alloc - min_alloc, 1),
                    "first_layer_trace": [
                        {
                            "label": s["label"],
                            "allocated_mb": round(s["allocated_mb"], 1),
                            "reserved_mb": round(s["reserved_mb"], 1),
                            "delta_mb": round(s["allocated_mb"] - first_layer[0]["allocated_mb"], 1),
                        }
                        for s in first_layer
                    ],
                }

        # 6. What estimator would have said
        estimator = {
            "free_gb": round(get_free_vram_gb(per_device=True), 2),
            "estimated_batch": calculate_max_batch_size(model, 512, mode='generation'),
        }

        # Clean up
        torch.cuda.empty_cache()

        results[f"batch_{batch_size}"] = {
            "baseline": baseline,
            "post_generation": post,
            "moe_profile": moe_summary,
            "estimator": estimator,
            "response_sample": responses[0][:100] if responses else None,
            "error": error,
        }

        if error:
            break  # Don't try larger batches if we OOMed

    return results


# ============ Entry Point ============
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Model Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", type=str, help="Pre-load model on startup")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load in 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load in 4-bit quantization")
    args = parser.parse_args()

    if args.model:
        print(f"Pre-loading model: {args.model}")
        load_model_endpoint(args.model, args.load_in_8bit, args.load_in_4bit)

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
