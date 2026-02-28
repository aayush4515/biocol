"""
Modal GPU function for LLM inference (no OpenAI API, no HTTP).

Loads an open-source instruct model once per container and exposes
generate_llm_json(system_prompt, user_prompt) -> dict for mutation proposals.
"""

from __future__ import annotations

import json
import re
from typing import Any

from protein_swarm.modal_app.app import app, llm_image

# Model loaded once per container and reused across calls.
# Use a very small instruct model by default; run on CPU for stability.
_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_pipeline_cache: dict[str, Any] = {}


def _get_pipeline(model: str):
    """Load the text-generation pipeline once per container (CPU, float32)."""
    if model not in _pipeline_cache:
        import torch
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            from transformers.pipelines import pipeline as hf_pipeline

        _pipeline_cache[model] = hf_pipeline(
            "text-generation",
            model=model,
            device=-1,
            torch_dtype=torch.float32,
        )
    return _pipeline_cache[model]


def _extract_first_json(text: str) -> dict[str, Any]:
    """Extract the first {...} JSON object from model output."""
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced braces")


@app.function(
    image=llm_image,
    timeout=120,
)
def generate_llm_json(system_prompt: str, user_prompt: str, model: str = _DEFAULT_MODEL) -> dict[str, Any]:
    """Run instruct model on GPU and return a dict matching mutation JSON schema.

    Returns dict with keys: proposed_residue, confidence, reason (and optionally
    position, current_residue). Caller fills position/current_residue if missing.
    """
    pipe = _get_pipeline(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    out = pipe(
        messages,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
    )
    # Handle chat-style output: list of message dicts or plain string
    raw = out[0]["generated_text"]
    if isinstance(raw, list):
        for m in reversed(raw):
            if isinstance(m, dict) and m.get("role") == "assistant":
                raw = m.get("content", "")
                break
        else:
            raw = str(raw[-1].get("content", "")) if raw else ""
    text = (raw if isinstance(raw, str) else str(raw)).strip()
    return _extract_first_json(text)
