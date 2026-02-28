"""
Pre-download the LLM weights into the Modal image at build time.

This is invoked from the llm_image definition via:
  .run_commands("python -m protein_swarm.modal_app.preload_llm")
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def main() -> None:
    # Download and cache tokenizer + model weights.
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForCausalLM.from_pretrained(MODEL_NAME)


if __name__ == "__main__":
    main()

