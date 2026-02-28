# Protein Swarm Design Engine

Distributed multi-agent protein sequence optimisation. Agents run on Modal (optional); LLM inference can use the OpenAI API or a Modal GPU function (open-source model, no API key).

## Modal setup (LLM on GPU, no OpenAI)

1. **Install Modal and log in**
   ```bash
   pip install modal
   modal token set
   ```

2. **Deploy the app** (registers agents + GPU LLM function)
   ```bash
   modal deploy protein_swarm/modal_app/functions.py
   ```

3. **Run design with LLM on Modal GPU** (`modal_local` = direct Modal function, no API)
   ```bash
   python -m protein_swarm.main design \
     --sequence "ACDEFGHIKLMNPQRSTVWY" \
     --objective "Design a stable helix-rich protein" \
     --use-llm \
     --llm-provider modal_local \
     --max-iterations 10
   ```

No `OPENAI_API_KEY` is required for `--llm-provider modal_local`. The GPU runs an open-source instruct model (e.g. Qwen2.5-7B-Instruct) inside Modal.

## Local run (no Modal)

```bash
python -m protein_swarm.main design \
  --sequence "ACDEFGHIKLMNPQRSTVWY" \
  --objective "Design a stable helix-rich protein" \
  --no-modal \
  --max-iterations 20
```

Use `--use-llm --llm-provider openai` with an API key for OpenAI-backed agents when running locally.
