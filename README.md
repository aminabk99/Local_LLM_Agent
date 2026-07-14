<div align="center">

# 🤖 AI LLM Coding Agent
### A Fully Local AI Agent That Writes, Reads, and Executes Code Autonomously

A Python coding agent powered by **Ollama + qwen2.5-coder:7b** that takes a task, breaks it into tool calls, and autonomously writes files, reads code, and runs shell commands — all in a structured **JSON action loop** with up to 20 agent steps. Exposed via a **FastAPI** `/chat` endpoint with a dark-themed web UI.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-qwen2.5--coder:7b-black?style=for-the-badge)
![Local](https://img.shields.io/badge/100%25_Local-No_API_Key-4CAF50?style=for-the-badge&logo=homeassistant&logoColor=white)

</div>

---

## How It Works

1. You send a task to the `/chat` endpoint — e.g. *"Write a Python script that scrapes headlines"*
2. The agent enters a **JSON action loop** — each step the LLM returns one action as a JSON object
3. The agent executes that action using one of four tools, appends the result to history, and loops
4. After up to **20 steps** the agent calls `finish` and returns the final response
5. All file operations are **sandboxed to the `workspace/` folder** — nothing outside can be touched

**Tools available:** 📁 `list_files` · 📖 `read_file` · ✏️ `write_file` · ⚙️ `run_command`

---

## Setup

**Requirements:** Python 3.11+ · [Ollama](https://ollama.com)

**1. Clone & install**
```bash
git clone https://github.com/aminabk99/AI_LLM
cd AI_LLM
python -m venv venv && source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

**2. Pull the model** (~4.7GB)
```bash
ollama pull qwen2.5-coder:7b
```

**3. Start Ollama**
```bash
ollama serve
```

**4. Run the server**
```bash
python main.py
```

Open `http://localhost:8000` in your browser to use the chat UI, or hit `/docs` for the interactive API docs.

---

## Project Structure

```
AI_LLM/
├── main.py                  # FastAPI server — /chat endpoint, session history, web UI
├── requirements.txt         # Python dependencies
├── agent/
│   ├── agent.py             # Agent runner — orchestrates the JSON action loop
│   ├── core.py              # CodingAgent class with step logic and approval gate
│   ├── llm_ollama.py        # Ollama API client with JSON recovery and 600s timeout
│   ├── prompts.py           # System prompt and user prompt builder
│   ├── tools.py             # Workspace (list/read/write files) + run_command
│   └── ui.html              # Dark-themed web UI served at /
└── workspace/               # Sandboxed folder — agent reads/writes only here
```

---

## API

**POST `/chat`**
```json
{
  "message": "Write a FastAPI hello world app",
  "session_id": ""
}
```
Returns:
```json
{
  "reply": "Done! I wrote main.py with a /hello endpoint...",
  "session_id": "abc-123",
  "steps_taken": 4
}
```

**GET `/history/{session_id}`** — retrieve full chat history for a session

**DELETE `/history/{session_id}`** — clear a session

---

## Hardest Part
**JSON recovery from a chatty model** — `qwen2.5-coder:7b` sometimes wraps its JSON in markdown code fences or adds explanation text before the object. The fallback parser scans for the first `{` and last `}` in the response and attempts to extract valid JSON from that slice — which handles ~95% of malformed outputs without needing to re-prompt.

## Most Interesting
**The 600s timeout** — on first run, Ollama has to load the full 7B model into memory which can take 30–60 seconds on a CPU. Without the extended timeout the agent would crash before the model even responded. A short retry message explains what happened if it still times out, rather than just throwing an exception.

---

## Security

- All file operations are restricted to `workspace/` — path traversal attempts raise a `ValueError`
- Shell commands run with `cwd=workspace` — the agent cannot touch files outside the sandbox
- No API keys, no cloud calls — the model runs entirely on your machine

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>

---

## Fine-Tuning with LoRA & DPO

The JSON action loop is a precision formatting task — ideal for LoRA fine-tuning. The model must emit exactly one JSON object per step with the right action key and required fields. Any deviation (markdown fences, prose prefix, wrong keys) breaks the agent loop.

### What's in `finetune/`

| File | Purpose |
|------|---------|
| `config.py` | Base model, LoRA rank/alpha, training hyperparameters |
| `data/generate_sft.py` | Generates 80 supervised (prompt → JSON) examples |
| `data/generate_dpo.py` | Generates 60 preference pairs (chosen: clean JSON vs. rejected: broken output) |
| `train_sft.py` | LoRA SFT using TRL's SFTTrainer — teaches output format |
| `train_dpo.py` | DPO from SFT checkpoint — reinforces preference via contrastive signal |
| `eval.py` | Measures JSON parse rate, action validity, and key completeness before vs. after |

**Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — same family as the Ollama `tinyllama` used in benchmarking, small enough to fine-tune on a single consumer GPU (6–8 GB VRAM) or in Google Colab.

**LoRA config:** rank r=16, alpha=32 (scaling=2.0), targeting all projection matrices (q/k/v/o + gate/up/down).

### Dataset

| Split | Examples | Covers |
|-------|----------|--------|
| SFT | 80 | All 5 action types: finish, list\_files, read\_file, write\_file, run |
| DPO chosen | 60 | Clean single JSON objects |
| DPO rejected | 60 | Markdown-wrapped (15), prose prefix (15), wrong keys (15), natural language only (15) |

### Quick start

```bash
# 1. Install fine-tuning dependencies
pip install torch transformers peft trl datasets accelerate

# 2. Generate datasets
python -m finetune.data.generate_sft
python -m finetune.data.generate_dpo

# 3. SFT training (~20 min on a single GPU)
python -m finetune.train_sft

# 4. DPO training (~10 min on a single GPU)
python -m finetune.train_dpo

# 5. Evaluate: JSON parse rate before vs. after
python -m finetune.eval --adapter both
```

### Expected results (illustrative)

| Model | JSON Parse Rate | Action Valid | Keys Complete |
|-------|----------------|--------------|---------------|
| Base TinyLlama | ~45% | ~38% | ~30% |
| + SFT (LoRA r=16) | ~82% | ~79% | ~74% |
| + DPO (SFT→DPO) | ~91% | ~88% | ~85% |

> **Why JSON parse rate is the key metric:** every step in the agent loop calls `json.loads()` on the model output. A failed parse = a broken step. The benchmark (`benchmark/run_benchmark.py`) measured the base tinyllama at ~40% JSON parse rate on action-format prompts. The fine-tuned adapter targets >85%.

### Running with the fine-tuned adapter in Ollama

After fine-tuning, convert the adapter to GGUF and push to Ollama:

```bash
# Merge LoRA weights into base model
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
model = PeftModel.from_pretrained(base, 'finetune/adapters/dpo')
merged = model.merge_and_unload()
merged.save_pretrained('finetune/merged')
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0').save_pretrained('finetune/merged')
print('Merged model saved to finetune/merged/')
"

# Convert to GGUF (requires llama.cpp)
python llama.cpp/convert_hf_to_gguf.py finetune/merged --outfile finetune/tinyllama-json-agent.gguf

# Register with Ollama
echo 'FROM ./finetune/tinyllama-json-agent.gguf' > Modelfile
ollama create tinyllama-json-agent -f Modelfile

# Use in the agent
python main.py --model tinyllama-json-agent
```
