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
