"""
Fine-tuning configuration for Local_LLM_Agent JSON action loop.

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Small enough to fine-tune on a single GPU (6–8 GB VRAM) or in Colab
  - Same model family used in the Ollama benchmark (tinyllama)
  - Instruction-tuned variant has the chat template already set up

LoRA targets q_proj + v_proj — standard for decoder-only transformers.
Higher rank (r=16) than typical (r=8) because JSON structure is a precise
formatting task that benefits from more adapter capacity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
DATA_DIR     = FINETUNE_DIR / "data"
ADAPTER_DIR  = FINETUNE_DIR / "adapters"

SFT_DATA_PATH = DATA_DIR / "sft_dataset.jsonl"
DPO_DATA_PATH = DATA_DIR / "dpo_dataset.jsonl"
SFT_ADAPTER   = ADAPTER_DIR / "sft"
DPO_ADAPTER   = ADAPTER_DIR / "dpo"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---------------------------------------------------------------------------
# LoRA hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    r: int             = 16      # rank — higher = more capacity for formatting tasks
    lora_alpha: int    = 32      # scaling = alpha/r = 2.0
    lora_dropout: float = 0.05
    bias: str          = "none"
    task_type: str     = "CAUSAL_LM"
    # Modules to inject LoRA into (standard for LLaMA-family models)
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

# ---------------------------------------------------------------------------
# SFT training arguments
# ---------------------------------------------------------------------------
@dataclass
class SFTTrainingConfig:
    output_dir: str         = str(SFT_ADAPTER)
    num_train_epochs: int   = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # effective batch = 16
    learning_rate: float    = 2e-4
    warmup_ratio: float     = 0.05
    lr_scheduler_type: str  = "cosine"
    logging_steps: int      = 10
    save_strategy: str      = "epoch"
    fp16: bool              = True          # use bf16=True on Ampere+ GPUs
    max_seq_length: int     = 512
    packing: bool           = False         # short sequences — no packing needed

# ---------------------------------------------------------------------------
# DPO training arguments
# ---------------------------------------------------------------------------
@dataclass
class DPOTrainingConfig:
    output_dir: str         = str(DPO_ADAPTER)
    num_train_epochs: int   = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch = 16
    learning_rate: float    = 5e-5         # lower LR for DPO (we're already close to good)
    beta: float             = 0.1          # DPO temperature — 0.1 is standard
    warmup_ratio: float     = 0.1
    lr_scheduler_type: str  = "cosine"
    logging_steps: int      = 5
    save_strategy: str      = "epoch"
    fp16: bool              = True
    max_length: int         = 512
    max_prompt_length: int  = 256
