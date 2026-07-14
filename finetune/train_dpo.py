#!/usr/bin/env python3
"""
DPO (Direct Preference Optimisation) fine-tuning.

Starts from the SFT adapter and further trains using preference pairs
from dpo_dataset.jsonl — the model learns to prefer clean JSON over
markdown-wrapped, prose-prefixed, or wrong-key responses.

Why DPO after SFT?
  SFT teaches the model the correct format.
  DPO reinforces it by showing the model *why* certain outputs are wrong —
  it learns the contrastive signal between good and bad responses.

Requirements:
    pip install torch transformers peft trl datasets accelerate

Usage:
    python -m finetune.train_dpo

Adapter saved to: finetune/adapters/dpo/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}\npip install torch transformers peft trl datasets accelerate")
        sys.exit(1)


def main() -> None:
    _check_deps()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, TaskType
    from trl import DPOTrainer, DPOConfig

    from finetune.config import (
        BASE_MODEL, DPO_DATA_PATH, SFT_ADAPTER, DPO_ADAPTER, LoRAConfig, DPOTrainingConfig,
    )
    from finetune.data.generate_dpo import generate as gen_dpo

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    if not DPO_DATA_PATH.exists():
        print("DPO dataset not found — generating...")
        examples = gen_dpo()
        DPO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DPO_DATA_PATH.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Generated {len(examples)} preference pairs.")

    raw = [json.loads(l) for l in DPO_DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # DPOTrainer expects: prompt, chosen, rejected (plain strings)
    def _format(ex: dict) -> dict:
        return {
            "prompt":   f"<|system|>\n{ex['system']}</s>\n<|user|>\n{ex['prompt']}</s>\n<|assistant|>\n",
            "chosen":   ex["chosen"] + "</s>",
            "rejected": ex["rejected"] + "</s>",
        }

    dataset = Dataset.from_list([_format(ex) for ex in raw])
    print(f"DPO dataset: {len(dataset)} preference pairs")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    # Load from SFT adapter if available (includes any tokenizer updates)
    tok_path = str(SFT_ADAPTER) if SFT_ADAPTER.exists() else BASE_MODEL
    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO prefers left-padding

    # ------------------------------------------------------------------
    # 3. Model — load base + SFT adapter
    # ------------------------------------------------------------------
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if SFT_ADAPTER.exists():
        print(f"Loading SFT adapter from: {SFT_ADAPTER}")
        model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER), is_trainable=True)
    else:
        print("WARNING: SFT adapter not found — starting DPO from base model.")
        print("Run python -m finetune.train_sft first for best results.")
        from peft import get_peft_model
        lora_cfg = LoRAConfig()
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_cfg.target_modules,
        )
        model = get_peft_model(base_model, peft_config)

    model.config.use_cache = False

    # Reference model (frozen SFT model — DPO KL penalty target)
    print("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        tok_path if SFT_ADAPTER.exists() else BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # 4. DPO training config
    # ------------------------------------------------------------------
    cfg = DPOTrainingConfig()

    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=False,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        report_to="none",
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\nStarting DPO training...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    DPO_ADAPTER.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(DPO_ADAPTER))
    tokenizer.save_pretrained(str(DPO_ADAPTER))
    print(f"\nDPO adapter saved → {DPO_ADAPTER}")
    print("Next step: python -m finetune.eval")


if __name__ == "__main__":
    main()
