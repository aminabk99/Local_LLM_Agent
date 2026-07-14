#!/usr/bin/env python3
"""
LoRA Supervised Fine-Tuning (SFT) for the JSON action loop.

Uses TRL's SFTTrainer + HuggingFace PEFT to attach LoRA adapters to
TinyLlama-1.1B-Chat-v1.0 and train on the synthetic SFT dataset.

The SFT stage teaches the model the output format (clean JSON only).
DPO then reinforces preference between good and bad outputs.

Requirements:
    pip install torch transformers peft trl datasets accelerate bitsandbytes

Usage:
    # Generate dataset first
    python -m finetune.data.generate_sft

    # Train (GPU recommended; CPU works but is slow)
    python -m finetune.train_sft

    # Adapter saved to finetune/adapters/sft/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy imports — fail gracefully with a helpful message if deps are missing
# ---------------------------------------------------------------------------
def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"Missing packages: {', '.join(missing)}\n"
            "Install with:\n"
            "  pip install torch transformers peft trl datasets accelerate\n"
        )
        sys.exit(1)


def main() -> None:
    _check_deps()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    from finetune.config import (
        BASE_MODEL, SFT_DATA_PATH, SFT_ADAPTER, LoRAConfig, SFTTrainingConfig,
    )
    from finetune.data.generate_sft import generate as gen_sft

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    if not SFT_DATA_PATH.exists():
        print("SFT dataset not found — generating...")
        examples = gen_sft()
        SFT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SFT_DATA_PATH.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Generated {len(examples)} examples.")

    raw = [json.loads(l) for l in SFT_DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Format: apply chat template so the model sees <|system|>...<|user|>...<|assistant|>...
    def _format(ex: dict) -> dict:
        text = (
            f"<|system|>\n{ex['system']}</s>\n"
            f"<|user|>\n{ex['prompt']}</s>\n"
            f"<|assistant|>\n{ex['completion']}</s>\n"
        )
        return {"text": text}

    dataset = Dataset.from_list([_format(ex) for ex in raw])
    print(f"Dataset: {len(dataset)} examples")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # 3. Base model
    # ------------------------------------------------------------------
    cfg = SFTTrainingConfig()
    lora_cfg = LoRAConfig()

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model on {device_map} ({dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ------------------------------------------------------------------
    # 4. LoRA adapter
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=False,
        report_to="none",           # set to "wandb" or "tensorboard" if desired
        dataloader_num_workers=0,
    )

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    # Train only on completions (not on the prompt/system text)
    response_template = "<|assistant|>\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("\nStarting SFT training...")
    trainer.train()

    # ------------------------------------------------------------------
    # 8. Save adapter
    # ------------------------------------------------------------------
    SFT_ADAPTER.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(SFT_ADAPTER))
    tokenizer.save_pretrained(str(SFT_ADAPTER))
    print(f"\nSFT adapter saved → {SFT_ADAPTER}")
    print("Next step: python -m finetune.train_dpo")


if __name__ == "__main__":
    main()
