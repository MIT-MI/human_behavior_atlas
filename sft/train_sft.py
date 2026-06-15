"""
Entry point for unified SFT training on Human Behavior Atlas parquet data.

Model-agnostic: the model class is selected from `model.arch` in the config:
  * "auto"         -> AutoModelForImageTextToText  (Qwen2.5-VL, Qwen3-VL, Gemma 3/4, ...)
  * "omni_thinker" -> Qwen2_5OmniThinkerForConditionalGeneration  (audio+vision omni)
  * "causal_lm"    -> AutoModelForCausalLM          (text-only)

Usage:
    accelerate launch --config_file configs/accelerate_config_sft_1gpu.yaml \
        train_sft.py --config configs/config_sft.yaml [overrides]
"""
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from trainer.sft_trainer import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SFT training (model-agnostic)")

    parser.add_argument("--config", type=str, default="configs/config_sft.yaml",
                        help="Path to YAML config file")

    # Overrides
    parser.add_argument("--train_files", type=str, nargs="+")
    parser.add_argument("--val_files", type=str, nargs="+")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_arch", type=str, choices=["auto", "omni_thinker", "causal_lm"])
    parser.add_argument("--modalities", type=str, help="comma-separated, e.g. 'images,videos'")
    parser.add_argument("--training_strategy", type=str, choices=["lora", "full"])
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--save_checkpoint_dir", type=str)
    parser.add_argument("--load_checkpoint_path", type=str)
    parser.add_argument("--max_prompt_length", type=int)
    parser.add_argument("--max_response_length", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--project", type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--save_every_n_steps", type=int)
    parser.add_argument("--validate_every_n_steps", type=int)

    return parser.parse_args()


def load_model(model_name, arch, torch_dtype, attn_implementation):
    """Load the base model according to the configured architecture."""
    print(f"[INFO] Loading model '{model_name}' (arch={arch}, dtype={torch_dtype})...")
    if arch == "omni_thinker":
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        )
        # accelerate FSDP crashes on this model's tied weights.
        model.config.tie_word_embeddings = False
        return model
    if arch == "causal_lm":
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
    # arch == "auto": vision-language / image-text-to-text models
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        trust_remote_code=True,
    )


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # Apply CLI overrides
    if args.train_files:
        cfg.data.train_files = args.train_files
    if args.val_files:
        cfg.data.val_files = args.val_files
    if args.model_name:
        cfg.model.name = args.model_name
    if args.model_arch:
        cfg.model.arch = args.model_arch
    if args.modalities:
        cfg.dataset_config.modalities = args.modalities
    if args.training_strategy:
        cfg.model.training_strategy = args.training_strategy
    if args.train_batch_size is not None:
        cfg.train.train_batch_size = args.train_batch_size
    if args.val_batch_size is not None:
        cfg.train.val_batch_size = args.val_batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.gradient_accumulation_steps is not None:
        cfg.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.save_checkpoint_dir:
        cfg.train.save_checkpoint_dir = args.save_checkpoint_dir
    if args.load_checkpoint_path:
        cfg.train.load_checkpoint_path = args.load_checkpoint_path
    if args.max_prompt_length is not None:
        cfg.dataset_config.max_prompt_length = args.max_prompt_length
    if args.max_response_length is not None:
        cfg.dataset_config.max_response_length = args.max_response_length
    if args.lora_r is not None:
        cfg.model.lora_config.r = args.lora_r
    if args.lora_alpha is not None:
        cfg.model.lora_config.alpha = args.lora_alpha
    if args.project:
        cfg.wandb.project = args.project
    if args.use_wandb:
        cfg.wandb.use = True
    if args.warmup_steps is not None:
        cfg.train.warmup_steps = args.warmup_steps
    if args.save_every_n_steps is not None:
        cfg.train.save_every_n_steps = args.save_every_n_steps
    if args.validate_every_n_steps is not None:
        cfg.train.validate_every_n_steps = args.validate_every_n_steps

    model_name = cfg.model.name
    arch = cfg.model.get("arch", "auto")
    training_strategy = cfg.model.training_strategy

    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Arch: {arch}")
    print(f"[INFO] Strategy: {training_strategy}")
    print(f"[INFO] Modalities: {cfg.dataset_config.get('modalities', 'images,videos')}")

    # Tokenizer + processor
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    torch_dtype = getattr(torch, cfg.model.get("torch_dtype", "bfloat16"))
    attn_impl = cfg.model.get("attn_implementation", "sdpa")
    model = load_model(model_name, arch, torch_dtype, attn_impl)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # LoRA / full fine-tuning
    if training_strategy == "lora":
        lora_cfg = cfg.model.lora_config
        lora_config = LoraConfig(
            r=int(lora_cfg.r),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.target_modules),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Parameters: {trainable_params:,} trainable / {total_params:,} total")

    def _parse_optional_int(val):
        if val is None or val == "None":
            return None
        return int(val)

    dataset_config = OmegaConf.create(dict(cfg.dataset_config))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        train_data_files=cfg.data.train_files if isinstance(cfg.data.train_files, list) else [cfg.data.train_files],
        val_data_files=cfg.data.val_files if isinstance(cfg.data.val_files, list) else [cfg.data.val_files],
        dataset_config=dataset_config,
        train_batch_size=int(cfg.train.train_batch_size),
        val_batch_size=int(cfg.train.val_batch_size),
        lr=float(cfg.train.lr),
        epochs=int(cfg.train.epochs),
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        num_workers=int(cfg.train.get("num_workers", 0)),
        save_checkpoint_dir=cfg.train.save_checkpoint_dir,
        load_checkpoint_path=cfg.train.get("load_checkpoint_path", None),
        save_every_n_steps=_parse_optional_int(cfg.train.get("save_every_n_steps", None)),
        save_every_n_epochs=_parse_optional_int(cfg.train.get("save_every_n_epochs", 1)),
        validate_every_n_steps=_parse_optional_int(cfg.train.get("validate_every_n_steps", None)),
        validate_every_n_epochs=_parse_optional_int(cfg.train.get("validate_every_n_epochs", 1)),
        use_scheduler=bool(cfg.train.get("use_scheduler", True)),
        scheduler_type=cfg.train.get("scheduler_type", "cosine"),
        warmup_steps=int(cfg.train.get("warmup_steps", 100)),
        max_grad_norm=float(cfg.train.get("max_grad_norm", 1.0)),
        use_wandb=bool(cfg.wandb.get("use", False)),
        wandb_project=cfg.wandb.get("project", "sft-hba"),
        wandb_entity=cfg.wandb.get("entity", None),
        seed=int(cfg.train.get("seed", 42)),
    )

    trainer.train()


if __name__ == "__main__":
    main()
