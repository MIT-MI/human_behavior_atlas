"""
Merge a LoRA adapter (produced by train_sft.py) into its base model and save a
full HuggingFace model directory that vLLM / verl can load directly for GRPO.

Two modes, selected by --arch:
  * auto / causal_lm : load the base model, attach the adapter, merge_and_unload,
                       save. This is the path for Qwen2.5-VL / Qwen3-VL / Gemma.
  * omni_thinker     : the SFT trained the Omni *thinker* with LoRA; load the
                       full Omni model, replace its thinker with the merged
                       weights, and save the full Omni model.

Usage:
    python merge_lora.py --arch auto \
        --base_model Qwen/Qwen3-VL-4B-Instruct \
        --adapter_path /path/to/sft_ckpt/step_30 \
        --output_path  /path/to/merged
"""
import argparse
import json
import os

import torch
from transformers import AutoProcessor, AutoTokenizer
from peft import PeftModel


def merge_simple(args, torch_dtype):
    """Plain LoRA merge for AutoModel VLM / CausalLM bases (non-omni)."""
    if args.arch == "causal_lm":
        from transformers import AutoModelForCausalLM as Loader
    else:
        from transformers import AutoModelForImageTextToText as Loader

    print(f"[INFO] Loading base model: {args.base_model}")
    base = Loader.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, device_map="cpu", trust_remote_code=True,
    )
    print(f"[INFO] Attaching LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path, device_map="cpu")
    print("[INFO] Merging LoRA weights...")
    model = model.merge_and_unload()
    print(f"[INFO] Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path, safe_serialization=True)


def merge_omni(args, torch_dtype):
    """Omni path: merge LoRA'd thinker, graft into the full Omni model."""
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    print(f"[INFO] Loading thinker from: {args.base_model}")
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, attn_implementation="sdpa", device_map="cpu",
    )
    print(f"[INFO] Loading LoRA adapter: {args.adapter_path}")
    thinker = PeftModel.from_pretrained(thinker, args.adapter_path, device_map="cpu")
    print("[INFO] Merging LoRA weights...")
    thinker = thinker.merge_and_unload()

    print(f"[INFO] Loading full Omni model: {args.base_model}")
    full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, attn_implementation="sdpa", device_map="cpu",
    )
    print("[INFO] Replacing thinker weights in full model...")
    full_model.thinker.load_state_dict(thinker.state_dict(), strict=False)
    del thinker

    print(f"[INFO] Saving full merged model to: {args.output_path}")
    full_model.save_pretrained(args.output_path, safe_serialization=True)

    # vLLM expects "Qwen2_5OmniModel"
    cfg_path = os.path.join(args.output_path, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["architectures"] = ["Qwen2_5OmniModel"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("[INFO] Patched config.json architectures for vLLM")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--arch", type=str, default="auto",
                        choices=["auto", "causal_lm", "omni_thinker"])
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="LoRA adapter checkpoint dir (contains adapter_model.safetensors)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    os.makedirs(args.output_path, exist_ok=True)

    if args.arch == "omni_thinker":
        merge_omni(args, torch_dtype)
    else:
        merge_simple(args, torch_dtype)

    # Save tokenizer + processor alongside the merged weights.
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(args.output_path)
    try:
        AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(args.output_path)
    except Exception as e:  # text-only models may have no processor
        print(f"[INFO] No processor saved ({e})")

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
