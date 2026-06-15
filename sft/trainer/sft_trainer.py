"""
Model-agnostic SFT trainer (accelerate + FSDP).

Trains with teacher-forced language modeling loss on (prompt, response) pairs.
Supports LoRA and full fine-tuning, multi-modal inputs (audio, video, images),
and any transformers model whose forward accepts the processor's outputs
(Qwen2.5-VL, Qwen3-VL, Gemma 3/4, Qwen2.5-Omni, ...).
"""
import os
import re
import sys
import json
import time
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from math import floor
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

# Put the sft/ package root (parent of trainer/) on the path so that
# `dataset.*` imports resolve regardless of the launch directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.sft_parquet_dataset import SFTParquetDataset
from dataset.dataset_utils import collate_fn

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)


class SFTTrainer:
    """
    Accelerate-based SFT trainer for Qwen2.5-Omni.

    Trains with standard cross-entropy loss on response tokens only
    (prompt tokens are masked via loss_mask).
    """

    def __init__(
        self,
        model,
        tokenizer,
        processor,
        train_data_files,
        val_data_files,
        dataset_config,
        train_batch_size=1,
        val_batch_size=1,
        lr=1e-5,
        epochs=3,
        gradient_accumulation_steps=8,
        num_workers=0,
        save_checkpoint_dir="./checkpoints",
        load_checkpoint_path=None,
        save_every_n_steps=None,
        save_every_n_epochs=1,
        validate_every_n_steps=None,
        validate_every_n_epochs=1,
        use_scheduler=True,
        scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
        use_wandb=False,
        wandb_project="sft-qwen-omni",
        wandb_entity=None,
        seed=42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_data_files = train_data_files
        self.val_data_files = val_data_files
        self.dataset_config = dataset_config

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.lr = lr
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.max_grad_norm = max_grad_norm

        self.save_checkpoint_dir = save_checkpoint_dir
        self.load_checkpoint_path = load_checkpoint_path
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.validate_every_n_steps = validate_every_n_steps
        self.validate_every_n_epochs = validate_every_n_epochs

        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.seed = seed

        # Initialize accelerator
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="bf16",
            log_with="wandb" if self.use_wandb else None,
            kwargs_handlers=[ddp_kwargs],
        )

        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                config={
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "batch_size": self.train_batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                },
            )

        set_seed(self.seed)

    def _get_dataloader(self, data_files, batch_size, shuffle=True, split="train"):
        dataset = SFTParquetDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.dataset_config,
            processor=self.processor,
            split=split,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _compute_loss(self, batch):
        """Compute cross-entropy loss on response tokens only."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]

        # Forward pass
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add multi-modal inputs if present
        if "multi_modal_inputs" in batch:
            mm = batch["multi_modal_inputs"]
            if isinstance(mm, dict):
                for k, v in mm.items():
                    forward_kwargs[k] = v

        outputs = self.model(**forward_kwargs)
        logits = outputs.logits  # [B, T, V]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous().float()

        # Cross-entropy loss
        loss_fct = CrossEntropyLoss(reduction="none")
        B, T, V = shift_logits.shape
        per_token_loss = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(B, T)

        # Apply loss mask (only response tokens contribute)
        masked_loss = per_token_loss * shift_loss_mask
        num_tokens = shift_loss_mask.sum()
        if num_tokens > 0:
            loss = masked_loss.sum() / num_tokens
        else:
            loss = masked_loss.sum()

        return loss, num_tokens.item()

    def _save_checkpoint(self, step, epoch):
        """Save model checkpoint + full training state for resuming."""
        save_dir = os.path.join(self.save_checkpoint_dir, f"step_{step}")
        self.accelerator.wait_for_everyone()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)

            # Save LoRA adapter (for inference / merging)
            unwrapped_model.save_pretrained(
                save_dir,
                save_function=self.accelerator.save,
            )
            self.tokenizer.save_pretrained(save_dir)

            # Save training state metadata
            state = {"step": step, "epoch": epoch}
            with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                json.dump(state, f)

        # Save full accelerate state (model, optimizer, scheduler, RNG)
        # for resuming training
        accel_state_dir = os.path.join(save_dir, "accelerate_state")
        self.accelerator.save_state(accel_state_dir)

        if self.accelerator.is_main_process:
            print(f"[Checkpoint] Saved to {save_dir}")

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract answer from model output.

        Handles: \\boxed{answer}, <answer>answer</answer>, or extracts the first
        meaningful phrase before punctuation/filler (e.g. 'neutral. well...' -> 'neutral').
        """
        # Try \boxed{...}
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        # Try <answer>...</answer>
        match = re.search(r'<answer>([^<]+)</answer>', text)
        if match:
            return match.group(1).strip()
        # Extract first phrase: take text before first period, comma, newline, or "well"
        text = text.strip()
        # Split on sentence-ending punctuation or common filler starts
        match = re.match(r'^([^.\n,!?]+)', text)
        if match:
            return match.group(1).strip()
        return text.strip()

    @torch.no_grad()
    def validate(self, val_dataloader):
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(val_dataloader, desc="Validating (loss)", disable=not self.accelerator.is_main_process):
            loss, n_tokens = self._compute_loss(batch)
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        self.model.train()
        return avg_loss

    @torch.no_grad()
    def validate_generation(self, val_dataloader, global_step=0, max_new_tokens=64, max_samples=1000):
        """
        Generation-based validation: generate text, compare with ground truth.
        Computes per-dataset accuracy and logs to wandb, matching TARPO evaluation.
        Caps at max_samples to keep generation time reasonable.
        """
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)

        all_predictions = []
        all_ground_truths = []
        all_datasets = []
        all_tasks = []
        sample_count = 0

        for batch in tqdm(val_dataloader, desc="Validating (gen)", disable=not self.accelerator.is_main_process):
            if sample_count >= max_samples:
                break
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            ground_truths = batch.get("ground_truth", [])
            dataset_names = batch.get("dataset_name", [])
            tasks = batch.get("task", [])
            prompt_lens = batch.get("prompt_len", [])

            # For generation, use only the prompt portion
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                p_len = int(prompt_lens[i]) if torch.is_tensor(prompt_lens[i]) else int(prompt_lens[i])
                p_ids = input_ids[i, :p_len].unsqueeze(0)
                p_mask = attention_mask[i, :p_len].unsqueeze(0)

                # Generate
                gen_kwargs = {
                    "input_ids": p_ids,
                    "attention_mask": p_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                try:
                    output_ids = unwrapped.generate(**gen_kwargs)
                    # Decode only the generated part
                    gen_ids = output_ids[0, p_len:]
                    prediction = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                except Exception as e:
                    prediction = f"[ERROR: {e}]"

                pred_answer = self._extract_answer(prediction).lower()
                gt = ground_truths[i]
                if isinstance(gt, bytes):
                    gt = gt.decode("utf-8")
                gt = str(gt).lower()

                ds = dataset_names[i]
                if isinstance(ds, bytes):
                    ds = ds.decode("utf-8")
                ds = str(ds)

                task = tasks[i]
                if isinstance(task, bytes):
                    task = task.decode("utf-8")
                task = str(task)

                all_predictions.append(pred_answer)
                all_ground_truths.append(gt)
                all_datasets.append(ds)
                all_tasks.append(task)
                sample_count += 1

        # Compute metrics
        metrics = self._compute_generation_metrics(
            all_predictions, all_ground_truths, all_datasets, all_tasks, global_step
        )

        self.model.train()
        return metrics

    def _compute_generation_metrics(self, predictions, ground_truths, datasets, tasks, global_step):
        """Compute per-dataset accuracy and aggregate metrics, matching TARPO evaluation."""
        metrics = {}

        # Overall accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        total = len(predictions)
        overall_acc = correct / max(total, 1)
        metrics["val/overall_accuracy"] = overall_acc
        metrics["val/total_samples"] = total

        # Per-dataset accuracy
        dataset_correct = defaultdict(int)
        dataset_total = defaultdict(int)
        for pred, gt, ds in zip(predictions, ground_truths, datasets):
            dataset_total[ds] += 1
            if pred == gt:
                dataset_correct[ds] += 1

        dataset_accs = []
        for ds in sorted(dataset_total.keys()):
            acc = dataset_correct[ds] / dataset_total[ds]
            metrics[f"val/{ds}/accuracy"] = acc
            metrics[f"val/{ds}/count"] = dataset_total[ds]
            dataset_accs.append(acc)

        # Mean accuracy across datasets (macro average)
        metrics["val/macro_accuracy"] = np.mean(dataset_accs) if dataset_accs else 0.0

        # Per-task accuracy
        task_correct = defaultdict(int)
        task_total = defaultdict(int)
        for pred, gt, task in zip(predictions, ground_truths, tasks):
            task_total[task] += 1
            if pred == gt:
                task_correct[task] += 1

        for task in sorted(task_total.keys()):
            acc = task_correct[task] / task_total[task]
            metrics[f"val/task/{task}/accuracy"] = acc

        # Save generation outputs to disk
        if self.accelerator.is_main_process:
            output_dir = os.path.join(self.save_checkpoint_dir, "val_generations")
            os.makedirs(output_dir, exist_ok=True)
            output_data = {
                "global_step": global_step,
                "overall_accuracy": overall_acc,
                "predictions": predictions,
                "ground_truths": ground_truths,
                "datasets": datasets,
                "tasks": tasks,
            }
            output_path = os.path.join(output_dir, f"step_{global_step}.json")
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"[Val Gen] Saved to {output_path}")

            # Print summary
            print(f"[Val Gen] Overall accuracy: {overall_acc:.4f} ({correct}/{total})")
            for ds in sorted(dataset_total.keys()):
                acc = dataset_correct[ds] / dataset_total[ds]
                print(f"  {ds}: {acc:.4f} ({dataset_correct[ds]}/{dataset_total[ds]})")

        return metrics

    def train(self):
        """Main training loop."""
        train_split = self.dataset_config.get("train_split", "train[:10%]")
        val_split = self.dataset_config.get("val_split", "validation")

        print(f"[SFTTrainer] Building dataloaders...")
        print(f"  Train split: {train_split}")
        print(f"  Val split:   {val_split}")
        train_dataloader = self._get_dataloader(
            self.train_data_files, self.train_batch_size, shuffle=True, split=train_split
        )
        val_dataloader = None
        if self.val_data_files:
            val_dataloader = self._get_dataloader(
                self.val_data_files, self.val_batch_size, shuffle=False, split=val_split
            )

        # Optimizer
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # Scheduler
        num_training_steps = len(train_dataloader) * self.epochs // self.gradient_accumulation_steps
        # Auto-scale warmup: if warmup_steps > total steps, use 10% of total
        warmup_steps = self.warmup_steps
        if warmup_steps >= num_training_steps:
            warmup_steps = max(1, int(num_training_steps * 0.1))
            print(f"[SFTTrainer] Warmup auto-scaled to {warmup_steps} steps (10% of {num_training_steps})")
        scheduler = None
        if self.use_scheduler:
            scheduler = get_scheduler(
                self.scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        # Prepare with accelerate
        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )
        if scheduler:
            scheduler = self.accelerator.prepare(scheduler)
        if val_dataloader:
            val_dataloader = self.accelerator.prepare(val_dataloader)

        # Resume from checkpoint if specified
        global_step = 0
        resume_epoch = 0
        best_val_loss = float("inf")

        if self.load_checkpoint_path:
            accel_state = os.path.join(self.load_checkpoint_path, "accelerate_state")
            if os.path.exists(accel_state):
                print(f"[SFTTrainer] Resuming from {self.load_checkpoint_path}")
                self.accelerator.load_state(accel_state)
                # Restore training state
                state_file = os.path.join(self.load_checkpoint_path, "training_state.json")
                if os.path.exists(state_file):
                    with open(state_file) as f:
                        state = json.load(f)
                    global_step = state.get("step", 0)
                    resume_epoch = state.get("epoch", 0)
                    print(f"[SFTTrainer] Resumed at step {global_step}, epoch {resume_epoch}")
            else:
                print(f"[SFTTrainer] No accelerate state found at {accel_state}, starting fresh")

        if self.accelerator.is_main_process:
            print(f"[SFTTrainer] Starting training")
            print(f"  Epochs: {self.epochs}")
            print(f"  Train batches/epoch: {len(train_dataloader)}")
            print(f"  Batch size: {self.train_batch_size}")
            print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"  Effective batch size: {self.train_batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes}")
            print(f"  Learning rate: {self.lr}")
            print(f"  Total training steps: {num_training_steps}")

        for epoch in range(resume_epoch, self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_steps = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    loss, n_tokens = self._compute_loss(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                cur_loss = loss.item()
                epoch_loss += cur_loss * n_tokens
                epoch_tokens += n_tokens
                epoch_steps += 1

                if self.accelerator.sync_gradients:
                    global_step += 1

                # Update progress bar every micro-step so loss is always visible
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                current_lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    loss=f"{cur_loss:.4f}", avg=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}", step=global_step
                )

                # Log to wandb on optimizer steps
                if self.accelerator.sync_gradients and self.use_wandb and self.accelerator.is_main_process:
                    self.accelerator.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/lr": current_lr,
                        "train/epoch": epoch + step / len(train_dataloader),
                        "train/global_step": global_step,
                    }, step=global_step)

                    # Step-based validation
                    if (
                        val_dataloader
                        and self.validate_every_n_steps
                        and global_step % self.validate_every_n_steps == 0
                    ):
                        val_loss = self.validate(val_dataloader)
                        gen_metrics = self.validate_generation(val_dataloader, global_step)
                        if self.accelerator.is_main_process:
                            print(f"\n[Step {global_step}] Val loss: {val_loss:.4f}, "
                                  f"accuracy: {gen_metrics.get('val/overall_accuracy', 0):.4f}")
                            if self.use_wandb:
                                self.accelerator.log({"val/loss": val_loss, **gen_metrics}, step=global_step)
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                self._save_checkpoint(global_step, epoch)

                    # Step-based checkpoint
                    if (
                        self.save_every_n_steps
                        and global_step % self.save_every_n_steps == 0
                    ):
                        self._save_checkpoint(global_step, epoch)

            # Epoch summary
            epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
            if self.accelerator.is_main_process:
                print(f"\n[Epoch {epoch + 1}] Avg train loss: {epoch_avg_loss:.4f}")

            # Epoch-based validation
            if (
                val_dataloader
                and self.validate_every_n_epochs
                and (epoch + 1) % self.validate_every_n_epochs == 0
            ):
                val_loss = self.validate(val_dataloader)
                gen_metrics = self.validate_generation(val_dataloader, global_step)
                if self.accelerator.is_main_process:
                    print(f"[Epoch {epoch + 1}] Val loss: {val_loss:.4f}, "
                          f"accuracy: {gen_metrics.get('val/overall_accuracy', 0):.4f}")
                    if self.use_wandb:
                        self.accelerator.log({"val/loss": val_loss, "val/epoch": epoch + 1, **gen_metrics}, step=global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

            # Epoch-based checkpoint
            if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(global_step, epoch)

        # Final save
        self._save_checkpoint(global_step, self.epochs - 1)

        if self.use_wandb:
            self.accelerator.end_training()

        if self.accelerator.is_main_process:
            print(f"[SFTTrainer] Training complete. Best val loss: {best_val_loss:.4f}")
