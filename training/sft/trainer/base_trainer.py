"""
Base trainer shared by all multi-task classification trainers.

Subclasses override:
  - get_dataloader()        — to swap dataset type (e.g. AddQAOmniClassifierDataset)
  - _datasets_to_domain_ids() — QA trainer adds -1 sentinel for QA rows
  - _extra_wandb_config()   — to append trainer-specific wandb fields
  - validate()              — BAM overrides to apply adapters; QA overrides for mixed loss
  - train()                 — QA and BAM override fully (different loss computation)
  - test()                  — inheritable; override if evaluation differs
"""
import os
import sys
import json
import time
import torch
from collections import defaultdict
from datetime import datetime
from math import floor
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.sft_dataset import OmniClassifierDataset
from dataset.dataset_utils import collate_fn
from utils.wandb_utils import init_wandb, log_metrics, finish
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
from evaluate.detailed_multi_task_evaluation import evaluate_predictions

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate.utils.fsdp_utils import load_fsdp_model

logger = get_logger(__name__)


class BaseMultiHeadTrainer:
    """
    Infrastructure shared by MultiHead, QA, and BAM trainers:
    accelerate setup, domain routing, checkpoint I/O, standard validation, and
    a classification training loop usable as-is by the MultiHead trainer.
    """

    def __init__(
        self,
        data_files,
        val_data_files,
        test_data_files,
        tokenizer,
        processor,
        config,
        batch_size,
        val_batch_size,
        test_batch_size,
        lr,
        epochs,
        save_checkpoint_dir,
        load_checkpoint_path,
        model,
        gradient_accumulation_steps,
        num_workers=0,
        use_lora=False,
        global_config=None,
    ):
        self.data_files = data_files
        self.val_data_files = val_data_files
        self.test_data_files = test_data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.label_key = config.get("label_key", "answer")

        self.global_config = global_config or {}
        self.full_label_scheme = self.global_config.get('FULL_LABEL_SCHEME', None)
        self.label_map = self.global_config.get('LABEL_MAP', {})
        self.label_map_path = self.global_config.get('LABEL_MAP_PATH', None)

        self.build_domain_routing()

        self.use_scheduler = self.global_config.get("USE_SCHEDULER", True)
        self.scheduler_type = self.global_config.get("SCHEDULER_TYPE", "cosine")
        self.warmup_steps = self.global_config.get("WARMUP_STEPS", None)

        self.checkpoint_dir = save_checkpoint_dir
        self.load_checkpoint_path = load_checkpoint_path
        self.validation_result_dir = self.global_config.get('VALIDATION_RESULT_DIR', None)

        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.steps_without_improvement = 0

        # Defaults for subclass attrs referenced by _extra_wandb_config() hooks,
        # which may be called during _init_wandb() before the subclass __init__ runs.
        self.task_type = self.global_config.get("TASK_TYPE", "cls")
        self.bam_hidden = self.global_config.get("BAM_HIDDEN", 128)
        self.qa_datasets = set()
        self.qa_loss_weight = 1.0

        use_wandb = self.global_config.get('USE_WANDB', False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision='fp16',
            log_with="wandb" if use_wandb else None,
            project_dir=save_checkpoint_dir if use_wandb else None,
            kwargs_handlers=self._accelerator_kwargs_handlers(),
        )
        set_seed(42)

        if use_wandb and self.accelerator.is_main_process:
            self._init_wandb()

        self.start_time = time.time()

    # -------------------------------------------------------------------------
    # Hooks for subclasses
    # -------------------------------------------------------------------------

    def _accelerator_kwargs_handlers(self) -> list:
        """Return kwargs_handlers for Accelerator. Override in subclasses."""
        return []

    # -------------------------------------------------------------------------
    # Wandb
    # -------------------------------------------------------------------------

    def _extra_wandb_config(self) -> dict:
        """Return extra wandb config entries. Override in subclasses."""
        return {}

    def _init_wandb(self):
        cfg = {
            "model_name": self.global_config.get('TOKENIZER_NAME', ''),
            "training_strategy": self.global_config.get('TRAINING_STRATEGY', ''),
            "batch_size": self.batch_size,
            "val_batch_size": self.val_batch_size,
            "test_batch_size": self.test_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "num_classes": self.global_config.get('NUM_CLASSES', 0),
            "validate_every_n_epochs": self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None),
            "validate_every_n_steps": self.global_config.get('VALIDATE_EVERY_N_STEPS', None),
            "save_every_n_epochs": self.global_config.get('SAVE_EVERY_N_EPOCHS', None),
            "save_every_n_steps": self.global_config.get('SAVE_EVERY_N_STEPS', None),
            "validation_result_dir": self.validation_result_dir,
            "save_checkpoint_dir": self.checkpoint_dir,
            "load_checkpoint_path": self.load_checkpoint_path,
            "early_stopping_patience": self.global_config.get('EARLY_STOPPING_PATIENCE', 0),
            "num_workers": self.num_workers,
            "lora_config": self.global_config.get('LORA_CONFIG', None),
            "label_map_path": self.global_config.get('LABEL_MAP_PATH', ''),
            "datasets": self.global_config.get('label_config', {}).get('datasets', []),
            "accelerate": True,
            "mixed_precision": "fp16",
            "use_scheduler": self.use_scheduler,
            "scheduler_type": self.scheduler_type if self.use_scheduler else None,
            "warmup_steps": self.warmup_steps if self.use_scheduler else None,
        }
        cfg.update(self._extra_wandb_config())
        init_wandb(
            project=self.global_config.get('WANDB_PROJECT', ''),
            entity=self.global_config.get('WANDB_ENTITY', ''),
            config=cfg,
            run_name=f"omni_classifier_accelerate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # -------------------------------------------------------------------------
    # Domain routing
    # -------------------------------------------------------------------------

    def build_domain_routing(self):
        meta = self.full_label_scheme.get("meta", {})
        global_classes = meta.get("global_classes", {})
        domain_names = list(global_classes.keys())
        self.domain_name_to_id = {d: i for i, d in enumerate(domain_names)}
        self.domain_id_to_global_indices = [[x["index"] for x in global_classes[d]] for d in domain_names]
        dataset_to_domain = meta.get("dataset_domain", {})
        self.dataset_to_domain_id = {ds: self.domain_name_to_id[dn] for ds, dn in dataset_to_domain.items()}

    def _datasets_to_domain_ids(self, dataset_names, device):
        ids = []
        for ds in dataset_names:
            if isinstance(ds, bytes):
                ds = ds.decode("utf-8")
            if ds not in self.dataset_to_domain_id:
                raise KeyError(f"Dataset '{ds}' not in label_map.meta.dataset_domain")
            ids.append(self.dataset_to_domain_id[ds])
        return torch.tensor(ids, dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Dataloader
    # -------------------------------------------------------------------------

    def get_dataloader(self, data_files, batch_size, num_workers=0, shuffle=True):
        dataset = OmniClassifierDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.label_key,
            label_map=self.label_map,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # -------------------------------------------------------------------------
    # Checkpoint I/O
    # -------------------------------------------------------------------------

    def _latest_checkpoint_dir(self, base_dir: str):
        if not os.path.isdir(base_dir):
            return None
        subs = [p for p in Path(base_dir).glob("step_*") if p.is_dir()]
        if not subs:
            return None
        subs.sort(key=lambda p: int(p.name.split("_")[-1]))
        return str(subs[-1])

    def save_checkpoint_unified(self, accelerator, model, epoch, batch_idx, len_train_dataloader, training_strategy, base_ckpt_dir):
        global_step = epoch * len_train_dataloader + (batch_idx + 1)
        ckpt_dir = os.path.join(base_ckpt_dir, f"step_{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        accelerator.save_state(ckpt_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            meta = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "len_train_dataloader": int(len_train_dataloader),
                "training_strategy": str(training_strategy),
                "saved_at_unix": time.time(),
            }
            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        accelerator.print(f"[save] checkpoint @ step {global_step} → {ckpt_dir}")
        return ckpt_dir

    def load_checkpoint_unified(
        self,
        accelerator,
        model,
        base_ckpt_dir: str,
        explicit_dir: str | None = None,
        expect_training_strategy: str | None = None,
        inference_only: bool = False,
    ):
        """
        Load a checkpoint. Pass inference_only=True in test/eval mode to skip loading
        optimizer and scheduler state (avoids mismatches when resuming across training stages).
        """
        ckpt_dir = explicit_dir or None
        if ckpt_dir is None:
            print(f"[load] finding latest checkpoint from {base_ckpt_dir}")
            ckpt_dir = self._latest_checkpoint_dir(base_ckpt_dir)
            print(f"[load] latest checkpoint found: {ckpt_dir}")

        print(f"[load] loading checkpoint from {ckpt_dir}")

        if ckpt_dir is None:
            accelerator.print("[load] no checkpoint found; starting fresh.")
            return 0, 0, 0, None, None

        meta_path = os.path.join(ckpt_dir, "meta.json")
        if not os.path.isfile(meta_path):
            accelerator.print(f"[load] missing meta.json in {ckpt_dir}; starting fresh.")
            return 0, 0, 0, None, None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if expect_training_strategy and meta.get("training_strategy") != expect_training_strategy:
            accelerator.print(f"[warn] strategy mismatch: expected {expect_training_strategy}, got {meta.get('training_strategy')}")

        if inference_only:
            # Load only model weights — avoids optimizer param-group size checks and
            # custom-checkpoint count mismatches when no optimizer is registered.
            load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, ckpt_dir, model_index=0)
        else:
            accelerator.load_state(ckpt_dir)

        global_step = int(meta["global_step"])
        len_dl = int(meta["len_train_dataloader"])
        if len_dl <= 0:
            accelerator.print("[load] invalid len_train_dataloader; starting at epoch 0.")
            return 0, 0, 0, meta, ckpt_dir

        start_epoch = floor((global_step - 1) / len_dl)
        start_batch_offset = (global_step - 1) % len_dl
        accelerator.print(f"[load] resumed {ckpt_dir} → epoch={start_epoch}, step={global_step}, offset={start_batch_offset}")
        return start_epoch, start_batch_offset, global_step, meta, ckpt_dir

    # -------------------------------------------------------------------------
    # Validation (standard classification — override in BAM / QA trainers)
    # -------------------------------------------------------------------------

    def validate(self, val_dataloader, split_name="validation", current_step=None):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_datasets = []
        criterion = CrossEntropyLoss()
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", total=len(val_dataloader), disable=not self.accelerator.is_main_process):
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)

                if 'dataset' not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")
                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)

                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                logits = self.model(input_ids, attention_mask=attention_mask, domain_ids=domain_ids)
                loss = criterion(logits, labels)
                total_loss += loss.item() * input_ids.size(0)

                preds = logits.argmax(dim=1)
                gathered_preds = self.accelerator.gather_for_metrics(preds)
                gathered_labels = self.accelerator.gather_for_metrics(labels)
                gathered_datasets = self.accelerator.gather_for_metrics(batch.get('dataset'))

                if self.accelerator.is_main_process:
                    all_predictions.extend(gathered_preds.cpu().numpy())
                    all_labels.extend(gathered_labels.cpu().numpy())
                    all_datasets.extend(gathered_datasets)

        avg_loss = total_loss / max(1, len(all_labels)) if self.accelerator.is_main_process else 0.0

        if self.accelerator.is_main_process:
            evaluation_results = evaluate_predictions(
                predictions=all_predictions,
                ground_truths=all_labels,
                datasets=all_datasets if all_datasets else None,
                split_name=split_name,
                save_path=self.validation_result_dir,
                global_steps=current_step,
                label_map_path=self.label_map_path,
            )
            aggregate_metrics = evaluation_results["aggregate_metrics"]
            accuracy = aggregate_metrics.get("micro_accuracy", 0.0)
            f1 = aggregate_metrics.get("micro_f1", 0.0)
            precision = aggregate_metrics.get("micro_precision", 0.0)
            recall = aggregate_metrics.get("micro_recall", 0.0)

            print(f"{split_name.capitalize()} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
            print(f"  Macro F1: {aggregate_metrics.get('macro_f1', 0.0):.4f} - Weighted F1: {aggregate_metrics.get('weighted_f1', 0.0):.4f}")

            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': all_predictions,
                'labels': all_labels,
                'evaluation_results': evaluation_results,
                'aggregate_metrics': aggregate_metrics,
            }
        return None

    # -------------------------------------------------------------------------
    # Training loop (standard classification — used by MultiHead; override in QA / BAM)
    # -------------------------------------------------------------------------

    def train(self):
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_dataloader = self.get_dataloader(self.val_data_files, self.val_batch_size, num_workers=self.num_workers, shuffle=False)

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = CrossEntropyLoss()
        total_updates = self.epochs * len(train_dataloader)

        if self.use_scheduler:
            scheduler = get_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_updates)
            print(f"[INFO] Using {self.scheduler_type} scheduler with {self.warmup_steps} warmup steps")
        else:
            scheduler = None
            print("[INFO] Scheduler disabled - using constant learning rate")

        if scheduler is not None:
            self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader, scheduler
            )
            self.accelerator.register_for_checkpointing(scheduler)
        else:
            self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader
            )

        start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
        )

        skipped_dataloader = None
        if start_batch_offset > 0:
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, start_batch_offset)

        validate_every_n_epochs = self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None)
        validate_every_n_steps = self.global_config.get('VALIDATE_EVERY_N_STEPS', None)
        save_every_n_epochs = self.global_config.get('SAVE_EVERY_N_EPOCHS', None)
        save_every_n_steps = self.global_config.get('SAVE_EVERY_N_STEPS', None)
        early_stopping_patience = self.global_config.get('EARLY_STOPPING_PATIENCE', 0)
        use_wandb = self.global_config.get('USE_WANDB', False)
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.model.train()

            is_resumed_epoch = (epoch == start_epoch and skipped_dataloader is not None)
            cur_loader = skipped_dataloader if is_resumed_epoch else train_dataloader
            base_offset = start_batch_offset if is_resumed_epoch else 0

            total_loss = 0.0
            correct = 0
            total = 0
            epoch_start_time = time.time()
            eff_loss = 0.0
            eff_correct = 0
            eff_total = 0

            for batch_idx, batch in tqdm(enumerate(cur_loader), desc="Training", total=len(cur_loader), disable=not self.accelerator.is_main_process):
                self.model.train()
                current_step = (epoch * len(train_dataloader)) + base_offset + batch_idx + 1

                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)

                if 'dataset' not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")
                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)

                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                with self.accelerator.accumulate(self.model):
                    logits = self.model(input_ids, attention_mask=attention_mask, domain_ids=domain_ids)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite logits encountered")
                    loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss encountered")
                    self.accelerator.backward(loss)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                    current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None and self.accelerator.sync_gradients else self.lr

                with torch.no_grad():
                    eff_loss += loss.item() * input_ids.size(0)
                    preds = logits.argmax(dim=1)
                    gathered_preds = self.accelerator.gather_for_metrics(preds)
                    gathered_labels = self.accelerator.gather_for_metrics(labels)

                    if self.accelerator.is_main_process:
                        eff_correct += (gathered_preds == gathered_labels).sum().item()
                        eff_total += gathered_labels.size(0)
                        correct += (gathered_preds == gathered_labels).sum().item()
                        total += gathered_labels.size(0)
                    total_loss += loss.item() * input_ids.size(0)

                    log_batch_training_metrics(
                        epoch=epoch, batch_idx=batch_idx, total_batches=len(train_dataloader),
                        loss=eff_loss, correct=eff_correct, total=eff_total,
                        epoch_start_time=epoch_start_time, start_time=self.start_time,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        batch_size=self.batch_size, epochs=self.epochs,
                        accelerator=self.accelerator, use_wandb=use_wandb,
                        current_lr=current_lr, current_step=current_step,
                    )

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        eff_loss = 0.0
                        eff_correct = 0
                        eff_total = 0

                    if validate_every_n_steps and current_step % validate_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Running step-based validation...")
                        val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                        if self.accelerator.is_main_process and val_results is not None:
                            val_f1 = val_results['f1']
                            if val_f1 > self.best_val_acc:
                                self.best_val_acc = val_f1
                                self.steps_without_improvement = 0
                            else:
                                self.steps_without_improvement += 1
                            val_results['best_val_f1'] = self.best_val_acc
                            val_results['steps_without_improvement'] = self.steps_without_improvement
                            log_validation_results(val_results=val_results, current_step=current_step, split_name="validation", accelerator=self.accelerator, use_wandb=use_wandb)

                    if save_every_n_steps and current_step % save_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Saving checkpoint...")
                        self.save_checkpoint_unified(
                            accelerator=self.accelerator, model=self.model, epoch=epoch,
                            batch_idx=base_offset + batch_idx, len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )

            skipped_dataloader = None

            avg_train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")

            if validate_every_n_epochs and (epoch + 1) % validate_every_n_epochs == 0:
                val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                if self.accelerator.is_main_process and val_results is not None:
                    val_f1 = val_results['f1']
                    if val_f1 > self.best_val_acc:
                        self.best_val_acc = val_f1
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                    print(f"Validation - Loss: {val_results['loss']:.4f} - Acc: {val_results['accuracy']:.4f} - F1: {val_f1:.4f}")
                    val_results['best_val_f1'] = self.best_val_acc
                    val_results['epochs_without_improvement'] = self.epochs_without_improvement
                    log_epoch_training_metrics(epoch=epoch, avg_train_loss=avg_train_loss, train_acc=train_acc, total_batches=len(train_dataloader), accelerator=self.accelerator, use_wandb=use_wandb, current_step=current_step)
                    log_validation_results(val_results=val_results, current_step=current_step, split_name="validation", accelerator=self.accelerator, use_wandb=use_wandb)
            else:
                log_epoch_training_metrics(epoch=epoch, avg_train_loss=avg_train_loss, train_acc=train_acc, total_batches=len(train_dataloader), accelerator=self.accelerator, use_wandb=use_wandb, current_step=current_step)

            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                self.save_checkpoint_unified(
                    accelerator=self.accelerator, model=self.model, epoch=epoch,
                    batch_idx=base_offset + batch_idx, len_train_dataloader=len(train_dataloader),
                    training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                    base_ckpt_dir=self.checkpoint_dir,
                )

            if validate_every_n_steps is not None:
                if self.steps_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} steps without improvement")
                    break
            else:
                if self.epochs_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    break

    # -------------------------------------------------------------------------
    # Test (eval only — no optimizer needed)
    # -------------------------------------------------------------------------

    def test(self):
        print("\n" + "="*50)
        print("STARTING TESTING PHASE")
        print("="*50)
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_dataloader = self.get_dataloader(self.test_data_files, self.test_batch_size, num_workers=self.num_workers, shuffle=False)

        # No optimizer registered — accelerator.load_state() won't attempt to restore
        # optimizer state, avoiding mismatches with checkpoints from different training stages.
        self.model, train_dataloader, test_dataloader = self.accelerator.prepare(
            self.model, train_dataloader, test_dataloader
        )

        _, _, _, _, _ = self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
            inference_only=True,
        )

        test_results = self.validate(test_dataloader, "test", current_step=1)

        if self.accelerator.is_main_process and test_results is not None:
            print(f"\nOverall TEST RESULTS:")
            print(f"Test Loss: {test_results['loss']:.4f}")
            print(f"Test Micro Accuracy: {test_results['accuracy']:.4f}")
            print(f"Test Micro F1: {test_results['f1']:.4f}")

            use_wandb = self.global_config.get('USE_WANDB', False)
            if use_wandb:
                log_metrics("test", test_results, step=1)

        return test_results
