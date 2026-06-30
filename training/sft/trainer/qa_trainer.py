"""
QA trainer: loads a frozen multi-head classification checkpoint, unfreezes
only lm_head, and trains with teacher-forced language-modeling loss.

Inherits all checkpoint I/O, dataloader building, and accelerate setup from
BaseMultiHeadTrainer. Only the unique QA-specific methods are defined here.
"""
import os
import json
import numpy as np
import torch
from math import floor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from accelerate.utils import DistributedDataParallelKwargs

from trainer.base_trainer import BaseMultiHeadTrainer
from dataset.sft_dataset import OmniClassifierDataset
from dataset.dataset_utils import collate_fn
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics


class QAMultiHeadTrainer(BaseMultiHeadTrainer):
    """
    Trainer for the QA stage: freeze multi-head model, unfreeze lm_head,
    train with teacher-forced LM loss on QA datasets.

    Overrides:
      - _accelerator_kwargs_handlers(): adds find_unused_parameters=True
      - _extra_wandb_config(): adds qa_datasets + qa_loss_weight
      - _datasets_to_domain_ids(): returns -1 sentinel for QA rows
      - get_dataloader(): uses OmniClassifierDataset
      - validate(): greedy decoding + JSON save (no CLS metrics)
      - train(): freeze + lm_head-only optimizer + LM loss
      - test(): inherits base (inference_only=True), then calls qa validate
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # EOS / PAD sanity (needed for generation)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an eos_token_id for QA SFT.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.backbone.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.backbone.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.qa_datasets = set(self.global_config.get('QA_DATASETS', []))
        self.qa_loss_weight = float(self.global_config.get('QA_LOSS_WEIGHT', 1.0))
        self.max_val_qa_tokens = 60

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def _accelerator_kwargs_handlers(self):
        return [DistributedDataParallelKwargs(find_unused_parameters=True)]

    def _extra_wandb_config(self):
        return {
            "qa_datasets": sorted(list(self.qa_datasets)),
            "qa_loss_weight": self.qa_loss_weight,
        }

    # -------------------------------------------------------------------------
    # Domain routing — -1 sentinel for QA rows
    # -------------------------------------------------------------------------

    def _datasets_to_domain_ids(self, dataset_names, device):
        ids = []
        for ds in dataset_names:
            if isinstance(ds, bytes):
                ds = ds.decode("utf-8")
            if ds in self.qa_datasets:
                ids.append(-1)
            elif ds not in self.dataset_to_domain_id:
                raise KeyError(f"Dataset '{ds}' not in label_map.meta.dataset_domain")
            else:
                ids.append(self.dataset_to_domain_id[ds])
        return torch.tensor(ids, dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Dataloader — uses OmniClassifierDataset
    # -------------------------------------------------------------------------

    def get_dataloader(self, data_files, batch_size, num_workers=0, shuffle=True):
        dataset = OmniClassifierDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.label_key,
            label_map=self.label_map,
            qa_datasets=self.qa_datasets,
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
    # QA-specific helpers
    # -------------------------------------------------------------------------

    def _split_qa_cls_indices(self, dataset_names):
        qa_idx, cls_idx = [], []
        for i, ds in enumerate(dataset_names):
            if isinstance(ds, bytes):
                ds = ds.decode("utf-8", errors="ignore")
            ds = str(ds).lower()
            (qa_idx if ds in self.qa_datasets else cls_idx).append(i)
        qa = torch.tensor(qa_idx, dtype=torch.long) if qa_idx else None
        cl = torch.tensor(cls_idx, dtype=torch.long) if cls_idx else None
        return qa, cl

    def _build_tf_inputs_and_labels(self, batch, qa_rows, seq_len, device):
        """
        Build teacher-forced inputs for QA rows.

        Returns:
            qa_input_ids : [Bq, T]  prompt + answer(+EOS), padded/truncated
            qa_attn      : [Bq, T]  1 on real tokens, 0 on pads
            lm_labels_q  : [Bq, T]  -100 on prompt/pads; answer tokens elsewhere
        """
        if qa_rows is None or qa_rows.numel() == 0:
            return None, None, None

        ids_all = batch["input_ids"]
        attn_all = batch.get("attention_mask", None)

        Bq = qa_rows.numel()
        T = seq_len
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        qa_input_ids = torch.full((Bq, T), pad_id, dtype=torch.long, device=device)
        qa_attn = torch.zeros((Bq, T), dtype=torch.long, device=device)
        lm_labels_q = torch.full((Bq, T), -100, dtype=torch.long, device=device)

        for j, idx in enumerate(qa_rows.tolist()):
            prompt_ids = ids_all[idx]
            if attn_all is not None:
                prompt_len = int(attn_all[idx].sum().item())
            else:
                prompt_len = (prompt_ids != pad_id).sum().item()
            prompt_len = min(prompt_len, T)

            qa_input_ids[j, :prompt_len] = prompt_ids[:prompt_len]
            qa_attn[j, :prompt_len] = 1

            ans = batch["lm_labels"][idx]
            if isinstance(ans, np.generic):
                ans = ans.item()
            if isinstance(ans, bytes):
                ans = ans.decode("utf-8", errors="ignore")
            ans = "" if ans is None else str(ans)

            ans_tok = self.tokenizer.encode(ans, add_special_tokens=False)
            if len(ans_tok) == 0 or ans_tok[-1] != eos_id:
                ans_tok = ans_tok + [eos_id]

            rem = T - prompt_len
            if rem > 0:
                ans_tok = ans_tok[:rem]
                qa_input_ids[j, prompt_len:prompt_len + len(ans_tok)] = torch.tensor(ans_tok, device=device)
                qa_attn[j, prompt_len:prompt_len + len(ans_tok)] = 1
                lm_labels_q[j, prompt_len:prompt_len + len(ans_tok)] = qa_input_ids[j, prompt_len:prompt_len + len(ans_tok)]

        return qa_input_ids, qa_attn, lm_labels_q

    @torch.no_grad()
    def _greedy_decode_no_generate(self, qa_input_ids, qa_attn, max_new_tokens=64):
        """
        Manual greedy decoding using model forward pass (no .generate()).

        Returns:
            cont_ids : [Bq, max_new_tokens] newly generated token ids
        """
        input_ids = qa_input_ids.clone()
        attn = qa_attn.clone() if qa_attn is not None else None

        device = input_ids.device
        Bq = input_ids.size(0)
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        finished = torch.zeros(Bq, dtype=torch.bool, device=device)
        generated = []

        for _ in range(max_new_tokens):
            domain_ids_q = torch.full((Bq,), -1, dtype=torch.long, device=device)
            dummy_lm_labels = torch.full_like(input_ids, -100)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                domain_ids=domain_ids_q,
                lm_labels=dummy_lm_labels,
            )
            next_logits = out["lm_output"].logits[:, -1, :]
            next_tokens = next_logits.argmax(dim=-1)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)
            generated.append(next_tokens.unsqueeze(1))

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            if attn is not None:
                attn = torch.cat([attn, torch.ones((Bq, 1), dtype=attn.dtype, device=device)], dim=1)

            finished = finished | (next_tokens == eos_id)

        return torch.cat(generated, dim=1)  # [Bq, max_new_tokens]

    # -------------------------------------------------------------------------
    # Validation — greedy QA generation + JSON save
    # -------------------------------------------------------------------------

    def validate(self, val_dataloader, split_name="validation", current_step=None):
        self.model.eval()

        all_pred_texts = []
        all_gold_texts = []
        all_qa_datasets = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", total=len(val_dataloader),
                              disable=not self.accelerator.is_main_process):
                if 'input_ids' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                lm_labels = batch['lm_labels']
                datasets = batch["dataset"]

                cont_ids = self._greedy_decode_no_generate(
                    input_ids, attention_mask, max_new_tokens=self.max_val_qa_tokens
                )

                g_cont_ids = self.accelerator.gather_for_metrics(cont_ids)
                gathered_lm_labels = self.accelerator.gather_for_metrics(lm_labels)
                gathered_datasets = self.accelerator.gather_for_metrics(datasets)

                if self.accelerator.is_main_process:
                    pred_texts = self.tokenizer.batch_decode(g_cont_ids, skip_special_tokens=True)
                    all_pred_texts.extend(pred_texts)
                    all_gold_texts.extend(gathered_lm_labels)
                    all_qa_datasets.extend(gathered_datasets)
                

        if self.accelerator.is_main_process:
            out_dir = self.validation_result_dir or "."
            os.makedirs(out_dir, exist_ok=True)
            step_tag = str(current_step) if current_step is not None else "final"
            out_path = os.path.join(out_dir, f"{split_name}_qa_preds_step_{step_tag}.json")

            qa_records = [
                {"dataset": d, "pred": p, "gold": g}
                for d, p, g in zip(all_qa_datasets, all_pred_texts, all_gold_texts)
            ]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(qa_records, f, ensure_ascii=False, indent=2)
            print(f"[QA] Saved {len(qa_records)} records to: {out_path}")

            return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "predictions": [], "labels": [], "evaluation_results": {}, "aggregate_metrics": {}}

        return None

    # -------------------------------------------------------------------------
    # Training — freeze all, unfreeze lm_head, LM loss only
    # -------------------------------------------------------------------------

    def train(self):
        train_dataloader = self.get_dataloader(
            self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        val_dataloader = self.get_dataloader(
            self.val_data_files, self.val_batch_size, num_workers=self.num_workers, shuffle=False
        )

        # Prepare model + dataloaders (no optimizer yet — we build it after freezing)
        train_dataloader, val_dataloader, self.model = self.accelerator.prepare(
            train_dataloader, val_dataloader, self.model
        )

        # Load previous-stage checkpoint (inference_only skips optimizer/scheduler state)
        self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
            inference_only=True,
        )

        # ---- freeze everything ----
        for p in self.model.parameters():
            p.requires_grad = False
        if hasattr(self.model, "heads"):
            for head in self.model.heads:
                for p in head.parameters():
                    p.requires_grad = False

        # ---- unfreeze only lm_head which is essentially the decoder ----
        assert hasattr(self.model.backbone, "lm_head"), "Expected backbone.lm_head"
        for p in self.model.backbone.lm_head.parameters():
            p.requires_grad = True

        # ---- untie lm_head from input embeddings ----
        self.model.backbone.config.tie_word_embeddings = False
        get_out = getattr(self.model.backbone, "get_output_embeddings", None)
        get_in = getattr(self.model.backbone, "get_input_embeddings", None)
        if callable(get_out) and callable(get_in):
            out_emb = get_out()
            in_emb = get_in()
            if out_emb is not None and in_emb is not None:
                out_emb.weight.data = out_emb.weight.detach().clone()
                for p in in_emb.parameters():
                    p.requires_grad = False
        for p in self.model.backbone.lm_head.parameters():
            p.requires_grad = True

        self.accelerator.wait_for_everyone()

        # ---- optimizer only over trainable params ----
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = Adam(trainable_params, lr=self.lr)
        optimizer = self.accelerator.prepare(optimizer)

        total_updates = self.epochs * len(train_dataloader)
        if self.use_scheduler:
            scheduler = get_scheduler(
                self.scheduler_type, optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_updates,
            )
            scheduler = self.accelerator.prepare(scheduler)
            print(f"[INFO] Using {self.scheduler_type} scheduler with {self.warmup_steps} warmup steps")
        else:
            scheduler = None
            print("[INFO] Scheduler disabled - using constant learning rate")

        validate_every_n_epochs = self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None)
        validate_every_n_steps = self.global_config.get('VALIDATE_EVERY_N_STEPS', None)
        save_every_n_epochs = self.global_config.get('SAVE_EVERY_N_EPOCHS', None)
        save_every_n_steps = self.global_config.get('SAVE_EVERY_N_STEPS', None)
        early_stopping_patience = self.global_config.get('EARLY_STOPPING_PATIENCE', 0)
        use_wandb = self.global_config.get('USE_WANDB', False)
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        import time
        for epoch in tqdm(range(self.epochs), desc="Epochs", position=0,
                          disable=not self.accelerator.is_main_process):
            self.model.train()
            total_loss = 0.0
            epoch_start_time = time.time()
            eff_loss = 0.0
            eff_correct = 0
            eff_total = 1  # avoid div-by-zero (no CLS metrics in QA stage)

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training",
                                         total=len(train_dataloader),
                                         disable=not self.accelerator.is_main_process):
                self.model.train()
                current_step = epoch * len(train_dataloader) + batch_idx + 1

                if 'input_ids' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                B, T = input_ids.size()
                device = input_ids.device

                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=device)

                # Build teacher-forced inputs for all rows (all treated as QA)
                qa_rows = torch.arange(B, device=device)
                qa_input_ids, qa_attn, lm_labels_q = self._build_tf_inputs_and_labels(
                    batch=batch, qa_rows=qa_rows, seq_len=T, device=device
                )

                input_ids_full = input_ids.clone()
                attn_full = attention_mask.clone() if attention_mask is not None else None
                input_ids_full.index_copy_(0, qa_rows, qa_input_ids)
                if attn_full is not None:
                    attn_full.index_copy_(0, qa_rows, qa_attn)

                lm_labels_full = torch.full((B, T), -100, dtype=torch.long, device=device)
                lm_labels_full.index_copy_(0, qa_rows, lm_labels_q)

                with self.accelerator.accumulate(self.model):
                    model_output = self.model(
                        input_ids=input_ids_full,
                        attention_mask=attn_full,
                        domain_ids=domain_ids,
                        lm_labels=lm_labels_full,
                    )
                    loss = self.qa_loss_weight * model_output["lm_loss"]
                    self.accelerator.backward(loss)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else self.lr

                with torch.no_grad():
                    total_loss += loss.item() * B
                    eff_loss += loss.item() * B

                    # Print a few decoded samples per step for monitoring
                    lm_output = model_output["lm_output"]
                    pred_ids = lm_output.logits.argmax(dim=-1)
                    active = lm_labels_q != -100
                    pred_ids_masked = torch.where(active, pred_ids, self.tokenizer.pad_token_id)
                    text_labels = lm_labels_q.masked_fill(lm_labels_q == -100, self.tokenizer.pad_token_id)

                    g_pred = self.accelerator.gather_for_metrics(pred_ids_masked)
                    g_gold = self.accelerator.gather_for_metrics(text_labels)
                    if self.accelerator.is_main_process:
                        pred_texts = self.tokenizer.batch_decode(g_pred, skip_special_tokens=True)
                        gold_texts = self.tokenizer.batch_decode(g_gold, skip_special_tokens=True)
                        for i in range(min(2, len(pred_texts))):
                            print(f"[QA pred] {pred_texts[i]}")
                            print(f"[QA gold] {gold_texts[i]}")

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
                        eff_total = 1

                    if validate_every_n_steps and current_step % validate_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Running step-based validation...")
                        self.accelerator.wait_for_everyone()
                        with self.accelerator.join_uneven_inputs(self.model, even_batches=False):
                            val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                        if self.accelerator.is_main_process and val_results is not None:
                            self.steps_without_improvement += 1
                            val_results['best_val_f1'] = self.best_val_acc
                            val_results['steps_without_improvement'] = self.steps_without_improvement
                            log_validation_results(val_results=val_results, current_step=current_step,
                                                   split_name="validation", accelerator=self.accelerator,
                                                   use_wandb=use_wandb)

                    if save_every_n_steps and current_step % save_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Saving checkpoint...")
                        self.save_checkpoint_unified(
                            accelerator=self.accelerator, model=self.model, epoch=epoch,
                            batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )

            avg_train_loss = total_loss / max(1, B * len(train_dataloader))
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}")

            if validate_every_n_epochs and (epoch + 1) % validate_every_n_epochs == 0:
                val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                if self.accelerator.is_main_process and val_results is not None:
                    log_epoch_training_metrics(
                        epoch=epoch, avg_train_loss=avg_train_loss, train_acc=0.0,
                        total_batches=len(train_dataloader), accelerator=self.accelerator,
                        use_wandb=use_wandb, current_step=current_step,
                    )
                    log_validation_results(val_results=val_results, current_step=current_step,
                                           split_name="validation", accelerator=self.accelerator,
                                           use_wandb=use_wandb)
            else:
                log_epoch_training_metrics(
                    epoch=epoch, avg_train_loss=avg_train_loss, train_acc=0.0,
                    total_batches=len(train_dataloader), accelerator=self.accelerator,
                    use_wandb=use_wandb, current_step=current_step,
                )

            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                self.save_checkpoint_unified(
                    accelerator=self.accelerator, model=self.model, epoch=epoch,
                    batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                    training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                    base_ckpt_dir=self.checkpoint_dir,
                )

            if validate_every_n_steps is not None:
                if self.steps_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} steps")
                    break
            else:
                if self.epochs_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} epochs")
                    break
