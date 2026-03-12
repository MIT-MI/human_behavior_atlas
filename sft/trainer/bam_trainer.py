"""
BAM (Residual Hidden Adapter) trainer.

Trains a single BAM model for a single dataset at a time.
  --task_type cls  →  BAMCLS model, CLS loss
  --task_type qa   →  BAMQA  model, LM (teacher-forcing) loss

Adapters (video_adapter, audio_adapter) are assigned as nn.Module submodules
on the model wrapper BEFORE accelerator.prepare(), so DDP/FSDP wraps model +
adapters as one unit and gradient sync is handled automatically.

Model forwards receive raw side-channel features (video_feats / audio_feats);
the model applies its adapters internally.

Inherits shared infrastructure from BaseMultiHeadTrainer.
"""
import os
import json
import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from transformers import get_scheduler

from trainer.base_trainer import BaseMultiHeadTrainer
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
from models.bam_utils import build_video_feats_batch, build_audio_feats_batch, maybe_build_hidden_adapters


class BAMTrainer(BaseMultiHeadTrainer):
    """
    Overrides:
      - _extra_wandb_config()
      - save_checkpoint_unified() — adds model_order to meta
      - load_checkpoint_unified() — adds bam_fresh_start support
      - validate()               — CLS evaluation (all rows)
      - train()                  — dispatches to _train_cls or _train_qa
      - test()                   — registers per-module opts for ckpt restore
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = self.global_config

        self.task_type = cfg.get("TASK_TYPE", "cls")   # "cls" or "qa"

        self.use_bam_video = cfg.get("USE_BAM_VIDEO", False)
        self.use_bam_audio = cfg.get("USE_BAM_AUDIO", False)

        # Stage: "bam_only" | "bam_and_classifier_heads_only" | "bam_and_full_model"
        self.bam_stage = cfg.get("BAM_STAGE", "bam_only")
        self.bam_fresh_start = bool(cfg.get("BAM_FRESH_START", False))

        self.bam_hidden = cfg.get("BAM_HIDDEN", 128)
        self.bam_hidden_video = int(cfg.get("BAM_HIDDEN_VIDEO", self.bam_hidden))
        self.bam_hidden_audio = int(cfg.get("BAM_HIDDEN_AUDIO", self.bam_hidden))

        self.bam_pv = cfg.get("BAM_P_MODDROP_VIDEO", 0.30)
        self.bam_pa = cfg.get("BAM_P_MODDROP_AUDIO", 0.30)

        self.video_temporal = cfg.get("BAM_VIDEO_TEMPORAL", "meanstd")
        self.video_norm     = cfg.get("BAM_VIDEO_NORM", None)
        self.audio_temporal = cfg.get("BAM_AUDIO_TEMPORAL", "none")
        self.audio_norm     = cfg.get("BAM_AUDIO_NORM", "l2")

        self.d_video_feat = cfg.get("D_VIDEO_FEAT", None)
        self.d_audio_feat = cfg.get("D_AUDIO_FEAT", None)

        self.qa_loss_weight = float(cfg.get("QA_LOSS_WEIGHT", 1.0))

        # Training config — read once here, used across train/validate loops
        self.validate_every_n_epochs = cfg.get("VALIDATE_EVERY_N_EPOCHS", None)
        self.validate_every_n_steps  = cfg.get("VALIDATE_EVERY_N_STEPS",  None)
        self.save_every_n_epochs     = cfg.get("SAVE_EVERY_N_EPOCHS",     None)
        self.save_every_n_steps      = cfg.get("SAVE_EVERY_N_STEPS",      None)
        self.early_stopping_patience = cfg.get("EARLY_STOPPING_PATIENCE", 0)
        self.use_wandb               = bool(cfg.get("USE_WANDB",           False))
        self.num_classes             = int(cfg.get("NUM_CLASSES",          0))

        self.video_adapter = None
        self.audio_adapter = None

    # -------------------------------------------------------------------------
    # Wandb
    # -------------------------------------------------------------------------

    def _extra_wandb_config(self):
        cfg = self.global_config
        return {
            "task_type":                   self.task_type,
            "bam_use_video":               bool(cfg.get("USE_BAM_VIDEO", False)),
            "bam_use_audio":               bool(cfg.get("USE_BAM_AUDIO", False)),
            "bam_stage":                   cfg.get("BAM_STAGE", "bam_only"),
            "bam_fresh_start":             bool(cfg.get("BAM_FRESH_START", False)),
            "bam_d_video_feat":            cfg.get("D_VIDEO_FEAT", None),
            "bam_d_audio_feat":            cfg.get("D_AUDIO_FEAT", None),
            "bam_video_temporal":          cfg.get("BAM_VIDEO_TEMPORAL", "meanstd"),
            "bam_video_norm":              cfg.get("BAM_VIDEO_NORM", None),
            "bam_audio_temporal":          cfg.get("BAM_AUDIO_TEMPORAL", "none"),
            "bam_audio_norm":              cfg.get("BAM_AUDIO_NORM", "l2"),
            "bam_hidden_global":           cfg.get("BAM_HIDDEN", 128),
            "bam_hidden_video":            int(cfg.get("BAM_HIDDEN_VIDEO", self.bam_hidden)),
            "bam_hidden_audio":            int(cfg.get("BAM_HIDDEN_AUDIO", self.bam_hidden)),
            "bam_p_moddrop_video":         cfg.get("BAM_P_MODDROP_VIDEO", 0.30),
            "bam_p_moddrop_audio":         cfg.get("BAM_P_MODDROP_AUDIO", 0.30),
            "bam_video_use_ln":            bool(cfg.get("BAM_VIDEO_USE_LN", False)),
            "bam_video_alpha_init":        float(cfg.get("BAM_VIDEO_ALPHA_INIT", 1.0)),
            "bam_audio_use_ln":            bool(cfg.get("BAM_AUDIO_USE_LN", False)),
            "bam_audio_alpha_init":        float(cfg.get("BAM_AUDIO_ALPHA_INIT", 1.0)),
            "base_lr":                     float(cfg.get("BASE_LR", self.lr * 0.25)),
            "bam_lr":                      float(cfg.get("BAM_LR", self.lr * 5.0)),
            "qa_loss_weight":              self.qa_loss_weight,
        }

    # -------------------------------------------------------------------------
    # Checkpoint — overrides to include model_order
    # -------------------------------------------------------------------------

    def _current_model_order(self):
        order = ["base"]
        if self.bam_stage in {"bam_only", "bam_and_classifier_heads_only", "bam_and_full_model"}:
            if self.video_adapter is not None:
                order.append("video")
            if self.audio_adapter is not None:
                order.append("audio")
        return order

    def save_checkpoint_unified(self, accelerator, model, epoch, batch_idx,
                                len_train_dataloader, training_strategy, base_ckpt_dir):
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
                "model_order": self._current_model_order(),
                "saved_at_unix": time.time(),
            }
            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        accelerator.print(f"[save] checkpoint @ step {global_step} → {ckpt_dir}")
        return ckpt_dir

    def load_checkpoint_unified(self, accelerator, model, base_ckpt_dir,
                                explicit_dir=None, expect_training_strategy=None,
                                inference_only=False):
        """Extended: supports inference_only for loading base-model weights without optimizer state."""
        from math import floor

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
            accelerator.print(
                f"[warn] strategy mismatch: expected {expect_training_strategy}, "
                f"got {meta.get('training_strategy')}"
            )

        if inference_only:
            from accelerate.utils.fsdp_utils import load_fsdp_model
            load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, ckpt_dir, model_index=0)
        else:
            accelerator.load_state(ckpt_dir)

        global_step = int(meta["global_step"])
        len_dl = int(meta["len_train_dataloader"])
        if len_dl <= 0:
            accelerator.print("[load] invalid len_train_dataloader; starting at epoch 0.")
            return 0, 0, 0, meta, ckpt_dir

        start_epoch        = floor((global_step - 1) / len_dl)
        start_batch_offset = (global_step - 1) % len_dl

        accelerator.print(
            f"[load] resumed {ckpt_dir} → epoch={start_epoch}, "
            f"step={global_step}, offset={start_batch_offset}"
        )
        return start_epoch, start_batch_offset, global_step, meta, ckpt_dir

    # -------------------------------------------------------------------------
    # Adapter build + preparation
    # -------------------------------------------------------------------------

    def _build_adapters(self, attach_to_model=True):
        """Build adapters and optionally register them as submodules on the model wrapper.

        attach_to_model=False is used in the bam_fresh_start path: the base model is
        already FSDP-wrapped before adapters are built, so we must NOT register them on
        self.model (the FSDP wrapper). _prepare_fresh_adapters_separately() attaches them
        to the inner model instead.
        """
        H = getattr(self.model, "hidden_size", None)
        if H is None:
            raise RuntimeError("Model must expose .hidden_size for BAM out_dim")
        cfg = self.global_config
        self.video_adapter, self.audio_adapter = maybe_build_hidden_adapters(
            use_bam_video=self.use_bam_video,
            use_bam_audio=self.use_bam_audio,
            bam_hidden_video=self.bam_hidden_video,
            bam_hidden_audio=self.bam_hidden_audio,
            p_moddrop_video=self.bam_pv,
            p_moddrop_audio=self.bam_pa,
            out_dim_hidden=H,
            d_video_feat=self.d_video_feat,
            d_audio_feat=self.d_audio_feat,
            video_use_ln=bool(cfg.get("BAM_VIDEO_USE_LN", False)),
            video_alpha_init=float(cfg.get("BAM_VIDEO_ALPHA_INIT", 1.0)),
            audio_use_ln=bool(cfg.get("BAM_AUDIO_USE_LN", False)),
            audio_alpha_init=float(cfg.get("BAM_AUDIO_ALPHA_INIT", 1.0)),
        )
        if attach_to_model:
            # Register as submodules so accelerator.prepare(model) wraps them together.
            # nn.Module.__setattr__ auto-registers nn.Module values as submodules.
            self.model.video_adapter = self.video_adapter
            self.model.audio_adapter = self.audio_adapter

    def _prepare_modules(self, train_dl, val_dl):
        """Prepare model (with adapters as submodules) + dataloaders with accelerator."""
        model, train_dl, val_dl = self.accelerator.prepare(self.model, train_dl, val_dl)
        self.model = model
        return train_dl, val_dl

    def _prepare_fresh_adapters_separately(self):
        """
        Diff-stage resume: build fresh adapters, prepare them as separate DDP modules,
        then assign to inner model so the model's forward can call them.
        """
        adapters = [a for a in (self.video_adapter, self.audio_adapter) if a is not None]
        if not adapters:
            return
        prepared = self.accelerator.prepare(*adapters)
        if not isinstance(prepared, (list, tuple)):
            prepared = [prepared]
        i = 0
        if self.video_adapter is not None:
            self.video_adapter = prepared[i]; i += 1
        if self.audio_adapter is not None and i < len(prepared):
            self.audio_adapter = prepared[i]

        # Assign prepared (DDP-wrapped) adapters to inner model so forward can call them.
        inner = getattr(self.model, "module", self.model)
        inner.video_adapter = self.video_adapter
        inner.audio_adapter = self.audio_adapter

    # -------------------------------------------------------------------------
    # Parameter groups + optimizers
    # -------------------------------------------------------------------------

    def prepare_params_for_training(self, base_lr=None, bam_lr=None):
        base_lr = self.lr if base_lr is None else base_lr
        bam_lr  = self.lr if bam_lr  is None else bam_lr

        def _set(module, flag):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = flag

        bundles = {"base": None, "video": None, "audio": None}

        if self.bam_stage == "bam_only":
            _set(self.model, False)
            _set(self.video_adapter, True)
            _set(self.audio_adapter, True)
            base_params = list(self.model.parameters())
            if base_params:
                bundles["base"] = {"params": base_params, "lr": 0.0, "weight_decay": 0.0}

        elif self.bam_stage == "bam_and_classifier_heads_only":
            _set(self.model, False)
            if hasattr(self.model, "heads"):
                _set(self.model.heads, True)
            _set(self.video_adapter, True)
            _set(self.audio_adapter, True)
            hp = [p for p in self.model.parameters() if p.requires_grad]
            if hp:
                bundles["base"] = {"params": hp, "lr": base_lr}

        elif self.bam_stage == "bam_and_full_model":
            _set(self.video_adapter, True)
            _set(self.audio_adapter, True)
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                bundles["base"] = {"params": params, "lr": base_lr}

        else:
            raise ValueError(f"Unknown BAM stage: {self.bam_stage}")

        if self.video_adapter is not None:
            vp = [p for p in self.video_adapter.parameters() if p.requires_grad]
            if vp:
                bundles["video"] = {"params": vp, "lr": bam_lr}
        if self.audio_adapter is not None:
            ap = [p for p in self.audio_adapter.parameters() if p.requires_grad]
            if ap:
                bundles["audio"] = {"params": ap, "lr": bam_lr}

        return bundles

    def _build_per_module_optimizers(self, bundles):
        opt_base  = Adam(bundles["base"]["params"],  lr=bundles["base"]["lr"])  if bundles["base"]  else None
        opt_video = Adam(bundles["video"]["params"], lr=bundles["video"]["lr"]) if bundles["video"] else None
        opt_audio = Adam(bundles["audio"]["params"], lr=bundles["audio"]["lr"]) if bundles["audio"] else None
        opts  = [o for o in (opt_base, opt_video, opt_audio) if o is not None]
        names = [n for n, o in zip(["base", "video", "audio"],
                                    (opt_base, opt_video, opt_audio)) if o is not None]
        return opts, names

    def _build_schedulers(self, opts, total_updates):
        if not self.use_scheduler:
            return []
        return [
            get_scheduler(self.scheduler_type, o,
                          num_warmup_steps=self.warmup_steps,
                          num_training_steps=total_updates)
            for o in opts
        ]

    def _prepare_opts_scheds(self, opts, scheds):
        if scheds:
            prepared = self.accelerator.prepare(*opts, *scheds)
            n = len(opts)
            self.prepared_opts  = list(prepared[:n])
            self.prepared_scheds = list(prepared[n:])
            for s in self.prepared_scheds:
                self.accelerator.register_for_checkpointing(s)
        else:
            self.prepared_opts   = list(self.accelerator.prepare(*opts))
            self.prepared_scheds = []

    def _step_all_opts(self):
        for opt in self.prepared_opts:
            opt.step()

    def _zero_all_opts(self):
        for opt in self.prepared_opts:
            opt.zero_grad(set_to_none=True)

    def _step_all_scheds(self):
        for sch in self.prepared_scheds:
            sch.step()

    def _current_lr(self):
        return self.prepared_opts[0].param_groups[0]["lr"] if self.prepared_opts else self.lr

    # -------------------------------------------------------------------------
    # Feature extraction helpers (called per batch)
    # -------------------------------------------------------------------------

    def _pool_feats(self, batch, device):
        """Extract and pool raw OpenPose / OpenSmile side features for the batch."""
        cfg = self.global_config

        pooled_video = None
        if self.use_bam_video and "video_feats" in batch and batch["video_feats"] is not None:
            pooled_video = build_video_feats_batch(
                batch["video_feats"], device=device,
                temporal_mode=cfg.get("BAM_VIDEO_TEMPORAL", "meanstd"),
                use_conf=cfg.get("BAM_VIDEO_USE_CONF", True),
                norm=self.video_norm,
                target_dim=self.d_video_feat,
            )

        pooled_audio = None
        if self.use_bam_audio and "audio_feats" in batch and batch["audio_feats"] is not None:
            pooled_audio = build_audio_feats_batch(
                batch["audio_feats"], device=device,
                temporal_mode=self.audio_temporal,
                norm=self.audio_norm,
                target_dim=self.d_audio_feat,
            )

        return pooled_video, pooled_audio

    # -------------------------------------------------------------------------
    # Teacher-forcing input builder (QA only)
    # -------------------------------------------------------------------------

    def _build_tf_inputs_and_labels(self, batch, seq_len, device):
        """
        Build teacher-forced inputs for all rows in a QA batch.

        Returns:
            qa_input_ids : [B, T]  prompt + answer(+EOS), padded/truncated
            qa_attn      : [B, T]  1 on real tokens, 0 on pads
            lm_labels    : [B, T]  -100 on prompt/pads; answer tokens elsewhere
        """
        ids_all  = batch["input_ids"]
        attn_all = batch.get("attention_mask", None)

        B     = ids_all.size(0)
        T     = seq_len
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        qa_input_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        qa_attn      = torch.zeros((B, T), dtype=torch.long, device=device)
        lm_labels    = torch.full((B, T), -100,   dtype=torch.long, device=device)

        for j in range(B):
            prompt_ids = ids_all[j]
            if attn_all is not None:
                prompt_len = int(attn_all[j].sum().item())
            else:
                prompt_len = (prompt_ids != pad_id).sum().item()
            prompt_len = min(prompt_len, T)

            qa_input_ids[j, :prompt_len] = prompt_ids[:prompt_len]
            qa_attn[j, :prompt_len] = 1

            ans = batch["lm_labels"][j]
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
                lm_labels[j, prompt_len:prompt_len + len(ans_tok)] = \
                    qa_input_ids[j, prompt_len:prompt_len + len(ans_tok)]

        return qa_input_ids, qa_attn, lm_labels

    # -------------------------------------------------------------------------
    # Validation (CLS)
    # -------------------------------------------------------------------------

    def validate(self, val_dataloader, split_name="validation", current_step=None):
        self.model.eval()

        from evaluate.detailed_multi_task_evaluation import evaluate_predictions

        total_loss      = 0.0
        all_predictions = []
        all_labels      = []
        all_datasets    = []
        criterion = CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", total=len(val_dataloader),
                              disable=not self.accelerator.is_main_process):
                if "input_ids" not in batch or "labels" not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")
                if "dataset" not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")

                input_ids      = batch["input_ids"]
                labels         = batch["labels"]
                attention_mask = batch.get("attention_mask", None)
                device         = input_ids.device

                domain_ids = self._datasets_to_domain_ids(batch["dataset"], device=device)

                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == self.num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                video_feats, audio_feats = self._pool_feats(batch, device)

                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    domain_ids=domain_ids,
                    video_feats=video_feats,
                    audio_feats=audio_feats,
                    train_mode=False,
                )
                logits = out["cls_logits"] if isinstance(out, dict) else out[0]

                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)

                gathered_preds    = self.accelerator.gather_for_metrics(preds)
                gathered_labels   = self.accelerator.gather_for_metrics(labels)
                gathered_datasets = self.accelerator.gather_for_metrics(batch["dataset"])

                if self.accelerator.is_main_process:
                    all_predictions.extend(gathered_preds.cpu().numpy())
                    all_labels.extend(gathered_labels.cpu().numpy())
                    all_datasets.extend(gathered_datasets)

        avg_loss = total_loss / max(1, len(all_labels)) if self.accelerator.is_main_process else 0.0

        if self.accelerator.is_main_process:
            evaluation_results = evaluate_predictions(
                predictions=all_predictions,
                ground_truths=all_labels,
                datasets=all_datasets or None,
                split_name=split_name,
                save_path=self.validation_result_dir,
                global_steps=current_step,
                label_map_path=self.label_map_path,
            )
            agg       = evaluation_results["aggregate_metrics"]
            accuracy  = agg.get("micro_accuracy",  0.0)
            f1        = agg.get("micro_f1",        0.0)
            precision = agg.get("micro_precision", 0.0)
            recall    = agg.get("micro_recall",    0.0)

            print(f"{split_name.capitalize()} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
            print(f"  Macro F1: {agg.get('macro_f1', 0.0):.4f} - Weighted F1: {agg.get('weighted_f1', 0.0):.4f}")

            return {
                "loss": avg_loss, "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1, "predictions": all_predictions,
                "labels": all_labels, "evaluation_results": evaluation_results,
                "aggregate_metrics": agg,
            }
        return None

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def _setup_training(self, train_dl, val_dl):
        """Shared setup: build adapters, prepare, load checkpoint, build optimizers."""
        # NOTE: _build_adapters() is called inside the branches below (not here) so that
        # the bam_fresh_start path can FSDP-wrap the base model before adapters are built.

        total_updates = self.epochs * len(train_dl)
        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        bam_lr  = self.global_config.get("BAM_LR",  self.lr * 5.0)
        self.prepared_opts, self.prepared_scheds = [], []

        if self.bam_fresh_start:
            # 1. FSDP-wrap the base model WITHOUT adapters registered.
            #    This ensures the strict checkpoint load succeeds (no adapter keys in model).
            train_dl, val_dl = self._prepare_modules(train_dl, val_dl)
            # 2. Load base checkpoint — strict load works because no adapter keys in model.
            self.load_checkpoint_unified(
                accelerator=self.accelerator, model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                inference_only=True,
            )
            # 3. Build fresh adapter objects but do NOT register on the FSDP-wrapped model.
            self._build_adapters(attach_to_model=False)
            # 4. DDP-prepare adapters and attach them to the inner model.
            self._prepare_fresh_adapters_separately()
            start_epoch, start_batch_offset = 0, 0
        else:
            # Normal resume: attach adapters before FSDP so they are one wrapped unit.
            self._build_adapters()
            train_dl, val_dl = self._prepare_modules(train_dl, val_dl)
            start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
                accelerator=self.accelerator, model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
            )

        bundles = self.prepare_params_for_training(base_lr=base_lr, bam_lr=bam_lr)
        opts, _ = self._build_per_module_optimizers(bundles)
        scheds  = self._build_schedulers(opts, total_updates)
        self._prepare_opts_scheds(opts, scheds)

        return train_dl, val_dl, start_epoch, start_batch_offset

    def train(self):
        train_dataloader = self.get_dataloader(
            self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        val_dataloader = self.get_dataloader(
            self.val_data_files, self.val_batch_size, num_workers=self.num_workers, shuffle=False
        )

        if self.task_type == "qa":
            self._train_qa(train_dataloader, val_dataloader)
        else:
            self._train_cls(train_dataloader, val_dataloader)

    # -------------------------------------------------------------------------
    # CLS training loop
    # -------------------------------------------------------------------------

    def _train_cls(self, train_dataloader, val_dataloader):

        # setting up the training, which involves preparing the adapters, models etc.
        train_dataloader, val_dataloader, start_epoch, start_batch_offset = \
            self._setup_training(train_dataloader, val_dataloader)

        criterion = CrossEntropyLoss()

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0,
                          disable=not self.accelerator.is_main_process):
            self.model.train()

            if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            total_loss   = 0.0
            correct      = 0
            total        = 0
            epoch_start_time = time.time()
            eff_loss     = 0.0
            eff_correct  = 0
            eff_total    = 0

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training",
                                         total=len(train_dataloader),
                                         disable=not self.accelerator.is_main_process):
                self.model.train()

                if epoch == start_epoch and batch_idx < start_batch_offset:
                    continue

                current_step = epoch * len(train_dataloader) + batch_idx + 1

                if "input_ids" not in batch or "labels" not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")
                if "dataset" not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")

                input_ids      = batch["input_ids"]
                labels         = batch["labels"]
                attention_mask = batch.get("attention_mask", None)
                device         = input_ids.device
                B              = input_ids.size(0)

                domain_ids = self._datasets_to_domain_ids(batch["dataset"], device=device)

                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == self.num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                video_feats, audio_feats = self._pool_feats(batch, device)

                with self.accelerator.accumulate(self.model):
                    logits, _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        domain_ids=domain_ids,
                        video_feats=video_feats,
                        audio_feats=audio_feats,
                        train_mode=True,
                    )
                    loss = criterion(logits, labels)

                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss encountered")

                    self.accelerator.backward(loss)
                    self._step_all_opts()
                    self._step_all_scheds()
                    self._zero_all_opts()

                current_lr = self._current_lr()

                with torch.no_grad():
                    eff_loss   += loss.item() * B
                    total_loss += loss.item() * B

                    preds    = logits.argmax(dim=1)
                    g_preds  = self.accelerator.gather_for_metrics(preds)
                    g_labels = self.accelerator.gather_for_metrics(labels)
                    if self.accelerator.is_main_process:
                        n_correct    = (g_preds == g_labels).sum().item()
                        eff_correct += n_correct
                        eff_total   += g_labels.size(0)
                        correct     += n_correct
                        total       += g_labels.size(0)

                    log_batch_training_metrics(
                        epoch=epoch, batch_idx=batch_idx, total_batches=len(train_dataloader),
                        loss=eff_loss, correct=eff_correct, total=max(1, eff_total),
                        epoch_start_time=epoch_start_time, start_time=self.start_time,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        batch_size=self.batch_size, epochs=self.epochs,
                        accelerator=self.accelerator, use_wandb=self.use_wandb,
                        current_lr=current_lr, current_step=current_step,
                    )

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        eff_loss    = 0.0
                        eff_correct = 0
                        eff_total   = 0

                    if self.validate_every_n_steps and current_step % self.validate_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Running step-based validation...")
                        val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                        if self.accelerator.is_main_process and val_results is not None:
                            val_f1 = val_results["f1"]
                            if val_f1 > self.best_val_acc:
                                self.best_val_acc              = val_f1
                                self.steps_without_improvement = 0
                            else:
                                self.steps_without_improvement += 1
                            val_results["best_val_f1"]               = self.best_val_acc
                            val_results["steps_without_improvement"]  = self.steps_without_improvement
                            log_validation_results(val_results=val_results, current_step=current_step,
                                                   split_name="validation", accelerator=self.accelerator,
                                                   use_wandb=self.use_wandb)

                    if self.save_every_n_steps and current_step % self.save_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Saving checkpoint...")
                        self.save_checkpoint_unified(
                            accelerator=self.accelerator, model=self.model, epoch=epoch,
                            batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )

            avg_train_loss = total_loss / max(1, total)
            train_acc      = correct / max(1, total)

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")

            if self.validate_every_n_epochs and (epoch + 1) % self.validate_every_n_epochs == 0:
                val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                if self.accelerator.is_main_process and val_results is not None:
                    val_f1 = val_results["f1"]
                    if val_f1 > self.best_val_acc:
                        self.best_val_acc               = val_f1
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                    val_results["best_val_f1"]                = self.best_val_acc
                    val_results["epochs_without_improvement"]  = self.epochs_without_improvement
                    log_epoch_training_metrics(
                        epoch=epoch, avg_train_loss=avg_train_loss, train_acc=train_acc,
                        total_batches=len(train_dataloader), accelerator=self.accelerator,
                        use_wandb=self.use_wandb, current_step=current_step,
                    )
                    log_validation_results(val_results=val_results, current_step=current_step,
                                           split_name="validation", accelerator=self.accelerator,
                                           use_wandb=self.use_wandb)
            else:
                log_epoch_training_metrics(
                    epoch=epoch, avg_train_loss=avg_train_loss, train_acc=train_acc,
                    total_batches=len(train_dataloader), accelerator=self.accelerator,
                    use_wandb=self.use_wandb, current_step=current_step,
                )

            if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint_unified(
                    accelerator=self.accelerator, model=self.model, epoch=epoch,
                    batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                    training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                    base_ckpt_dir=self.checkpoint_dir,
                )

            if self.validate_every_n_steps is not None:
                if self.steps_without_improvement >= self.early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {self.early_stopping_patience} steps")
                    break
            else:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {self.early_stopping_patience} epochs")
                    break

    # -------------------------------------------------------------------------
    # QA training loop
    # -------------------------------------------------------------------------

    def _train_qa(self, train_dataloader, val_dataloader):
        train_dataloader, val_dataloader, start_epoch, start_batch_offset = \
            self._setup_training(train_dataloader, val_dataloader)

        save_every_n_epochs = self.global_config.get("SAVE_EVERY_N_EPOCHS", None)
        save_every_n_steps  = self.global_config.get("SAVE_EVERY_N_STEPS",  None)
        use_wandb           = self.global_config.get("USE_WANDB",           False)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0,
                          disable=not self.accelerator.is_main_process):
            self.model.train()

            if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            total_loss       = 0.0
            total_samples    = 0
            epoch_start_time = time.time()

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training (QA)",
                                         total=len(train_dataloader),
                                         disable=not self.accelerator.is_main_process):
                self.model.train()

                if epoch == start_epoch and batch_idx < start_batch_offset:
                    continue

                current_step = epoch * len(train_dataloader) + batch_idx + 1

                if "input_ids" not in batch or "lm_labels" not in batch:
                    raise KeyError(f"QA batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch["input_ids"]
                device    = input_ids.device
                B, T      = input_ids.shape

                # domain_ids = -1 for all QA rows (no classification heads used)
                domain_ids = torch.full((B,), -1, dtype=torch.long, device=device)

                # Build teacher-forced inputs
                tf_input_ids, tf_attn, lm_labels = self._build_tf_inputs_and_labels(
                    batch=batch, seq_len=T, device=device
                )

                video_feats, audio_feats = self._pool_feats(batch, device)

                with self.accelerator.accumulate(self.model):
                    out = self.model(
                        input_ids=tf_input_ids,
                        attention_mask=tf_attn,
                        domain_ids=domain_ids,
                        lm_labels=lm_labels,
                        video_feats=video_feats,
                        audio_feats=audio_feats,
                        train_mode=True,
                    )
                    lm_loss = out["lm_loss"]

                    if lm_loss is None or not torch.isfinite(lm_loss):
                        raise FloatingPointError("Non-finite or missing lm_loss")

                    loss = self.qa_loss_weight * lm_loss

                    self.accelerator.backward(loss)
                    self._step_all_opts()
                    self._step_all_scheds()
                    self._zero_all_opts()

                current_lr = self._current_lr()

                with torch.no_grad():
                    total_loss    += loss.item() * B
                    total_samples += B

                    log_batch_training_metrics(
                        epoch=epoch, batch_idx=batch_idx, total_batches=len(train_dataloader),
                        loss=total_loss, correct=0, total=max(1, total_samples),
                        epoch_start_time=epoch_start_time, start_time=self.start_time,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        batch_size=self.batch_size, epochs=self.epochs,
                        accelerator=self.accelerator, use_wandb=self.use_wandb,
                        current_lr=current_lr, current_step=current_step,
                    )

                    if self.save_every_n_steps and current_step % self.save_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Saving checkpoint...")
                        self.save_checkpoint_unified(
                            accelerator=self.accelerator, model=self.model, epoch=epoch,
                            batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )

            avg_train_loss = total_loss / max(1, total_samples)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - QA Train Loss: {avg_train_loss:.4f}")

            log_epoch_training_metrics(
                epoch=epoch, avg_train_loss=avg_train_loss, train_acc=0.0,
                total_batches=len(train_dataloader), accelerator=self.accelerator,
                use_wandb=self.use_wandb, current_step=current_step,
            )

            if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint_unified(
                    accelerator=self.accelerator, model=self.model, epoch=epoch,
                    batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                    training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                    base_ckpt_dir=self.checkpoint_dir,
                )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    def test(self):
        print("\n" + "=" * 50)
        print("STARTING TESTING PHASE")
        print("=" * 50)

        train_dataloader = self.get_dataloader(
            self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        test_dataloader = self.get_dataloader(
            self.test_data_files, self.test_batch_size, num_workers=self.num_workers, shuffle=False
        )

        # BAM checkpoints save base model and adapters as separate files
        # (pytorch_model_fsdp.bin = index 0, pytorch_model_fsdp_1.bin = index 1+).
        # Mirror the training structure so accelerate maps files correctly.

        # 1. FSDP-wrap base model WITHOUT adapters.
        train_dataloader, test_dataloader = self._prepare_modules(train_dataloader, test_dataloader)

        # 2. Build adapter objects and DDP-prepare them separately (model index 1+).
        self._build_adapters(attach_to_model=False)
        self._prepare_fresh_adapters_separately()

        # 3. Prepare optimizers/schedulers (needed for accelerator.load_state() file count).
        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        bam_lr  = self.global_config.get("BAM_LR",  self.lr * 5.0)
        total_updates = max(1, self.epochs * len(train_dataloader))
        bundles = self.prepare_params_for_training(base_lr=base_lr, bam_lr=bam_lr)
        opts, _ = self._build_per_module_optimizers(bundles)
        scheds  = self._build_schedulers(opts, total_updates)
        self._prepare_opts_scheds(opts, scheds)

        # 4. Load BAM checkpoint.
        self.load_checkpoint_unified(
            accelerator=self.accelerator, model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
        )

        test_results = self.validate(test_dataloader, "test", current_step=1)

        if self.accelerator.is_main_process and test_results is not None:
            print(f"\nOverall TEST RESULTS:")
            print(f"Test Loss:           {test_results['loss']:.4f}")
            print(f"Test Micro Accuracy: {test_results['accuracy']:.4f}")
            print(f"Test Micro F1:       {test_results['f1']:.4f}")

            use_wandb = self.global_config.get("USE_WANDB", False)
            log_validation_results(val_results=test_results, current_step=1, split_name="test",
                                   accelerator=self.accelerator, use_wandb=self.use_wandb)

        return test_results
