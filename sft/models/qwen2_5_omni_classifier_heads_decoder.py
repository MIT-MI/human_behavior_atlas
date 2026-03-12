# models/qwen2_5_omni_classifier_heads_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

NEG_INF = -1e9  # safe mask for "irrelevant" classes

def build_domain_specs_from_labelscheme(full_label_scheme: dict):
    """
    Returns:
      domain_name_to_id: {'sentiment_intensity':0, 'emotion':1, 'mental_health':2}
      domain_id_to_global_indices: List[List[int]] e.g. [[0..6], [7..14], [15..20]]
      dataset_to_domain_id: map dataset_name -> domain_id using meta.dataset_domain
      global_num_classes: int (= 21)
    """
    meta = full_label_scheme["meta"]
    global_classes = meta["global_classes"]
    # preserve the ordering in the JSON
    domain_names = list(global_classes.keys())  # ['sentiment_intensity','emotion','mental_health']
    domain_name_to_id = {d:i for i,d in enumerate(domain_names)}
    domain_id_to_global_indices = []
    for d in domain_names:
        # each entry has {'index': int, 'label': str}
        idxs = [x["index"] for x in global_classes[d]]
        domain_id_to_global_indices.append(idxs)

    dataset_to_domain_name = meta["dataset_domain"]  # e.g. 'mosei_senti':'sentiment_intensity'
    dataset_to_domain_id = {ds: domain_name_to_id[dn] for ds, dn in dataset_to_domain_name.items()}

    global_num_classes = full_label_scheme.get("num_classes", max(max(g) for g in domain_id_to_global_indices)+1)

    return domain_name_to_id, domain_id_to_global_indices, dataset_to_domain_id, global_num_classes


class MultiHeadOmniClassifier(nn.Module):
    def __init__(self,
                 full_label_scheme: dict,
                 freeze_backbone="head_only",
                 backbone_name='Qwen/Qwen2.5-Omni-7B',
                 lora_config=None,
                 device_map="auto",
                 torch_dtype=torch.float16,
                 **from_pretrained_kwargs):
        super().__init__()

        # basically the full label scheme; not just the label mapping;
        # this includes the meta data
        self.full_label_scheme = full_label_scheme 

        # obtaining the domain name, ids, datasets and global number of classes
        (self.domain_name_to_id,
         self.domain_id_to_global_indices,
         self.dataset_to_domain_id,
         self.global_num_classes) = build_domain_specs_from_labelscheme(self.full_label_scheme)

        # === Backbone ===
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            **from_pretrained_kwargs
        }
        self.backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            backbone_name, attn_implementation="flash_attention_2", **model_kwargs
        )
        self.device_map = device_map

        # obtain the hidden size of the backbone
        hidden_size = self._resolve_hidden_size(self.backbone)
        self.hidden_size = hidden_size

        # === One head per domain ===
        self.heads = nn.ModuleList()
        for global_indices in self.domain_id_to_global_indices:
            k_t = len(global_indices)
            self.heads.append(nn.Linear(hidden_size, k_t))

        # Training strategy (freeze/LORA/full)
        self._setup_training_strategy(freeze_backbone, lora_config)

        # Align heads’ device/dtype with pooled backbone output
        self._ensure_heads_alignment()

    # ---------- utils ----------
    @staticmethod
    def _resolve_hidden_size(backbone):
        cfg = backbone.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        for sub in ("language_config", "vision_config", "audio_config", "encoder_config"):
            if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), "hidden_size"):
                return getattr(cfg, sub).hidden_size
        # last-resort probe
        device = next(backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.ones(1, 1, dtype=torch.long, device=device)
            out = backbone(input_ids=dummy, output_hidden_states=True)
            return out.hidden_states[-1].shape[-1]

    def _apply_lora(self, lora_config):
        cfg = {'r':16, 'alpha':32, 'dropout':0.1,
               'target_modules': ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]}
        cfg.update(lora_config or {})
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=cfg['r'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'],
            target_modules=cfg['target_modules'], bias="none",
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        print(f"Applied LoRA r={cfg['r']} alpha={cfg['alpha']} dropout={cfg['dropout']}")

    def _setup_training_strategy(self, freeze_backbone, lora_config):
        if freeze_backbone == "lora":
            if lora_config is None:
                raise ValueError("lora_config must be provided when freeze_backbone='lora'")
            self._apply_lora(lora_config)
            print("Training strategy: LoRA (backbone unfrozen)")
        elif freeze_backbone == "head_only" or freeze_backbone is True:
            for p in self.backbone.parameters(): p.requires_grad = False
            print("Training strategy: Head-only (backbone frozen)")
        elif freeze_backbone == "full" or freeze_backbone is False:
            print("Training strategy: Full fine-tuning")
        else:
            for p in self.backbone.parameters(): p.requires_grad = False
            print("Training strategy: Head-only (default)")

    def _ensure_heads_alignment(self):
        # Place heads where pooled features live
        if self.device_map == "auto":
            with torch.no_grad():
                dummy_input = torch.ones(1, 10, dtype=torch.long)
                dummy_mask  = torch.ones(1, 10, dtype=torch.long)
                out = self.backbone(input_ids=dummy_input, attention_mask=dummy_mask, output_hidden_states=True)
                h = out.hidden_states[-2].mean(dim=1)
                target_device, target_dtype = h.device, h.dtype
            for head in self.heads:
                head.to(device=target_device, dtype=target_dtype)
        else:
            dev = next(self.backbone.parameters()).device
            dt  = next(self.backbone.parameters()).dtype
            for head in self.heads:
                head.to(device=dev, dtype=dt)

    def _heads_from_pooled(self, pooled: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """Route pooled hidden state through per-domain heads into global logit space."""
        B = pooled.size(0)
        device, dtype = pooled.device, pooled.dtype
        neg_inf = torch.finfo(dtype).min / 2
        logits_all = torch.full((B, self.global_num_classes), neg_inf, device=device, dtype=dtype)

        domain_ids = domain_ids.to(device)
        for d in domain_ids.unique(sorted=True).tolist():
            rows = (domain_ids == d).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = torch.as_tensor(self.domain_id_to_global_indices[d], device=device, dtype=torch.long)
            local_logits = self.heads[d](pooled.index_select(0, rows)).to(dtype)  # [B_d, K_d]
            block = torch.full((rows.numel(), self.global_num_classes), neg_inf, device=device, dtype=dtype)
            block = block.scatter(1, cols.unsqueeze(0).expand(rows.numel(), cols.numel()), local_logits)
            logits_all = logits_all.index_copy(0, rows, block)
        return logits_all

    # ---------- forward ----------
    def forward(self, input_ids, attention_mask=None, domain_ids=None, lm_labels=None, **kwargs):
        """
        Args:
          input_ids:    [B, T]
          attention_mask: [B, T]
          domain_ids:   [B] int in {0..D-1}; -1 for QA rows (receives neg_inf logits)
          lm_labels:    [B, T] or None.  When provided, runs QA path and returns a dict.
                        When None, runs CLS path and returns logits tensor [B, global_num_classes].
        Returns (CLS path):
          logits_all: [B, global_num_classes], NEG_INF for irrelevant classes.
        Returns (QA path):
          {"cls_logits": [B, global_num_classes], "lm_loss": scalar, "lm_output": backbone output}
        """
        if domain_ids is None:
            raise ValueError("domain_ids must be provided: a per-sample domain id is required for multi-head routing.")

        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False, **kwargs
        )
        hidden_states = out.hidden_states
        h = hidden_states[-2]  # [B, T, H]

        if attention_mask is not None:
            pooled = (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            pooled = h.mean(dim=1)  # [B, H]

        # --- CLS path (default) ---
        if lm_labels is None:
            return self._heads_from_pooled(pooled, domain_ids)

        # --- QA path ---
        cls_logits = self._heads_from_pooled(pooled, domain_ids)

        # Apply backbone norm + lm_head to last hidden state
        maybe_model = getattr(self.backbone, "model", None)
        h_last = hidden_states[-1]
        if maybe_model is not None and hasattr(maybe_model, "norm"):
            h_for_lm = maybe_model.norm(h_last)
        else:
            h_for_lm = h_last
        lm_head = getattr(self.backbone, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Backbone has no lm_head")
        lm_logits = lm_head(h_for_lm)  # [B, T, V]

        # Teacher-forcing LM loss
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = lm_labels[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        out.logits = lm_logits
        out.loss   = lm_loss
        return {"cls_logits": cls_logits, "lm_loss": lm_loss, "lm_output": out}

    # Convenience: expose mappings
    @property
    def dataset_to_domain_id(self):
        # property already set in __init__, but kept for API symmetry
        return self._dataset_to_domain_id
    @dataset_to_domain_id.setter
    def dataset_to_domain_id(self, v):
        self._dataset_to_domain_id = v

    def get_trainable_parameters(self):
        """
        Get the number of trainable parameters for monitoring LoRA efficiency.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.backbone.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Backbonetrainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")
        return trainable_params, all_param
