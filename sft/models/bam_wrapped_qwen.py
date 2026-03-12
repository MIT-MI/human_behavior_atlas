# models/bam_wrapped_qwen.py
"""
BAM-augmented Qwen wrappers for classification and QA.

  BAMBase — shared base class (adapter slots, penultimate pooling, adapter application)
  BAMCLS  — classification-only forward (multi-head CE loss)
  BAMQA   — QA forward with LM loss; also returns cls_logits for domain_id=-1 rows

Adapters (video_adapter, audio_adapter) are assigned as nn.Module submodules by the
trainer BEFORE accelerator.prepare(), so DDP/FSDP wraps model+adapters as one unit.
Model forwards take raw feats (video_feats, audio_feats) and call the adapters internally.
Partial modality is handled gracefully: if an adapter is None or feats are None, that
modality is skipped silently.
"""
import torch
import torch.nn.functional as F

from .qwen2_5_omni_classifier_heads_decoder import MultiHeadOmniClassifier


class BAMBase(MultiHeadOmniClassifier):
    """
    Shared base for BAMCLS and BAMQA.

    Provides:
      _pool_penultimate  — attention-masked mean of backbone penultimate hidden layer → [B, H]
      _apply_adapters    — sequentially applies video then audio adapter deltas to pooled repr
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Placeholder attrs — trainer assigns nn.Module instances here before
        # accelerator.prepare(), which auto-registers them as submodules.
        self.video_adapter = None
        self.audio_adapter = None

    def _pool_penultimate(self, hidden_states, attention_mask) -> torch.Tensor:
        """Pool the penultimate hidden layer to [B, H] using attention mask weighting."""
        h = hidden_states[-2]  # [B, T, H]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return h.mean(dim=1)

    def _apply_adapters(
        self,
        pooled_base: torch.Tensor,
        video_feats: "torch.Tensor | None",
        audio_feats: "torch.Tensor | None",
        train_mode: bool,
    ) -> torch.Tensor:
        """
        Apply video and audio adapter deltas as sequential residuals.
        Skips silently if adapter is None or feats are None (partial modality).
        """
        h = pooled_base
        if self.video_adapter is not None and video_feats is not None:
            delta = self.video_adapter(video_feats.to(h.dtype), train_mode=train_mode)
            if delta is not None:
                h = h + delta
        if self.audio_adapter is not None and audio_feats is not None:
            delta = self.audio_adapter(audio_feats.to(h.dtype), train_mode=train_mode)
            if delta is not None:
                h = h + delta
        return h


class BAMCLS(BAMBase):
    """
    BAM-augmented multi-head classifier (CLS only).

    Forward returns (logits [B, C_domain], pooled_eff [B, H]).
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        domain_ids=None,
        *,
        video_feats: "torch.Tensor | None" = None,  # [B, D_v]
        audio_feats: "torch.Tensor | None" = None,  # [B, D_a]
        train_mode: bool = False,
        **kwargs,
    ):
        if domain_ids is None:
            raise ValueError("domain_ids required")

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        pooled_base = self._pool_penultimate(out.hidden_states, attention_mask)
        pooled_eff  = self._apply_adapters(pooled_base, video_feats, audio_feats, train_mode)
        logits      = self._heads_from_pooled(pooled_eff, domain_ids)
        return logits, pooled_eff


class BAMQA(BAMBase):
    """
    BAM-augmented model supporting QA (LM loss) training.

    Returns a dict:
        {
            "cls_logits": [B, C_global]  — classification logits (QA rows get domain_id=-1 → neg_inf)
            "lm_loss"   : scalar or None — teacher-forcing LM loss
            "lm_output" : HF output with BAM-injected lm_logits
        }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        domain_ids=None,
        lm_labels=None,
        *,
        video_feats: "torch.Tensor | None" = None,  # [B, D_v]
        audio_feats: "torch.Tensor | None" = None,  # [B, D_a]
        train_mode: bool = False,
        **kwargs,
    ):
        if domain_ids is None:
            raise ValueError("domain_ids required")

        # 1) Single backbone pass
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        hidden_states = out.hidden_states

        # 2) Pool penultimate + apply BAM adapter deltas
        pooled_base = self._pool_penultimate(hidden_states, attention_mask)
        pooled_eff  = self._apply_adapters(pooled_base, video_feats, audio_feats, train_mode)

        # 3) Inject BAM delta into last hidden layer, recompute through lm_head
        delta      = (pooled_eff - pooled_base).to(hidden_states[-1].dtype)  # [B, H]
        h_last_mod = hidden_states[-1] + delta.unsqueeze(1)                   # [B, T, H]

        maybe_model = getattr(self.backbone, "model", None)
        if maybe_model is not None and hasattr(maybe_model, "norm"):
            h_for_lm = maybe_model.norm(h_last_mod)
        else:
            h_for_lm = h_last_mod

        lm_head = getattr(self.backbone, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Backbone has no lm_head")
        lm_logits = lm_head(h_for_lm)  # [B, T, V]

        # 4) Teacher-forcing LM loss
        lm_loss = None
        if lm_labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # 5) Classification logits (domain_id=-1 rows receive neg_inf via _heads_from_pooled)
        cls_logits = self._heads_from_pooled(pooled_eff, domain_ids)

        out.logits = lm_logits
        out.loss   = lm_loss
        return {"cls_logits": cls_logits, "lm_loss": lm_loss, "lm_output": out}
