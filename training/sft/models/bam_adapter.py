# models/bam_adapter.py
from typing import Optional
import torch
import torch.nn as nn


class BehavioralAdapterModule(nn.Module):
    """
    Residual adapter at the pooled hidden (penultimate) state.

    Computes a delta in backbone H-space from side-channel features (video/audio):
        delta = MLP(feats) * alpha   # [B, H]

    The trainer adds this delta to the base pooled representation before the
    classification heads or LM-head injection.  No confidence features or domain
    routing are used.
    """

    def __init__(
        self,
        *,
        feat_dim: int,       # side feature dim (video/audio)
        hidden: int,         # MLP hidden units
        out_dim: int,        # backbone hidden size H
        p_moddrop: float = 0.3,
        use_ln: bool = False,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        self.p_moddrop = p_moddrop
        self.use_ln = use_ln

        if use_ln:
            self.ln_feats = nn.LayerNorm(feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))

    def _maybe_drop(self, feats: torch.Tensor, train_mode: bool) -> torch.Tensor:
        if not train_mode or self.p_moddrop <= 0:
            return feats
        keep = (torch.rand(feats.size(0), device=feats.device) >= self.p_moddrop).float().unsqueeze(-1)
        return feats * keep

    def forward(
        self,
        feats: Optional[torch.Tensor],  # [B, D_feat]
        train_mode: bool = False,
    ) -> Optional[torch.Tensor]:
        """Returns [B, H] delta tensor, or None if feats is None."""
        if feats is None:
            return None
        feats = self._maybe_drop(feats, train_mode)
        if self.use_ln:
            feats = self.ln_feats(feats)
        return self.mlp(feats) * self.alpha.view(1, 1)   # [B, H]
