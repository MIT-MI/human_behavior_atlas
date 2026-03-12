"""
Multi-head classification trainer.

Uses LoRA (or other strategies) to train the full model end-to-end with
per-dataset classification heads. All shared logic lives in BaseMultiHeadTrainer;
this class inherits everything without override.
"""
from trainer.base_trainer import BaseMultiHeadTrainer


class CLSTrainer(BaseMultiHeadTrainer):
    """
    Standard multi-head classification trainer.
    Inherits train / validate / test / checkpoint I/O fully from BaseMultiHeadTrainer.
    """
    pass
