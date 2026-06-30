"""
Custom FSDP wrapping policy for models with classifier heads.
This allows you to wrap transformer layers while keeping the classifier head unwrapped.
"""

import torch.nn as nn
from typing import Set, Type
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def custom_transformer_wrap_policy(
    transformer_layer_cls_to_wrap: Set[Type[nn.Module]],
    transformer_layer_cls_to_wrap_with_classifier: Set[Type[nn.Module]] = None,
) -> callable:
    """
    Custom wrapping policy that wraps transformer layers but keeps classifier heads unwrapped.
    
    Args:
        transformer_layer_cls_to_wrap: Set of transformer layer classes to wrap
        transformer_layer_cls_to_wrap_with_classifier: Set of additional classes to wrap (optional)
    
    Returns:
        Wrapping policy function
    """
    
    def custom_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        # Don't wrap the classifier head
        if isinstance(module, nn.Linear) and hasattr(module, 'out_features'):
            # This is likely a classifier head
            return False
        
        # Don't wrap the main classifier module
        if hasattr(module, 'classifier'):
            return False
        
        # Use the standard transformer wrapping policy for transformer layers
        if transformer_layer_cls_to_wrap_with_classifier is None:
            transformer_layer_cls_to_wrap_with_classifier = set()
        
        all_transformer_classes = transformer_layer_cls_to_wrap.union(transformer_layer_cls_to_wrap_with_classifier)
        
        return transformer_auto_wrap_policy(
            transformer_layer_cls_to_wrap=all_transformer_classes,
            transformer_layer_cls_to_wrap_with_classifier=transformer_layer_cls_to_wrap_with_classifier,
        )(module, recurse, nonwrapped_numel)
    
    return custom_wrap_policy


# Common transformer layer classes for different model families
QWEN_TRANSFORMER_LAYERS = {
    "Qwen2DecoderLayer",  # For Qwen2 models
    "Qwen2MoeDecoderLayer",  # For Qwen2 MoE models
    "Qwen2_5OmniDecoderLayer",  # For Qwen2.5 Omni models
}

LLAMA_TRANSFORMER_LAYERS = {
    "LlamaDecoderLayer",
    "LlamaMoeDecoderLayer",
}

MISTRAL_TRANSFORMER_LAYERS = {
    "MistralDecoderLayer",
}

# You can add more model families as needed
TRANSFORMER_LAYER_MAPPING = {
    "qwen": QWEN_TRANSFORMER_LAYERS,
    "llama": LLAMA_TRANSFORMER_LAYERS,
    "mistral": MISTRAL_TRANSFORMER_LAYERS,
}


def get_wrap_policy_for_model(model_name: str):
    """
    Get the appropriate wrapping policy for a given model.
    
    Args:
        model_name: Name of the model (e.g., "Qwen/Qwen2.5-Omni-7B")
    
    Returns:
        Wrapping policy function
    """
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        transformer_layers = QWEN_TRANSFORMER_LAYERS
    elif "llama" in model_name_lower:
        transformer_layers = LLAMA_TRANSFORMER_LAYERS
    elif "mistral" in model_name_lower:
        transformer_layers = MISTRAL_TRANSFORMER_LAYERS
    else:
        # Default to no wrapping if model type is unknown
        print(f"Warning: Unknown model type '{model_name}', using NO_WRAP policy")
        return None
    
    return custom_transformer_wrap_policy(transformer_layers)
