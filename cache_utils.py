"""Optimized cache utilities for transformers v5 compatibility.

This module provides enhanced cache classes that maintain backward compatibility
with the original TSV project while leveraging transformers v5 optimizations.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Any
import torch

if TYPE_CHECKING:
    from transformers.cache_utils import Cache as TransformersCache
else:
    try:
        from transformers.cache_utils import (
            Cache as TransformersCache,
            DynamicCache,
            StaticCache,
            StaticCache as SlidingWindowCache,  # Aliased for compatibility
            QuantizedCache,
            EncoderDecoderCache,
        )
    except ImportError:
        raise ImportError(
            "Failed to import cache classes from transformers v5. "
            "Please ensure you have transformers >= 5.0.0 installed."
        )


class Cache(TransformersCache):
    """Enhanced cache with backward compatibility and optimizations.
    
    This class extends the transformers v5 Cache class to maintain compatibility
    with the original TSV project's interface while adding performance optimizations.
    """
    
    def __init__(self, *args, **kwargs):
        # Handle transformers v5 Cache constructor requirements
        if not args and not kwargs:
            # Provide default layer_class_to_replicate for backward compatibility
            from transformers.cache_utils import DynamicCache
            kwargs['layer_class_to_replicate'] = DynamicCache
        
        super().__init__(*args, **kwargs)
        self._device_cache: Optional[torch.device] = None
        # Initialize cache attributes for backward compatibility
        self.key_cache = []
        self.value_cache = []
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Backward compatible method with layer_idx parameter.
        
        Args:
            layer_idx: Layer index (for backward compatibility)
            
        Returns:
            Sequence length of cached states
        """
        # Try transformers v5 method first
        if hasattr(super(), 'get_seq_length'):
            try:
                return super().get_seq_length()
            except (TypeError, AttributeError):
                pass
        
        # Fallback to layer-based approach
        if hasattr(self, 'layers') and layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if hasattr(layer, 'get_seq_length'):
                return layer.get_seq_length()
            elif hasattr(layer, 'keys') and layer.keys is not None:
                return layer.keys.shape[-2]  # seq_len dimension
        
        # Final fallback for backward compatibility
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx]:
            return self.key_cache[layer_idx].shape[-2]
        
        return 0
    
    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object."""
        # Try transformers v5 method first
        if hasattr(super(), 'get_max_cache_shape'):
            try:
                return super().get_max_cache_shape()
            except (TypeError, AttributeError):
                pass
        
        # Fallback implementation
        if hasattr(self, 'layers') and self.layers:
            return max(layer.get_max_cache_shape() for layer in self.layers)
        
        return None
    
    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache.
        
        Args:
            new_seq_length: Length of new sequence to add
            layer_idx: Layer index (for backward compatibility)
            
        Returns:
            Usable cache length
        """
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        
        # Cache without size limit -> all cache is usable
        if max_length is None:
            return previous_seq_length
            
        # Cache with size limit -> check if eviction is needed
        if previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search with optimizations.
        
        Args:
            beam_idx: Selected beam indices for reordering
        """
        # Try transformers v5 method first
        if hasattr(super(), 'reorder_cache'):
            try:
                super().reorder_cache(beam_idx)
                return
            except (TypeError, AttributeError):
                pass
        
        # Optimized fallback implementation
        if not beam_idx.numel() > 0:
            return
            
        beam_idx_device = beam_idx.device
        
        # Process layers efficiently
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if hasattr(layer, 'keys') and layer.keys is not None:
                    device = self._get_device(layer.keys)
                    if device != beam_idx_device:
                        beam_idx = beam_idx.to(device)
                    layer.keys = layer.keys.index_select(0, beam_idx)
                    layer.values = layer.values.index_select(0, beam_idx)
        
        # Backward compatibility for key_cache/value_cache
        if hasattr(self, 'key_cache') and hasattr(self, 'value_cache'):
            for layer_idx in range(len(self.key_cache)):
                key_cache = self.key_cache[layer_idx]
                value_cache = self.value_cache[layer_idx]
                
                # Skip empty caches early
                if not key_cache and not value_cache:
                    continue
                    
                # Batch device transfers
                if key_cache is not None and len(key_cache) > 0:
                    device = self._get_device(key_cache)
                    if device != beam_idx_device:
                        beam_idx = beam_idx.to(device)
                    self.key_cache[layer_idx] = key_cache.index_select(0, beam_idx)
                    
                if value_cache is not None and len(value_cache) > 0:
                    device = self._get_device(value_cache)
                    if device != beam_idx_device:
                        beam_idx = beam_idx.to(device)
                    self.value_cache[layer_idx] = value_cache.index_select(0, beam_idx)
    
    def _get_device(self, tensor: torch.Tensor) -> torch.device:
        """Cache device information to avoid repeated device calls.
        
        Args:
            tensor: Tensor to get device from
            
        Returns:
            Device of the tensor
        """
        if self._device_cache is None or self._device_cache != tensor.device:
            self._device_cache = tensor.device
        return self._device_cache
    
    @property
    def seen_tokens(self):
        """Backward compatibility property for seen tokens.
        
        Returns:
            Number of seen tokens or None if not available
        """
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        elif hasattr(super(), 'seen_tokens'):
            return super().seen_tokens
        else:
            return None
    
    def get_max_length(self) -> Optional[int]:
        """Backward compatibility method.
        
        Returns:
            Maximum cache length (deprecated in favor of get_max_cache_shape)
        """
        return self.get_max_cache_shape()


# Re-export all cache classes for backward compatibility
__all__ = [
    "Cache",
    "DynamicCache", 
    "StaticCache",
    "SlidingWindowCache",
    "QuantizedCache",
    "EncoderDecoderCache",
]
