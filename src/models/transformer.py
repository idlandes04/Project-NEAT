"""
Memory-efficient transformer implementation with extension points.

This module provides a memory-efficient transformer implementation with
extension points for integrating different architectures.
"""
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """
    Efficient attention implementation with optimized memory usage.
    
    This class implements a memory-efficient attention mechanism that
    reduces memory usage during training and inference.
    """
    
    def __init__(self, config):
        """Initialize the flash attention module."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Check if hidden size is divisible by number of heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
        
        # Projection matrices
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Use flash attention if available
        self.use_flash_attention = getattr(config.hardware, "use_flash_attention", True)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for the flash attention module.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, 1, 1, seq_len]
            head_mask: Mask to nullify selected heads of the self-attention
            past_key_value: Cached past key and value projection states
            output_attentions: Whether to return attention weights
            
        Returns:
            A tuple of (context_layer, attention_probs, past_key_value)
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use past key value if available
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # Save current key and value for future use
        past_key_value = (key_states, value_states)
        
        # Compute attention scores
        if self.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
            # Use PyTorch's built-in flash attention if available
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            attn_weights = None  # Flash attention doesn't return attention weights
        else:
            # Compute attention scores with manual implementation
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Normalize attention weights
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Apply dropout to attention weights
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply head mask if provided
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
            
            # Compute attention output
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        # Project attention output
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, attn_weights, past_key_value


class FeedForward(nn.Module):
    """
    Feed-forward network with optimized memory usage.
    
    This class implements a memory-efficient feed-forward network with
    activation functions and dropout.
    """
    
    def __init__(self, config):
        """Initialize the feed-forward network."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Projection matrices
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        
        # Activation function
        self.act_fn = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward network.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class TransformerLayer(nn.Module):
    """
    Transformer layer with optimized memory usage.
    
    This class implements a memory-efficient transformer layer with
    attention and feed-forward networks.
    """
    
    def __init__(self, config):
        """Initialize the transformer layer."""
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.attention = FlashAttention(config)
        
        # Feed-forward network
        self.feed_forward = FeedForward(config)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.ln2 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Extension points for architecture integration
        self.pre_attention_extension = None
        self.post_attention_extension = None
        self.pre_feed_forward_extension = None
        self.post_feed_forward_extension = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for the transformer layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, 1, 1, seq_len]
            head_mask: Mask to nullify selected heads of the self-attention
            past_key_value: Cached past key and value projection states
            output_attentions: Whether to return attention weights
            
        Returns:
            A tuple of (hidden_states, attention_weights, past_key_value)
        """
        # Pre-attention extension point
        if self.pre_attention_extension is not None:
            hidden_states = self.pre_attention_extension(hidden_states)
        
        # Self-attention with residual connection and layer normalization
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attn_output, attn_weights, past_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        
        hidden_states = residual + attn_output
        
        # Post-attention extension point
        if self.post_attention_extension is not None:
            hidden_states = self.post_attention_extension(hidden_states)
        
        # Pre-feed-forward extension point
        if self.pre_feed_forward_extension is not None:
            hidden_states = self.pre_feed_forward_extension(hidden_states)
        
        # Feed-forward network with residual connection and layer normalization
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        # Post-feed-forward extension point
        if self.post_feed_forward_extension is not None:
            hidden_states = self.post_feed_forward_extension(hidden_states)
        
        outputs = (hidden_states, attn_weights, past_key_value)
        
        return outputs


class MemoryEfficientTransformer(nn.Module):
    """
    Memory-efficient transformer with extension points for architecture integration.
    
    This class implements a memory-efficient transformer with extension points
    for integrating different architectures, including Titans, Transformer²,
    MVoT, and BLT.
    """
    
    def __init__(self, config):
        """Initialize the memory-efficient transformer."""
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Extension points for architecture integration
        self.extension_points = {
            "pre_embedding": None,  # For BLT
            "post_embedding": None,  # For custom embeddings
            "pre_layer": [None] * config.num_layers,  # For Titans
            "post_layer": [None] * config.num_layers,  # For MVoT
            "pre_output": None,  # For custom output processing
            "post_output": None,  # For Transformer²
        }
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights if needed
        self.tie_weights()
    
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization with ones and zeros
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        self.lm_head.weight = self.embeddings.weight
    
    def get_input_embeddings(self):
        """Get the input embeddings."""
        return self.embeddings
    
    def set_input_embeddings(self, new_embeddings):
        """Set the input embeddings."""
        self.embeddings = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the memory-efficient transformer.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            position_ids: Position IDs of shape [batch_size, seq_len]
            head_mask: Mask to nullify selected heads of the self-attention
            inputs_embeds: Input embeddings of shape [batch_size, seq_len, hidden_size]
            past_key_values: Cached past key and value projection states
            labels: Labels for language modeling of shape [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            A dictionary of model outputs
        """
        # Set default values for optional parameters
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        # Pre-embedding extension point (BLT)
        if self.extension_points["pre_embedding"] is not None and input_ids is not None:
            input_ids = self.extension_points["pre_embedding"](input_ids)
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        
        # Get position embeddings
        if position_ids is None:
            # Create position IDs
            seq_length = inputs_embeds.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids) if input_ids is not None else position_ids.unsqueeze(0)
        
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Post-embedding extension point
        if self.extension_points["post_embedding"] is not None:
            hidden_states = self.extension_points["post_embedding"](hidden_states)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Extend attention mask for multi-head attention
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert mask to float and set large negative value for masked positions
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Prepare head mask
        if head_mask is not None:
            # Extend head mask for multi-head attention
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(self.config.num_layers, -1, -1, -1, -1)
        else:
            head_mask = [None] * self.config.num_layers
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.config.num_layers
        
        # Initialize lists for hidden states and attentions
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            # Add hidden state to list if needed
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Pre-layer extension point (Titans)
            if self.extension_points["pre_layer"][i] is not None:
                hidden_states = self.extension_points["pre_layer"][i](hidden_states)
            
            # Process through transformer layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[i],
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            # Post-layer extension point (MVoT)
            if self.extension_points["post_layer"][i] is not None:
                hidden_states = self.extension_points["post_layer"][i](hidden_states, token_type_ids)
            
            # Add attention weights to list if needed
            if output_attentions:
                all_attentions.append(layer_outputs[1])
            
            # Update past key values
            past_key_values[i] = layer_outputs[2]
        
        # Add final hidden state to list if needed
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Apply final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Pre-output extension point
        if self.extension_points["pre_output"] is not None:
            hidden_states = self.extension_points["pre_output"](hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Post-output extension point (Transformer²)
        if self.extension_points["post_output"] is not None:
            logits = self.extension_points["post_output"](logits)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        # Return outputs
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "past_key_values": past_key_values,
        }
