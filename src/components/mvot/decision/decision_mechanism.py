"""
Text/image generation decision mechanism for MVoT.

This module provides components for determining whether to generate
text or image tokens during multimodal reasoning, implementing
heuristics for visualization benefit assessment.
"""
import re
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityType(Enum):
    """Enum for different modality types in MVoT."""
    TEXT = 0
    IMAGE = 1


class VisualizationBenefitHeuristic(Enum):
    """Enum for different types of visualization benefit heuristics."""
    SPATIAL = "spatial"
    VISUAL = "visual"
    COMPLEXITY = "complexity"
    REASONING = "reasoning"
    SPECIFICITY = "specificity"


class VisualizationBenefitAssessor:
    """
    Assesses whether visualization would benefit the reasoning process.
    
    This class implements various heuristics to determine if generating
    an image would be beneficial for the current reasoning step.
    """
    
    def __init__(self, config):
        """
        Initialize the visualization benefit assessor.
        
        Args:
            config: Configuration object with visualization settings
        """
        self.config = config
        
        # Load spatial keywords if available
        self.spatial_keywords = set([
            "position", "location", "orientation", "direction", "rotation",
            "left", "right", "top", "bottom", "above", "below", "under", "over",
            "north", "south", "east", "west", "behind", "in front", "beside",
            "inside", "outside", "center", "middle", "edge", "corner", "side",
            "diagonal", "parallel", "perpendicular", "intersection", "overlap",
            "adjacent", "distance", "proximity", "far", "near", "move", "rotate",
            "translate", "shift", "angle", "degree", "radian", "coordinate",
            "height", "width", "depth", "dimension", "size", "scale", "shape",
            "circle", "square", "rectangle", "triangle", "polygon", "sphere",
            "cube", "cylinder", "cone", "grid", "map", "path", "route", "layout",
            "arrangement", "configuration", "pattern", "sequence", "series",
            "cross", "align", "stack", "row", "column", "tilt", "twist", "turn"
        ])
        
        # Load visual appearance keywords
        self.visual_keywords = set([
            "color", "shade", "hue", "brightness", "contrast", "saturation",
            "pattern", "texture", "appearance", "look", "image", "picture",
            "visual", "visible", "invisible", "transparent", "opaque", "dark",
            "light", "red", "green", "blue", "yellow", "purple", "orange",
            "black", "white", "gray", "brown", "pink", "violet", "indigo",
            "colorful", "monochrome", "gradient", "shadow", "reflection",
            "highlight", "outline", "border", "edge", "surface", "shiny",
            "dull", "matte", "glossy", "rough", "smooth", "bumpy", "flat",
            "curved", "striped", "spotted", "dotted", "checkered", "wavy",
            "zigzag", "symmetrical", "asymmetrical", "logo", "icon", "symbol",
            "diagram", "illustration", "visualization", "render", "display"
        ])
        
        # Load complexity indicators
        self.complexity_indicators = set([
            "complex", "complicated", "intricate", "elaborate", "sophisticated",
            "convoluted", "involved", "detailed", "multifaceted", "layered",
            "nested", "multiple", "several", "many", "numerous", "various",
            "different", "diverse", "heterogeneous", "mixed", "assorted",
            "collection", "set", "array", "matrix", "network", "graph",
            "tree", "hierarchy", "structure", "system", "organization",
            "interconnected", "linked", "related", "associated", "coupled",
            "joined", "combined", "integrated", "unified", "merged", "fused"
        ])
        
        # Load reasoning terms
        self.reasoning_terms = set([
            "therefore", "thus", "hence", "consequently", "so", "as a result",
            "because", "since", "due to", "leads to", "causes", "results in",
            "implies", "suggests", "indicates", "shows", "demonstrates",
            "proves", "confirms", "verifies", "validates", "corroborates",
            "supports", "reinforces", "establishes", "if", "then", "else",
            "otherwise", "either", "or", "neither", "nor", "both", "and",
            "also", "additionally", "furthermore", "moreover", "besides",
            "in addition", "similarly", "likewise", "in the same way",
            "for example", "for instance", "specifically", "in particular",
            "namely", "such as", "including", "in conclusion", "to conclude",
            "in summary", "to summarize", "finally", "ultimately", "overall"
        ])
        
        # Keywords indicating specificity
        self.specificity_terms = set([
            "exactly", "precisely", "specifically", "particularly", "especially",
            "notably", "remarkably", "distinctly", "uniquely", "exclusively",
            "solely", "only", "merely", "just", "simply", "purely", "entirely",
            "completely", "totally", "wholly", "fully", "thoroughly", "utterly",
            "absolutely", "definitely", "certainly", "undoubtedly", "indubitably",
            "unquestionably", "incontrovertibly", "irrefutably", "unmistakably"
        ])
        
        # Thresholds for different heuristics
        self.spatial_threshold = getattr(config.mvot, "spatial_threshold", 0.15)
        self.visual_threshold = getattr(config.mvot, "visual_threshold", 0.15)
        self.complexity_threshold = getattr(config.mvot, "complexity_threshold", 0.10)
        self.reasoning_threshold = getattr(config.mvot, "reasoning_threshold", 0.20)
        self.specificity_threshold = getattr(config.mvot, "specificity_threshold", 0.10)
        
        # Combined keyword set for faster lookup
        self.all_keywords = (
            self.spatial_keywords | 
            self.visual_keywords | 
            self.complexity_indicators | 
            self.reasoning_terms |
            self.specificity_terms
        )
    
    def assess_text_for_visualization(self, text: str) -> Dict[str, float]:
        """
        Assess text to determine if visualization would be beneficial.
        
        Args:
            text: The text to assess
            
        Returns:
            Dictionary with heuristic scores
        """
        # Normalize text
        normalized_text = text.lower()
        words = re.findall(r'\b\w+\b', normalized_text)
        word_count = len(words)
        
        if word_count == 0:
            return {
                "spatial_score": 0.0,
                "visual_score": 0.0,
                "complexity_score": 0.0,
                "reasoning_score": 0.0,
                "specificity_score": 0.0,
                "combined_score": 0.0,
                "visualization_recommended": False
            }
        
        # Count occurrences of each type of keyword
        spatial_count = sum(1 for word in words if word in self.spatial_keywords)
        visual_count = sum(1 for word in words if word in self.visual_keywords)
        complexity_count = sum(1 for word in words if word in self.complexity_indicators)
        reasoning_count = sum(1 for word in words if word in self.reasoning_terms)
        specificity_count = sum(1 for word in words if word in self.specificity_terms)
        
        # Calculate scores
        spatial_score = min(1.0, spatial_count / word_count)
        visual_score = min(1.0, visual_count / word_count)
        complexity_score = min(1.0, complexity_count / word_count)
        reasoning_score = min(1.0, reasoning_count / word_count)
        specificity_score = min(1.0, specificity_count / word_count)
        
        # Calculate combined score with weights
        spatial_weight = getattr(self.config.mvot, "spatial_weight", 1.0)
        visual_weight = getattr(self.config.mvot, "visual_weight", 1.0)
        complexity_weight = getattr(self.config.mvot, "complexity_weight", 0.7)
        reasoning_weight = getattr(self.config.mvot, "reasoning_weight", 0.5)
        specificity_weight = getattr(self.config.mvot, "specificity_weight", 0.8)
        
        weighted_sum = (
            spatial_score * spatial_weight +
            visual_score * visual_weight +
            complexity_score * complexity_weight +
            reasoning_score * reasoning_weight +
            specificity_score * specificity_weight
        )
        
        total_weight = (
            spatial_weight + 
            visual_weight + 
            complexity_weight + 
            reasoning_weight +
            specificity_weight
        )
        
        combined_score = weighted_sum / total_weight
        
        # Additional score boost for combinations of features
        if (spatial_score > 0 and visual_score > 0):
            combined_score *= 1.2  # 20% boost for spatial + visual
        
        if (reasoning_score > 0.1 and (spatial_score > 0 or visual_score > 0)):
            combined_score *= 1.15  # 15% boost for reasoning + spatial/visual
        
        # Determine if visualization is recommended
        visualization_recommended = (
            spatial_score >= self.spatial_threshold or
            visual_score >= self.visual_threshold or
            complexity_score >= self.complexity_threshold or
            combined_score >= 0.15 or  # Allow combined score to trigger recommendation
            (reasoning_score >= self.reasoning_threshold and 
             (spatial_score > 0 or visual_score > 0)) or
            (specificity_score >= self.specificity_threshold and
             (spatial_score > 0 or visual_score > 0))
        )
        
        return {
            "spatial_score": float(spatial_score),
            "visual_score": float(visual_score),
            "complexity_score": float(complexity_score),
            "reasoning_score": float(reasoning_score),
            "specificity_score": float(specificity_score),
            "combined_score": float(combined_score),
            "visualization_recommended": bool(visualization_recommended)
        }
    
    def detect_diagram_request(self, text: str) -> bool:
        """
        Detect explicit requests for diagrams or visualizations.
        
        Args:
            text: The text to assess
            
        Returns:
            Whether the text contains an explicit request for visualization
        """
        # Regular expressions for detecting visualization requests
        diagram_patterns = [
            r"\bshow\s+(?:a|the)?\s*diagram\b",
            r"\bdraw\s+(?:a|the)?\s*diagram\b",
            r"\bcreate\s+(?:a|the)?\s*diagram\b",
            r"\bvisuali[sz]e\s+this\b",
            r"\bshow\s+(?:a|the)?\s*(?:image|picture|illustration|figure|visualization|drawing)\b",
            r"\bdraw\s+(?:a|the)?\s*(?:image|picture|illustration|figure|visualization|drawing)\b",
            r"\bcreate\s+(?:a|the)?\s*(?:image|picture|illustration|figure|visualization|drawing)\b",
            r"\bproduce\s+(?:a|the)?\s*(?:image|picture|illustration|figure|visualization|drawing|diagram)\b",
            r"\bdepict\s+(?:this|it)\b",
            r"\billustrate\s+(?:this|it)\b",
            r"\bcan\s+(?:you|I)\s+see\b",
            r"\bshow\s+me\b",
            r"I('ll| will)\s+draw\b"  # Match "I'll draw" or "I will draw"
        ]
        
        # Check for matches
        for pattern in diagram_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        return False


class ContextAwareDecider(nn.Module):
    """
    Context-aware decision logic for MVoT token generation.
    
    This class implements a neural network-based decision mechanism for
    determining whether to generate text or image tokens based on the
    context.
    """
    
    def __init__(self, config):
        """
        Initialize the context-aware decider.
        
        Args:
            config: Configuration object with model settings
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Neural network for decision making
        self.decision_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.hidden_size // 2, 2)  # 2 outputs: text or image
        )
        
        # Context aggregation
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=max(1, config.num_attention_heads // 4),
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        # Learned position embeddings for type of token
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Trainable parameters for context weighting
        self.context_weighting = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0]))
        
        # Thresholds for decisions
        self.image_threshold = getattr(config.mvot, "image_threshold", 0.7)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the context-aware decider.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            
        Returns:
            Dictionary with decision logits and probabilities
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # Get token type embeddings
        token_type_embed = self.token_type_embeddings(token_type_ids)
        
        # Add token type embeddings to hidden states
        contextualized_input = hidden_states + token_type_embed
        contextualized_input = self.layer_norm(contextualized_input)
        
        # Use attention to aggregate context information
        context_features, _ = self.context_attention(
            contextualized_input,
            contextualized_input,
            contextualized_input,
            key_padding_mask=(attention_mask == 0)
        )
        
        # Get the last token representation for decision
        last_token_index = attention_mask.sum(dim=1) - 1
        last_token_index = torch.clamp(last_token_index, min=0).to(torch.long)
        
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        last_token_features = context_features[batch_indices, last_token_index]
        
        # Make the decision
        decision_logits = self.decision_network(last_token_features)
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        # Get decision as argmax
        decision = torch.argmax(decision_probs, dim=-1)
        
        # Return with detailed information
        return {
            "logits": decision_logits,
            "probabilities": decision_probs,
            "decision": decision,
            "image_prob": decision_probs[:, ModalityType.IMAGE.value],
            "text_prob": decision_probs[:, ModalityType.TEXT.value]
        }
    
    def should_generate_image(self, decision_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Determine if an image should be generated based on the decision output.
        
        Args:
            decision_output: Output from the forward pass
            
        Returns:
            Boolean tensor indicating whether to generate an image
        """
        image_prob = decision_output["image_prob"]
        return image_prob > self.image_threshold


class GenerationDecisionMechanism(nn.Module):
    """
    Decision mechanism for MVoT text/image generation.
    
    This class combines heuristic-based and neural network-based approaches
    for determining whether to generate text or image tokens during
    multimodal reasoning.
    """
    
    def __init__(self, config):
        """
        Initialize the generation decision mechanism.
        
        Args:
            config: Configuration object with model settings
        """
        super().__init__()
        self.config = config
        
        # Create the heuristic-based assessor
        self.visualization_assessor = VisualizationBenefitAssessor(config)
        
        # Create the neural network-based decider
        self.context_decider = ContextAwareDecider(config)
        
        # Decision strategy
        self.decision_strategy = getattr(config.mvot, "decision_strategy", "hybrid")
        self.heuristic_weight = getattr(config.mvot, "heuristic_weight", 0.5)
        self.neural_weight = getattr(config.mvot, "neural_weight", 0.5)
        
        # Maximum number of images to generate
        self.max_images = getattr(config.mvot, "max_images", 5)
        
        # Minimum tokens between images
        self.min_tokens_between_images = getattr(config.mvot, "min_tokens_between_images", 20)
    
    def get_last_text_segment(self, text_tokens, tokenizer):
        """
        Get the last segment of text from the tokens.
        
        Args:
            text_tokens: The token IDs
            tokenizer: The tokenizer to convert tokens to text
            
        Returns:
            The last segment of text (up to 100 tokens)
        """
        # Get the last 100 tokens
        last_tokens = text_tokens[-100:] if len(text_tokens) > 100 else text_tokens
        
        # Convert to text
        try:
            last_text = tokenizer.decode(last_tokens)
            return last_text
        except:
            # If tokenizer is not available, return empty string
            return ""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_text: Optional[str] = None,
        tokens_since_last_image: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for the generation decision mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            input_text: Optional text context for heuristic assessment
            tokens_since_last_image: Number of tokens generated since the last image
            
        Returns:
            Dictionary with decision information
        """
        batch_size = hidden_states.shape[0]
        
        # Count existing images
        image_count = torch.sum(token_type_ids == ModalityType.IMAGE.value, dim=1)
        
        # Get neural network-based decision
        neural_decision = self.context_decider(hidden_states, token_type_ids, attention_mask)
        
        # Initialize heuristic decision
        heuristic_decision = {
            "visualization_recommended": torch.zeros(batch_size, dtype=torch.bool, device=hidden_states.device)
        }
        
        # If input text is provided, use heuristic assessment
        if input_text is not None:
            # For batched input, process each example
            if isinstance(input_text, list):
                for i, text in enumerate(input_text):
                    assessment = self.visualization_assessor.assess_text_for_visualization(text)
                    explicit_request = self.visualization_assessor.detect_diagram_request(text)
                    
                    # Set visualization recommended flag
                    heuristic_decision["visualization_recommended"][i] = (
                        assessment["visualization_recommended"] or explicit_request
                    )
            else:
                # Single input
                assessment = self.visualization_assessor.assess_text_for_visualization(input_text)
                explicit_request = self.visualization_assessor.detect_diagram_request(input_text)
                
                # Set visualization recommended flag for all examples in batch
                heuristic_decision["visualization_recommended"][:] = (
                    assessment["visualization_recommended"] or explicit_request
                )
        
        # Combine decisions based on strategy
        if self.decision_strategy == "neural":
            # Use only neural network-based decision
            generate_image = neural_decision["image_prob"] > self.context_decider.image_threshold
        elif self.decision_strategy == "heuristic":
            # Use only heuristic-based decision
            generate_image = heuristic_decision["visualization_recommended"]
        else:  # "hybrid"
            # Combine both decisions
            neural_score = neural_decision["image_prob"]
            
            # Convert to float if it's a boolean tensor
            if isinstance(heuristic_decision["visualization_recommended"], torch.Tensor):
                if heuristic_decision["visualization_recommended"].dtype == torch.bool:
                    heuristic_score = heuristic_decision["visualization_recommended"].float()
                else:
                    heuristic_score = heuristic_decision["visualization_recommended"]
            else:
                # Convert Python boolean to float tensor
                heuristic_score = torch.tensor(
                    [float(heuristic_decision["visualization_recommended"])] * batch_size,
                    device=hidden_states.device
                )
            
            combined_score = (
                neural_score * self.neural_weight + 
                heuristic_score * self.heuristic_weight
            )
            
            generate_image = combined_score > 0.5
        
        # Apply constraints
        # 1. Maximum number of images
        max_images_constraint = image_count < self.max_images
        
        # 2. Minimum tokens between images
        if tokens_since_last_image is not None:
            if isinstance(tokens_since_last_image, torch.Tensor):
                min_tokens_constraint = tokens_since_last_image > self.min_tokens_between_images
            else:
                min_tokens_constraint = torch.tensor(
                    [tokens_since_last_image > self.min_tokens_between_images] * batch_size,
                    device=hidden_states.device
                )
        else:
            min_tokens_constraint = torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)
        
        # Final decision
        final_decision = generate_image & max_images_constraint & min_tokens_constraint
        
        return {
            "should_generate_image": final_decision,
            "neural_decision": neural_decision,
            "heuristic_decision": heuristic_decision,
            "image_count": image_count,
            "max_images_constraint": max_images_constraint,
            "min_tokens_constraint": min_tokens_constraint
        }
    
    def should_generate_image(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_text: Optional[str] = None,
        tokens_since_last_image: Optional[int] = None
    ) -> torch.Tensor:
        """
        Determine if an image should be generated.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            input_text: Optional text context for heuristic assessment
            tokens_since_last_image: Number of tokens generated since the last image
            
        Returns:
            Boolean tensor indicating whether to generate an image
        """
        decision = self.forward(
            hidden_states, 
            token_type_ids, 
            attention_mask, 
            input_text, 
            tokens_since_last_image
        )
        
        return decision["should_generate_image"]


def create_decision_mechanism(config) -> GenerationDecisionMechanism:
    """
    Factory function to create a generation decision mechanism.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        GenerationDecisionMechanism instance
    """
    return GenerationDecisionMechanism(config)