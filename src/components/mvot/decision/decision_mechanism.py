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
            "cross", "align", "stack", "row", "column", "tilt", "twist", "turn",
            # Enhanced spatial keywords
            "between", "in-between", "around", "surrounding", "encircling", "enclosing",
            "horizontal", "vertical", "diagonal", "orthogonal", "sloping", "curved",
            "clockwise", "counterclockwise", "upward", "downward", "inward", "outward",
            "frontward", "backward", "forward", "reverse", "offset", "staggered",
            "concentric", "radial", "adjacent", "layered", "stacked", "nested",
            "orientation", "positioned", "situated", "placed", "located", "arranged"
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
            "diagram", "illustration", "visualization", "render", "display",
            # Enhanced visual keywords
            "blend", "mixing", "fade", "transition", "vibrant", "muted", "pastel",
            "bold", "subtle", "rich", "deep", "pale", "dim", "bright", "illuminated",
            "shadowed", "highlighted", "contoured", "silhouette", "profile", "backdrop",
            "foreground", "background", "filter", "effect", "textured", "patterned",
            "translucent", "clearness", "opacity", "glowing", "luminous", "fluorescent",
            "reflective", "mirrored", "matte", "spectral", "chromatic", "tinted",
            "pixelated", "resolution", "clarity", "fidelity", "render", "rendering"
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
            "joined", "combined", "integrated", "unified", "merged", "fused",
            # Enhanced complexity keywords
            "tangled", "woven", "intertwined", "interwoven", "interlaced",
            "compound", "composite", "multidimensional", "interrelated",
            "recursive", "cyclical", "nonlinear", "branching", "diverging",
            "converging", "cascading", "tiered", "stratified", "segmented",
            "compartmentalized", "modular", "systemic", "procedural", "algorithmic",
            "interdependent", "correlative", "interactive", "dynamic", "adaptive",
            "emergent", "evolutionary", "transformational", "metamorphic", "chaotic"
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
            "in summary", "to summarize", "finally", "ultimately", "overall",
            # Enhanced reasoning keywords
            "infer", "deduce", "conclude", "derive", "reason", "rationalize",
            "analyze", "synthesize", "compare", "contrast", "distinguish", "differentiate",
            "evaluate", "assess", "appraise", "gauge", "measure", "quantify",
            "qualify", "classify", "categorize", "organize", "arrange", "sequence",
            "predict", "forecast", "project", "extrapolate", "hypothesize", "theorize",
            "assume", "presuppose", "postulate", "construct", "formulate", "conceptualize",
            "consider", "contemplate", "deliberate", "ponder", "reflect", "examine"
        ])
        
        # Keywords indicating specificity
        self.specificity_terms = set([
            "exactly", "precisely", "specifically", "particularly", "especially",
            "notably", "remarkably", "distinctly", "uniquely", "exclusively",
            "solely", "only", "merely", "just", "simply", "purely", "entirely",
            "completely", "totally", "wholly", "fully", "thoroughly", "utterly",
            "absolutely", "definitely", "certainly", "undoubtedly", "indubitably",
            "unquestionably", "incontrovertibly", "irrefutably", "unmistakably",
            # Enhanced specificity keywords
            "explicitly", "unambiguously", "clearly", "manifestly", "patently",
            "evidently", "obviously", "plainly", "expressly", "directly",
            "quintessentially", "fundamentally", "intrinsically", "inherently",
            "characteristically", "distinctively", "singularly", "peculiarly",
            "idiosyncratically", "explicitly", "unmistakably", "unerringly",
            "invariably", "unfailingly", "consistently", "persistently", "steadily",
            "uniformly", "homogeneously", "comprehensively", "exhaustively"
        ])
        
        # Mathematical and technical terms that often benefit from visualization
        self.math_technical_terms = set([
            "equation", "formula", "function", "variable", "coefficient", "parameter",
            "logarithm", "exponential", "derivative", "integral", "calculus", 
            "vector", "matrix", "tensor", "scalar", "linear", "nonlinear", "polynomial",
            "algorithm", "flowchart", "pseudocode", "iteration", "recursion",
            "data structure", "tree", "graph", "list", "array", "stack", "queue",
            "database", "schema", "entity", "relationship", "attribute",
            "architecture", "component", "module", "interface", "protocol",
            "circuit", "logic gate", "transistor", "resistor", "capacitor",
            "wavelength", "frequency", "amplitude", "oscillation", "resonance",
            "threshold", "boundary", "limit", "asymptote", "convergence", "divergence"
        ])
        
        # Thresholds for different heuristics
        self.spatial_threshold = getattr(config.mvot, "spatial_threshold", 0.15)
        self.visual_threshold = getattr(config.mvot, "visual_threshold", 0.15)
        self.complexity_threshold = getattr(config.mvot, "complexity_threshold", 0.10)
        self.reasoning_threshold = getattr(config.mvot, "reasoning_threshold", 0.20)
        self.specificity_threshold = getattr(config.mvot, "specificity_threshold", 0.10)
        self.math_technical_threshold = getattr(config.mvot, "math_technical_threshold", 0.12)
        
        # Adaptive thresholds for context-aware assessment
        self.use_adaptive_thresholds = getattr(config.mvot, "use_adaptive_thresholds", True)
        self.adaptive_threshold_scale = getattr(config.mvot, "adaptive_threshold_scale", 0.8)
        
        # Sequential pattern detection
        self.sequential_step_threshold = getattr(config.mvot, "sequential_step_threshold", 0.6)
        self.history = []
        self.max_history = getattr(config.mvot, "viz_history_length", 5)
        
        # Combined keyword set for faster lookup
        self.all_keywords = (
            self.spatial_keywords | 
            self.visual_keywords | 
            self.complexity_indicators | 
            self.reasoning_terms |
            self.specificity_terms |
            self.math_technical_terms
        )
    
    def _detect_sequential_pattern(self, scores: Dict[str, float]) -> float:
        """
        Detect sequential reasoning patterns that might benefit from visualization.
        
        Args:
            scores: Current assessment scores
            
        Returns:
            Pattern boost factor (1.0 = no boost)
        """
        # If we don't have enough history, no pattern boost
        if len(self.history) < 2:
            return 1.0
        
        # Look for increasing complexity or reasoning patterns
        complexity_trend = []
        reasoning_trend = []
        
        for past_score in self.history[-3:]:  # Look at last 3 entries
            complexity_trend.append(past_score.get("complexity_score", 0))
            reasoning_trend.append(past_score.get("reasoning_score", 0))
        
        # Current scores
        current_complexity = scores.get("complexity_score", 0)
        current_reasoning = scores.get("reasoning_score", 0)
        
        # Check for increasing trends
        if (len(complexity_trend) >= 2 and 
            all(b >= a for a, b in zip(complexity_trend, complexity_trend[1:])) and
            current_complexity > complexity_trend[-1]):
            return 1.3  # 30% boost for increasing complexity
        
        if (len(reasoning_trend) >= 2 and 
            all(b >= a for a, b in zip(reasoning_trend, reasoning_trend[1:])) and
            current_reasoning > reasoning_trend[-1] * 1.1):
            return 1.25  # 25% boost for increasing reasoning complexity
        
        # Look for alternating patterns (potential compare/contrast scenarios)
        if len(self.history) >= 4:
            visual_pattern = [score.get("visual_score", 0) > self.visual_threshold for score in self.history[-4:]]
            if visual_pattern[0] != visual_pattern[1] and visual_pattern[1] != visual_pattern[2] and visual_pattern[2] != visual_pattern[3]:
                return 1.15  # 15% boost for alternating visual emphasis
        
        return 1.0
    
    def _get_adaptive_thresholds(self, text_length: int) -> Dict[str, float]:
        """
        Get adaptive thresholds based on text length and other factors.
        
        Args:
            text_length: Length of the text in words
            
        Returns:
            Dictionary of adjusted thresholds
        """
        if not self.use_adaptive_thresholds:
            return {
                "spatial": self.spatial_threshold,
                "visual": self.visual_threshold,
                "complexity": self.complexity_threshold,
                "reasoning": self.reasoning_threshold,
                "specificity": self.specificity_threshold,
                "math_technical": self.math_technical_threshold
            }
        
        # For very short texts, we need higher keyword density to trigger visualization
        if text_length < 10:
            length_factor = 1.2  # Increase thresholds by 20%
        elif text_length < 20:
            length_factor = 1.1  # Increase thresholds by 10%
        elif text_length > 100:
            length_factor = 0.9  # Decrease thresholds by 10%
        else:
            length_factor = 1.0  # No adjustment
        
        # Apply adjustment factor
        return {
            "spatial": self.spatial_threshold * length_factor,
            "visual": self.visual_threshold * length_factor,
            "complexity": self.complexity_threshold * length_factor,
            "reasoning": self.reasoning_threshold * length_factor,
            "specificity": self.specificity_threshold * length_factor,
            "math_technical": self.math_technical_threshold * length_factor
        }
    
    def detect_phrase_patterns(self, text: str) -> Dict[str, float]:
        """
        Detect multi-word phrases that indicate visualization benefit.
        
        Args:
            text: The text to assess
            
        Returns:
            Dictionary with pattern scores
        """
        text_lower = text.lower()
        
        # Multi-word patterns that strongly suggest visualization would help
        pattern_scores = {
            # Comparison patterns
            "compare and contrast": 0.8,
            "side by side": 0.8,
            "in comparison to": 0.7,
            "as opposed to": 0.6,
            "on the other hand": 0.5,
            "in contrast": 0.5,
            
            # Process/sequence patterns
            "step by step": 0.7,
            "sequential process": 0.8,
            "in sequence": 0.6,
            "one after another": 0.6,
            "in succession": 0.5,
            "in order": 0.5,
            
            # Relationship patterns
            "connected to": 0.6,
            "linked with": 0.6,
            "related to": 0.5,
            "depends on": 0.5,
            "correlates with": 0.7,
            "proportional to": 0.7,
            
            # Spatial relationships
            "next to each other": 0.7,
            "surrounding the": 0.6,
            "in the middle of": 0.6,
            "at the intersection": 0.8,
            "in relation to": 0.6,
            "spatial relationship": 0.9,
            
            # Transformation patterns
            "transforms into": 0.7,
            "changes from": 0.6,
            "converts to": 0.6,
            "evolves from": 0.6,
            "turns into": 0.5,
            "becomes a": 0.5,
            
            # Technical/mathematical patterns
            "can be calculated": 0.6,
            "can be computed": 0.6,
            "can be derived": 0.6,
            "is defined as": 0.5,
            "is represented by": 0.7,
            "is visualized as": 0.9,
            
            # Explicit visualization value
            "easier to understand visually": 0.9,
            "best seen in a diagram": 0.9,
            "visual representation": 0.8,
            "visual explanation": 0.8,
            "graphical representation": 0.9,
            "graphical model": 0.8
        }
        
        matched_patterns = {}
        for pattern, score in pattern_scores.items():
            if pattern in text_lower:
                matched_patterns[pattern] = score
        
        # Get the max score if multiple patterns match
        max_score = max(matched_patterns.values()) if matched_patterns else 0.0
        
        return {
            "pattern_score": max_score,
            "matched_patterns": matched_patterns
        }
        
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
                "math_technical_score": 0.0,
                "combined_score": 0.0,
                "visualization_recommended": False
            }
        
        # Get adaptive thresholds
        thresholds = self._get_adaptive_thresholds(word_count)
        
        # Count occurrences of each type of keyword
        spatial_count = sum(1 for word in words if word in self.spatial_keywords)
        visual_count = sum(1 for word in words if word in self.visual_keywords)
        complexity_count = sum(1 for word in words if word in self.complexity_indicators)
        reasoning_count = sum(1 for word in words if word in self.reasoning_terms)
        specificity_count = sum(1 for word in words if word in self.specificity_terms)
        math_technical_count = sum(1 for word in words if word in self.math_technical_terms)
        
        # Calculate scores
        spatial_score = min(1.0, spatial_count / word_count)
        visual_score = min(1.0, visual_count / word_count)
        complexity_score = min(1.0, complexity_count / word_count)
        reasoning_score = min(1.0, reasoning_count / word_count)
        specificity_score = min(1.0, specificity_count / word_count)
        math_technical_score = min(1.0, math_technical_count / word_count)
        
        # Assess phrase patterns
        phrase_patterns = self.detect_phrase_patterns(text)
        pattern_score = phrase_patterns.get("pattern_score", 0.0)
        
        # Calculate combined score with weights
        spatial_weight = getattr(self.config.mvot, "spatial_weight", 1.0)
        visual_weight = getattr(self.config.mvot, "visual_weight", 1.0)
        complexity_weight = getattr(self.config.mvot, "complexity_weight", 0.7)
        reasoning_weight = getattr(self.config.mvot, "reasoning_weight", 0.5)
        specificity_weight = getattr(self.config.mvot, "specificity_weight", 0.8)
        math_technical_weight = getattr(self.config.mvot, "math_technical_weight", 1.0)
        pattern_weight = getattr(self.config.mvot, "pattern_weight", 1.2)
        
        weighted_sum = (
            spatial_score * spatial_weight +
            visual_score * visual_weight +
            complexity_score * complexity_weight +
            reasoning_score * reasoning_weight +
            specificity_score * specificity_weight +
            math_technical_score * math_technical_weight +
            pattern_score * pattern_weight
        )
        
        total_weight = (
            spatial_weight + 
            visual_weight + 
            complexity_weight + 
            reasoning_weight +
            specificity_weight +
            math_technical_weight +
            (pattern_weight if pattern_score > 0 else 0)
        )
        
        combined_score = weighted_sum / total_weight
        
        # Additional score boosts for combinations of features
        if (spatial_score > 0 and visual_score > 0):
            combined_score *= 1.2  # 20% boost for spatial + visual
        
        if (reasoning_score > 0.1 and (spatial_score > 0 or visual_score > 0)):
            combined_score *= 1.15  # 15% boost for reasoning + spatial/visual
            
        if (math_technical_score > 0.15 and complexity_score > 0):
            combined_score *= 1.25  # 25% boost for technical + complexity
        
        # Sequential pattern detection
        scores = {
            "spatial_score": spatial_score,
            "visual_score": visual_score,
            "complexity_score": complexity_score,
            "reasoning_score": reasoning_score,
            "specificity_score": specificity_score,
            "math_technical_score": math_technical_score,
            "pattern_score": pattern_score
        }
        
        pattern_boost = self._detect_sequential_pattern(scores)
        combined_score *= pattern_boost
        
        # Determine if visualization is recommended
        visualization_recommended = (
            spatial_score >= thresholds["spatial"] or
            visual_score >= thresholds["visual"] or
            complexity_score >= thresholds["complexity"] or
            pattern_score >= 0.7 or
            math_technical_score >= thresholds["math_technical"] or
            combined_score >= 0.15 or  # Allow combined score to trigger recommendation
            (reasoning_score >= thresholds["reasoning"] and 
             (spatial_score > 0 or visual_score > 0)) or
            (specificity_score >= thresholds["specificity"] and
             (spatial_score > 0 or visual_score > 0))
        )
        
        # Store in history for sequential pattern detection
        result = {
            "spatial_score": float(spatial_score),
            "visual_score": float(visual_score),
            "complexity_score": float(complexity_score),
            "reasoning_score": float(reasoning_score),
            "specificity_score": float(specificity_score),
            "math_technical_score": float(math_technical_score),
            "pattern_score": float(pattern_score),
            "combined_score": float(combined_score),
            "pattern_boost": float(pattern_boost),
            "matched_patterns": phrase_patterns.get("matched_patterns", {}),
            "visualization_recommended": bool(visualization_recommended)
        }
        
        # Update history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return result
    
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
            r"I('ll| will)\s+draw\b",  # Match "I'll draw" or "I will draw"
            # Enhanced patterns
            r"\bdisplay\s+(?:a|the)?\s*(?:diagram|visualization|model|chart|graph)\b",
            r"\brender\s+(?:a|the)?\s*(?:scene|image|visualization|view)\b",
            r"\bplot\s+(?:a|the)?\s*(?:graph|chart|function|data|points|line)\b",
            r"\bsketch\s+(?:a|the)?\s*(?:outline|diagram|drawing|figure|draft)\b",
            r"\bchart\s+(?:a|the)?\s*(?:course|path|direction|relationship|correlation)\b",
            r"\bmap\s+(?:a|the)?\s*(?:process|flow|structure|organization|hierarchy)\b",
            r"\bproject\s+(?:a|the)?\s*(?:image|view|model|representation)\b",
            r"\bwould\s+(?:look|appear)\s+like\b",
            r"\bpictorially\b",
            r"\bvisually\s+represent\b"
        ]
        
        # Check for matches
        for pattern in diagram_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        return False
    
    def reset_history(self):
        """Reset the history of assessments."""
        self.history = []


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
        
        # Neural network for decision making with enhanced architecture
        self.decision_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob * 0.8),  # Slightly reduced dropout
            nn.Linear(self.hidden_size // 4, 2)  # 2 outputs: text or image
        )
        
        # Context aggregation with improved attention
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
        
        # Trainable parameters for context weighting (expanded)
        self.context_weighting = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0, 0.5, 0.8]))
        
        # Thresholds for decisions
        self.image_threshold = getattr(config.mvot, "image_threshold", 0.7)
        
        # Decision history tracking (maintains state between calls)
        self.reset_history()
        
        # Domain adaptation parameters
        self.domain_specific = nn.Parameter(torch.randn(4, self.hidden_size // 8))
        self.domain_projection = nn.Linear(self.hidden_size, 4)
        
        # Whether model is in training mode for visualization decisions
        self.training_decision_model = getattr(config.mvot, "train_decision_model", False)
        
        # Loss function for training
        self.decision_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    def reset_history(self):
        """Reset the decision history tracking."""
        self.decision_history = []
        self.context_history = []
        self.max_history_len = 10  # Keep track of last 10 decisions
    
    def update_history(self, decision, context_features):
        """
        Update decision history with new decision.
        
        Args:
            decision: The decision made (0=text, 1=image)
            context_features: Features used for the decision
        """
        # Add to history
        self.decision_history.append(decision.detach().cpu())
        
        # Store summarized context features
        if isinstance(context_features, torch.Tensor):
            self.context_history.append(
                torch.mean(context_features, dim=0, keepdim=True).detach().cpu()
            )
        
        # Trim history if too long
        if len(self.decision_history) > self.max_history_len:
            self.decision_history.pop(0)
            self.context_history.pop(0)
    
    def get_domain_weights(self, hidden_states):
        """
        Get domain-specific weights for current input.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Domain weights for current input
        """
        # Average hidden states across sequence
        avg_hidden = torch.mean(hidden_states, dim=1)
        
        # Project to domain space and get weights
        domain_logits = self.domain_projection(avg_hidden)
        domain_weights = F.softmax(domain_logits, dim=-1)
        
        return domain_weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        history_weight: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the context-aware decider.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            history_weight: Weight to give to decision history (0.0-1.0)
            
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
        
        # Get domain-specific weights
        domain_weights = self.get_domain_weights(hidden_states)
        
        # Use attention to aggregate context information
        context_features, attention_weights = self.context_attention(
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
        
        # Apply domain-specific adaptation
        domain_feature = torch.matmul(domain_weights, self.domain_specific)
        domain_feature = domain_feature.view(batch_size, -1)
        
        # Adjust features based on decision history if available
        history_features = None
        if self.decision_history and history_weight > 0:
            # Convert history to tensor
            history_tensor = torch.cat(self.context_history, dim=0).to(hidden_states.device)
            
            # Calculate similarity to current features
            similarity = F.cosine_similarity(
                last_token_features.unsqueeze(1),
                history_tensor.unsqueeze(0),
                dim=2
            )
            
            # Weight by recency (more recent = higher weight)
            recency_weights = torch.linspace(
                0.5, 1.0, len(self.decision_history), 
                device=hidden_states.device
            )
            
            weighted_similarity = similarity * recency_weights.unsqueeze(0)
            
            # Get weighted average of history features
            attention_weights = F.softmax(weighted_similarity, dim=1)
            history_features = torch.matmul(
                attention_weights.unsqueeze(1), 
                history_tensor
            ).squeeze(1)
            
            # Combine with current features
            last_token_features = (
                (1 - history_weight) * last_token_features + 
                history_weight * history_features
            )
        
        # Make the decision
        decision_input = last_token_features
        decision_logits = self.decision_network(decision_input)
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        # Get decision as argmax
        decision = torch.argmax(decision_probs, dim=-1)
        
        # Update history if not in training mode
        if not self.training or not self.training_decision_model:
            with torch.no_grad():
                self.update_history(decision, last_token_features)
        
        # Return with detailed information
        return {
            "logits": decision_logits,
            "probabilities": decision_probs,
            "decision": decision,
            "image_prob": decision_probs[:, ModalityType.IMAGE.value],
            "text_prob": decision_probs[:, ModalityType.TEXT.value],
            "attention_weights": attention_weights,
            "domain_weights": domain_weights,
            "context_features": last_token_features
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
        
    def compute_loss(self, 
                    decision_output: Dict[str, torch.Tensor], 
                    targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for training the decision model.
        
        Args:
            decision_output: Output from the forward pass
            targets: Target labels (1 for image, 0 for text)
            
        Returns:
            Loss value
        """
        # Get logits
        logits = decision_output["logits"]
        
        # Convert targets to one-hot
        if targets.dim() == 1:
            batch_size = targets.size(0)
            target_one_hot = torch.zeros(
                batch_size, 2, 
                device=logits.device
            )
            target_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets = target_one_hot
        
        # Compute BCE loss
        loss = self.decision_loss_fn(logits, targets)
        
        return loss
    
    def train_on_examples(self, 
                         hidden_states: torch.Tensor,
                         token_type_ids: torch.Tensor,
                         targets: torch.Tensor,
                         optimizer: torch.optim.Optimizer,
                         attention_mask: Optional[torch.Tensor] = None) -> float:
        """
        Train the decision model on examples.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            targets: Target labels (1 for image, 0 for text)
            optimizer: Optimizer to use for training
            attention_mask: Attention mask of shape [batch_size, seq_len]
            
        Returns:
            Training loss
        """
        # Set to training mode
        self.train()
        self.training_decision_model = True
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        decision_output = self.forward(
            hidden_states, 
            token_type_ids, 
            attention_mask
        )
        
        # Compute loss
        loss = self.compute_loss(decision_output, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Return loss value
        return loss.item()


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
        
        # Enhanced combination methods
        self.use_adaptive_weighting = getattr(config.mvot, "use_adaptive_weighting", True)
        self.use_context_boost = getattr(config.mvot, "use_context_boost", True)
        
        # Maximum number of images to generate
        self.max_images = getattr(config.mvot, "max_images", 5)
        
        # Minimum tokens between images
        self.min_tokens_between_images = getattr(config.mvot, "min_tokens_between_images", 20)
        
        # Adaptive spacing (allow more frequent images during key reasoning sections)
        self.adaptive_spacing = getattr(config.mvot, "adaptive_token_spacing", True)
        self.min_spacing_reduction_factor = getattr(config.mvot, "min_spacing_reduction_factor", 0.5)
        
        # Domain-specific adjustments
        self.domain_sensitive = getattr(config.mvot, "domain_sensitive_decisions", True)
        self.domain_biases = {
            # Domain idx 0: General text (neutral)
            0: 0.0,
            # Domain idx 1: Scientific/technical (favor visualization)
            1: 0.15,
            # Domain idx 2: Abstract reasoning (favor visualization)
            2: 0.1,
            # Domain idx 3: Narrative/conversational (less visualization)
            3: -0.1
        }
        
        # Decision history
        self.decision_history = []
        self.max_history_len = 10
        
        # Visualization quality tracking
        self.visualization_quality = []
        self.use_quality_feedback = getattr(config.mvot, "use_visualization_quality_feedback", True)
    
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
    
    def update_decision_history(self, decision: bool, scores: Dict[str, Any]):
        """
        Update decision history with new decision.
        
        Args:
            decision: Whether to generate an image
            scores: Scores used for the decision
        """
        self.decision_history.append({
            "decision": decision,
            "scores": scores
        })
        
        # Trim history if too long
        if len(self.decision_history) > self.max_history_len:
            self.decision_history.pop(0)
    
    def provide_quality_feedback(self, quality_score: float, decision_idx: int = -1):
        """
        Provide feedback on the quality of a generated visualization.
        
        Args:
            quality_score: Quality score (0.0 to 1.0)
            decision_idx: Index in decision history (-1 for most recent)
        """
        self.visualization_quality.append(quality_score)
        
        # Limit size of quality history
        if len(self.visualization_quality) > self.max_history_len:
            self.visualization_quality.pop(0)
            
    def get_adaptive_weights(self, neural_decision: Dict[str, torch.Tensor], 
                           heuristic_assessment: Dict[str, Any]) -> Tuple[float, float]:
        """
        Get adaptive weights for neural and heuristic components.
        
        Args:
            neural_decision: Output from neural decision model
            heuristic_assessment: Output from heuristic assessment
            
        Returns:
            Tuple of (neural_weight, heuristic_weight)
        """
        if not self.use_adaptive_weighting:
            return self.neural_weight, self.heuristic_weight
        
        # Start with default weights
        neural_w = self.neural_weight
        heuristic_w = self.heuristic_weight
        
        # Confidence-based adjustment
        # If neural network is very confident (probs near 0 or 1), increase its weight
        if "probabilities" in neural_decision:
            probs = neural_decision["probabilities"]
            confidence = torch.max(probs, dim=-1)[0].mean().item()
            
            if confidence > 0.85:
                neural_w += 0.1  # Increase neural weight when very confident
                heuristic_w -= 0.1
        
        # If heuristic detected strong signals, increase its weight
        if isinstance(heuristic_assessment, dict):
            # Strong pattern match
            if heuristic_assessment.get("pattern_score", 0) > 0.75:
                heuristic_w += 0.2
                neural_w -= 0.2
            
            # Strong feature match
            strong_signals = (
                heuristic_assessment.get("spatial_score", 0) > 0.3 or
                heuristic_assessment.get("visual_score", 0) > 0.3 or
                heuristic_assessment.get("math_technical_score", 0) > 0.25
            )
            
            if strong_signals:
                heuristic_w += 0.15
                neural_w -= 0.15
        
        # Use historical quality information if available
        if self.use_quality_feedback and len(self.visualization_quality) > 3:
            avg_quality = sum(self.visualization_quality) / len(self.visualization_quality)
            
            # If visualizations have been high quality, trust the system more
            if avg_quality > 0.7:
                neural_w *= 1.1
                heuristic_w *= 1.1
            elif avg_quality < 0.4:
                # Poor quality visualizations, be more conservative
                neural_w *= 0.9
                heuristic_w *= 0.9
        
        # Normalize weights to sum to 1.0
        total = neural_w + heuristic_w
        return neural_w / total, heuristic_w / total
        
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
        
        # Get neural network-based decision with history awareness
        neural_decision = self.context_decider(
            hidden_states, 
            token_type_ids, 
            attention_mask,
            history_weight=0.2  # Use context history in decision
        )
        
        # Initialize heuristic decision
        heuristic_decision = {
            "visualization_recommended": torch.zeros(batch_size, dtype=torch.bool, device=hidden_states.device)
        }
        heuristic_assessment = None
        
        # If input text is provided, use heuristic assessment
        if input_text is not None:
            # For batched input, process each example
            if isinstance(input_text, list):
                for i, text in enumerate(input_text):
                    # Enhanced assessment with phrases and patterns
                    assessment = self.visualization_assessor.assess_text_for_visualization(text)
                    explicit_request = self.visualization_assessor.detect_diagram_request(text)
                    
                    # Store assessment for first example (for adaptive weighting)
                    if i == 0:
                        heuristic_assessment = assessment
                    
                    # Set visualization recommended flag
                    heuristic_decision["visualization_recommended"][i] = (
                        assessment["visualization_recommended"] or explicit_request
                    )
            else:
                # Single input - enhanced assessment
                assessment = self.visualization_assessor.assess_text_for_visualization(input_text)
                explicit_request = self.visualization_assessor.detect_diagram_request(input_text)
                
                # Store assessment (for adaptive weighting)
                heuristic_assessment = assessment
                
                # Set visualization recommended flag for all examples in batch
                heuristic_decision["visualization_recommended"][:] = (
                    assessment["visualization_recommended"] or explicit_request
                )
        
        # Get domain-specific adjustments if enabled
        domain_bias = 0.0
        if self.domain_sensitive and "domain_weights" in neural_decision:
            domain_weights = neural_decision["domain_weights"]
            
            # Apply domain-specific biases
            for domain_idx, bias in self.domain_biases.items():
                if domain_idx < domain_weights.shape[1]:
                    # Weight the bias by the domain probability
                    domain_bias += bias * domain_weights[0, domain_idx].item()
        
        # Get adaptive weights based on confidence and context
        neural_weight, heuristic_weight = self.get_adaptive_weights(
            neural_decision, heuristic_assessment
        )
        
        # Combine decisions based on strategy
        if self.decision_strategy == "neural":
            # Use only neural network-based decision
            generate_image = neural_decision["image_prob"] > (self.context_decider.image_threshold - domain_bias)
        elif self.decision_strategy == "heuristic":
            # Use only heuristic-based decision
            generate_image = heuristic_decision["visualization_recommended"]
        else:  # "hybrid"
            # Combine both decisions with adaptive weights
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
            
            # Apply adaptive weighting
            combined_score = (
                neural_score * neural_weight + 
                heuristic_score * heuristic_weight
            )
            
            # Apply domain bias adjustment
            threshold = 0.5 - domain_bias
            
            generate_image = combined_score > threshold
        
        # Apply constraints
        # 1. Maximum number of images
        max_images_constraint = image_count < self.max_images
        
        # 2. Adaptive minimum tokens between images
        min_tokens_required = self.min_tokens_between_images
        
        # If adaptive spacing is enabled, adjust based on context
        if self.adaptive_spacing and heuristic_assessment is not None:
            # For highly visual or technical content, allow more frequent visualization
            importance_score = max(
                heuristic_assessment.get("visual_score", 0),
                heuristic_assessment.get("math_technical_score", 0),
                heuristic_assessment.get("pattern_score", 0)
            )
            
            if importance_score > 0.3:
                # Reduce the spacing for important visualizations, but not below 50%
                reduction_factor = max(
                    1.0 - importance_score, 
                    self.min_spacing_reduction_factor
                )
                min_tokens_required = int(min_tokens_required * reduction_factor)
        
        if tokens_since_last_image is not None:
            if isinstance(tokens_since_last_image, torch.Tensor):
                min_tokens_constraint = tokens_since_last_image > min_tokens_required
            else:
                min_tokens_constraint = torch.tensor(
                    [tokens_since_last_image > min_tokens_required] * batch_size,
                    device=hidden_states.device
                )
        else:
            min_tokens_constraint = torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)
        
        # Final decision
        final_decision = generate_image & max_images_constraint & min_tokens_constraint
        
        # Store decision history for future context
        if heuristic_assessment is not None:
            self.update_decision_history(
                final_decision[0].item() if isinstance(final_decision, torch.Tensor) else final_decision,
                heuristic_assessment
            )
        
        # Return detailed decision information
        result = {
            "should_generate_image": final_decision,
            "neural_decision": neural_decision,
            "heuristic_decision": heuristic_decision,
            "image_count": image_count,
            "max_images_constraint": max_images_constraint,
            "min_tokens_constraint": min_tokens_constraint,
            "neural_weight": neural_weight,
            "heuristic_weight": heuristic_weight,
            "domain_bias": domain_bias
        }
        
        # Add heuristic assessment details if available
        if heuristic_assessment is not None:
            result["assessment_details"] = heuristic_assessment
            
        return result
    
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
    
    def reset_history(self):
        """Reset decision history and visualization quality tracking."""
        self.decision_history = []
        self.visualization_quality = []
        
        # Also reset component histories
        self.context_decider.reset_history()
        self.visualization_assessor.reset_history()


def create_decision_mechanism(config) -> GenerationDecisionMechanism:
    """
    Factory function to create a generation decision mechanism.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        GenerationDecisionMechanism instance
    """
    return GenerationDecisionMechanism(config)