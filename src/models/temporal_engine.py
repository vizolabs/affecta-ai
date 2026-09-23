"""Temporal smoothing and transition detection."""

from typing import List, Dict, Optional
from collections import deque


class TemporalEngine:
    """Temporal smoothing and expression transition detection.
    
    Features:
        - Exponential Moving Average (EMA) smoothing
        - Transition detection
        - Expression persistence tracking
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        history_size: int = 10,
        transition_threshold: float = 0.2
    ):
        """Initialize temporal engine.
        
        Args:
            alpha: EMA smoothing factor (0-1)
            history_size: Number of frames to keep in history
            transition_threshold: Threshold for transition detection
        """
        self.alpha = alpha
        self.history_size = history_size
        self.transition_threshold = transition_threshold
    
    def smooth(
        self,
        history: List[Dict[str, float]],
        current: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply EMA smoothing to emotion probabilities.
        
        Args:
            history: List of historical probability dicts
            current: Current probability dict
        
        Returns:
            Smoothed probabilities
        """
        if not history:
            return current
        
        smoothed = {}
        for emotion in current.keys():
            # Compute EMA
            ema = history[0].get(emotion, 0)
            for h in history[1:]:
                ema = self.alpha * h.get(emotion, 0) + (1 - self.alpha) * ema
            
            smoothed[emotion] = self.alpha * current[emotion] + (1 - self.alpha) * ema
        
        # Normalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        
        return smoothed
    
    def detect_transition(
        self,
        previous_emotion: Optional[str],
        current_emotion: str
    ) -> Optional[Dict]:
        """Detect expression transition.
        
        Args:
            previous_emotion: Previous dominant emotion
            current_emotion: Current dominant emotion
        
        Returns:
            Transition dict if detected, None otherwise
        """
        if previous_emotion is None or previous_emotion == current_emotion:
            return None
        
        return {
            'type': 'transition',
            'from': previous_emotion,
            'to': current_emotion
        }
    
    def compute_temporal_evidence(
        self,
        history: List[Dict[str, float]],
        current: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute temporal evidence for each emotion.
        
        Args:
            history: Historical probability dicts
            current: Current probabilities
        
        Returns:
            Dict of emotion -> temporal evidence
        """
        if len(history) < 2:
            return {e: 0.0 for e in current.keys()}
        
        temporal = {}
        for emotion in current.keys():
            # Compute historical average
            hist_avg = sum(h.get(emotion, 0) for h in history) / len(history)
            
            # Temporal evidence = current - historical
            temporal[emotion] = current[emotion] - hist_avg
        
        return temporal


def get_temporal_engine(
    alpha: float = 0.3,
    history_size: int = 10
) -> TemporalEngine:
    """Factory function to get temporal engine."""
    return TemporalEngine(alpha=alpha, history_size=history_size)
