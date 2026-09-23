"""Uncertainty and ambiguity detection engine."""

from typing import Tuple, List


class UncertaintyEngine:
    """Detects when the model cannot reliably classify the expression.
    
    States:
        - CERTAIN: High confidence, clear margin
        - AMBIGUOUS: Low margin between top-2 emotions
        - INSUFFICIENT: Low max probability
    """
    
    def __init__(
        self,
        margin_threshold: float = 0.15,
        probability_threshold: float = 0.30
    ):
        """Initialize uncertainty engine.
        
        Args:
            margin_threshold: Minimum gap between top-2 for certainty
            probability_threshold: Minimum probability for certainty
        """
        self.margin_threshold = margin_threshold
        self.probability_threshold = probability_threshold
    
    def compute(
        self,
        probs: List[float]
    ) -> Tuple[str, float]:
        """Compute uncertainty state from emotion probabilities.
        
        Args:
            probs: 7 softmax probabilities
        
        Returns:
            Tuple of (state, margin)
        """
        sorted_probs = sorted(probs, reverse=True)
        top1 = sorted_probs[0]
        top2 = sorted_probs[1]
        
        margin = top1 - top2
        
        if top1 < self.probability_threshold:
            return 'insufficient', margin
        elif margin < self.margin_threshold:
            return 'ambiguous', margin
        else:
            return 'certain', margin
    
    def optimize_thresholds(
        self,
        val_probs: List[List[float]],
        val_targets: List[int]
    ) -> Tuple[float, float]:
        """Optimize thresholds on validation set.
        
        Args:
            val_probs: List of probability vectors
            val_targets: List of true emotion indices
        
        Returns:
            Optimal (margin_threshold, probability_threshold)
        """
        best_threshold = (0.15, 0.40)
        best_score = -1
        
        for margin_thresh in [0.10, 0.15, 0.20, 0.25]:
            for prob_thresh in [0.30, 0.35, 0.40, 0.45]:
                self.margin_threshold = margin_thresh
                self.probability_threshold = prob_thresh
                
                correct_certain = 0
                total_certain = 0
                total = len(val_probs)
                
                for probs, target in zip(val_probs, val_targets):
                    state, _ = self.compute(probs)
                    if state == 'certain':
                        total_certain += 1
                        pred = probs.index(max(probs))
                        if pred == target:
                            correct_certain += 1
                
                if total_certain > 0:
                    acc_certain = correct_certain / total_certain
                    abstention_rate = 1 - total_certain / total
                    score = acc_certain - 0.5 * abstention_rate
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = (margin_thresh, prob_thresh)
        
        return best_threshold
    
    @staticmethod
    def get_display(state: str) -> str:
        """Get display text for uncertainty state."""
        displays = {
            'certain': '✓ CERTAIN',
            'ambiguous': '⚠ AMBIGUOUS',
            'insufficient': '⚠ INSUFFICIENT EVIDENCE'
        }
        return displays.get(state, 'UNKNOWN')


def get_uncertainty_engine(
    margin_threshold: float = 0.15,
    probability_threshold: float = 0.30
) -> UncertaintyEngine:
    """Factory function to get uncertainty engine."""
    return UncertaintyEngine(
        margin_threshold=margin_threshold,
        probability_threshold=probability_threshold
    )
