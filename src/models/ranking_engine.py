"""Evidence-scored ranking engine."""

from typing import Dict, Optional, List
import numpy as np

from src.core.config import EMOTIONS, AU_TO_EMOTION


class RankingEngine:
    """Evidence-scored ranking engine for facial expression analysis.
    
    Computes emotion scores based on:
        - P_e: Classifier probability
        - A_e: AU evidence score
        - R_e: LRP regional evidence (optional)
        - T_e: Temporal evidence (optional)
    
    Formula:
        S_e = w_p * P_e + w_a * A_e + w_r * R_e + w_t * T_e
    """
    
    AU_LIST = ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12']
    
    def __init__(
        self,
        w_p: float = 0.4,
        w_a: float = 0.25,
        w_r: float = 0.25,
        w_t: float = 0.1
    ):
        """Initialize ranking engine.
        
        Args:
            w_p: Weight for classifier probability
            w_a: Weight for AU evidence
            w_r: Weight for LRP regional evidence
            w_t: Weight for temporal evidence
        """
        assert abs(w_p + w_a + w_r + w_t - 1.0) < 1e-6, "Weights must sum to 1"
        
        self.weights = {
            'w_p': w_p,
            'w_a': w_a,
            'w_r': w_r,
            'w_t': w_t
        }
    
    def compute_au_evidence(self, au_activations: List[float], emotion: str) -> float:
        """Compute AU evidence score for a specific emotion.
        
        Args:
            au_activations: List of 6 AU activations (0-1)
            emotion: Emotion label
        
        Returns:
            Evidence score (0-1)
        """
        au_weights = AU_TO_EMOTION.get(emotion, {})
        if not au_weights:
            return 0.0
        
        evidence = 0.0
        for au_name, weight in au_weights.items():
            au_idx = self.AU_LIST.index(au_name)
            evidence += weight * au_activations[au_idx]
        
        return evidence
    
    def compute_preliminary(
        self,
        probs,
        au_activations
    ) -> Dict[str, object]:
        """Compute preliminary ranking (fast path, no LRP).

        Contract consumed by the inference pipeline and tests:
        returns {'ranked_emotions': [...], 'evidence_scores': {...}}

        Args:
            probs: 7-class softmax probabilities
            au_activations: 6 AU activations

        Returns:
            Dict with ranked emotion list (descending) and per-emotion scores
        """
        probs = [float(p) for p in probs]
        au = [float(a) for a in au_activations]
        scores = self.preliminary_ranking(probs, au)
        return {
            'ranked_emotions': list(scores.keys()),
            'evidence_scores': scores,
        }

    def preliminary_ranking(
        self,
        probs: List[float],
        au_activations: List[float]
    ) -> Dict[str, float]:
        """Compute preliminary ranking (fast path, no LRP).
        
        Args:
            probs: 7-class softmax probabilities
            au_activations: 6 AU activations
        
        Returns:
            Dict of emotion -> score, sorted descending
        """
        scores = {}
        for i, emotion in enumerate(EMOTIONS):
            P_e = probs[i]
            A_e = self.compute_au_evidence(au_activations, emotion)
            
            score = self.weights['w_p'] * P_e + self.weights['w_a'] * A_e
            scores[emotion] = score
        
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return dict(sorted(scores.items(), key=lambda x: -x[1]))
    
    def final_ranking(
        self,
        probs: List[float],
        au_activations: List[float],
        region_evidence: Optional[Dict[str, float]] = None,
        temporal_evidence: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Compute final ranking (slow path, with LRP).
        
        Args:
            probs: 7-class softmax probabilities
            au_activations: 6 AU activations
            region_evidence: Dict of emotion -> regional relevance (optional)
            temporal_evidence: Dict of emotion -> temporal evidence (optional)
        
        Returns:
            Dict of emotion -> score, sorted descending
        """
        scores = {}
        for i, emotion in enumerate(EMOTIONS):
            P_e = probs[i]
            A_e = self.compute_au_evidence(au_activations, emotion)
            R_e = region_evidence.get(emotion, 0.0) if region_evidence else 0.0
            T_e = temporal_evidence.get(emotion, 0.0) if temporal_evidence else 0.0
            
            score = (
                self.weights['w_p'] * P_e +
                self.weights['w_a'] * A_e +
                self.weights['w_r'] * R_e +
                self.weights['w_t'] * T_e
            )
            scores[emotion] = score
        
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return dict(sorted(scores.items(), key=lambda x: -x[1]))
    
    def compute_region_ranking(self, region_relevance: Dict[str, float]) -> Dict[str, float]:
        """Compute region contribution ranking.
        
        Args:
            region_relevance: Dict of region -> relevance score
        
        Returns:
            Dict of region -> percentage, sorted descending
        """
        total = sum(region_relevance.values())
        if total == 0:
            return region_relevance
        
        percentages = {k: v / total * 100 for k, v in region_relevance.items()}
        return dict(sorted(percentages.items(), key=lambda x: -x[1]))
    
    def compute_au_ranking(self, au_activations: List[float]) -> Dict[str, str]:
        """Compute AU evidence ranking.
        
        Args:
            au_activations: List of 6 AU activations
        
        Returns:
            Dict of AU -> strength level
        """
        ranking = {}
        for i, au_name in enumerate(self.AU_LIST):
            activation = au_activations[i]
            if activation > 0.7:
                strength = 'Strong'
            elif activation > 0.4:
                strength = 'Moderate'
            elif activation > 0.2:
                strength = 'Weak'
            else:
                strength = 'Absent'
            ranking[au_name] = strength
        
        return dict(sorted(ranking.items(), key=lambda x: -au_activations[self.AU_LIST.index(x[0])]))


def get_ranking_engine(
    w_p: float = 0.4,
    w_a: float = 0.25,
    w_r: float = 0.25,
    w_t: float = 0.1
) -> RankingEngine:
    """Factory function to get ranking engine."""
    return RankingEngine(w_p=w_p, w_a=w_a, w_r=w_r, w_t=w_t)
