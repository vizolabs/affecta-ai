"""Real-time inference pipeline."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from src.core.config import InferenceConfig, EMOTIONS
from src.models.expression_classifier import get_expression_model
from src.models.au_detector import get_au_detector
from src.models.ranking_engine import get_ranking_engine
from src.models.intensity_estimator import get_intensity_estimator
from src.models.uncertainty_engine import get_uncertainty_engine
from src.models.temporal_engine import get_temporal_engine
from ml.xai.lrp import get_lrp_engine
from ml.xai.region_mapping import aggregate_relevance_by_region


class InferencePipeline:
    """Real-time facial expression analysis pipeline.
    
    Stages:
        1. Face Detection (MTCNN/RetinaFace)
        2. Preprocessing
        3. Expression Classification (VGG16/ResNet50/EfficientNetB2)
        4. AU Detection (ResNet-18)
        5. Ranking Engine
        6. Intensity Estimation
        7. Uncertainty Detection
        8. XAI (LRP/Grad-CAM) - on-demand
        9. Temporal Smoothing
        10. Commentary Generation
    """
    
    def __init__(self, config: InferenceConfig = None):
        """Initialize pipeline.
        
        Args:
            config: Inference configuration
        """
        if config is None:
            config = InferenceConfig()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._init_models()
        
        # State
        self.face_history: Dict[int, deque] = {}
        self.previous_emotions: Dict[int, str] = {}
        self.frame_count = 0
    
    def _init_models(self):
        """Initialize all models."""
        # Expression classifier
        self.expression_model = get_expression_model(
            backbone='vgg16',
            pretrained=False
        ).to(self.device)
        
        # AU detector
        self.au_model = get_au_detector(pretrained=False).to(self.device)
        
        # Ranking engine
        self.ranking_engine = get_ranking_engine()
        
        # Intensity estimator
        self.intensity_estimator = get_intensity_estimator().to(self.device)
        
        # Uncertainty engine
        self.uncertainty_engine = get_uncertainty_engine(
            margin_threshold=self.config.margin_threshold,
            probability_threshold=self.config.probability_threshold
        )
        
        # Temporal engine
        self.temporal_engine = get_temporal_engine(
            alpha=self.config.temporal_smoothing_alpha
        )
        
        # LRP engine (on-demand)
        self.lrp_engine = get_lrp_engine()
    
    def load_model_weights(
        self,
        expression_path: str,
        au_path: str
    ):
        """Load trained model weights.
        
        Args:
            expression_path: Path to expression model weights
            au_path: Path to AU model weights
        """
        self.expression_model.load_state_dict(
            torch.load(expression_path, map_location=self.device)
        )
        self.au_model.load_state_dict(
            torch.load(au_path, map_location=self.device)
        )
        
        self.expression_model.eval()
        self.au_model.eval()
    
    def process_frame(
        self,
        frame: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        landmarks: Optional[List[np.ndarray]] = None
    ) -> List[Dict]:
        """Process a single frame.
        
        Args:
            frame: RGB frame [H, W, 3]
            face_locations: List of (top, right, bottom, left)
            landmarks: Optional list of [68, 2] landmarks per face
        
        Returns:
            List of analysis results per face
        """
        self.frame_count += 1
        results = []
        
        for face_idx, (top, right, bottom, left) in enumerate(face_locations):
            # Crop and preprocess face
            face_img = frame[top:bottom, left:right]
            face_tensor = self._preprocess(face_img)
            
            # Stage 1-2: Classification and AU detection
            expression_probs = self._classify_expression(face_tensor)
            au_activations = self._detect_aus(face_tensor)
            
            # Stage 3: Ranking
            evidence_scores = self.ranking_engine.compute_preliminary(
                expression_probs, au_activations
            )
            
            # Stage 4: Intensity
            intensity = self._estimate_intensity(face_tensor, au_activations)
            
            # Stage 5: Uncertainty
            uncertainty_state, margin = self.uncertainty_engine.compute(
                expression_probs.tolist()
            )
            
            # Stage 6: Temporal smoothing
            if face_idx not in self.face_history:
                self.face_history[face_idx] = deque(maxlen=10)
            
            smoothed_probs = self.temporal_engine.smooth(
                list(self.face_history[face_idx]),
                {e: p for e, p in zip(EMOTIONS, expression_probs.tolist())}
            )
            self.face_history[face_idx].append(smoothed_probs)
            
            # Get dominant emotion
            dominant_emotion = max(smoothed_probs, key=smoothed_probs.get)
            
            # Stage 7: Transition detection
            transition = None
            if face_idx in self.previous_emotions:
                transition = self.temporal_engine.detect_transition(
                    self.previous_emotions[face_idx],
                    dominant_emotion
                )
            self.previous_emotions[face_idx] = dominant_emotion
            
            # Stage 8: XAI (on-demand)
            xai_result = None
            if self._should_compute_xai():
                xai_result = self._compute_xai(face_tensor, dominant_emotion)
            
            # Build result
            result = {
                'face_id': face_idx,
                'bounding_box': (top, right, bottom, left),
                'dominant_emotion': dominant_emotion,
                'emotion_probabilities': smoothed_probs,
                'au_activations': {
                    f'AU{i+1}': float(au_activations[i])
                    for i in range(len(au_activations))
                },
                'evidence_scores': evidence_scores,
                'intensity': {
                    'value': float(intensity),
                    'category': self.intensity_estimator.get_category(float(intensity))
                },
                'uncertainty': {
                    'state': uncertainty_state,
                    'margin': float(margin),
                    'display': self.uncertainty_engine.get_display(uncertainty_state)
                },
                'transition': transition,
                'xai': xai_result
            }
            
            results.append(result)
        
        return results
    
    def _preprocess(self, face_img: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        import cv2
        
        # Resize
        face_img = cv2.resize(face_img, (224, 224))
        
        # Normalize
        face_img = face_img.astype(np.float32) / 255.0
        face_img = (face_img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Convert to tensor
        face_tensor = torch.FloatTensor(face_img.transpose(2, 0, 1))
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        return face_tensor
    
    def _classify_expression(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Classify facial expression."""
        with torch.no_grad():
            output = self.expression_model(face_tensor)
            probs = torch.softmax(output, dim=1)
        return probs.squeeze()
    
    def _detect_aus(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Detect Action Units."""
        with torch.no_grad():
            output = self.au_model(face_tensor)
        return output.squeeze()
    
    def _estimate_intensity(
        self,
        face_tensor: torch.Tensor,
        au_activations: torch.Tensor
    ) -> torch.Tensor:
        """Estimate expression intensity."""
        # Get features from expression model
        features = self.expression_model.get_features(face_tensor)
        
        with torch.no_grad():
            intensity = self.intensity_estimator(features, au_activations.unsqueeze(0))
        return intensity.squeeze()
    
    def _should_compute_xai(self) -> bool:
        """Determine if XAI should be computed this frame."""
        return self.frame_count % self.config.xai_frame_interval == 0
    
    def _compute_xai(
        self,
        face_tensor: torch.Tensor,
        target_emotion: str
    ) -> Optional[Dict]:
        """Compute XAI explanation."""
        try:
            target_class = EMOTIONS.index(target_emotion)
            
            # LRP
            relevance_map = self.lrp_engine.compute(
                self.expression_model,
                face_tensor,
                target_class
            )
            
            return {
                'type': 'lrp',
                'relevance_map': relevance_map.cpu().numpy().tolist(),
                'target_emotion': target_emotion
            }
        except Exception as e:
            return {'type': 'lrp', 'error': str(e)}
    
    def generate_commentary(self, results: List[Dict]) -> str:
        """Generate natural language commentary for results."""
        if not results:
            return "No faces detected."
        
        commentaries = []
        
        for result in results:
            emotion = result['dominant_emotion']
            intensity = result['intensity']['value']
            state = result['uncertainty']['state']
            
            if state == 'ambiguous':
                commentaries.append(
                    f"Face {result['face_id']}: Expression appears ambiguous "
                    f"between {emotion} and other emotions."
                )
            elif state == 'insufficient':
                commentaries.append(
                    f"Face {result['face_id']}: Insufficient evidence to "
                    f"confidently determine expression."
                )
            else:
                commentaries.append(
                    f"Face {result['face_id']}: {emotion.capitalize()} "
                    f"detected with {intensity:.0f}% intensity."
                )
            
            if result['transition']:
                t = result['transition']
                commentaries.append(
                    f"  → Transition: {t['from'].capitalize()} → {t['to'].capitalize()}"
                )
        
        return "\n".join(commentaries)


def get_pipeline(config: InferenceConfig = None) -> InferencePipeline:
    """Factory function to get inference pipeline."""
    return InferencePipeline(config)
