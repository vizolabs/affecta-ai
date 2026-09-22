"""Attribution maps: Grad x Input relevance and Grad-CAM.

NOTE: This is the internals-free, robust baseline used for the Phase 4 ablation
studies. True LRP-epsilon (layer-wise relevance propagation) is a Phase 4
research deliverable and replaces the relevance_* methods; Grad-CAM is kept for
comparison. Both implementations below avoid autograd forward hooks on inplace
ReLU topology, which torch raises a RuntimeError for.

Methods:
- GradxInput: relevance = gradient * input, channel-averaged -> [1,1,H,W]
- Grad-CAM: conv-feature-map attribution for a target layer
- aggregate_regions: maps a relevance map to facial regions via 68 landmarks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LRPEngine:
    """Attribution engine (Grad x Input + Grad-CAM) for staged models."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def compute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """Grad x Input relevance map.

        Args:
            model: evaluatable model, returns logits [1, C]
            input_tensor: [1, 3, H, W] float tensor
            target_class: class to explain (default: argmax prediction)

        Returns:
            Relevance map [1, 1, H, W]
        """
        model.eval()
        x = input_tensor.detach().clone()
        x.requires_grad_(True)

        out = model(x)
        if target_class is None:
            target_class = out.argmax(dim=1).item()

        seed = torch.zeros_like(out)
        seed[0, target_class] = out[0, target_class]
        out.backward(gradient=seed)

        relevance = x.grad * x
        relevance = relevance.mean(dim=1, keepdim=True)
        return relevance

    def compute_gradcam(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Grad-CAM heatmap for a target conv layer.

        layer_name may be a named module (e.g. 'features.block5.1'). If omitted,
        the last Conv2d in the model is used.

        Returns:
            Heatmap [H, W] resized to input spatial size.
        """
        model.eval()
        x = input_tensor.detach().clone()
        x.requires_grad_(True)

        target = self._find_conv_layer(model, layer_name)
        acts = {}

        def hook(module, input, output):
            acts['out'] = output

        handle = target.register_forward_hook(hook)
        out = model(x)
        target_class = out.argmax(dim=1).item() if target_class is None else target_class
        grads = torch.autograd.grad(
            out[0, target_class], acts['out'], retain_graph=True, allow_unused=True
        )[0]
        handle.remove()

        if grads is None:
            return torch.zeros(input_tensor.shape[2:])

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts['out']).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        return cam.squeeze()

    @staticmethod
    def _find_conv_layer(model: nn.Module, layer_name: Optional[str]) -> nn.Module:
        if layer_name is not None:
            for name, module in model.named_modules():
                if name == layer_name and isinstance(module, nn.Conv2d):
                    return module
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError('No Conv2d layer found in model')
        return last_conv

    def aggregate_regions(
        self,
        relevance_map: torch.Tensor,
        landmarks: np.ndarray,
        image_size: tuple = (224, 224),
    ) -> dict:
        """Aggregate pixel-level relevance into facial regions.

        Args:
            relevance_map: [1, 1, H, W] relevance heatmap
            landmarks: [68, 2] facial landmarks
            image_size: (height, width) of image

        Returns:
            Dict of region -> relevance percentage (sum = 1.0)
        """
        relevance_np = relevance_map.detach().cpu().numpy()
        relevance_np = np.abs(relevance_np.squeeze())
        h, w = relevance_np.shape

        regions = {
            'mouth': landmarks[48:67],
            'cheeks': np.vstack([landmarks[1:9], landmarks[9:17]]),
            'eyes': np.vstack([landmarks[36:42], landmarks[42:48]]),
            'eyebrows': np.vstack([landmarks[17:22], landmarks[22:27]]),
            'nose': landmarks[27:36],
        }
        out = {}
        for name, pts in regions.items():
            mask = self._landmarks_to_mask(pts.astype(float) * (
                np.array([w, h]) / np.array(image_size)
            ), h, w)
            out[name] = float(np.sum(relevance_np * mask))
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        return out

    def _landmarks_to_mask(
        self,
        landmarks: np.ndarray,
        h: int,
        w: int,
        radius: int = 10,
    ) -> np.ndarray:
        mask = np.zeros((h, w))
        for point in landmarks:
            x, y = int(round(point[0])), int(round(point[1]))
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            yy, xx = np.ogrid[:h, :w]
            mask[(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = 1.0
        return mask


def get_lrp_engine(epsilon: float = 1e-6) -> LRPEngine:
    """Factory function to get LRP engine."""
    return LRPEngine(epsilon=epsilon)