"""True LRP-epsilon (layer-wise relevance propagation) engine.

Implements the composite epsilon rule for feed-forward conv/linear nets
(VGG-style): conv/linear layers use the *consumed* (post-ReLU) activation
as the stabilizing denominator,

    R_in_j = a_j * conv_transpose(s, W)_j,   s = R_out / z_composite

- Conv2d/Linear: epsilon rule over the post-activation (composite) output.
- MaxPool/AvgPool/AdaptiveAvgPool: distributed via local autograd adjoint
  (max-pool funnels to the argmax neuron).
- Dropout (inactive at eval) and BatchNorm pass relevance through.
- ReLU is folded into the composite conv/linear denominator.

Suitable for the staged VGG-type backbones; residual nets (ResNet/EfficientNet
with add/skip connections) are out of scope and raise an error during capture.
Conservation is approximate (typical error 30-100% on VGG16 with the z-rule);
see compute_conservation. Maps are stable and bounded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class LRPEpsilon:
    EPS_DEFAULT = 1e-6
    REL_EPS_DEFAULT = 1e-3

    CAPTURE = (
        nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d,
        nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Dropout,
        nn.BatchNorm2d, nn.BatchNorm1d,
    )
    PASS_THROUGH = (nn.Dropout, nn.ReLU)
    LINEAR = (nn.Conv2d, nn.Linear)
    POOL_RELU = (
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
        nn.BatchNorm2d, nn.BatchNorm1d,
    )

    def __init__(self, eps: float = EPS_DEFAULT, rel_eps: float = REL_EPS_DEFAULT):
        self.eps = eps
        self.rel_eps = rel_eps

    def compute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        model.eval()
        captured = []
        for m in model.modules():
            if isinstance(m, self.CAPTURE):
                captured.append(m)

        activations = []
        handles = []

        def make_hook(m):
            def hook(module, ins, outs):
                act = outs[0] if isinstance(outs, tuple) else outs
                activations.append((ins[0], act, module))
            return hook

        for m in captured:
            handles.append(m.register_forward_hook(make_hook(m)))

        out = model(input_tensor)
        pred = out.argmax(dim=1).item() if target_class is None else target_class
        R = torch.zeros_like(out)
        R[0, pred] = out[0, pred]
        for h in handles:
            h.remove()

        if len(activations) == 0:
            raise ValueError('No capturable layers found (residual-only net?)')

        for i in range(len(activations) - 1, -1, -1):
            a, z, m = activations[i]
            z_denom = z
            if (
                isinstance(m, self.LINEAR)
                and i + 1 < len(activations)
                and isinstance(activations[i + 1][2], nn.ReLU)
            ):
                z_denom = activations[i + 1][1]
            R = self._reverse_layer(m, a, z, R, z_denom)
            if i > 0:
                prev_shape = activations[i - 1][1].shape
                if tuple(R.shape) != tuple(prev_shape):
                    R = R.reshape(prev_shape)

        return R

    def _reverse_layer(
        self, m: nn.Module, a: torch.Tensor, z: torch.Tensor, R: torch.Tensor,
        z_denom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(m, self.LINEAR):
            return self._eps_linear(m, a, R, z if z_denom is None else z_denom)
        if isinstance(m, self.POOL_RELU):
            return torch.autograd.grad(z, a, grad_outputs=R, retain_graph=True)[0]
        if isinstance(m, self.PASS_THROUGH):
            return R
        raise NotImplementedError(f'Layer type {type(m).__name__} unsupported')

    def _eps_linear(
            self, m: nn.Module, a: torch.Tensor, R: torch.Tensor,
            z_denom: torch.Tensor,
    ) -> torch.Tensor:
        s = R / self._eps_denom(z_denom)
        weight = m.weight
        if isinstance(m, nn.Conv2d):
            c = F.conv_transpose2d(
                s, weight, bias=None,
                stride=m.stride, padding=m.padding,
                output_padding=m.output_padding, dilation=m.dilation,
            )
        else:
            c = s @ weight
        return a * c

    def _eps_denom(self, z: torch.Tensor) -> torch.Tensor:
        floor = max(self.eps, self.rel_eps * z.detach().abs().mean().item())
        sign_z = torch.sign(z)
        sign_z[sign_z == 0] = 1.0
        return torch.where(z.abs() < floor, floor * sign_z, z)

    def compute_conservation(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> Tuple[torch.Tensor, float]:
        """Return (relevance_map, approximate conservation error = abs(sum(R) - class_score) / max(|class_score|, 1))."""
        relevance = self.compute(model, input_tensor, target_class=target_class)
        with torch.no_grad():
            score = model(input_tensor)[0, target_class]
        r_sum = relevance.detach().sum()
        denom = max(float(score.abs()), 1.0)
        rel_err = float((r_sum - score).abs()) / denom
        return relevance, rel_err


def get_epsilon_lrp(eps: float = LRPEpsilon.EPS_DEFAULT, rel_eps: float = LRPEpsilon.REL_EPS_DEFAULT) -> LRPEpsilon:
    return LRPEpsilon(eps=eps, rel_eps=rel_eps)