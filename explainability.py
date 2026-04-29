from __future__ import annotations

import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from typing import Tuple

from utils import DEVICE, EVAL_TRANSFORM


class GradCAM:
    """
    Hooks into MobileNetV2 features[-1] (last conv block)
    to capture activations and gradients during inference.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._handles    = []
        target = self.model.features[-1]
        self._handles.append(target.register_forward_hook(self._save_act))
        self._handles.append(target.register_full_backward_hook(self._save_grad))

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    @torch.enable_grad()
    def generate(
        self,
        image: Image.Image,
        target_class: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, float, list]:
        """
        Returns (heatmap_rgb_uint8, cam_raw_01, pred_class, confidence, all_probs)
        """
        self.model.eval()
        tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        logits    = self.model(tensor)
        probs     = torch.softmax(logits, dim=1).squeeze()
        all_probs = probs.detach().cpu().tolist()

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[target_class].item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        w, h    = image.size
        cam_raw = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

        heatmap = cv2.applyColorMap(
            (cam_raw * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return heatmap, cam_raw, target_class, confidence, all_probs


def overlay_heatmap(
    original: Image.Image,
    heatmap : np.ndarray,
    alpha   : float = 0.45,
) -> Image.Image:
    """Alpha-blend GradCAM heatmap onto the original image."""
    orig_np = np.array(original.convert("RGB")).astype(np.float32)
    heat_np = heatmap.astype(np.float32)
    blended = np.clip((1 - alpha) * orig_np + alpha * heat_np, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def cam_region_description(cam_raw: np.ndarray) -> str:
    """
    Convert a raw CAM (0-1 float) to human-readable text about
    which spatial region had highest activation. Used in the VLM prompt.
    """
    h, w = cam_raw.shape
    regions = {
        "top-left":      cam_raw[:h//3,       :w//3      ],
        "top-centre":    cam_raw[:h//3,        w//3:2*w//3],
        "top-right":     cam_raw[:h//3,        2*w//3:    ],
        "centre-left":   cam_raw[h//3:2*h//3,  :w//3      ],
        "centre":        cam_raw[h//3:2*h//3,  w//3:2*w//3],
        "centre-right":  cam_raw[h//3:2*h//3,  2*w//3:    ],
        "bottom-left":   cam_raw[2*h//3:,      :w//3      ],
        "bottom-centre": cam_raw[2*h//3:,      w//3:2*w//3],
        "bottom-right":  cam_raw[2*h//3:,      2*w//3:    ],
    }
    scores = {k: float(v.mean()) for k, v in regions.items()}
    top3   = sorted(scores, key=scores.get, reverse=True)[:3]
    if scores[top3[0]] < 0.15:
        return "diffuse activation across the entire lesion"
    return f"the {top3[0]}, {top3[1]}, and {top3[2]} regions"


def explain_prediction(model: torch.nn.Module, image: Image.Image, alpha: float = 0.45) -> dict:
    """
    One-call wrapper. Returns dict with keys:
      pred_class, confidence, all_probs,
      heatmap_pil, overlay_pil, cam_raw, region_desc
    """
    gcam = GradCAM(model)
    try:
        heatmap, cam_raw, pred_class, confidence, all_probs = gcam.generate(image)
        overlay = overlay_heatmap(image, heatmap, alpha=alpha)
        return {
            "pred_class":  pred_class,
            "confidence":  confidence,
            "all_probs":   all_probs,
            "heatmap_pil": Image.fromarray(heatmap),
            "overlay_pil": overlay,
            "cam_raw":     cam_raw,
            "region_desc": cam_region_description(cam_raw),
        }
    finally:
        gcam.remove_hooks()
