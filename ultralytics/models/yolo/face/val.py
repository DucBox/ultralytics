# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.pose import PoseValidator

# Calibrated OKS sigma for 5 face landmarks.
# Derived from annotation variability of face landmarks (similar to COCO eyes/nose sigma).
# With sigma=0.025 and a 100px face: OKS=0.5 when landmark is ~3px off (vs ~23px with uniform 0.2).
# Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth
FACE5_SIGMA = np.array([0.025, 0.025, 0.026, 0.035, 0.035], dtype=np.float32)


class FaceValidator(PoseValidator):
    """Validator for YOLO face detection models with 5-point landmark evaluation.

    Extends PoseValidator with face-specific configuration: uses uniform OKS sigmas
    for 5 face landmarks, skips COCO JSON evaluation (no WIDER Face annotation JSON),
    and reports metrics as face-specific labels.

    Attributes:
        args (dict): Validator arguments with task='face'.
        kpt_shape (list): Shape of face landmarks, e.g. [5, 3].
        sigma (np.ndarray): Calibrated OKS sigma values for 5 face landmarks (FACE5_SIGMA).

    Examples:
        >>> from ultralytics.models.yolo.face import FaceValidator
        >>> args = dict(model="yolo11n-face.pt", data="widerface.yaml")
        >>> validator = FaceValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize FaceValidator for face detection with landmarks.

        Args:
            dataloader: DataLoader to be used for validation.
            save_dir: Directory to save validation results.
            args (dict, optional): Validator arguments.
            _callbacks (dict, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "face"

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize face validation metrics with calibrated OKS sigma for 5 face landmarks."""
        super().init_metrics(model)
        self.sigma = FACE5_SIGMA  # override uniform sigma=0.2 with calibrated face landmark sigma

    def get_desc(self) -> str:
        """Return description of face evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Lmk(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return stats without COCO JSON evaluation (no WIDER Face annotation JSON available).

        Args:
            stats (dict): Computed validation statistics.

        Returns:
            (dict): Stats unchanged (skipping COCO eval).
        """
        return stats
