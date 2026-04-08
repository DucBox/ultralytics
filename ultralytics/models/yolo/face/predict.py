# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG


class FacePredictor(PosePredictor):
    """Predictor for YOLO face detection models with 5-point landmark output.

    Extends PosePredictor with face task configuration. Inherits all keypoint coordinate decoding and scaling from
    PosePredictor, which already handles arbitrary kpt_shape values.

    Attributes:
        args (namespace): Configuration arguments with task='face'.

    Examples:
        >>> from ultralytics.models.yolo.face import FacePredictor
        >>> from ultralytics.utils import ASSETS
        >>> args = dict(model="yolo11n-face.pt", source=ASSETS)
        >>> predictor = FacePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize FacePredictor for face detection with landmark output.

        Args:
            cfg: Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "face"
