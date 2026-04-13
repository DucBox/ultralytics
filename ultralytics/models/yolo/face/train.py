# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import FaceModel
from ultralytics.utils import DEFAULT_CFG


class FaceTrainer(yolo.pose.PoseTrainer):
    """Trainer for YOLO face detection models with 5-point landmark support.

    Extends PoseTrainer with face-specific configuration: WingLoss criterion,
    face task registration, and appropriate loss naming. Uses the existing pose
    infrastructure (Pose head, keypoint-aware augmentations, dataset loading).

    Attributes:
        args (dict): Configuration arguments for training.
        model (FaceModel): The face detection model being trained.
        data (dict): Dataset configuration including kpt_shape and flip_idx.
        loss_names (tuple): Names of the loss components used in training.

    Examples:
        >>> from ultralytics.models.yolo.face import FaceTrainer
        >>> args = dict(model="yolo11n-face.yaml", data="widerface.yaml", epochs=10)
        >>> trainer = FaceTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize FaceTrainer for face detection with landmarks.

        Args:
            cfg (dict, optional): Default configuration dictionary.
            overrides (dict, optional): Parameter overrides for the default configuration.
            _callbacks (dict, optional): Callback functions to be executed during training.

        Notes:
            Sets task to 'face' and bypasses PoseTrainer.__init__ which forces task='pose'.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "face"
        # Call DetectionTrainer.__init__ directly to avoid PoseTrainer overwriting task="pose"
        yolo.detect.DetectionTrainer.__init__(self, cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> FaceModel:
        """Get face detection model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to pretrained weights.
            verbose (bool): Whether to display model info.

        Returns:
            (FaceModel): Initialized face detection model with WingLoss criterion.
        """
        model = FaceModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Set keypoint shape and name attributes on the face model."""
        super().set_model_attributes()  # sets kpt_shape and kpt_names from PoseTrainer

    def get_validator(self):
        """Return a FaceValidator instance for model evaluation."""
        from ultralytics.utils.torch_utils import unwrap_model

        model = unwrap_model(self.model)
        criterion = getattr(model, "criterion", None) or model.init_criterion()
        if type(criterion).__name__ == "v8YOLOv6FaceLoss":
            self.loss_names = "box_loss", "lmk_loss", "kobj_loss", "cls_loss", "dfl_loss", "repgt_loss", "repbox_loss"
        else:
            self.loss_names = "box_loss", "lmk_loss", "kobj_loss", "cls_loss", "dfl_loss"
            
        return yolo.face.FaceValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and verify it contains the required kpt_shape key.

        Returns:
            (dict): Dataset configuration dictionary.

        Raises:
            KeyError: If kpt_shape is missing from the dataset YAML.
        """
        data = yolo.detect.DetectionTrainer.get_dataset(self)
        if "kpt_shape" not in data:
            raise KeyError(
                f"No `kpt_shape` in {self.args.data}. "
                "Add `kpt_shape: [5, 3]` to your dataset YAML for face landmark detection."
            )
        return data
