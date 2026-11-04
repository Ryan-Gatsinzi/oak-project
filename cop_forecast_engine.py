"""Real-time inference wrapper for the trained COP_forecast recurrent model.

The COP_forecast repo stores its COP/TOI models as state dicts for
``training/recurrent_models.py``.  This module reproduces the exact live
preprocessing used in ``analysis/live_frame_rate_test.py`` so the checkpoint can
be driven from the OAK-D stereo stream.

Note: the TOI checkpoint (``82_time_base.torch``) is not included in the
COP_forecast folder.  ``toi_weights`` is optional; when omitted only COP is
predicted.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COP_TRAINING_DIR = os.path.join(_THIS_DIR, "COP_forecast", "training")
if _COP_TRAINING_DIR not in sys.path:
    sys.path.insert(0, _COP_TRAINING_DIR)

from recurrent_models import model as COPForecastModel, sequence_regressor_live  # noqa: E402

DEFAULT_COP_WEIGHTS = os.path.join(
    _THIS_DIR, "COP_forecast", "base_models_pred_cop", "23.torch"
)

# From analysis/plot_results_with_speed.py:
# SHOE_SIZE_LOOKUP = {3: 236.8 / 10, 5: 261.1 / 10}
SHOE_SIZE_FOOT_LENGTH_MM = {3: 236.8, 5: 261.1}
DEFAULT_FOOT_LENGTH_MM = SHOE_SIZE_FOOT_LENGTH_MM[5]

# Model input resolution (from analysis/live_frame_rate_test.py)
_PREPROCESS_SIZE = (427, 240)  # (width, height)
_MODEL_INPUT_SIZE = (50, 25)   # (width, height)
_MODEL_HEIGHT = 25
_MODEL_WIDTH = 50

# TOI labels are normalized by 60 frames / 15 intervals:
# unnormalized_ms = 1000 * 15 * normalized / 60
_TOI_MS_SCALE = 1000.0 * 15.0 / 60.0


def preprocess_stereo(left_raw, right_raw):
    """Convert a raw grayscale stereo pair to the model's (2, 25, 50) input.

    Mirrors ``live_frame_rate_test.py``:
    hstack -> resize (240, 427) -> /255 -> resize (25, 50).
    """
    stack = np.stack([left_raw, right_raw], axis=0).astype(np.float32)
    resized = np.stack(
        [
            cv2.resize(channel, _PREPROCESS_SIZE, interpolation=cv2.INTER_LINEAR)
            for channel in stack
        ],
        axis=0,
    )
    normalized = resized / 255.0
    small = np.stack(
        [
            cv2.resize(channel, _MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            for channel in normalized
        ],
        axis=0,
    )
    return small.astype(np.float32)


def cop_norm_to_mm(cop_norm, foot_length_mm):
    """Map normalized COP to mm along the foot.

    From analysis/plot_results_with_speed.py:
    ``cop_mm = (cop_norm * 10 + 5) * SHOE_SIZE_LOOKUP[size]``
    which simplifies to ``(cop_norm + 0.5) * foot_length_mm``.
    """
    cop_mm = (float(cop_norm) + 0.5) * float(foot_length_mm)
    return float(np.clip(cop_mm, 0.0, foot_length_mm))


class COPForecastEngine:
    """Stateful real-time COP (and optional TOI) forecaster."""

    def __init__(
        self,
        cop_weights=DEFAULT_COP_WEIGHTS,
        toi_weights=None,
        foot_length_mm=DEFAULT_FOOT_LENGTH_MM,
        device=None,
        warmup=True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.foot_length_mm = float(foot_length_mm)
        self.cop_model = self._load_model(cop_weights, "COP")

        self.has_toi = bool(toi_weights and os.path.exists(toi_weights))
        self.toi_model = None
        if toi_weights and not self.has_toi:
            print(f"[WARN] TOI weights not found at {toi_weights}; TOI disabled.")
        if self.has_toi:
            self.toi_model = self._load_model(toi_weights, "TOI")

        self.reset()

        if warmup:
            self._warmup()

    def _load_model(self, path, label):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"{label} weights not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        net = COPForecastModel()
        net.SEQ = sequence_regressor_live()
        net.load_state_dict(state_dict)
        net.to(self.device).eval()
        print(f"[INFO] Loaded {label} weights: {path}")
        return net

    def reset(self):
        self.hidden1_cop = torch.zeros(2, 1, 60, device=self.device)
        self.hidden2_cop = torch.zeros(2, 1, 1, device=self.device)
        if self.has_toi:
            self.hidden1_toi = torch.zeros(2, 1, 60, device=self.device)
            self.hidden2_toi = torch.zeros(2, 1, 1, device=self.device)

    def _warmup(self):
        dummy = np.zeros((1, 2, _MODEL_HEIGHT, _MODEL_WIDTH), dtype=np.float32)
        self.step(stereo_np=dummy[0])
        self.reset()

    def _forward_model(self, net, stereo_np, hidden1, hidden2):
        tensor = (
            torch.from_numpy(np.ascontiguousarray(stereo_np))
            .view(1, 2, 1, _MODEL_HEIGHT, _MODEL_WIDTH)
            .to(self.device)
        )
        with torch.inference_mode():
            features = net.CNN(tensor)
            prediction, hidden1, hidden2 = net.SEQ(features, hidden1, hidden2)
        return float(prediction.reshape(-1)[0]), hidden1, hidden2

    def step(self, left_raw=None, right_raw=None, stereo_np=None):
        """Run one timestep.

        Args:
            left_raw, right_raw: raw grayscale uint8 frames, or
            stereo_np: already-preprocessed (2, 25, 50) float32 array.

        Returns:
            toi_ms (float | None), cop_mm (float), latency_ms (float),
            stereo_np (2, 25, 50)
        """
        start = time.perf_counter()

        if stereo_np is None:
            if left_raw is None or right_raw is None:
                raise ValueError("Provide either stereo_np or left_raw/right_raw")
            stereo_np = preprocess_stereo(left_raw, right_raw)

        cop_norm, self.hidden1_cop, self.hidden2_cop = self._forward_model(
            self.cop_model, stereo_np, self.hidden1_cop, self.hidden2_cop
        )

        toi_ms = None
        if self.has_toi:
            toi_norm, self.hidden1_toi, self.hidden2_toi = self._forward_model(
                self.toi_model, stereo_np, self.hidden1_toi, self.hidden2_toi
            )
            toi_ms = float(np.clip(toi_norm * _TOI_MS_SCALE, 0.0, 250.0))

        latency_ms = (time.perf_counter() - start) * 1000.0
        cop_mm = cop_norm_to_mm(cop_norm, self.foot_length_mm)
        return toi_ms, cop_mm, latency_ms, stereo_np
