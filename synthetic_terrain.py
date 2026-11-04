"""
Synthetic labeled stereo terrain generator.

The real OAK-D pipeline predicts two quantities from a short stereo sequence:

    * TOI  - time until the foot impacts the stair, in milliseconds
    * COP  - center of pressure along the foot (heel -> toe), in millimetres

We have no labelled real data, so this module renders a controllable side-view
stair scene where both labels are known exactly from the scene geometry:

    TOI  <- vertical gap between the boot sole and the step surface
    COP  <- horizontal position of the stair nosing under the boot

The same generator is used to build the training set *and* to drive the
visualizer's synthetic live feed, so what you see is what the model was trained
on. This is a benchmark, not a substitute for real labelled exoboot data.
"""

from __future__ import annotations

import numpy as np
import cv2

WIDTH, HEIGHT = 50, 25
RAW_W, RAW_H = 160, 100
FOOT_LEN_MM = 263.2
FPS = 60
TOI_MAX_MS = 250.0


def preprocess(raw: np.ndarray) -> np.ndarray:
    """Grayscale raw frame -> (25, 50) float32 in [0, 1] (absolute scaling)."""
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return np.clip(small.astype(np.float32) / 255.0, 0.0, 1.0)


def stereo_tensor(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack two raw frames into the (2, 25, 50) network input."""
    return np.stack([preprocess(left), preprocess(right)], axis=0)


class TerrainSceneGenerator:
    """Renders a boot above a stair; labels fall out of the geometry."""

    def __init__(self, seed: int | None = None, raw_w: int = RAW_W,
                 raw_h: int = RAW_H, seq_len: int = 15):
        self.rng = np.random.default_rng(seed)
        self.raw_w, self.raw_h = raw_w, raw_h
        self.seq_len = seq_len
        self._loop = False
        self.t = 0
        self._params = None

    # ------------------------------------------------------------------ params
    def _sample_params(self, loop: bool) -> dict:
        r = self.rng
        foot_len = r.uniform(0.42, 0.62) * self.raw_w
        center = r.uniform(0.42, 0.62) * self.raw_w
        cop_frac = r.uniform(0.06, 0.94)
        step_top = r.uniform(0.55, 0.68) * self.raw_h
        floor_y = min(self.raw_h - 4.0, step_top + r.uniform(0.07, 0.16) * self.raw_h)

        if loop:
            start_toi = TOI_MAX_MS
            toi_rate = TOI_MAX_MS / max(1, self.seq_len - 1)
        else:
            start_toi = r.uniform(90.0, TOI_MAX_MS)
            toi_rate = start_toi / max(1, self.seq_len - 1) * r.uniform(0.85, 1.15)

        p = {
            "foot_len": foot_len,
            "foot_x0": center - foot_len / 2.0,
            "foot_x1": center + foot_len / 2.0,
            "cop_frac": cop_frac,
            "step_top": step_top,
            "floor_y": floor_y,
            "descent_px": r.uniform(0.18, 0.34) * self.raw_h,
            "start_toi": start_toi,
            "toi_rate": toi_rate,
            "base_gray": r.uniform(70, 150),
            "step_gray": r.uniform(55, 120),
            "foot_gray": r.uniform(150, 235),
            "edge_strength": r.uniform(0.6, 1.0),
            "stereo_base": r.uniform(1.5, 4.0),
            "stereo_gain": r.uniform(6.0, 16.0),
            "mirror": bool(r.random() < 0.5),
            "bob_amp": r.uniform(0.0, 2.5),
            "blur": int(r.integers(0, 2)),
            "noise": r.uniform(3.0, 10.0),
            "texture_phase": r.uniform(0, 1000),
        }
        return p

    def reset(self, loop: bool = False) -> None:
        self.t = 0
        self._loop = loop
        self._params = self._sample_params(loop)

    def _params_at(self, t: int) -> dict:
        p = self._params
        toi = max(0.0, p["start_toi"] - t * p["toi_rate"])
        return p, toi

    # ----------------------------------------------------------------- render
    def _render(self, p: dict, toi: float, t: int) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.raw_w, self.raw_h
        r = self.rng

        gap = (toi / TOI_MAX_MS) * p["descent_px"]
        bob = p["bob_amp"] * np.sin(0.7 * t + p["texture_phase"])
        sole_y = p["step_top"] - gap + bob

        x0, x1 = p["foot_x0"], p["foot_x1"]
        edge_x = x0 + p["cop_frac"] * (x1 - x0)
        step_top, floor_y = p["step_top"], p["floor_y"]

        # Background wall with faint horizontal texture (single-channel).
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        bg = p["base_gray"] + 12.0 * np.sin(yy * 0.15 + p["texture_phase"] * 0.01)
        bg = bg + 4.0 * np.sin(xx * 0.05 + p["texture_phase"])
        img = np.clip(bg, 0, 255).astype(np.uint8)

        # Lower floor (left of the nosing) and the step body (right of it).
        cv2.rectangle(img, (0, int(floor_y)), (w, h), 40, -1)
        cv2.rectangle(img, (int(edge_x), int(step_top)), (w, h), 58, -1)

        # Step surface + nosing edge.
        cv2.line(img, (int(edge_x), int(step_top)), (w, int(step_top)), 120, 2)
        cv2.line(img, (int(edge_x), int(step_top)), (int(edge_x), h), 170, 2)
        cv2.line(img, (0, int(floor_y)), (int(edge_x), int(floor_y)), 72, 2)

        # Boot: sole plus an ankle block above it.
        sole_thick = max(3, int(0.05 * h))
        cv2.line(img, (int(x0), int(sole_y)), (int(x1), int(sole_y)), 200, sole_thick)
        cv2.ellipse(img, (int(x0), int(sole_y)), (sole_thick, sole_thick), 0, 0, 360, 200, -1)
        cv2.ellipse(img, (int(x1), int(sole_y)), (sole_thick, sole_thick), 0, 0, 360, 200, -1)

        ankle_h = int(0.22 * h)
        ankle_x0 = int(x0 + 0.22 * (x1 - x0))
        ankle_x1 = int(x0 + 0.72 * (x1 - x0))
        boot = np.array([
            [int(x0) + sole_thick, int(sole_y) - sole_thick // 2],
            [int(x1) - sole_thick, int(sole_y) - sole_thick // 2],
            [ankle_x1, int(sole_y) - ankle_h],
            [ankle_x0, int(sole_y) - ankle_h],
        ], dtype=np.int32)
        cv2.fillPoly(img, [boot], 135)

        # Heel / toe markers make the COP direction unambiguous.
        cv2.circle(img, (int(x0), int(sole_y)), max(2, sole_thick // 2), 245, -1)
        cv2.circle(img, (int(x1), int(sole_y)), max(2, sole_thick // 2), 175, -1)

        if p["mirror"]:
            img = cv2.flip(img, 1)

        if p["blur"]:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        gray = img.astype(np.float32)
        gray += r.normal(0.0, p["noise"], gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Crude stereo: shift the right eye by a distance-dependent disparity.
        d = int(round(p["stereo_base"] + p["stereo_gain"] * (1.0 - toi / TOI_MAX_MS)))
        d = int(np.clip(d, 1, self.raw_w // 4))
        right = np.empty_like(gray)
        right[:, :d] = gray[:, :1]
        right[:, d:] = gray[:, :-d]
        right = np.clip(right.astype(np.float32) + r.normal(0.0, p["noise"] * 0.5, right.shape), 0, 255).astype(np.uint8)

        return gray, right

    # --------------------------------------------------------------- sequences
    def sample_sequence(self, length: int | None = None):
        """Return one labelled sequence for training.

        Returns left (T,H,W) uint8, right (T,H,W) uint8, x (T,2,25,50) float32,
        toi_ms (T,) float32, cop_mm (T,) float32.
        """
        length = length or self.seq_len
        self.reset(loop=False)
        left, right, toi_l, cop_l = [], [], [], []
        for t in range(length):
            p, toi = self._params_at(t)
            left_raw, right_raw = self._render(p, toi, t)
            left.append(left_raw)
            right.append(right_raw)
            toi_l.append(toi)
            cop_l.append(p["cop_frac"] * FOOT_LEN_MM)
        x = np.stack([stereo_tensor(l, r_) for l, r_ in zip(left, right)], axis=0)
        return (np.stack(left), np.stack(right), x.astype(np.float32),
                np.asarray(toi_l, dtype=np.float32), np.asarray(cop_l, dtype=np.float32))

    def step(self):
        """Advance the continuous synthetic live feed by one frame."""
        if self._params is None:
            self.reset(loop=True)
        p, toi = self._params_at(self.t)
        left_raw, right_raw = self._render(p, toi, self.t)
        self.t += 1
        if self.t >= self.seq_len:
            self.reset(loop=True)
        return (left_raw, right_raw, stereo_tensor(left_raw, right_raw),
                float(toi), float(p["cop_frac"] * FOOT_LEN_MM))


if __name__ == "__main__":
    gen = TerrainSceneGenerator(seed=0)
    l, r, x, toi, cop = gen.sample_sequence()
    print("left", l.shape, l.dtype, "x", x.shape, x.dtype)
    print("toi range", float(toi.min()), float(toi.max()))
    print("cop range", float(cop.min()), float(cop.max()))
    assert x.shape == (15, 2, HEIGHT, WIDTH)
    assert 0.0 <= x.min() and x.max() <= 1.0
    print("synthetic generator smoke test passed.")
