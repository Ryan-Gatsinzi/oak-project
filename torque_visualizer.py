#!/usr/bin/env python3
"""
oak_torque_visualizer.py
========================
Live OAK-D stereo feed -> trained CNN-RNN forecaster (TOI + COP) -> torque
profile selection -> real-time HUD.

What it does
------------
* Captures left/right frames from the OAK-D (CAM_B / CAM_C) and feeds each new
  frame, in order, through the recurrent model (state carried across frames).
* Smooths TOI/COP, picks a profile (Collins level ground / Murray stair heel /
  Murray stair toe), and runs a small state machine:
        IDLE  -> ARMED (impact imminent) -> STANCE (profile plays on a real-time
        clock that starts at the predicted impact) -> IDLE
* Draws the profile, a cursor driven by that clock, and live gauges.

What it does NOT do
-------------------
* It does not command the exoskeleton. Visualization only.
* After impact, the camera cannot tell you gait phase. The cursor is a
  time-based estimate (--stride-time), not a measurement. Real phase needs an
  IMU / encoder / foot switch.

Items you MUST confirm against the professors' code (search for "CONFIRM"):
  1. TerrainForecaster API (constructor args, initial_state, forward_frame)
  2. Output scaling (--output-map, --toi-range, --cop-range) and output order
  3. Preprocessing (--resize) and any cropping used in training
  4. The terrain decision rule (decide in StrideController._classify)
  5. Gear ratio for motor current (--gear-ratio)
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError:  # handled at camera creation
    dai = None

WIDTH, HEIGHT = 50, 25          # network input resolution
RAW_W, RAW_H = 640, 400         # camera output resolution
FOOT_LEN_MM = 263.2
MID_FOOT_MM = FOOT_LEN_MM / 2.0

# Fallback Murray stair polynomial coefficients (x in % gait cycle, output Nm/kg).
# Used only if the professors' JSON files are not found. Startup prints which
# source was used.
FALLBACK_HEEL = [-1.3418952053455997e-07, 1.7845671326326224e-05, -0.0008115159297602824,
                 0.014257060759835865, -0.05348055449407867, 0.05081448731224058]
FALLBACK_TOE = [-2.6283254641545553e-07, 3.629269408838788e-05, -0.001707299481976467,
                0.029826242570955334, -0.08755150988293577, 0.1387268050697176]

# BGR colors
C_BG, C_PANEL, C_BORDER = (24, 28, 34), (35, 41, 51), (55, 65, 81)
C_TEXT, C_DIM = (240, 243, 246), (140, 150, 160)
C_GREEN, C_RED, C_WARN = (80, 210, 100), (80, 80, 240), (50, 185, 255)
PROFILE_COLOR = {"LEVEL": (235, 175, 40), "HEEL": (50, 185, 255),
                 "TOE": (40, 110, 255), "BLEND": (150, 120, 255)}
WINDOW = "OAK-D Real-Time Torque Profile Visualizer"


# =============================================================================
# 1. TORQUE PROFILES
# =============================================================================
class TorqueProfiles:
    """Collins level-ground spline and Murray stair polynomials."""

    def __init__(self, params_dir=None):
        self.x = np.linspace(0.0, 100.0, 400, dtype=np.float32)
        if not params_dir:
            for cand in ("Dephy-Soft-Sensor-integrate_cop_toi_ml/params", "params", "../params"):
                if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "heel_params.json")):
                    params_dir = cand
                    break
        self.heel, self.heel_src = self._load(params_dir, "heel_params.json", FALLBACK_HEEL)
        self.toe, self.toe_src = self._load(params_dir, "toe_params.json", FALLBACK_TOE)
        self._cache_key, self._cache = None, None

    @staticmethod
    def _load(params_dir, name, fallback):
        if params_dir:
            path = os.path.join(params_dir, name)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        coeffs = json.load(f)
                    if len(coeffs) == 6:
                        return [float(c) for c in coeffs], f"file: {path}"
                    print(f"[WARN] {path} does not hold 6 coefficients; using fallback.")
                except Exception as e:
                    print(f"[WARN] could not read {path}: {e}; using fallback.")
        return list(fallback), "built-in fallback"

    @staticmethod
    def collins(x, mass, peak_norm=0.20):
        """Zhang & Collins level-ground cubic-spline profile (Nm)."""
        t0, t1, t2, t3 = 0.0, 27.1, 50.4, 62.7
        ts, tp = 2.0, mass * peak_norm
        x = np.asarray(x, dtype=np.float64)
        tau = np.zeros_like(x)
        m1 = (x >= t0) & (x <= t1)
        tau[m1] = ts / (t1 - t0) * (x[m1] - t0)
        m2 = (x > t1) & (x <= t2)
        d = (t1 - t2) ** 3
        a1 = 2.0 * (tp - ts) / d
        b1 = -3.0 * (t1 + t2) * (tp - ts) / d
        c1 = 6.0 * t1 * t2 * (tp - ts) / d
        d1 = -(-(t1 ** 3) * tp + 3 * t1 ** 2 * t2 * tp - 3 * t1 * t2 ** 2 * ts + t2 ** 3 * ts) / d
        tau[m2] = a1 * x[m2] ** 3 + b1 * x[m2] ** 2 + c1 * x[m2] + d1
        m3 = (x > t2) & (x <= t3)
        d = 2.0 * (t2 - t3) ** 3
        a2 = -(tp - ts) / d
        b2 = 3.0 * t3 * (tp - ts) / d
        c2 = 3.0 * (t2 ** 2 - 2 * t2 * t3) * (tp - ts) / d
        d2 = -(3 * t2 ** 2 * t3 * tp - 6 * t2 * t3 ** 2 * tp + 2 * t3 ** 3 * tp
               - 2 * t2 ** 3 * ts + 3 * t2 ** 2 * t3 * ts) / d
        tau[m3] = a2 * x[m3] ** 3 + b2 * x[m3] ** 2 + c2 * x[m3] + d2
        return np.maximum(tau, 0.0).astype(np.float32)

    @staticmethod
    def poly5(x, coeffs, mass, scale):
        """Murray stair polynomial (coeffs in Nm/kg), scaled to exo assistance."""
        x = np.asarray(x, dtype=np.float64)
        val = sum(c * x ** (5 - i) for i, c in enumerate(coeffs))
        mask = (x > 3.0) & (x <= 60.0) & (val > 0.0)
        tau = np.zeros_like(x)
        tau[mask] = val[mask] * mass * scale
        return np.maximum(tau, 0.0).astype(np.float32)

    def curves(self, mass, assist_scale):
        key = (round(mass, 3), round(assist_scale, 3))
        if key != self._cache_key:
            self._cache = {
                "LEVEL": self.collins(self.x, mass),
                "HEEL": self.poly5(self.x, self.heel, mass, assist_scale),
                "TOE": self.poly5(self.x, self.toe, mass, assist_scale),
            }
            self._cache_key = key
        return self._cache


def motor_current_ma(torque_nm, kt_nmm_per_a, gear_ratio, efficiency=1.0) -> Optional[float]:
    """
    Ankle torque -> motor current. Needs the transmission ratio: motor torque is
    ankle torque divided by (gear_ratio * efficiency). Returns None if unknown.
    """
    if not gear_ratio:
        return None
    motor_nmm = torque_nm * 1000.0 / (gear_ratio * efficiency)
    return motor_nmm / kt_nmm_per_a * 1000.0


# =============================================================================
# 2. CAMERAS
# =============================================================================
@dataclass
class FrameBundle:
    left: np.ndarray
    right: np.ndarray
    tensor: np.ndarray            # (1, 2, HEIGHT, WIDTH) float32 in [0, 1]
    seq: Optional[int]
    capture_latency_ms: float


def preprocess(left, right, interp=cv2.INTER_AREA):
    """Resize each image to 50x25 and min-max normalize it independently."""
    def one(img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(img, (WIDTH, HEIGHT), interpolation=interp)
        norm = cv2.normalize(small.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return np.clip(norm, 0.0, 1.0)      # cv2 can return ~-1e-9 from rounding
    return np.stack([one(left), one(right)], axis=0)[None].astype(np.float32)


class OakCamera:
    synthetic = False

    def __init__(self, fps=60, interp=cv2.INTER_AREA):
        if dai is None:
            raise RuntimeError("depthai is not installed in this Python environment.")
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            raise RuntimeError("No OAK device found on the network/USB.")
        info = devices[0]
        getter = getattr(info, "getDeviceId", None) or getattr(info, "getMxId", None)
        print(f"[INFO] OAK device: {getter() if getter else 'unknown id'}")
        self.interp = interp

        self.pipeline = dai.Pipeline()
        left_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        left_out = left_cam.requestOutput((RAW_W, RAW_H), type=dai.ImgFrame.Type.GRAY8, fps=fps)
        right_out = right_cam.requestOutput((RAW_W, RAW_H), type=dai.ImgFrame.Type.GRAY8, fps=fps)
        sync = self.pipeline.create(dai.node.Sync)
        sync.setRunOnHost(True)
        sync.setSyncThreshold(timedelta(milliseconds=15))
        left_out.link(sync.inputs["left"])
        right_out.link(sync.inputs["right"])
        self.queue = sync.out.createOutputQueue(maxSize=4, blocking=False)
        self.pipeline.start()
        print("[INFO] OAK-D stereo streaming.")

    def read(self) -> Optional[FrameBundle]:
        group = self.queue.tryGet()
        if group is None:
            return None
        lf, rf = group["left"], group["right"]
        left, right = lf.getCvFrame(), rf.getCvFrame()
        try:
            seq = int(lf.getSequenceNum())
        except Exception:
            seq = None
        try:  # device->host latency; falls back to 0 if the clock API differs
            lat = (dai.Clock.now() - lf.getTimestamp()).total_seconds() * 1000.0
            lat = lat if 0.0 <= lat < 200.0 else 0.0
        except Exception:
            lat = 0.0
        return FrameBundle(left, right, preprocess(left, right, self.interp), seq, lat)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class SyntheticCamera:
    """Paced 60 FPS test pattern. Predictions on it are NOT meaningful."""
    synthetic = True

    def __init__(self, fps=60, interp=cv2.INTER_AREA):
        self.period, self.interp = 1.0 / fps, interp
        self.next_t, self.phase, self.seq = time.perf_counter(), 0.0, 0
        yy = np.arange(RAW_H)[:, None]
        self.base = (((yy % 80) < 40).astype(np.float32) * 160 + 40) * np.ones((1, RAW_W), np.float32)

    def read(self) -> Optional[FrameBundle]:
        now = time.perf_counter()
        if now < self.next_t:
            return None
        self.next_t += self.period
        if now - self.next_t > 0.2:
            self.next_t = now + self.period
        self.phase += 1.0
        img = np.roll(self.base, int(self.phase), axis=0)
        left = np.clip(img + np.random.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
        right = np.roll(left, -12, axis=1)
        self.seq += 1
        return FrameBundle(left, right, preprocess(left, right, self.interp), self.seq, 0.0)

    def close(self):
        pass


# =============================================================================
# 3. FORECASTER ENGINE (torch imported lazily)
# =============================================================================
class ForecasterEngine:
    """
    Wraps the professors' TerrainForecaster.

    CONFIRM against their model.py:
        TerrainForecaster(hidden1=, hidden2=, num_outputs=)
        model.initial_state(batch_size=1, device=device)
        pred, state = model.forward_frame(x[1,2,25,50], state)   # pred: [1, num_outputs]
    Use --check-model to verify this before running the full visualizer.
    """

    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            from model import TerrainForecaster
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from model import TerrainForecaster

        def build(num_outputs, weights):
            m = TerrainForecaster(hidden1=args.hidden1, hidden2=args.hidden2,
                                  num_outputs=num_outputs).to(self.device)
            for attr in ("initial_state", "forward_frame"):
                if not hasattr(m, attr):
                    raise RuntimeError(f"model.TerrainForecaster has no '{attr}'. "
                                       "Paste your model.py so the adapter can be matched.")
            if weights:
                sd = torch.load(weights, map_location=self.device)
                if isinstance(sd, dict):
                    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
                m.load_state_dict(sd)          # strict: wrong shapes fail loudly
                print(f"[INFO] loaded weights: {weights}")
            m.eval()
            return m

        if args.weights:
            self.models = [build(2, args.weights)]
            order = [s.strip() for s in args.combined_order.split(",")]
            self.idx = {order[0]: 0, order[1]: 1}
            self.trained = True
        elif args.toi_weights and args.cop_weights:
            self.models = [build(1, args.toi_weights), build(1, args.cop_weights)]
            self.idx = {"toi": 0, "cop": 1}
            self.trained = True
        elif args.untrained:
            self.models = [build(2, None)]
            self.idx = {"toi": 0, "cop": 1}
            self.trained = False
        else:
            print("[INFO] No weights specified (--weights). Starting with uninitialized model (use 'A' for manual control).")
            self.models = [build(2, None)]
            self.idx = {"toi": 0, "cop": 1}
            self.trained = False
        self.reset()
        self._seen, self._max_abs = 0, 0.0
        self.suspect_normalized = False

    def reset(self):
        self.states = [m.initial_state(batch_size=1, device=self.device) for m in self.models]

    def _map(self, raw, lo, hi):
        mode = self.args.output_map
        if mode == "raw":
            return float(raw)
        s = 1.0 / (1.0 + np.exp(-raw)) if mode == "sigmoid" else raw
        return float(lo + s * (hi - lo))

    def step(self, x_np):
        """x_np: (1,2,25,50) float32. Returns (toi_ms, cop_mm, infer_ms)."""
        t0 = time.perf_counter()
        x = self.torch.from_numpy(x_np).to(self.device)
        outs = []
        with self.torch.no_grad():
            for i, m in enumerate(self.models):
                pred, self.states[i] = m.forward_frame(x, self.states[i])
                outs.append(pred.reshape(-1).float().cpu().numpy())
        raw = np.concatenate(outs)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        if raw.size != 2:
            raise RuntimeError(f"expected 2 outputs total, model produced {raw.size}")
        toi = self._map(raw[self.idx["toi"]], *self.args.toi_range)
        cop = self._map(raw[self.idx["cop"]], *self.args.cop_range)
        self._seen += 1
        self._max_abs = max(self._max_abs, abs(toi), abs(cop))
        if self._seen == 120 and self.args.output_map == "raw" and self._max_abs <= 1.5:
            self.suspect_normalized = True
            print("[WARN] Outputs stayed within +/-1.5 for 120 frames. They may be normalized "
                  "(0-1). Try --output-map unit --toi-range ... --cop-range ...")
        return float(np.clip(toi, 0.0, 2000.0)), float(np.clip(cop, 0.0, FOOT_LEN_MM)), infer_ms


# =============================================================================
# 4. INFERENCE WORKER (camera + model on their own thread)
# =============================================================================
@dataclass
class Prediction:
    t: float            # perf_counter when the prediction was ready
    toi_ms: float       # latency-compensated: time from t until predicted impact
    cop_mm: float
    infer_ms: float


class Worker(threading.Thread):
    def __init__(self, camera, engine, comm_latency_ms=0.0):
        super().__init__(daemon=True)
        self.camera, self.engine, self.comm = camera, engine, comm_latency_ms
        self.q = deque(maxlen=1024)
        self.lock = threading.Lock()
        self.latest = None
        self.stop_evt = threading.Event()
        self.paused = False
        self.reset_req = False
        self.skipped, self.frames, self.error = 0, 0, None
        self.last_seq = None
        self.rate = 0.0
        self.infer_ms = 0.0

    def run(self):
        stamps = deque(maxlen=60)
        try:
            while not self.stop_evt.is_set():
                b = self.camera.read()
                if b is None:
                    time.sleep(0.0005)
                    continue
                if self.paused:
                    self.last_seq = None
                    continue
                if self.reset_req:
                    self.engine.reset()
                    self.reset_req = False
                if b.seq is not None and self.last_seq is not None and b.seq - self.last_seq > 1:
                    self.skipped += b.seq - self.last_seq - 1
                self.last_seq = b.seq
                toi, cop, infer_ms = self.engine.step(b.tensor)
                t = time.perf_counter()
                # Remaining time to impact now = forecast - time already elapsed since capture.
                toi_adj = max(0.0, toi - infer_ms - b.capture_latency_ms - self.comm)
                with self.lock:
                    self.latest = (b.left, b.right, b.tensor)
                self.q.append(Prediction(t, toi_adj, cop, infer_ms))
                self.frames += 1
                self.infer_ms = infer_ms
                stamps.append(t)
                if len(stamps) > 2:
                    self.rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        except Exception:
            self.error = traceback.format_exc()


# =============================================================================
# 5. STRIDE CONTROLLER
# =============================================================================
class StrideController:
    """
    Profile selection + IDLE -> ARMED -> STANCE state machine.

    CONFIRM: _classify() carries over the rule from the original script
    (TOI > EMD => level ground; otherwise stair heel/toe by COP vs midfoot).
    As written it cannot select LEVEL while the impact is imminent, so during
    ARMED/STANCE it will always pick a stair profile. If the professors' control
    code uses a different rule, change _classify() only.
    """
    IDLE, ARMED, STANCE = "IDLE", "ARMED", "STANCE"

    def __init__(self, emd_ms=50.0, hyst_toi_ms=10.0, hyst_cop_mm=8.0, stride_s=1.2,
                 stance_end_pct=65.0, smooth=3, arm_timeout_s=0.4):
        self.emd, self.h_toi, self.h_cop = emd_ms, hyst_toi_ms, hyst_cop_mm
        self.stride_s, self.stance_end, self.timeout = stride_s, stance_end_pct, arm_timeout_s
        self.toi_hist, self.cop_hist = deque(maxlen=smooth), deque(maxlen=smooth)
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.stair, self.strike = False, "HEEL"
        self.rearm_ok = True
        self.impact_time = self.arm_time = self.stance_start = None
        self.frozen_profile = self.frozen_cop = None
        self.toi = self.cop = None
        self.profile = "LEVEL"
        self.toi_hist.clear()
        self.cop_hist.clear()

    def _classify(self, toi, cop):
        if self.stair:
            if toi > self.emd + self.h_toi:
                self.stair = False
        elif toi <= self.emd:
            self.stair = True
            self.strike = "HEEL" if cop <= MID_FOOT_MM else "TOE"
        if not self.stair:
            return "LEVEL"
        if self.strike == "HEEL" and cop > MID_FOOT_MM + self.h_cop:
            self.strike = "TOE"
        elif self.strike == "TOE" and cop < MID_FOOT_MM - self.h_cop:
            self.strike = "HEEL"
        return self.strike

    def update(self, t, toi_ms, cop_mm, allow_arm=True):
        self.toi_hist.append(toi_ms)
        self.cop_hist.append(cop_mm)
        self.toi, self.cop = float(np.median(self.toi_hist)), float(np.median(self.cop_hist))
        if self.state != self.STANCE:
            self.profile = self._classify(self.toi, self.cop)
        if not allow_arm:
            return
        if self.state == self.IDLE:
            if self.toi > self.emd + self.h_toi:
                self.rearm_ok = True
            if self.rearm_ok and self.toi <= self.emd:
                self.state, self.arm_time, self.rearm_ok = self.ARMED, t, False
        if self.state == self.ARMED:
            if self.toi > self.emd + self.h_toi or t - self.arm_time > self.timeout:
                self.state = self.IDLE                      # false alarm / timeout
            else:
                # Use the latest RAW forecast for timing: the median used for decisions
                # lags by ~1 frame and would put the impact ~16 ms late.
                self.impact_time = t + toi_ms / 1000.0

    def _start_stance(self, t_start):
        self.state, self.stance_start = self.STANCE, t_start
        self.frozen_profile, self.frozen_cop = self.profile, self.cop

    def force_stance(self, t):
        self._start_stance(t)

    def tick(self, t):
        """Call every render. Returns gait phase in % (None unless in stance)."""
        if self.state == self.ARMED and t >= self.impact_time:
            self._start_stance(self.impact_time)
        if self.state == self.STANCE:
            phase = (t - self.stance_start) / self.stride_s * 100.0
            if phase >= self.stance_end:
                self.state = self.IDLE
                return None
            return phase
        return None

    def display_profile(self):
        return self.frozen_profile if self.state == self.STANCE else self.profile

    def display_cop(self):
        return self.frozen_cop if (self.state == self.STANCE and self.frozen_cop is not None) else self.cop

    def countdown_ms(self, t):
        return max(0.0, (self.impact_time - t) * 1000.0) if self.state == self.ARMED else None


# =============================================================================
# 6. HUD
# =============================================================================
def _put(img, text, org, color=C_TEXT, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Hud:
    def __init__(self, width=1280, height=720):
        self.w, self.h = width, height
        self.canvas = np.zeros((height, width, 3), np.uint8)
        self.gx0, self.gy0, self.gw, self.gh = 550, 100, 690, 405
        # applyColorMap is slow per call; build the viridis lookup table once.
        ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
        self.lut = cv2.applyColorMap(ramp, cv2.COLORMAP_VIRIDIS).reshape(256, 3)

    def panel(self, x, y, w, h):
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), C_PANEL, -1)
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), C_BORDER, 1)

    def _grid(self, max_t):
        c, x0, y0, gw, gh = self.canvas, self.gx0, self.gy0, self.gw, self.gh
        cv2.rectangle(c, (x0, y0), (x0 + gw, y0 + gh), (18, 22, 28), -1)
        cv2.rectangle(c, (x0, y0), (x0 + gw, y0 + gh), C_BORDER, 1)
        for pct in range(0, 101, 20):
            px = int(x0 + pct / 100.0 * gw)
            cv2.line(c, (px, y0), (px, y0 + gh), (32, 38, 48), 1)
            _put(c, f"{pct}%", (px - 14, y0 + gh + 22), C_DIM, 0.45)
        step = 5 if max_t <= 25 else 10 if max_t <= 50 else 20
        for v in np.arange(0, max_t + 0.1, step):
            py = int(y0 + gh - v / max_t * gh)
            cv2.line(c, (x0, py), (x0 + gw, py), (32, 38, 48), 1)
            _put(c, f"{int(v)}", (x0 - 32, py + 5), C_DIM, 0.45)
        _put(c, "Gait cycle (%)", (x0 + gw // 2 - 45, y0 + gh + 45), C_TEXT, 0.55)
        _put(c, "Torque (Nm)", (x0 - 35, y0 - 15), C_TEXT, 0.55)

    def _curve(self, xs, ys, max_t, color, thick, bright):
        pts = np.stack([self.gx0 + xs / 100.0 * self.gw,
                        self.gy0 + self.gh - ys / max_t * self.gh], axis=1).astype(np.int32)
        col = color if bright else tuple(int(v * 0.35) for v in color)
        cv2.polylines(self.canvas, [pts.reshape(-1, 1, 2)], False, col, thick, cv2.LINE_AA)

    def render(self, v):
        c = self.canvas
        c[:] = C_BG
        key = v["profile_key"]
        color = PROFILE_COLOR[key]

        # header
        self.panel(20, 15, self.w - 40, 60)
        _put(c, "OAK-D Terrain Forecaster & Torque Profile", (40, 40), C_TEXT, 0.7, 2)
        tag, tcol = ("[ SYNTHETIC INPUT ]", C_WARN) if v["synthetic"] else ("[ OAK-D LIVE ]", C_GREEN)
        _put(c, tag, (610, 40), tcol, 0.6, 2)
        _put(c, f"Inference {v['rate']:.1f}/s ({v['infer_ms']:.1f} ms)  |  HUD {v['hud_fps']:.0f} fps",
             (880, 40), C_DIM, 0.45)
        if v["warnings"]:
            _put(c, "  |  ".join(v["warnings"])[:150], (40, 64), C_WARN, 0.46)

        # camera panel
        self.panel(20, 90, 490, 240)
        _put(c, "STEREO FEED (CAM_B left / CAM_C right)", (35, 115), C_TEXT, 0.5)
        if v["left"] is not None:
            tw, th = 215, 135
            for img, x in ((v["left"], 35), (v["right"], 265)):
                t = cv2.resize(img, (tw, th))
                if t.ndim == 2:
                    t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
                c[130:130 + th, x:x + tw] = t
            _put(c, "Left", (40, 280), C_DIM, 0.45)
            _put(c, "Right", (270, 280), C_DIM, 0.45)
            _put(c, "NN input 50x25:", (35, 312), C_DIM, 0.42)
            for ch, x in ((0, 190), (1, 290)):
                m = self.lut[(v["tensor"][0, ch] * 255).astype(np.uint8)]
                m = cv2.resize(m, (90, 45), interpolation=cv2.INTER_NEAREST)
                c[283:283 + 45, x:x + 90] = m
        else:
            _put(c, "waiting for first frame...", (35, 200), C_DIM, 0.55)

        # forecast panel
        self.panel(20, 345, 490, 230)
        _put(c, "TERRAIN INTERACTION FORECAST", (35, 375), C_TEXT, 0.55)
        bx, bw, bh = 35, 440, 16
        toi_max = v["toi_max"]
        _put(c, f"Time-to-impact (TOI): {v['toi']:.1f} ms", (35, 410), C_TEXT, 0.5)
        cv2.rectangle(c, (bx, 420), (bx + bw, 420 + bh), (25, 30, 38), -1)
        fill = int(np.clip(v["toi"] / toi_max, 0, 1) * bw)
        cv2.rectangle(c, (bx, 420), (bx + fill, 420 + bh),
                      PROFILE_COLOR["LEVEL"] if v["toi"] > v["emd"] else C_WARN, -1)
        cv2.rectangle(c, (bx, 420), (bx + bw, 420 + bh), C_BORDER, 1)
        ex = int(bx + min(v["emd"] / toi_max, 1.0) * bw)
        cv2.line(c, (ex, 416), (ex, 440), C_RED, 2)
        _put(c, f"EMD ({v['emd']:.0f} ms)", (ex - 30, 456), C_RED, 0.4)

        _put(c, f"Center of pressure (COP): {v['cop']:.1f} mm from heel", (35, 480), C_TEXT, 0.5)
        cv2.rectangle(c, (bx, 490), (bx + bw, 490 + bh), (25, 30, 38), -1)
        fill = int(np.clip(v["cop"] / FOOT_LEN_MM, 0, 1) * bw)
        cv2.rectangle(c, (bx, 490), (bx + fill, 490 + bh),
                      PROFILE_COLOR["HEEL"] if v["cop"] <= MID_FOOT_MM else PROFILE_COLOR["TOE"], -1)
        cv2.rectangle(c, (bx, 490), (bx + bw, 490 + bh), C_BORDER, 1)
        mx = int(bx + MID_FOOT_MM / FOOT_LEN_MM * bw)
        cv2.line(c, (mx, 486), (mx, 510), (255, 255, 255), 2)
        _put(c, "Heel", (bx, 524), C_DIM, 0.4)
        _put(c, f"Midfoot ({MID_FOOT_MM:.0f} mm)", (mx - 45, 524), (255, 255, 255), 0.4)
        _put(c, "Toe", (bx + bw - 25, 524), C_DIM, 0.4)

        cv2.rectangle(c, (35, 532), (35 + bw, 568), (28, 34, 44), -1)
        cv2.rectangle(c, (35, 532), (35 + bw, 568), color, 2)
        _put(c, v["banner"], (45, 556), color, 0.55, 2)

        # graph
        max_t = v["max_torque"]
        self._grid(max_t)
        xs = v["x"]
        for k in ("LEVEL", "HEEL", "TOE"):
            self._curve(xs, v["curves"][k], max_t, PROFILE_COLOR[k], 1, False)
        self._curve(xs, v["active_curve"], max_t, color, 3, True)
        if v["phase"] is not None:
            px = int(self.gx0 + v["phase"] / 100.0 * self.gw)
            py = int(self.gy0 + self.gh - v["torque"] / max_t * self.gh)
            cv2.line(c, (px, self.gy0), (px, self.gy0 + self.gh), (90, 110, 140), 1, cv2.LINE_AA)
            cv2.circle(c, (px, py), 7, color, -1, cv2.LINE_AA)
            cv2.circle(c, (px, py), 11, (255, 255, 255), 1, cv2.LINE_AA)
            _put(c, f"{v['torque']:.1f} Nm", (px + 10, max(self.gy0 + 40, py - 10)), (255, 255, 255), 0.5)
        lx, ly = self.gx0 + 20, self.gy0 + 25
        for i, (k, label) in enumerate((("LEVEL", "Collins (level)"),
                                        ("HEEL", f"Murray heel x{v['assist']:.2f}"),
                                        ("TOE", f"Murray toe x{v['assist']:.2f}"))):
            ox = lx + i * 215
            cv2.line(c, (ox, ly), (ox + 25, ly), PROFILE_COLOR[k], 2)
            _put(c, label, (ox + 33, ly + 4), C_TEXT, 0.45)

        # metrics + controls
        self.panel(self.gx0, 575, self.gw, 50)
        cur = "set --gear-ratio" if v["current_ma"] is None else f"{v['current_ma']:.0f} mA"
        _put(c, f"Peak torque: {v['peak']:.1f} Nm  |  Motor current at peak: {cur}  |  Mass: {v['mass']:.0f} kg",
             (self.gx0 + 20, 606), C_TEXT, 0.52)
        self.panel(20, 640, self.w - 40, 65)
        _put(c, "[Q] quit  [SPACE] pause  [A] auto/manual  [C/H/T] manual presets  [G] start stride (manual)  [R] reset RNN",
             (40, 668), C_TEXT, 0.46)
        _put(c, "[I/K or UP/DOWN] TOI +/-10 ms  [J/L or LEFT/RIGHT] COP +/-10 mm  [+/-] mass  [B] bio/assist scale",
             (40, 692), C_DIM, 0.44)
        return c


# =============================================================================
# 7. MAIN LOOP
# =============================================================================
ARROWS = {2490368: "up", 2621440: "down", 2424832: "left", 2555904: "right",
          65362: "up", 65364: "down", 65361: "left", 65363: "right"}


def run(args, camera, engine):
    profiles = TorqueProfiles(args.params_dir)
    print(f"[INFO] heel coefficients: {profiles.heel_src}")
    print(f"[INFO] toe coefficients:  {profiles.toe_src}")
    ctrl = StrideController(emd_ms=args.emd_ms, stride_s=args.stride_time, stance_end_pct=args.stance_end)
    hud = Hud()
    worker = Worker(camera, engine, args.comm_latency_ms)
    worker.start()

    mass = args.mass
    assist = 1.0 if args.bio else 0.25
    auto, paused = True, False
    manual_toi, manual_cop = 120.0, 80.0
    last_state, hud_fps, t_prev = None, 0.0, time.perf_counter()
    t_start, frame = time.perf_counter(), None
    period = 1.0 / 60.0

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            loop_t0 = time.perf_counter()
            if worker.error:
                print("[ERROR] inference thread crashed:\n" + worker.error)
                break
            if args.max_seconds and loop_t0 - t_start > args.max_seconds:
                break

            # 1. feed controller from new predictions, in order
            last_pred = None
            while worker.q:
                p = worker.q.popleft()
                last_pred = p
                if auto and not paused:
                    # Resolve any pending impact up to this prediction's time first,
                    # so a post-impact TOI jump is not mistaken for a false alarm.
                    ctrl.tick(p.t)
                    ctrl.update(p.t, p.toi_ms, p.cop_mm)
                    ctrl.tick(p.t)
            now = time.perf_counter()
            if not auto:
                ctrl.update(now, manual_toi, manual_cop, allow_arm=False)
            stance_phase = None if paused else ctrl.tick(now)
            if stance_phase is not None:
                phase = stance_phase
            elif not paused:
                phase = ((now - t_start) / args.stride_time * 100.0) % 100.0
            else:
                phase = None

            if ctrl.state != last_state:
                print(f"[{now - t_start:7.2f}s] {ctrl.state:6s} profile={ctrl.display_profile()} "
                      f"toi={ctrl.toi if ctrl.toi is not None else float('nan'):.0f}ms "
                      f"cop={ctrl.cop if ctrl.cop is not None else float('nan'):.0f}mm")
                last_state = ctrl.state

            # 2. what to draw
            curves = profiles.curves(mass, assist)
            key = ctrl.display_profile()
            cop_disp = ctrl.display_cop() if ctrl.display_cop() is not None else 0.0
            if args.blend and key in ("HEEL", "TOE"):
                alpha = float(np.clip((cop_disp - 40.0) / 180.0, 0.0, 1.0))
                active = (1 - alpha) * curves["HEEL"] + alpha * curves["TOE"]
                key_draw, name = "BLEND", f"STAIR BLEND (alpha={alpha:.2f})"
            else:
                active = curves[key]
                key_draw = key
                name = {"LEVEL": "LEVEL GROUND (COLLINS)", "HEEL": "STAIR HEEL STRIKE (MURRAY)",
                        "TOE": "STAIR TOE STRIKE (MURRAY)"}[key]
            if ctrl.state == ctrl.ARMED:
                banner = f"{name} | IMPACT IN {ctrl.countdown_ms(now):.0f} ms"
            elif ctrl.state == ctrl.STANCE and stance_phase is not None:
                banner = f"{name} | STAIR STANCE ({stance_phase:.0f}%)"
            elif phase is not None:
                banner = f"{name} | GAIT CYCLE {phase:.0f}%"
            else:
                banner = f"{name} | PAUSED"
            torque = float(np.interp(phase, profiles.x, active)) if phase is not None else 0.0
            peak = float(active.max())

            warnings = []
            if camera.synthetic:
                warnings.append("SYNTHETIC INPUT: predictions not meaningful")
            if not getattr(engine, "trained", True):
                warnings.append("UNTRAINED WEIGHTS")
            if not auto:
                warnings.append("MANUAL OVERRIDE")
            if getattr(engine, "suspect_normalized", False):
                warnings.append("OUTPUTS LOOK NORMALIZED: check --output-map")
            if worker.skipped:
                warnings.append(f"SKIPPED FRAMES: {worker.skipped}")
            if paused:
                warnings.append("PAUSED")

            with worker.lock:
                latest = worker.latest
            left, right, tensor = latest if latest else (None, None, None)
            shown_toi = (manual_toi if not auto else (ctrl.toi if ctrl.toi is not None else 0.0))
            shown_cop = (manual_cop if not auto else (ctrl.cop if ctrl.cop is not None else 0.0))
            all_max = max(float(c.max()) for c in curves.values())
            frame = hud.render(dict(
                synthetic=camera.synthetic, warnings=warnings, rate=worker.rate,
                infer_ms=worker.infer_ms, hud_fps=hud_fps, left=left, right=right, tensor=tensor,
                toi=shown_toi, cop=shown_cop, toi_max=max(args.toi_range[1], 1.0), emd=args.emd_ms,
                banner=banner, profile_key=key_draw, x=profiles.x, curves=curves,
                active_curve=active, max_torque=max(10.0, all_max * 1.15), phase=phase,
                torque=torque, assist=assist, peak=peak, mass=mass,
                current_ma=motor_current_ma(peak, args.kt, args.gear_ratio)))

            # 3. display + keys
            if args.headless:
                time.sleep(max(0.0, period - (time.perf_counter() - loop_t0)))
            else:
                cv2.imshow(WINDOW, frame)
                wait = max(1, int((period - (time.perf_counter() - loop_t0)) * 1000))
                k = cv2.waitKeyEx(wait)
                if k != -1:
                    arrow = ARROWS.get(k)
                    ch = chr(k & 0xFF).lower() if arrow is None else ""
                    if ch == "q" or k == 27:
                        break
                    elif ch == " ":
                        paused = not paused
                        worker.paused = paused
                        if not paused:
                            worker.reset_req = True
                            ctrl.reset()
                    elif ch == "a":
                        auto = not auto
                    elif ch in ("c", "h", "t"):
                        auto = False
                        manual_toi, manual_cop = {"c": (180.0, manual_cop), "h": (30.0, 70.0),
                                                  "t": (30.0, 190.0)}[ch]
                    elif ch == "g":
                        ctrl.force_stance(time.perf_counter())
                    elif ch == "r":
                        worker.reset_req = True
                        ctrl.reset()
                    elif ch == "b":
                        assist = 0.25 if assist == 1.0 else 1.0
                    elif ch in ("+", "="):
                        mass += 5.0
                    elif ch in ("-", "_"):
                        mass = max(30.0, mass - 5.0)
                    elif ch == "i" or arrow == "up":
                        manual_toi, auto = min(args.toi_range[1], manual_toi + 10.0), False
                    elif ch == "k" or arrow == "down":
                        manual_toi, auto = max(0.0, manual_toi - 10.0), False
                    elif ch == "j" or arrow == "left":
                        manual_cop, auto = max(0.0, manual_cop - 10.0), False
                    elif ch == "l" or arrow == "right":
                        manual_cop, auto = min(FOOT_LEN_MM, manual_cop + 10.0), False

            dt = time.perf_counter() - t_prev
            t_prev = time.perf_counter()
            if dt > 0:
                hud_fps = 0.9 * hud_fps + 0.1 / dt if hud_fps else 1.0 / dt
    finally:
        worker.stop_evt.set()
        camera.close()
        if args.save_frame and frame is not None:
            cv2.imwrite(args.save_frame, frame)
            print(f"[INFO] saved HUD frame to {args.save_frame}")
        if not args.headless:
            cv2.destroyAllWindows()


def build_parser():
    p = argparse.ArgumentParser(description="Live OAK-D torque profile visualizer")
    p.add_argument("--weights", help="combined checkpoint (model with 2 outputs)")
    p.add_argument("--toi-weights", help="TOI-only checkpoint (1 output)")
    p.add_argument("--cop-weights", help="COP-only checkpoint (1 output)")
    p.add_argument("--untrained", action="store_true", help="random weights (plumbing test only)")
    p.add_argument("--hidden1", type=int, default=32, help="CONFIRM with the trained model")
    p.add_argument("--hidden2", type=int, default=64, help="CONFIRM with the trained model")
    p.add_argument("--combined-order", default="toi,cop", help="output order of a combined model")
    p.add_argument("--output-map", choices=["raw", "unit", "sigmoid"], default="unit",
                   help="unit: map normalized [0,1] to ranges below (default). raw: outputs already in ms/mm")
    p.add_argument("--toi-range", type=float, nargs=2, default=[0.0, 250.0], metavar=("LO", "HI"))
    p.add_argument("--cop-range", type=float, nargs=2, default=[0.0, FOOT_LEN_MM], metavar=("LO", "HI"))
    p.add_argument("--resize", choices=["area", "linear"], default="linear",
                   help="downscaling filter; defaults to linear matching repo preprocessing")
    p.add_argument("--device", default=None)
    p.add_argument("--demo", action="store_true", help="synthetic camera (no OAK-D)")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--mass", type=float, default=75.0)
    p.add_argument("--blend", action="store_true", help="blend heel/toe curves by COP")
    p.add_argument("--bio", action="store_true", help="show Murray curves at 100%% biological torque")
    p.add_argument("--params-dir", default=None, help="folder with heel_params.json / toe_params.json")
    p.add_argument("--emd-ms", type=float, default=50.0)
    p.add_argument("--stride-time", type=float, default=1.2, help="seconds per gait cycle (time-based cursor)")
    p.add_argument("--stance-end", type=float, default=65.0, help="percent of cycle where the cursor ends")
    p.add_argument("--comm-latency-ms", type=float, default=0.0)
    p.add_argument("--kt", type=float, default=140.0, help="motor torque constant, Nmm/A")
    p.add_argument("--gear-ratio", type=float, default=None, help="transmission ratio (needed for motor current)")
    p.add_argument("--check-model", action="store_true", help="run 3 random frames through the model and exit")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--save-frame", default=None)
    return p


def main():
    args = build_parser().parse_args()
    interp = cv2.INTER_AREA if args.resize == "area" else cv2.INTER_LINEAR

    try:
        engine = ForecasterEngine(args)
    except Exception as e:
        print(f"[ERROR] could not set up the model: {e}")
        traceback.print_exc()
        sys.exit(1)

    if args.check_model:
        x = np.random.rand(1, 2, HEIGHT, WIDTH).astype(np.float32)
        for i in range(3):
            toi, cop, ms = engine.step(x)
            print(f"frame {i}: toi={toi:.3f}  cop={cop:.3f}  ({ms:.1f} ms)")
        print("Model API check passed (shapes OK). Values on random input are meaningless.")
        return

    if args.demo:
        camera = SyntheticCamera(args.fps, interp)
    else:
        try:
            camera = OakCamera(args.fps, interp)
        except Exception as e:
            print(f"[INFO] Physical OAK-D not detected ({e}).")
            print("[INFO] Automatically switching to synthetic test camera.")
            camera = SyntheticCamera(args.fps, interp)
    run(args, camera, engine)


if __name__ == "__main__":
    main()