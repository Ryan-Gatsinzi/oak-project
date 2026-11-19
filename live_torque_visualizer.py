"""
Live OAK-D Camera Feed & Real-Time Predicted Torque Profile Visualizer
======================================================================
Integrates OAK-D stereo sensing with the continuous COP/TOI terrain forecasting
model and the Dephy ExoBoot torque control laws (Zhang/Collins Level Ground & 
Murray 5th-Degree Polynomial Stair Ascent Profiles).

Features:
- Real-time DepthAI stereo capture (Left & Right cameras @ 60 FPS)
- Automatic fallback to synthetic test bench if no physical camera is detected
- Recurrent neural network inference predicting COP (Center-of-Pressure) and, when TOI weights are supplied, TOI (Time-of-Impact)
- Dynamic switching / continuous blending between:
    * Collins Level-Ground Profile (Cubic Splines)
    * Murray Stair Ascent Heel-Strike Profile (5th-degree Polynomial)
    * Murray Stair Ascent Toe-Strike Profile (5th-degree Polynomial)
- High-performance, zero-latency OpenCV HUD and real-time interactive torque graph
"""

import os
import sys
import time
import argparse
from datetime import timedelta
from collections import deque

import cv2
import numpy as np

# Import the validated COP_forecast live inference engine
try:
    from cop_forecast_engine import (
        COPForecastEngine,
        DEFAULT_COP_WEIGHTS,
        DEFAULT_FOOT_LENGTH_MM,
    )
except ImportError:
    # If executed from a different directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cop_forecast_engine import (
        COPForecastEngine,
        DEFAULT_COP_WEIGHTS,
        DEFAULT_FOOT_LENGTH_MM,
    )

from torque_profiles import (
    collins_profile as _collins_profile,
    load_profile_coefficients,
    poly5_profile as _poly5_profile,
)
from torque_interpolation import (
    HEEL,
    MID,
    TOE,
    blend_profiles,
    cop_interpolation_weights,
    select_discrete_profile,
)

# Optional DepthAI import
try:
    import depthai as dai
    HAS_DEPTHAI = True
except ImportError:
    HAS_DEPTHAI = False


# =============================================================================
# 1. BIOMECHANICS & TORQUE PROFILE LAWS
# =============================================================================

class TorqueProfileManager:
    """Computes Collins level ground and Murray stair ascent profiles."""

    def __init__(self, base_dir=None, user_mass=75.0, kt=140.0):
        self.user_mass = user_mass
        self.kt = kt  # Torque constant: 140 Nmm/A
        self.mid_foot_mm = 263.2 / 2.0  # 131.6 mm
        self.emd_ms = 50.0  # Electromechanical delay in ms

        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        coefficients = load_profile_coefficients(base_dir)
        self.heel_params = coefficients[HEEL]
        self.mid_params = coefficients[MID]
        self.toe_params = coefficients[TOE]

    def collins_profile(self, percent_gait, peak_torque_normalized=0.20):
        return _collins_profile(percent_gait, self.user_mass, peak_torque_normalized)

    def poly5_profile(self, percent_gait, coeffs, scale_factor=0.25):
        return _poly5_profile(percent_gait, coeffs, self.user_mass, scale_factor)

    def heel_profile(self, percent_gait, scale_factor=0.25):
        return self.poly5_profile(percent_gait, self.heel_params, scale_factor=scale_factor)

    def mid_profile(self, percent_gait, scale_factor=0.25):
        return self.poly5_profile(percent_gait, self.mid_params, scale_factor=scale_factor)

    def toe_profile(self, percent_gait, scale_factor=0.25):
        return self.poly5_profile(percent_gait, self.toe_params, scale_factor=scale_factor)

    def reference_curves(self, percent_gait, scale_factor=0.25):
        return {
            "collins": self.collins_profile(percent_gait),
            HEEL: self.heel_profile(percent_gait, scale_factor=scale_factor),
            MID: self.mid_profile(percent_gait, scale_factor=scale_factor),
            TOE: self.toe_profile(percent_gait, scale_factor=scale_factor),
        }

    def torque_to_motor_current_ma(self, torque_nm):
        """Converts Nm to mA for Dephy ExoBoot actuator (kt = 140 Nmm/A)."""
        # torque_nmm = torque_nm * 1000.0
        # current_a = torque_nmm / kt
        # current_ma = current_a * 1000.0
        return (torque_nm * 1000.0 / self.kt) * 1000.0


# =============================================================================
# 2. CAMERA PIPELINE & SYNTHETIC BENCH
# =============================================================================

class CameraStreamManager:
    """Manages DepthAI stereo camera or synthetic benchmark fallback."""

    def __init__(self, fps=60, width=50, height=25, force_demo=False,
                 mono_resolution=400):
        self.fps = fps
        self.width = width
        self.height = height
        self.mono_resolution = mono_resolution
        self.is_synthetic = force_demo
        self.pipeline = None
        self.queue = None
        self.sim_phase = 0.0
        self.last_left = None
        self.last_right = None
        self.last_tensor = None

        if not force_demo and HAS_DEPTHAI:
            self._init_depthai()
        else:
            self.is_synthetic = True

    def _init_depthai(self):
        try:
            available_devices = dai.Device.getAllAvailableDevices()
            if len(available_devices) == 0:
                print("[INFO] No physical OAK-D device detected. Initializing synthetic benchmark.")
                self.is_synthetic = True
                return

            info = available_devices[0]
            device_id = getattr(info, "deviceId", None) or getattr(info, "getMxId", lambda: "unknown")()
            print(f"[INFO] Found OAK-D device: {device_id}. Starting DepthAI pipeline...")
            pipeline = dai.Pipeline()
            left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

            if self.mono_resolution == 720:
                output_size = (1280, 720)
            else:
                output_size = (640, 400)
            left_out = left_cam.requestOutput(output_size, type=dai.ImgFrame.Type.GRAY8, fps=self.fps)
            right_out = right_cam.requestOutput(output_size, type=dai.ImgFrame.Type.GRAY8, fps=self.fps)

            sync = pipeline.create(dai.node.Sync)
            sync.setRunOnHost(True)
            sync.setSyncThreshold(timedelta(milliseconds=15))

            left_out.link(sync.inputs["left"])
            right_out.link(sync.inputs["right"])

            self.queue = sync.out.createOutputQueue(maxSize=2, blocking=False)
            pipeline.start()
            self.pipeline = pipeline
            self.is_synthetic = False
            print(f"[INFO] OAK-D stereo streaming {output_size[0]}x{output_size[1]} @ {self.fps} FPS.")

            try:
                usb_speed = pipeline.getDefaultDevice().getUsbSpeed()
                print(f"[INFO] OAK-D USB speed: {usb_speed}")
                if output_size == (1280, 720) and usb_speed in (dai.UsbSpeed.HIGH, dai.UsbSpeed.FULL):
                    print("[WARN] 720p stereo needs USB3. Use --resolution 400 if the device crashes.")
            except Exception as exc:
                print(f"[WARN] Could not read OAK-D USB speed: {exc}")
        except Exception as e:
            print(f"[WARN] DepthAI initialization failed: {e}. Switching to synthetic benchmark.")
            self.is_synthetic = True

    def get_stereo_pair(self):
        """
        Returns:
            left_raw: (720, 1280) uint8 for OAK-D, (400, 640) in synthetic mode
            right_raw: same shape as left_raw
            stereo_np: (2, height, width) float32 normalized [0, 1]
        """
        if not self.is_synthetic and self.queue is not None:
            group = self.queue.tryGet()
            if group is not None:
                try:
                    left_raw = group["left"].getCvFrame()
                    right_raw = group["right"].getCvFrame()
                    self.last_left = left_raw
                    self.last_right = right_raw
                except Exception as e:
                    print(f"[WARN] Failed to decode DepthAI frame: {e}")
                    left_raw = self.last_left
                    right_raw = self.last_right
            else:
                left_raw = self.last_left
                right_raw = self.last_right

            if left_raw is None or right_raw is None:
                return None, None, None
        else:
            # Synthetic frame generator simulating descending/approaching stair patterns
            self.sim_phase += 0.05
            h, w = 400, 640
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            stair_bars = ((y_coords + int(self.sim_phase * 20)) % 80 < 40).astype(np.float32) * 160 + 40
            noise = np.random.normal(0, 8, (h, w)).astype(np.float32)
            left_raw = np.clip(stair_bars + noise, 0, 255).astype(np.uint8)
            # Add horizontal stereo disparity
            right_raw = np.roll(left_raw, shift=-12, axis=1)

        # Match COP_forecast training preprocessing: 640x400 -> 427x240 -> 50x25, /255
        def _prep(img):
            img = cv2.resize(img, (427, 240), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            return img.astype(np.float32) / 255.0

        l_small = _prep(left_raw)
        r_small = _prep(right_raw)
        stereo_np = np.stack([l_small, r_small], axis=0)  # (2, 25, 50)

        return left_raw, right_raw, stereo_np

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass


# =============================================================================
# 3. REAL-TIME FORECASTER ENGINE
# =============================================================================

class ForecasterEngine:
    """Wraps the trained COP_forecast recurrent model for real-time COP/TOI."""

    def __init__(self, device=None, cop_weights=None, toi_weights=None,
                 foot_length_mm=DEFAULT_FOOT_LENGTH_MM):
        self.engine = COPForecastEngine(
            cop_weights=cop_weights or DEFAULT_COP_WEIGHTS,
            toi_weights=toi_weights,
            foot_length_mm=foot_length_mm,
            device=device,
        )
        self.has_toi = self.engine.has_toi
        self.foot_length_mm = self.engine.foot_length_mm

    def forward(self, stereo_np):
        """
        Passes one preprocessed stereo timestep through the recurrent model.
        Returns:
            toi_ms: float [0, 250] ms, or None when no TOI weights are loaded
            cop_mm: float [0, foot_length_mm]
            latency_ms: float
        """
        toi_ms, cop_mm, latency_ms, _ = self.engine.step(stereo_np=stereo_np)

        if toi_ms is not None:
            # Latency compensation: TOI_adj = TOI - (t_infer + t_comm)
            toi_ms = max(0.0, toi_ms - latency_ms)

        return toi_ms, cop_mm, latency_ms


# =============================================================================
# 4. HIGH-PERFORMANCE OPENCV DASHBOARD & GRAPH RENDERER
# =============================================================================

class RealTimeVisualizerHUD:
    """Renders real-time HUD with camera feeds, biomechanics gauges, and the torque profile graph."""

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Color Palette (BGR)
        self.C_BG = (24, 28, 34)
        self.C_PANEL = (35, 41, 51)
        self.C_BORDER = (55, 65, 81)
        self.C_TEXT = (240, 243, 246)
        self.C_DIM = (140, 150, 160)
        self.C_CYAN = (235, 175, 40)      # Level ground / Collins
        self.C_AMBER = (50, 185, 255)     # Stair Heel
        self.C_MID = (90, 220, 130)       # Stair Midfoot
        self.C_ORANGE = (40, 110, 255)    # Stair Toe
        self.C_CONTINUOUS = (220, 90, 220)
        self.C_GREEN = (80, 210, 100)
        self.C_RED = (80, 80, 240)

        # Graph coordinates within the canvas
        self.gx0 = 550
        self.gy0 = 100
        self.gw = 690
        self.gh = 450

    def draw_panel(self, x, y, w, h, bg_color=None, border_color=None):
        if bg_color is None:
            bg_color = self.C_PANEL
        if border_color is None:
            border_color = self.C_BORDER
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), bg_color, -1)
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), border_color, 1)

    def draw_graph_grid(self, max_torque):
        # Fill plot background
        cv2.rectangle(self.canvas, (self.gx0, self.gy0), (self.gx0 + self.gw, self.gy0 + self.gh), (18, 22, 28), -1)
        cv2.rectangle(self.canvas, (self.gx0, self.gy0), (self.gx0 + self.gw, self.gy0 + self.gh), self.C_BORDER, 1)

        # X-axis ticks (Gait % from 0 to 100)
        for pct in range(0, 101, 20):
            px = int(self.gx0 + (pct / 100.0) * self.gw)
            cv2.line(self.canvas, (px, self.gy0), (px, self.gy0 + self.gh), (32, 38, 48), 1)
            cv2.putText(self.canvas, f"{pct}%", (px - 14, self.gy0 + self.gh + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_DIM, 1, cv2.LINE_AA)

        # Y-axis ticks (Torque in Nm)
        y_step = 10 if max_torque <= 40 else 20
        y_ticks = np.arange(0, max_torque + 5, y_step)
        for val in y_ticks:
            py = int(self.gy0 + self.gh - (val / max_torque) * self.gh)
            if self.gy0 <= py <= self.gy0 + self.gh:
                cv2.line(self.canvas, (self.gx0, py), (self.gx0 + self.gw, py), (32, 38, 48), 1)
                cv2.putText(self.canvas, f"{int(val)}", (self.gx0 - 32, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_DIM, 1, cv2.LINE_AA)

        # Axis Labels
        cv2.putText(self.canvas, "Gait Cycle (%)", (self.gx0 + self.gw // 2 - 45, self.gy0 + self.gh + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(self.canvas, "Torque (Nm)", (self.gx0 - 35, self.gy0 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.C_TEXT, 1, cv2.LINE_AA)

    def plot_curve(self, x_pts, y_pts, max_torque, color, thickness=2, is_active=True):
        pts = []
        for x, y in zip(x_pts, y_pts):
            px = int(self.gx0 + (x / 100.0) * self.gw)
            py = int(self.gy0 + self.gh - (y / max_torque) * self.gh)
            pts.append([px, py])

        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        if is_active:
            cv2.polylines(self.canvas, [pts], False, color, thickness + 2, cv2.LINE_AA)
        else:
            # Dimmed reference line
            dim_color = tuple(int(c * 0.35) for c in color)
            cv2.polylines(self.canvas, [pts], False, dim_color, 1, cv2.LINE_AA)

    def draw_gait_cursor(self, percent_gait, current_torque, max_torque, active_color):
        px = int(self.gx0 + (percent_gait / 100.0) * self.gw)
        py = int(self.gy0 + self.gh - (current_torque / max_torque) * self.gh)

        # Vertical cursor line
        cv2.line(self.canvas, (px, self.gy0), (px, self.gy0 + self.gh), (90, 110, 140), 1, cv2.LINE_AA)

        # Animated contact point
        cv2.circle(self.canvas, (px, py), 7, active_color, -1, cv2.LINE_AA)
        cv2.circle(self.canvas, (px, py), 11, (255, 255, 255), 1, cv2.LINE_AA)

        # Instantaneous readout tooltip
        tip = f"{current_torque:.1f} Nm"
        cv2.putText(self.canvas, tip, (px + 10, max(self.gy0 + 20, py - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    def render_frame(self, left_raw, right_raw, stereo_np, toi_ms, cop_mm,
                     state_str, active_curve_y, x_grid, all_curves,
                     continuous_curve_y, weights,
                     current_gait_pct, current_torque, user_mass, fps, latency_ms,
                     is_synthetic=False, is_paused=False, assist_scale=0.25,
                     toi_source="MODEL", mid_foot_mm=131.6,
                     toe_mm=263.2, controller_mode="continuous"):

        # Reset canvas background
        self.canvas[:] = self.C_BG

        # ----------------- Top Header Bar -----------------
        self.draw_panel(20, 15, self.width - 40, 60)
        title = "OAK-D Real-Time Terrain Forecaster & ExoBoot Torque Profile"
        cv2.putText(self.canvas, title, (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.C_TEXT, 2, cv2.LINE_AA)

        # Source mode tag
        tag_str = "[ SYNTHETIC BENCH ]" if is_synthetic else "[ OAK-D STEREO ACTIVE ]"
        tag_color = self.C_AMBER if is_synthetic else self.C_GREEN
        cv2.putText(self.canvas, tag_str, (820, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, tag_color, 2, cv2.LINE_AA)

        fps_str = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms"
        cv2.putText(self.canvas, fps_str, (1060, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.C_DIM, 1, cv2.LINE_AA)

        # ----------------- Left Panel: Camera Feeds -----------------
        cam_panel_w = 490
        self.draw_panel(20, 90, cam_panel_w, 240)
        cv2.putText(self.canvas, "STEREO FEED (CAM_B & CAM_C)", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.C_TEXT, 1, cv2.LINE_AA)

        if left_raw is not None and right_raw is not None:
            # Resize left/right thumbnails
            thumb_w = 215
            thumb_h = 135
            l_thumb = cv2.resize(left_raw, (thumb_w, thumb_h))
            r_thumb = cv2.resize(right_raw, (thumb_w, thumb_h))

            # Convert to BGR if grayscale
            if len(l_thumb.shape) == 2:
                l_thumb = cv2.cvtColor(l_thumb, cv2.COLOR_GRAY2BGR)
                r_thumb = cv2.cvtColor(r_thumb, cv2.COLOR_GRAY2BGR)

            self.canvas[130:130+thumb_h, 35:35+thumb_w] = l_thumb
            self.canvas[130:130+thumb_h, 265:265+thumb_w] = r_thumb

            cv2.putText(self.canvas, "Left (CAM_B)", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_DIM, 1)
            cv2.putText(self.canvas, "Right (CAM_C)", (270, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_DIM, 1)

            # Neural Network Input Preview (50x25 enlarged)
            l_small_map = (stereo_np[0] * 255).astype(np.uint8)
            l_small_bgr = cv2.applyColorMap(l_small_map, cv2.COLORMAP_VIRIDIS)
            l_small_view = cv2.resize(l_small_bgr, (90, 45), interpolation=cv2.INTER_NEAREST)
            self.canvas[270:270+45, 390:390+90] = l_small_view
            cv2.putText(self.canvas, "NN 25x50", (390, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_DIM, 1)

        # ----------------- Middle Panel: Biomechanical Prediction HUD -----------------
        self.draw_panel(20, 345, cam_panel_w, 230)
        cv2.putText(self.canvas, "TERRAIN INTERACTION FORECAST", (35, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.C_TEXT, 1, cv2.LINE_AA)

        # 1. TOI Gauge Bar
        cv2.putText(self.canvas, f"Time-of-Impact (TOI): {toi_ms:.1f} ms  [{toi_source}]", (35, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.C_TEXT, 1)
        bar_x, bar_y, bar_w, bar_h = 35, 420, 440, 16
        cv2.rectangle(self.canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (25, 30, 38), -1)
        fill_toi = int(np.clip(toi_ms / 250.0, 0.0, 1.0) * bar_w)
        toi_color = self.C_CYAN if toi_ms > 50.0 else self.C_AMBER
        cv2.rectangle(self.canvas, (bar_x, bar_y), (bar_x + fill_toi, bar_y + bar_h), toi_color, -1)
        cv2.rectangle(self.canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.C_BORDER, 1)
        # EMD Threshold marker at 50 ms
        emd_x = int(bar_x + (50.0 / 250.0) * bar_w)
        cv2.line(self.canvas, (emd_x, bar_y - 4), (emd_x, bar_y + bar_h + 4), self.C_RED, 2)
        cv2.putText(self.canvas, "EMD (50ms)", (emd_x - 30, bar_y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_RED, 1)

        # 2. COP Gauge Bar
        cv2.putText(self.canvas, f"Center-of-Pressure (COP): {cop_mm:.1f} mm", (35, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.C_TEXT, 1)
        bar2_y = 485
        cv2.rectangle(self.canvas, (bar_x, bar2_y), (bar_x + bar_w, bar2_y + bar_h), (25, 30, 38), -1)
        fill_cop = int(np.clip(cop_mm / toe_mm, 0.0, 1.0) * bar_w)
        if controller_mode == "continuous":
            cop_color = self.C_CONTINUOUS
        else:
            cop_color = self.C_AMBER if cop_mm <= mid_foot_mm else self.C_ORANGE
        cv2.rectangle(self.canvas, (bar_x, bar2_y), (bar_x + fill_cop, bar2_y + bar_h), cop_color, -1)
        cv2.rectangle(self.canvas, (bar_x, bar2_y), (bar_x + bar_w, bar2_y + bar_h), self.C_BORDER, 1)
        # Midfoot threshold
        mf_x = int(bar_x + (mid_foot_mm / toe_mm) * bar_w)
        cv2.line(self.canvas, (mf_x, bar2_y - 4), (mf_x, bar2_y + bar_h + 4), (255, 255, 255), 2)
        cv2.putText(self.canvas, "Heel", (bar_x, bar2_y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_DIM, 1)
        cv2.putText(self.canvas, f"Midfoot ({mid_foot_mm:.0f}mm)", (mf_x - 40, bar2_y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(self.canvas, "Toe", (bar_x + bar_w - 25, bar2_y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_DIM, 1)

        # 3. Active State Banner
        if "LEVEL" in state_str:
            state_color = self.C_CYAN
        elif "CONTINUOUS" in state_str:
            state_color = self.C_CONTINUOUS
        elif "HEEL" in state_str:
            state_color = self.C_AMBER
        elif "MID" in state_str:
            state_color = self.C_MID
        else:
            state_color = self.C_ORANGE
        cv2.rectangle(self.canvas, (35, 525), (35 + bar_w, 565), (28, 34, 44), -1)
        cv2.rectangle(self.canvas, (35, 525), (35 + bar_w, 565), state_color, 2)
        cv2.putText(self.canvas, f"STATE: {state_str}", (55, 553), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2, cv2.LINE_AA)

        # ----------------- Right Panel: Real-Time Torque Graph -----------------
        max_torque = 35.0 if assist_scale <= 0.3 else 90.0
        self.draw_graph_grid(max_torque)

        reference_curves = (
            ("collins", self.C_CYAN, "LEVEL"),
            (HEEL, self.C_AMBER, "HEEL"),
            (MID, self.C_MID, "MID"),
            (TOE, self.C_ORANGE, "TOE"),
        )
        for key, color, state_token in reference_curves:
            active = controller_mode == "discrete" and state_token in state_str
            self.plot_curve(x_grid, all_curves[key], max_torque, color, thickness=1, is_active=active)

        self.plot_curve(
            x_grid, continuous_curve_y, max_torque, self.C_CONTINUOUS,
            thickness=2, is_active=controller_mode == "continuous",
        )
        self.plot_curve(x_grid, active_curve_y, max_torque, state_color, thickness=3, is_active=True)
        self.draw_gait_cursor(current_gait_pct, current_torque, max_torque, state_color)

        leg_x = self.gx0 + 20
        leg_y = self.gy0 + 25
        legend_entries = (
            (self.C_CYAN, "Collins (level)"),
            (self.C_AMBER, f"Heel x{assist_scale:.2f}"),
            (self.C_MID, f"Mid x{assist_scale:.2f}"),
            (self.C_ORANGE, f"Toe x{assist_scale:.2f}"),
            (self.C_CONTINUOUS, "Continuous COP"),
        )
        for index, (color, label) in enumerate(legend_entries):
            ox = leg_x + (index % 3) * 215
            oy = leg_y + (index // 3) * 22
            cv2.line(self.canvas, (ox, oy), (ox + 25, oy), color, 2)
            cv2.putText(self.canvas, label, (ox + 33, oy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_TEXT, 1)

        # ----------------- Bottom Metrics & Controls -----------------
        self.draw_panel(self.gx0, 565, self.gw, 72)
        peak_torque = float(np.max(active_curve_y))
        peak_current_ma = (peak_torque * 1000.0 / 140.0) * 1000.0  # kt = 140 Nmm/A
        metrics_str = f"Peak Torque: {peak_torque:.1f} Nm | Peak Current: {peak_current_ma:.0f} mA | User Mass: {user_mass:.0f} kg"
        cv2.putText(self.canvas, metrics_str, (self.gx0 + 25, 598), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.C_TEXT, 1, cv2.LINE_AA)
        weights_str = (
            f"COP: {cop_mm:.1f} mm | w_heel: {weights[HEEL]:.2f} | "
            f"w_mid: {weights[MID]:.2f} | w_toe: {weights[TOE]:.2f}"
        )
        cv2.putText(self.canvas, weights_str, (self.gx0 + 25, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.C_DIM, 1, cv2.LINE_AA)

        # Bottom Shortcut Legend
        self.draw_panel(20, 640, self.width - 40, 65)
        ctrl_title = "CONTROLS: [Q] Quit | [SPACE] Pause | [A] Auto/Interactive | [V] Continuous/Discrete | [C/H/T] Presets"
        ctrl_sub = "[UP/DOWN] TOI | [LEFT/RIGHT] COP | [+ / -] Mass | [B] Bio/Assist"
        cv2.putText(self.canvas, ctrl_title, (40, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(self.canvas, ctrl_sub, (40, 692), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_DIM, 1, cv2.LINE_AA)

        return self.canvas


# =============================================================================
# 5. MAIN APPLICATION CONTROLLER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real-Time OAK-D Predicted Torque Profile Visualizer")
    parser.add_argument("--demo", action="store_true", help="Force synthetic test bench without searching for OAK-D camera")
    parser.add_argument("--cop-weights", type=str, default=DEFAULT_COP_WEIGHTS,
                        help="Path to trained COP_forecast COP checkpoint (default: base_models_pred_cop/23.torch)")
    parser.add_argument("--toi-weights", type=str, default=None,
                        help="Optional path to trained COP_forecast TOI checkpoint (e.g. 82_time_base.torch)")
    parser.add_argument("--weights", type=str, default=None, help="Deprecated alias for --cop-weights")
    parser.add_argument("--foot-length-mm", type=float, default=DEFAULT_FOOT_LENGTH_MM,
                        help=f"Foot length in mm used to rescale COP (default: {DEFAULT_FOOT_LENGTH_MM:.1f})")
    parser.add_argument("--toi-ms", type=float, default=30.0,
                        help="Manual TOI fallback in ms when no TOI weights are loaded (default: 30.0)")
    parser.add_argument("--resolution", type=int, choices=[400, 720], default=400,
                        help="OAK-D mono resolution: 400 (stable, USB2/3) or 720 (needs USB3). Default: 400")
    parser.add_argument("--mass", type=float, default=75.0, help="Subject body mass in kg (default: 75.0)")
    parser.add_argument("--controller", choices=["continuous", "discrete"], default="continuous",
                        help="COP controller: continuous heel/mid/toe interpolation, or original discrete controller")
    parser.add_argument("--blend", action="store_true", help="Legacy linear heel/toe blend; only used with --controller discrete")
    parser.add_argument("--bio", action="store_true", help="Display full biological torque scale instead of exo assistive scale")
    args = parser.parse_args()

    print("==================================================================")
    print("   OAK-D LIVE PREDICTIVE TORQUE PROFILE VISUALIZER               ")
    print("==================================================================")

    # Initialize subsystems
    profile_manager = TorqueProfileManager(user_mass=args.mass)
    profile_manager.mid_foot_mm = args.foot_length_mm / 2.0
    cam_manager = CameraStreamManager(
        fps=60, force_demo=args.demo, mono_resolution=args.resolution
    )
    cop_weights = args.cop_weights or args.weights or DEFAULT_COP_WEIGHTS
    forecaster = ForecasterEngine(
        cop_weights=cop_weights,
        toi_weights=args.toi_weights,
        foot_length_mm=args.foot_length_mm,
    )
    hud = RealTimeVisualizerHUD(width=1280, height=720)

    if not forecaster.has_toi:
        print(f"[WARN] No TOI weights loaded. Using manual TOI fallback: {args.toi_ms:.1f} ms")
        print("[WARN] Pass --toi-weights <path> to enable learned TOI (e.g. 82_time_base.torch).")

    # State variables
    auto_mode = True
    manual_toi = args.toi_ms  # ms
    manual_cop = forecaster.foot_length_mm / 2.0  # mm
    is_paused = False
    assist_scale = 1.0 if args.bio else 0.25
    controller_mode = args.controller

    if args.blend and controller_mode == "continuous":
        print("[WARN] --blend only affects --controller discrete; continuous interpolation is active.")

    # Simulation gait sweep
    gait_pct = 0.0
    gait_speed = 0.6  # % per frame at 60 FPS (~1.6 seconds per stride)

    # Rolling FPS tracker
    fps_history = deque(maxlen=60)
    last_time = time.perf_counter()

    # Precompute reference curves (0 to 100% gait cycle with 200 points)
    x_grid = np.linspace(0, 100, 200)

    cv2.namedWindow("OAK-D Real-Time Predictive Torque Visualizer", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            t_now = time.perf_counter()
            dt = t_now - last_time
            last_time = t_now
            if dt > 0:
                fps_history.append(1.0 / dt)
            cur_fps = sum(fps_history) / max(len(fps_history), 1)

            if not is_paused:
                # 1. Fetch stereo camera pair
                left_raw, right_raw, stereo_np = cam_manager.get_stereo_pair()
                if stereo_np is None:
                    # Waiting for the first camera frame to arrive from DepthAI
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    continue

                # 2. Run model inference or manual override
                if auto_mode:
                    toi_model, cop_pred, latency_ms = forecaster.forward(stereo_np)
                    toi_pred = toi_model if toi_model is not None else manual_toi
                else:
                    toi_pred = manual_toi
                    cop_pred = manual_cop
                    latency_ms = 0.5

                toi_source = "MODEL" if (auto_mode and forecaster.has_toi) else "MANUAL"

                all_curves = profile_manager.reference_curves(x_grid, scale_factor=assist_scale)
                weights = cop_interpolation_weights(
                    cop_pred,
                    heel_mm=0.0,
                    mid_mm=profile_manager.mid_foot_mm,
                    toe_mm=forecaster.foot_length_mm,
                )
                continuous_curve = blend_profiles(all_curves, weights)

                if toi_pred > profile_manager.emd_ms:
                    state_str = "LEVEL GROUND (COLLINS PROFILE)"
                    active_curve = all_curves["collins"]
                elif controller_mode == "continuous":
                    state_str = f"STAIR CONTINUOUS [COP={cop_pred:.0f}mm]"
                    active_curve = continuous_curve
                else:
                    profile_key = select_discrete_profile(cop_pred, profile_manager.mid_foot_mm)
                    if profile_key == HEEL:
                        state_str = f"STAIR ASCENT [HEEL STRIKE, COP={cop_pred:.0f}mm]"
                    else:
                        state_str = f"STAIR ASCENT [TOE STRIKE, COP={cop_pred:.0f}mm]"
                    active_curve = all_curves[profile_key]

                    if args.blend:
                        alpha = float(np.clip((cop_pred - 40.0) / 180.0, 0.0, 1.0))
                        active_curve = (1.0 - alpha) * all_curves[HEEL] + alpha * all_curves[TOE]
                        state_str = f"STAIR BLEND [COP={cop_pred:.0f}mm, α={alpha:.2f}]"

                gait_pct = (gait_pct + gait_speed) % 100.0

                idx = int(np.clip((gait_pct / 100.0) * len(x_grid), 0, len(x_grid) - 1))
                current_torque = float(active_curve[idx])

                # 4. Render visualizer frame
                frame = hud.render_frame(
                    left_raw=left_raw,
                    right_raw=right_raw,
                    stereo_np=stereo_np,
                    toi_ms=toi_pred,
                    cop_mm=cop_pred,
                    state_str=state_str,
                    active_curve_y=active_curve,
                    x_grid=x_grid,
                    all_curves=all_curves,
                    continuous_curve_y=continuous_curve,
                    weights=weights,
                    current_gait_pct=gait_pct,
                    current_torque=current_torque,
                    user_mass=profile_manager.user_mass,
                    fps=cur_fps,
                    latency_ms=latency_ms,
                    is_synthetic=cam_manager.is_synthetic,
                    is_paused=is_paused,
                    assist_scale=assist_scale,
                    toi_source=toi_source,
                    mid_foot_mm=profile_manager.mid_foot_mm,
                    toe_mm=forecaster.foot_length_mm,
                    controller_mode=controller_mode,
                )

                cv2.imshow("OAK-D Real-Time Predictive Torque Visualizer", frame)

            # 5. Handle Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("[INFO] Exiting visualizer.")
                break
            elif key == ord(' '):  # SPACE: pause/resume
                is_paused = not is_paused
                print(f"[INFO] Pause toggled: {is_paused}")
            elif key == ord('a'):  # Toggle Auto / Interactive manual mode
                auto_mode = not auto_mode
                print(f"[INFO] Auto-Inference mode: {auto_mode}")
            elif key == ord('v'):
                controller_mode = "discrete" if controller_mode == "continuous" else "continuous"
                print(f"[INFO] Controller: {controller_mode}")
            elif key == ord('c'):  # Force Collins
                auto_mode = False
                manual_toi = 180.0
                print("[INFO] Forced Collins Level Ground")
            elif key == ord('h'):  # Force Stair Heel
                auto_mode = False
                manual_toi = 30.0
                manual_cop = 70.0
                print("[INFO] Forced Stair Heel Strike")
            elif key == ord('t'):  # Force Stair Toe
                auto_mode = False
                manual_toi = 30.0
                manual_cop = 190.0
                print("[INFO] Forced Stair Toe Strike")
            elif key == ord('b'):  # Toggle Bio vs Assistive scale
                assist_scale = 0.25 if assist_scale == 1.0 else 1.0
                print(f"[INFO] Assist scale set to: {assist_scale}")
            elif key in [ord('+'), ord('=')]:
                profile_manager.user_mass += 5.0
                print(f"[INFO] User mass: {profile_manager.user_mass} kg")
            elif key in [ord('-'), ord('_')]:
                profile_manager.user_mass = max(30.0, profile_manager.user_mass - 5.0)
                print(f"[INFO] User mass: {profile_manager.user_mass} kg")
            # Arrow keys for manual TOI / COP manipulation (standard Windows keycodes)
            elif key == 2490368 or key == 0:  # UP
                manual_toi = min(250.0, manual_toi + 10.0)
                auto_mode = False
            elif key == 2621440 or key == 1:  # DOWN
                manual_toi = max(0.0, manual_toi - 10.0)
                auto_mode = False
            elif key == 2424832 or key == 2:  # LEFT
                manual_cop = max(0.0, manual_cop - 10.0)
                auto_mode = False
            elif key == 2555904 or key == 3:  # RIGHT
                manual_cop = min(forecaster.foot_length_mm, manual_cop + 10.0)
                auto_mode = False

    except RuntimeError as exc:
        print(f"\n[ERROR] OAK-D stream stopped: {exc}")
        print("[HINT] Unplug/replug the camera and retry, or use --resolution 400.")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Exiting visualizer.")
    finally:
        cam_manager.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
