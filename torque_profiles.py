import json
import os

import numpy as np

HEEL_COEFFICIENTS = [
    -1.3418952053455997e-07, 1.7845671326326224e-05, -0.0008115159297602824,
    0.014257060759835865, -0.05348055449407867, 0.05081448731224058,
]
MID_COEFFICIENTS = [
    -1.8018684186184578e-07, 2.393482843276188e-05, -0.0010667010483254012,
    0.01686895559962403, -0.013305050756574633, 0.052798802400771463,
]
TOE_COEFFICIENTS = [
    -2.6283254641545553e-07, 3.629269408838788e-05, -0.001707299481976467,
    0.029826242570955334, -0.08755150988293577, 0.1387268050697176,
]

FALLBACK_COEFFICIENTS = {
    "heel": HEEL_COEFFICIENTS,
    "mid": MID_COEFFICIENTS,
    "toe": TOE_COEFFICIENTS,
}


def load_coefficients(params_dir, name, fallback):
    if params_dir:
        path = os.path.join(params_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "r") as handle:
                    coefficients = json.load(handle)
                if len(coefficients) == 6:
                    return [float(coefficient) for coefficient in coefficients]
                print(f"[WARN] {path} does not hold 6 coefficients; using fallback.")
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}. Using fallback.")
    return list(fallback)


def load_profile_coefficients(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    params_dir = os.path.join(
        base_dir, "Dephy-Soft-Sensor-integrate_cop_toi_ml", "params"
    )
    return {
        "heel": load_coefficients(params_dir, "heel_params.json", HEEL_COEFFICIENTS),
        "mid": load_coefficients(params_dir, "mid_params.json", MID_COEFFICIENTS),
        "toe": load_coefficients(params_dir, "toe_params.json", TOE_COEFFICIENTS),
    }


def collins_profile(percent_gait, user_mass, peak_torque_normalized=0.20):
    t0 = 0.0
    t1 = 27.1
    t2 = 50.4
    t3 = 62.7
    ts = 2.0
    tp = user_mass * peak_torque_normalized

    x = np.asarray(percent_gait, dtype=np.float32)
    tau = np.zeros_like(x)

    m1 = (x >= t0) & (x <= t1)
    tau[m1] = ts / (t1 - t0) * (x[m1] - t0)

    m2 = (x > t1) & (x <= t2)
    denom1 = (t1 - t2) ** 3
    a1 = (2.0 * (tp - ts)) / denom1
    b1 = -3.0 * (t1 + t2) * (tp - ts) / denom1
    c1 = 6.0 * t1 * t2 * (tp - ts) / denom1
    d1 = -(-(t1 ** 3) * tp + 3 * (t1 ** 2) * t2 * tp - 3 * t1 * (t2 ** 2) * ts + (t2 ** 3) * ts) / denom1
    tau[m2] = a1 * (x[m2] ** 3) + b1 * (x[m2] ** 2) + c1 * x[m2] + d1

    m3 = (x > t2) & (x <= t3)
    denom2 = 2.0 * ((t2 - t3) ** 3)
    a2 = -(tp - ts) / denom2
    b2 = 3.0 * t3 * (tp - ts) / denom2
    c2 = 3.0 * (t2 ** 2 - 2 * t2 * t3) * (tp - ts) / denom2
    d2 = -(3 * (t2 ** 2) * t3 * tp - 6 * t2 * (t3 ** 2) * tp + 2 * (t3 ** 3) * tp - 2 * (t2 ** 3) * ts + 3 * (t2 ** 2) * t3 * ts) / denom2
    tau[m3] = a2 * (x[m3] ** 3) + b2 * (x[m3] ** 2) + c2 * x[m3] + d2

    return np.maximum(tau, 0.0)


def poly5_profile(percent_gait, coeffs, user_mass, scale_factor=0.25):
    x = np.asarray(percent_gait, dtype=np.float32)
    val = sum(c * (x ** (5 - i)) for i, c in enumerate(coeffs))
    mask = (x > 3.0) & (x <= 60.0) & (val > 0.0)
    tau = np.zeros_like(x)
    tau[mask] = val[mask] * user_mass * scale_factor
    return np.maximum(tau, 0.0)
