import math

import numpy as np

HEEL = "heel"
MID = "mid"
TOE = "toe"
PROFILE_KEYS = (HEEL, MID, TOE)


def smoothstep(u):
    u = float(u)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * (3.0 - 2.0 * u)


def cop_interpolation_weights(cop_mm, heel_mm=0.0, mid_mm=131.6, toe_mm=263.2):
    heel_mm = float(heel_mm)
    mid_mm = float(mid_mm)
    toe_mm = float(toe_mm)

    if not all(math.isfinite(value) for value in (heel_mm, mid_mm, toe_mm)):
        raise ValueError("COP reference positions must be finite")
    if not heel_mm < mid_mm < toe_mm:
        raise ValueError("COP reference positions must satisfy heel < mid < toe")

    cop_mm = float(cop_mm)
    if math.isnan(cop_mm):
        cop_mm = mid_mm

    if cop_mm <= heel_mm:
        return {HEEL: 1.0, MID: 0.0, TOE: 0.0}
    if cop_mm >= toe_mm:
        return {HEEL: 0.0, MID: 0.0, TOE: 1.0}

    if cop_mm <= mid_mm:
        u = (cop_mm - heel_mm) / (mid_mm - heel_mm)
        blend = smoothstep(u)
        return {HEEL: 1.0 - blend, MID: blend, TOE: 0.0}

    u = (cop_mm - mid_mm) / (toe_mm - mid_mm)
    blend = smoothstep(u)
    return {HEEL: 0.0, MID: 1.0 - blend, TOE: blend}


def blend_profiles(profiles, weights):
    profile_arrays = {key: np.asarray(profiles[key]) for key in PROFILE_KEYS}
    shapes = {profile.shape for profile in profile_arrays.values()}
    if len(shapes) != 1:
        raise ValueError("All profiles must have the same shape")

    values = [float(weights[key]) for key in PROFILE_KEYS]
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("Interpolation weights must be finite and nonnegative")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError("Interpolation weights must sum to one")

    result = (
        values[0] * profile_arrays[HEEL]
        + values[1] * profile_arrays[MID]
        + values[2] * profile_arrays[TOE]
    )
    return result.astype(profile_arrays[HEEL].dtype, copy=False)


def select_discrete_profile(cop_mm, mid_mm=131.6):
    cop_mm = float(cop_mm)
    if math.isnan(cop_mm):
        cop_mm = mid_mm
    return HEEL if cop_mm <= mid_mm else TOE
