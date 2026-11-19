import numpy as np

from torque_interpolation import (
    HEEL,
    MID,
    TOE,
    blend_profiles,
    cop_interpolation_weights,
    select_discrete_profile,
)
from torque_profiles import load_profile_coefficients, poly5_profile

DEFAULT_HEEL_MM = 0.0
DEFAULT_MID_MM = 261.1 / 2.0
DEFAULT_TOE_MM = 261.1


def build_reference_curves(x_grid, mass=75.0, assist_scale=0.25):
    coefficients = load_profile_coefficients()
    return {
        HEEL: poly5_profile(x_grid, coefficients[HEEL], mass, assist_scale),
        MID: poly5_profile(x_grid, coefficients[MID], mass, assist_scale),
        TOE: poly5_profile(x_grid, coefficients[TOE], mass, assist_scale),
    }


def cop_sweep(toe_mm=DEFAULT_TOE_MM, step_mm=10.0):
    values = np.arange(DEFAULT_HEEL_MM, toe_mm, step_mm)
    return np.append(values, toe_mm)


def fluctuating_cops(mid_mm=DEFAULT_MID_MM, toe_mm=DEFAULT_TOE_MM):
    values = [
        DEFAULT_HEEL_MM,
        mid_mm - 60.0,
        mid_mm - 20.0,
        mid_mm - 5.0,
        mid_mm - 1.0,
        mid_mm + 1.0,
        mid_mm + 5.0,
        mid_mm + 20.0,
        mid_mm + 60.0,
        toe_mm,
    ]
    values = values + list(reversed(values[:-1]))
    return np.clip(np.asarray(values, dtype=np.float64), DEFAULT_HEEL_MM, toe_mm)


def continuous_curves_for_cops(profiles, cops, heel_mm, mid_mm, toe_mm):
    result = []
    for cop_mm in cops:
        weights = cop_interpolation_weights(cop_mm, heel_mm, mid_mm, toe_mm)
        result.append(blend_profiles(profiles, weights))
    return result


def discrete_curves_for_cops(profiles, cops, mid_mm):
    result = []
    for cop_mm in cops:
        profile_key = select_discrete_profile(cop_mm, mid_mm)
        result.append(profiles[profile_key])
    return result


def curve_jump(curves):
    stacked = np.stack(curves, axis=0)
    return float(np.max(np.abs(np.diff(stacked, axis=0))))


def torque_at_phase(curves, x_grid, phase_pct):
    return np.asarray([float(np.interp(phase_pct, x_grid, curve)) for curve in curves])


def torque_rate(torques, dt_s):
    return float(np.max(np.abs(np.diff(torques)) / dt_s))


def cop_sensitivity(torques, cops):
    torque_deltas = np.abs(np.diff(torques))
    cop_deltas = np.abs(np.diff(cops))
    valid = cop_deltas > 1e-9
    ratios = torque_deltas[valid] / cop_deltas[valid]
    return float(np.mean(ratios)), float(np.max(ratios))


def midfoot_crossings(cops, mid_mm, max_step_mm=None):
    side = cops <= mid_mm
    crossings = np.flatnonzero(side[:-1] != side[1:])
    if max_step_mm is not None:
        steps = np.abs(cops[crossings + 1] - cops[crossings])
        crossings = crossings[steps <= max_step_mm]
    return crossings


def switch_jump(curves, crossings):
    if crossings.size == 0:
        return 0.0
    stacked = np.stack(curves, axis=0)
    return float(np.max(np.abs(stacked[crossings + 1] - stacked[crossings])))


def switch_rate(torques, crossings, dt_s):
    if crossings.size == 0:
        return 0.0
    return float(np.max(np.abs(torques[crossings + 1] - torques[crossings]) / dt_s))


def compare_controllers(
    mass=75.0,
    assist_scale=0.25,
    phase_pct=25.0,
    dt_s=1.0 / 60.0,
    heel_mm=DEFAULT_HEEL_MM,
    mid_mm=DEFAULT_MID_MM,
    toe_mm=DEFAULT_TOE_MM,
):
    x_grid = np.linspace(0.0, 100.0, 200)
    profiles = build_reference_curves(x_grid, mass, assist_scale)
    cops = np.concatenate([cop_sweep(toe_mm), fluctuating_cops(mid_mm, toe_mm)])

    continuous = continuous_curves_for_cops(profiles, cops, heel_mm, mid_mm, toe_mm)
    discrete = discrete_curves_for_cops(profiles, cops, mid_mm)

    continuous_phase = torque_at_phase(continuous, x_grid, phase_pct)
    discrete_phase = torque_at_phase(discrete, x_grid, phase_pct)
    continuous_mean_sensitivity, continuous_max_sensitivity = cop_sensitivity(continuous_phase, cops)
    discrete_mean_sensitivity, discrete_max_sensitivity = cop_sensitivity(discrete_phase, cops)

    crossings = midfoot_crossings(cops, mid_mm, max_step_mm=20.0)
    continuous_max_jump = curve_jump(continuous)
    discrete_max_jump = curve_jump(discrete)
    continuous_switch_jump = switch_jump(continuous, crossings)
    discrete_switch_jump = switch_jump(discrete, crossings)
    continuous_max_rate = torque_rate(continuous_phase, dt_s)
    discrete_max_rate = torque_rate(discrete_phase, dt_s)
    continuous_switch_rate = switch_rate(continuous_phase, crossings, dt_s)
    discrete_switch_rate = switch_rate(discrete_phase, crossings, dt_s)

    return {
        "continuous": {
            "max_curve_jump_nm": continuous_max_jump,
            "max_switch_jump_nm": continuous_switch_jump,
            "max_torque_nm": float(max(float(curve.max()) for curve in continuous)),
            "max_torque_rate_nm_per_s": continuous_max_rate,
            "max_switch_rate_nm_per_s": continuous_switch_rate,
            "mean_cop_sensitivity_nm_per_mm": continuous_mean_sensitivity,
            "max_cop_sensitivity_nm_per_mm": continuous_max_sensitivity,
        },
        "discrete": {
            "max_curve_jump_nm": discrete_max_jump,
            "max_switch_jump_nm": discrete_switch_jump,
            "max_torque_nm": float(max(float(curve.max()) for curve in discrete)),
            "max_torque_rate_nm_per_s": discrete_max_rate,
            "max_switch_rate_nm_per_s": discrete_switch_rate,
            "mean_cop_sensitivity_nm_per_mm": discrete_mean_sensitivity,
            "max_cop_sensitivity_nm_per_mm": discrete_max_sensitivity,
        },
        "reduction": {
            "max_curve_jump_nm": discrete_max_jump - continuous_max_jump,
            "max_switch_jump_nm": discrete_switch_jump - continuous_switch_jump,
            "max_torque_rate_nm_per_s": discrete_max_rate - continuous_max_rate,
            "max_switch_rate_nm_per_s": discrete_switch_rate - continuous_switch_rate,
            "max_cop_sensitivity_nm_per_mm": discrete_max_sensitivity - continuous_max_sensitivity,
        },
    }


def format_report(report):
    lines = []
    for controller in ("continuous", "discrete"):
        metrics = report[controller]
        lines.append(f"{controller}:")
        lines.append(f"  max curve jump (full): {metrics['max_curve_jump_nm']:.3f} Nm")
        lines.append(f"  max switch jump (<=20mm COP step): {metrics['max_switch_jump_nm']:.3f} Nm")
        lines.append(f"  max torque:            {metrics['max_torque_nm']:.3f} Nm")
        lines.append(f"  max torque rate (full):{metrics['max_torque_rate_nm_per_s']:8.1f} Nm/s")
        lines.append(f"  max switch rate:       {metrics['max_switch_rate_nm_per_s']:.1f} Nm/s")
        lines.append(f"  mean COP sensitivity:{metrics['mean_cop_sensitivity_nm_per_mm']:8.4f} Nm/mm")
        lines.append(f"  max COP sensitivity: {metrics['max_cop_sensitivity_nm_per_mm']:.4f} Nm/mm")
    reduction = report["reduction"]
    lines.append("reduction (discrete - continuous):")
    lines.append(f"  max curve jump (full): {reduction['max_curve_jump_nm']:.3f} Nm")
    lines.append(f"  max switch jump:       {reduction['max_switch_jump_nm']:.3f} Nm")
    lines.append(f"  max torque rate (full):{reduction['max_torque_rate_nm_per_s']:8.1f} Nm/s")
    lines.append(f"  max switch rate:       {reduction['max_switch_rate_nm_per_s']:.1f} Nm/s")
    lines.append(f"  max COP sensitivity:   {reduction['max_cop_sensitivity_nm_per_mm']:.4f} Nm/mm")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(compare_controllers()))
