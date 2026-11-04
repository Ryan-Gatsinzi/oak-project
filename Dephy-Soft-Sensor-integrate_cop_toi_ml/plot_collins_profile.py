import numpy as np
import matplotlib.pyplot as plt

def collins_profile(percent_gait, t0, t1, t2, t3, ts, tp):
    """
    Compute the Collins profile torque as a function of percent gait.
    Arguments:
        percent_gait: array-like, percent gait (0-100)
        t0, t1, t2, t3: float, percent gait time points
        ts: float, onset torque (Nm)
        tp: float, peak torque (Nm)
    Returns:
        tau: array-like, torque profile
    """
    tau = np.zeros_like(percent_gait, dtype=float)
    # Rising ramp
    mask1 = (percent_gait >= t0) & (percent_gait <= t1)
    tau[mask1] = ts / (t1 - t0) * (percent_gait[mask1] - t0)
    # Rising spline
    mask2 = (percent_gait > t1) & (percent_gait <= t2)
    a1 = (2 * (tp - ts)) / (t1 - t2) ** 3
    b1 = -3 * (t1 + t2) * (tp - ts) / (t1 - t2) ** 3
    c1 = 6 * t1 * t2 * (tp - ts) / (t1 - t2) ** 3
    d1 = -(-t1 ** 3 * tp + 3 * t1 ** 2 * t2 * tp - 3 * t1 * t2 ** 2 * ts + t2 ** 3 * ts) / (t1 - t2) ** 3
    tau[mask2] = a1 * percent_gait[mask2] ** 3 + b1 * percent_gait[mask2] ** 2 + c1 * percent_gait[mask2] + d1
    # Falling spline
    mask3 = (percent_gait > t2) & (percent_gait <= t3)
    a2 = -(tp - ts) / (2 * (t2 - t3) ** 3)
    b2 = 3 * t3 * (tp - ts) / (2 * (t2 - t3) ** 3)
    c2 = 3 * (t2 ** 2 - 2 * t2 * t3) * (tp - ts) / (2 * (t2 - t3) ** 3)
    d2 = -(
        3 * t2 ** 2 * t3 * tp - 6 * t2 * t3 ** 2 * tp + 2 * t3 ** 3 * tp - 2 * t2 ** 3 * ts + 3 * t2 ** 2 * t3 * ts
    ) / (2 * (t2 - t3) ** 3)
    tau[mask3] = a2 * percent_gait[mask3] ** 3 + b2 * percent_gait[mask3] ** 2 + c2 * percent_gait[mask3] + d2
    # After t3, slack
    tau[percent_gait > t3] = 0
    return tau

def plot_collins_profile(
    t0=0, t1=27.1, t2=50.4, t3=62.7, ts=2, peak_torque_normalized=0.20, user_mass=75
):
    """
    Plot the Collins profile torque curve as a function of percent gait.
    """
    percent_gait = np.linspace(0, 100, 1000)
    tp = user_mass * peak_torque_normalized
    tau = collins_profile(percent_gait, t0, t1, t2, t3, ts, tp)
    plt.figure(figsize=(8, 4))
    plt.plot(percent_gait, tau, label="Collins Profile")
    plt.xlabel("Percent Gait (%)")
    plt.ylabel("Torque (Nm)")
    plt.title("Collins Profile Torque Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_collins_profile_as_current(
    t0=0, t1=27.1, t2=50.4, t3=62.7, ts=2, peak_torque_normalized=0.20, user_mass=75, k_t=140 # Torque constant (Nmm/A)
):
    """
    Plot the Collins profile torque curve as a function of percent gait.
    k_t is the torque constant (Nmm/A).
    """
    percent_gait = np.linspace(0, 100, 1000)
    tp = user_mass * peak_torque_normalized
    tau = collins_profile(percent_gait, t0, t1, t2, t3, ts, tp)
    def NM_TO_NMM(torque):
        return (torque) * 1000
    current = NM_TO_NMM(tau) / k_t # current in Amperes
    plt.figure(figsize=(8, 4))
    plt.plot(percent_gait, current, label="Collins Profile")
    plt.xlabel("Percent Gait (%)")
    plt.ylabel("Current (A)")
    plt.title("Collins Profile Current Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_collins_profile()
    plot_collins_profile_as_current()
