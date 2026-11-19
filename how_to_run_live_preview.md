# How to Run the Real-Time Torque Visualizer

### 1. Running with a Physical OAK-D Camera

Make sure your OAK camera is connected via USB 3.0, and run using the Python environment with `depthai` installed:

```powershell
# Using the OAK virtual environment directly:
& "C:\Users\Ryan Udatinya\oak-venv\Scripts\python.exe" live_torque_visualizer.py
```

Or activate the environment first:
```powershell
& "C:\Users\Ryan Udatinya\oak-venv\Scripts\Activate.ps1"
python live_torque_visualizer.py
```

When connected:
* The terminal will display: `[INFO] Found OAK-D device: <Device MxID>. Starting DepthAI pipeline...`
* The top-right badge will show: `[ OAK-D STEREO ACTIVE ]` in green.
* The stereo camera feeds in the upper-left panel will stream live from your physical camera.

---

### 2. Running in Synthetic Demo Mode (No Camera Required)

If no camera is plugged in (or to test algorithms without hardware), run:

```powershell
& "C:\Users\Ryan Udatinya\oak-venv\Scripts\python.exe" live_torque_visualizer.py --demo
```

---

### 3. CLI Command Options

| Option | Description |
| :--- | :--- |
| `--demo` | Forces synthetic test bench mode (runs simulated stair approaches even without a camera). |
| `--cop-weights <path>` | COP checkpoint (default: `COP_forecast/base_models_pred_cop/23.torch`). |
| `--toi-weights <path>` | Optional TOI checkpoint (e.g., `82_time_base.torch`). Not included in this repo. |
| `--weights <path>` | Deprecated alias for `--cop-weights`. |
| `--resolution <400\|720>` | OAK-D mono resolution (default: `400`). Use `720` only on USB3; 720p60 stereo can crash on USB2. |
| `--foot-length-mm <mm>` | Foot length used to rescale normalized COP to mm (default: `261.1`, shoe size 5). |
| `--toi-ms <ms>` | Manual TOI fallback when no TOI weights are loaded (default: `30.0`). |
| `--mass <kg>` | Set subject mass in kg (default: `75.0`). |
| `--controller <continuous\|discrete>` | Selects smooth heel→midfoot→toe smoothstep interpolation (default) or the original discrete controller. |
| `--blend` | Legacy linear heel/toe blend; only used with `--controller discrete`. |
| `--bio` | Toggles full biological torque scale (0–90 Nm) vs exoskeleton assistive scale (0–35 Nm). |

### COP-only mode

The included weights predict COP only. Without `--toi-weights`, the HUD shows
`[MANUAL]` next to TOI and uses `--toi-ms` (default `30 ms`, i.e. stair branch)
so the predicted COP actually drives heel/toe profile selection. Adjust TOI at
runtime with the arrow keys, or load TOI weights to restore learned TOI.

### COP interpolation mode

The default controller blends the existing Murray heel, midfoot, and toe profiles
with smoothstep weights over `COP = 0`, `foot_length / 2`, and `foot_length` mm.
The HUD shows all three reference profiles, the continuous curve, the current COP,
and the applied weights. `--controller discrete` restores the original switching
behavior for benchmarking. Learned TOI still selects the separate Collins level-ground
profile when `TOI > EMD`.

---

### 4. Live Interactive Keyboard Controls

| Key | Action |
| :--- | :--- |
| **Q** or **ESC** | Quit visualizer |
| **SPACE** | Freeze / pause frame |
| **A** | Toggle Auto-inference vs Interactive Manual Mode |
| **V** | Toggle continuous interpolation vs original discrete controller |
| **C** | Force Level Ground (Collins profile) |
| **H** | Force Stair Ascent Heel-Strike profile |
| **T** | Force Stair Ascent Toe-Strike profile |
| **↑ / ↓** | Adjust Time-of-Impact (±10 ms) |
| **← / →** | Adjust Center-of-Pressure (±10 mm) |
| **+ / -** | Increase / decrease user mass |
| **B** | Toggle Biological vs Assistive torque scaling |
