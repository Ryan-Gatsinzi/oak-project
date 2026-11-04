
import time
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np
import torch

from model import TerrainForecaster

WIDTH = 50
HEIGHT = 25
FPS = 60

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TerrainForecaster().to(device)
model.eval()

# These weights are random until we load a trained checkpoint.
state = model.initial_state(batch_size=1, device=device)

with dai.Pipeline() as pipeline:
    left_cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )
    right_cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C
    )

    left_output = left_cam.requestOutput(
        (640, 400),
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )
    right_output = right_cam.requestOutput(
        (640, 400),
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    sync = pipeline.create(dai.node.Sync)
    sync.setRunOnHost(True)
    sync.setSyncThreshold(timedelta(milliseconds=10))

    left_output.link(sync.inputs["left"])
    right_output.link(sync.inputs["right"])

    queue = sync.out.createOutputQueue()

    pipeline.start()
    print(f"Live inference on {device}. Press Q to quit.")
    print("WARNING: outputs are meaningless until trained weights are loaded.")

    count = 0
    last_report = time.perf_counter()
    start_time = last_report

    try:
        while pipeline.isRunning():
            group = queue.get()

            left_frame = group["left"].getCvFrame()
            right_frame = group["right"].getCvFrame()

            left_small = cv2.resize(
                left_frame, (WIDTH, HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )
            right_small = cv2.resize(
                right_frame, (WIDTH, HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )

            left_normalized = cv2.normalize(
                left_small.astype(np.float32), None,
                0.0, 1.0, cv2.NORM_MINMAX
            )
            right_normalized = cv2.normalize(
                right_small.astype(np.float32), None,
                0.0, 1.0, cv2.NORM_MINMAX
            )

            stereo_input = np.stack(
                [left_normalized, right_normalized], axis=0
            )

            # (2, 25, 50) -> (1, 2, 25, 50)
            frame_tensor = torch.from_numpy(
                stereo_input
            ).unsqueeze(0).to(device)

            with torch.inference_mode():
                output, state = model.forward_frame(
                    frame_tensor, state
                )

            count += 1
            now = time.perf_counter()

            if now - last_report >= 1.0:
                elapsed = now - start_time
                rate = count / elapsed

                print(
                    f"Untrained output: {output.item():.4f} | "
                    f"Processing rate: {rate:.1f} pairs/sec"
                )
                last_report = now

            cv2.imshow(
                "Left camera",
                cv2.resize(left_small, (500, 250),
                           interpolation=cv2.INTER_NEAREST)
            )
            cv2.imshow(
                "Right camera",
                cv2.resize(right_small, (500, 250),
                           interpolation=cv2.INTER_NEAREST)
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
