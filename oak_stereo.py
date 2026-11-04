
import depthai as dai
import cv2
import numpy as np
import time
from datetime import timedelta
from collections import deque

WIDTH = 50
HEIGHT = 25
FPS = 60
SEQUENCE_LENGTH = 15

# Stores the most recent 15 stereo pairs.
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

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
        fps=FPS
    )
    right_output = right_cam.requestOutput(
        (640, 400),
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS
    )

    sync = pipeline.create(dai.node.Sync)
    sync.setRunOnHost(True)
    sync.setSyncThreshold(timedelta(milliseconds=10))

    left_output.link(sync.inputs["left"])
    right_output.link(sync.inputs["right"])

    queue = sync.out.createOutputQueue()

    pipeline.start()
    print("Building the rolling buffer. Press Q to quit.")

    frame_count = 0
    start_time = time.perf_counter()
    last_report = start_time

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

            # Shape: (2, 25, 50), dtype: float32.
            left_normalized = cv2.normalize(
            left_small.astype(np.float32),
            None,
            alpha=0.0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX
            )

            right_normalized = cv2.normalize(
            right_small.astype(np.float32),
            None,
            alpha=0.0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX
            )

            stereo_input = np.stack(
            [left_normalized, right_normalized],
            axis=0
            )

            # Add this stereo pair to the rolling buffer.
            frame_buffer.append(stereo_input.copy())
            frame_count += 1

            # Once we have 15 pairs, assemble the full sequence.
            if len(frame_buffer) == SEQUENCE_LENGTH:
                sequence = np.stack(list(frame_buffer), axis=0)

                # Expected shape: (15, 2, 25, 50).
                assert sequence.shape == (
                    SEQUENCE_LENGTH, 2, HEIGHT, WIDTH
                )
                assert sequence.dtype == np.float32

            # Display the images at their original orientation.
            cv2.imshow(
                "Left - 50x25",
                cv2.resize(left_small, (500, 250),
                           interpolation=cv2.INTER_NEAREST)
            )
            cv2.imshow(
                "Right - 50x25",
                cv2.resize(right_small, (500, 250),
                           interpolation=cv2.INTER_NEAREST)
            )

            # Report the processing rate approximately once per second.
            now = time.perf_counter()
            if now - last_report >= 1.0:
                elapsed = now - start_time
                measured_fps = frame_count / elapsed

                if len(frame_buffer) < SEQUENCE_LENGTH:
                    status = (
                        f"Buffer: {len(frame_buffer)}/"
                        f"{SEQUENCE_LENGTH}"
                    )
                else:
                    status = (
                        f"Buffer full | Sequence shape: "
                        f"{sequence.shape}"
                    )

                print(
                    f"{status} | Processing rate: "
                    f"{measured_fps:.1f} pairs/sec"
                )
                last_report = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
