import depthai as dai
import cv2

with dai.Pipeline() as pipeline:
    # 1. Initialize the Center Camera node (CAM_A)
    cam_center = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )

    # 2. Request a 1280x720 BGR Color stream from the center camera
    output_rgb = cam_center.requestOutput(
        (1280, 720),
        type=dai.ImgFrame.Type.BGR888i
    )

    # 3. Request a 1280x720 Grayscale stream from the SAME center camera (CAM_A)
    output_right = cam_center.requestOutput(
        (1280, 720),
        type=dai.ImgFrame.Type.GRAY8
    )

    # 4. Create host PC queues for both streams (Syntax error fixed)
    queue_rgb = output_rgb.createOutputQueue()
    queue_right = output_right.createOutputQueue()

    # Start the camera pipeline
    pipeline.start()
    print("Cameras running. Press Q to quit.")

    while pipeline.isRunning():
        # Fetch and process frames from both queues
        frame_rgb = queue_rgb.get().getCvFrame()
        frame_right = queue_right.get().getCvFrame()

        # Display both camera feeds in separate windows
        cv2.imshow("OAK-D S2 Center Camera (RGB)", frame_rgb)
        cv2.imshow("OAK-D S2 Center Camera (Grayscale)", frame_right)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
