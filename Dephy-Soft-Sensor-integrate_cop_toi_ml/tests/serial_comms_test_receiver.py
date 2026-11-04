import serial  # "386_proj" env
import time
import numpy as np
import struct
from collections import deque  # <-- Added for rolling FPS

## This file must integrate with the sender side `serial_comms_test_sender.py`

max_uint8 = np.iinfo(np.uint8).max  # 255
min_uint8 = np.iinfo(np.uint8).min  # 0

sample_rate = 400  # sample rate of your pi/receiver pc


class HandShakeReceiver:
    def __init__(self, ser: serial):
        self.bytedepth = 10  # this must match the bytedepth on sender side
        self.ser = ser

    def wait_for_sender(self):  # ENSURE THAT SERIAL HAS timeout=None
        print("Handshake: Waiting for sender...")
        handshake_id = self.ser.read(self.bytedepth)
        print(f"Found ID, {handshake_id}")
        self.ser.write(handshake_id)
        print("Handshake: Rebound to sender, wait for data...")


class ReceiverFlowControl:
    def __init__(self, ser: serial):
        self.ser = ser

    def receive_predictions_and_send_ack(self):
        packet = self.ser.read(4)  # 4 depth for reading (uint16 and 2 uint8)
        COP, TOI, cyclic_index = struct.unpack("<HBB", packet)
        # send ack
        self.ser.write(bytes([cyclic_index]))

        return TOI, COP


if __name__ == "__main__":
    count_preds = True
    if count_preds:
        total_preds = 0
        rolling_window = deque(maxlen=100)  # <-- rolling FPS window

    ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=None) # for rasp pi
    #ser = serial.Serial('COM3', 921600, timeout=None) # for windows
    receiver = ReceiverFlowControl(ser=ser)
    shaker = HandShakeReceiver(ser=ser)
    shaker.wait_for_sender()
    start_time = time.time()

    while True:
        TOI, COP = receiver.receive_predictions_and_send_ack()
        print(f"TOI: {TOI}, COP: {COP}", end = " ")

        if count_preds:
            total_preds += 1
            now = time.time()
            rolling_window.append(now)

            if len(rolling_window) > 1:
                duration = rolling_window[-1] - rolling_window[0]
                rolling_fps = (len(rolling_window) - 1) / duration if duration > 0 else 0
                print(f"Rolling FPS: {rolling_fps:.2f}")
        # time.sleep(1/sample_rate)

