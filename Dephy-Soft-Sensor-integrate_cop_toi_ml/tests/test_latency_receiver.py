import serial
import time
# Open serial connection on COM3 with baudrate 115200

print("ON WINDOWS YOU MUST CHANGE THE LATENCY TIMER ON THE COMM PORT TO THE SMALLEST VALUE. ON RASP YOU CAN DO THIS TOO. THIS IS NOT EASY ON MAC SO IT IS NOT SUGGESTED")

ser = serial.Serial('COM3', 921600, timeout=None)

print("Listening on COM3... Press Ctrl+C to stop.")


while True:
    data = ser.read(3)
    #if data == b"ping":
    #ser.write(b"pong")
    ser.write(data)

        #time.sleep(0.001)