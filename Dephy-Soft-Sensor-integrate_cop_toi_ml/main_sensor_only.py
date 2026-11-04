# 1. Import libraries
import csv
import warnings
import time
import serial # make sure pyserial library is installed instead of serial
import serial.tools.list_ports
from datetime import datetime
import os

from Class_SoftSensor import *

#2. Initialize Global Variables
task_flag = 0
# USER UPDATE BELOW VARIABLES !!!!!!!!!!!!!!!!!!!!!!!
use_mocap = 0
path = "CSV Logs"
os.chdir(path)
# filename = datetime.now().strftime('_%H_%M.csv')
# filename = datetime.now().strftime('CY_new_garment_test_dig_hip_sensor_t2_%H_%M.csv')
# filename = datetime.now().strftime('CY_motor_fatigue_t5_%H_%M.csv')
filename = datetime.now().strftime('Test_t1_%H_%M.csv')

start_time = 0

if __name__ == "__main__":
    print('communicating with Arduino')
    soft_sensor = Arduino()

    print("start recording")

    # msg = 0
    # stop_at = 10000000
    skip_initial = 10
    while True:
        #print(time.monotonic())
        while ((time.monotonic()-start_time) <= (1/100)) : # if you run boot interface loop faster than the dephy frequency the cue will fill up and it will take a long time shut down and to have the data come in.  This just waits till the boot period has passed then runs.
            pass
        print(time.monotonic() - start_time)
        start_time = time.monotonic()

        #for _ in range(skip_initial):
         #   sensor_data = soft_sensor.readline()
        # msg += 1
        # if msg >= stop_at:
        # break
        # while (UMLCapSensorPCB.inWaiting()==0):
        #     pass
        # try:

        # record sensor data
        # print("about to readline")
        soft_sensor.obtain_softsensor_data()
        #print('New Read')
        #left_data = soft_sensor.left_sensor_data
