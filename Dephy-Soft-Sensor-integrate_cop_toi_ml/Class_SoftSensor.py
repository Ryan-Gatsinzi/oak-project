import os
import sys
import serial
import serial.tools.list_ports
import time
import warnings


class Arduino:

    def __init__(self):
        self.left_sensor_data = [-1, -1, -1]
        self.right_sensor_data = [-1, -1, -1]
        print('connecting to arduino')
        self.arduino = self._connectToArduino()  # connect to the ardunio
        # self.obtain_softsensor_data()  # read in the current data from the system

    def _connectToArduino(self):
        esp32_ports = [
            esp32.device
            for esp32 in serial.tools.list_ports.comports()
            if 'CP210x' in esp32.description
        ]
        if not esp32_ports:
            raise IOError("No ESP32 devices found")
        if len(esp32_ports) > 1:
            warnings.warn('Multiple ESP32 devices found - using the first')
        arduino = serial.Serial(esp32_ports[0], baudrate=57600, timeout=1)
        arduino.setDTR(False)
        arduino.setRTS(False)
        # UMLCapSensorPCB.open()
        print(arduino.name)
        # UMLCapSensorPCB = serial.Serial(port='COM4', baudrate=115200)
        # 115200 is from the ESP32 website, the arduino nano code ran at 57600. The higher baud rate didn't return data, the lower one doesn't return all of it
        # Other ESP32 boards may require setting the DTR and RTS ports to false, Adafruit states this is not necessary for the Huzzah32
        arduino.reset_input_buffer()
        return arduino
    def obtain_softsensor_data(self):
        soft_sensor = self.arduino
        #print(soft_sensor.inWaiting())

        #time.sleep(.01)
        while (soft_sensor.inWaiting() == 0):
            #print('waiting for data')


            pass
        try:
            #soft_sensor.flush()
            #data = soft_sensor.readline()
            #data_array = data.decode().rstrip().split('\t')

            data = soft_sensor.read(56)
            soft_sensor.flushInput()
            #data = soft_sensor.read(56)
            #data = soft_sensor.readlines(20)

            #print(data)

            lines = data.decode().rstrip().split('\n')
            data_array = [0] * 10
            for line in lines:
                if len(line) >= 35:
                    data_line = line
                    data_array = data_line.split('\t')
                    break


            #soft_sensor.reset_input_buffer()
            print(data_array)
            current_time = time.time()
            self.left_sensor_data[0] = current_time  # float(dataarray[0])  # sensor time
            self.left_sensor_data[1] = float(data_array[2])  # raw value of the textile sensor 1, usually left hip
            self.left_sensor_data[2] = float(data_array[3])  # raw value of the textile sensor 2, usually left knee

            self.right_sensor_data[0] = current_time  # float(dataarray[0])  # sensor time
            self.right_sensor_data[1] = float(data_array[0])  # raw value of the textile sensor 3, usually right hip
            self.right_sensor_data[2] = float(data_array[1])  # raw value of the textile sensor 4, usually right knee
            #print(dataarray[2])
            #return dataarray[3] #rawdata2, rawdata3, rawdata4
        except (KeyboardInterrupt, SystemExit, IndexError, ValueError):
            pass