import serial
import time
import struct
import random
import numpy as np
from datetime import datetime
import os

import cv2
import depthai as dai
import torch
import torchvision
import numpy as np
import sys
import torch.nn.functional as F


from models import model, sequence_regressor_live


###START THE RECEIVER FIRST

sample_rate = 60

max_uint8 = np.iinfo(np.uint8).max  # 255
min_uint8 = np.iinfo(np.uint8).min  # 0

min_uint16 = np.iinfo(np.uint16).min  
max_uint16 = np.iinfo(np.uint16).max  

ser = serial.Serial('/dev/ttyTHS0', 921600, timeout=None)
time.sleep(2)

# For COP, our range exceeds that of [0, 255] which means we cannot use 8 bit integers (which would be faster)
# for TOI, our range does logically (it could, but we can force it) exceed [0, 250], so 8 bit unsigned integers work in this case
# technically using floats wont slow it down toooo much and its probably okay, but this is best practice for now



class HandShakeSender:
    def __init__(self, ser : serial):
        #SERIAL timeout should be None
        self.agreed = False # has the handshake been done
        self.ser = ser
        self.handshake_id = None
        self.bytedepth = 10
    def generate_handshake(self):
        self.handshake_id = os.urandom(self.bytedepth) # generate random 10 bytes
    def start(self): # determines sends and waits for the handshake, timeout in ms
        if self.handshake_id == None:
            raise ValueError("Please generate handshake using the `generate_handshake` method")
        ser.write(self.handshake_id)
        response = ser.read(self.bytedepth) # MUST USE SERIAL TIMEOUT AS NONE FOR THIS TO LIKELY WORK PROPERLY, TODO RESOLVE IF TIMEOUT != NONE IS NEEDED
        if response == self.handshake_id:
            print(f"Handshake successful with ID, {response}")
            return True
        else:
            raise Exception(f"BAD HANDSHAKE, expected {self.handshake_id}, but received {response}")

class SenderFlowControl:
    def __init__(self, ser : serial, model):
        self.ser = ser
        self.model_call = model # must return (TOI,COP) in that order
        self.cyclic_index = 0
        self.last_key = None

    def process_predictions(self, cyclic_index,delay = 0.00055):  # here we assume (based on computation) that the uart transmission delay is 0.55 ms
        start_time = time.time() # want to compute the latency from generating cop and toi from the model, technically this should be computed for when the input is passed into the model
        toi_prediction, cop_prediction = self.model_call()
        end_time = time.time()
        toi_prediction_adj = toi_prediction - (end_time-start_time + delay)
        
        # additional processing to turn into ints    
        cop_prediction = np.uint16(np.clip(cop_prediction, min_uint16, max_uint16))
        toi_prediction_adj = np.uint8(np.clip(toi_prediction_adj, min_uint8, max_uint8))

        packed = struct.pack('<HBB', cop_prediction, toi_prediction_adj, cyclic_index) 

        return packed
        
    def transmit_and_await_feedback(self, verbose = False): # Transmits the data and waits for feedback
        data = self.process_predictions(self.cyclic_index)
        key = data[-1]
        if verbose:
            start = time.time()
        
        if self.last_key is not None:
            response = self.ser.read(1)[0] # key is 1 byte long
            assert self.last_key == response
            if verbose:
                duration = time.time() - start
                print(f"KEY --> RESPONSE : {key} --> {response}, LATENCY = {duration/2 * 1000} ms") # ESTIMATED AS ROUND TRIP TIME / 2. THIS IS NOT SUPER PRECISE, GIVEN THAT WE TRANSMIT 4 BYTE AND GET 1 BYTE BACK, BUT PRACTICALLY SPEAKING ITS OKAY
            
        self.ser.write(data)
        self.last_key = key

        self.cyclic_index += 1
        self.cyclic_index = self.cyclic_index % 256


class VisionClass:
    def __init__(self, MONO_RES = 720, MONO_FRAME_RATE = 60, GPU_INTERPOLATE = False, USE_HALF_PRECISION = False,
                cop_model_dir = "/home/michael/Documents/step_reg_ieee/step_reg_ieee-aside_notebook_for_submission/analysis/jetson/models/23.torch",
                toi_model_dir = "/home/michael/Documents/step_reg_ieee/step_reg_ieee-aside_notebook_for_submission/analysis/jetson/models/82_time_base.torch",):
        
        self.MONO_RES = MONO_RES
        self.MONO_FRAME_RATE = MONO_FRAME_RATE
        self.GPU_INTERPOLATE = GPU_INTERPOLATE
        self.USE_HALF_PRECISION = USE_HALF_PRECISION
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SHOE_SIZE_LOOKUP = {
                                3: 236.8/10,
                                5: 261.1/10
                                }


        pipeline = dai.Pipeline()

        def getMonoCamera(pipeline, isLeft):
            mono = pipeline.createMonoCamera()
            if MONO_RES == 720:
                mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            elif MONO_RES == 400:
                mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono.setFps(MONO_FRAME_RATE)
            if isLeft:
                mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
            else:
                mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            return mono

        monoLeft = getMonoCamera(pipeline, isLeft=True)
        monoRight = getMonoCamera(pipeline, isLeft=False)

        xoutLeft = pipeline.createXLinkOut()
        xoutLeft.setStreamName("left")
        xoutRight = pipeline.createXLinkOut()
        xoutRight.setStreamName("right")

        monoLeft.out.link(xoutLeft.input)
        monoRight.out.link(xoutRight.input)

        # Model setup for two independent models
        self.hidden1_a = torch.zeros(2, 1, 60).to(self.DEVICE)
        self.hidden2_a = torch.zeros(2, 1, 1).to(self.DEVICE)
        if USE_HALF_PRECISION:
            self.hidden1_a = self.hidden1_a.half()
            self.hidden2_a = self.hidden2_a.half()
        # COP model
        state_dict_cop = torch.load(cop_model_dir)['state_dict']
        net_a = model().to(self.DEVICE)
        net_a.SEQ = sequence_regressor_live().to(self.DEVICE)
        net_a.load_state_dict(state_dict_cop)
        if USE_HALF_PRECISION:
            net_a = net_a.half()
            net_a.SEQ = net_a.SEQ.half()
        FE_a = net_a.CNN
        SR_a = net_a.SEQ

        self.hidden1_b = torch.zeros(2, 1, 60).to(self.DEVICE)
        self.hidden2_b = torch.zeros(2, 1, 1).to(self.DEVICE)
        if USE_HALF_PRECISION:
            self.hidden1_b = self.hidden1_b.half()
            self.hidden2_b = self.hidden2_b.half()
        # TOI model
        state_dict_toi = torch.load(toi_model_dir)['state_dict']
        net_b = model().to(self.DEVICE)
        net_b.SEQ = sequence_regressor_live().to(self.DEVICE)
        net_b.load_state_dict(state_dict_toi)
        if USE_HALF_PRECISION:
            net_b = net_b.half()
            net_b.SEQ = net_b.SEQ.half()
        FE_b = net_b.CNN
        SR_b = net_b.SEQ

        # transform = torchvision.transforms.Resize((25,50))
        self.transform_cpu = torchvision.transforms.Resize((25,50))

        # TorchScript tracing for speedup
        dummy_input = torch.randn(1, 2, 1, 25, 50).to(self.DEVICE)
        dummy_features = torch.randn(1, 1, 384).to(self.DEVICE)
        dummy_hidden1 = torch.zeros(2, 1, 60).to(self.DEVICE)
        dummy_hidden2 = torch.zeros(2, 1, 1).to(self.DEVICE)
        if USE_HALF_PRECISION:
            dummy_input = dummy_input.half()
            dummy_features = dummy_features.half()
            dummy_hidden1 = dummy_hidden1.half()
            dummy_hidden2 = dummy_hidden2.half()

        FE_a = torch.jit.trace(FE_a, dummy_input)
        FE_b = torch.jit.trace(FE_b, dummy_input)
        SR_a = torch.jit.trace(SR_a, (dummy_features, dummy_hidden1, dummy_hidden2))
        SR_b = torch.jit.trace(SR_b, (dummy_features, dummy_hidden1, dummy_hidden2))

        net_a.eval()
        net_b.eval()
        FE_a.eval()
        FE_b.eval()
        SR_a.eval()
        SR_b.eval()

        self.net_a = net_a
        self.net_b = net_b
        self.FE_a = FE_a
        self.FE_b = FE_b
        self.SR_a = SR_a
        self.SR_b = SR_b

        self.pipeline = pipeline
        # Initialize device and queues ONCE
        self.device = dai.Device(self.pipeline)
        self.leftQueue = self.device.getOutputQueue(name="left", maxSize=1)
        self.rightQueue = self.device.getOutputQueue(name="right", maxSize=1)

    def get_predictions(self):
        # Use already-initialized device and queues
        start_time = time.time()
        with torch.no_grad():
            leftFrame = self.leftQueue.get().getCvFrame()
            rightFrame = self.rightQueue.get().getCvFrame()
            if self.MONO_RES == 400:
                imOut_p = np.hstack((leftFrame, rightFrame)).reshape(2,400,640).astype(np.float32)
            elif self.MONO_RES == 720:
                imOut_p = np.hstack((leftFrame, rightFrame)).reshape(2,720,1280).astype(np.float32)
            imOut_p = torch.from_numpy(imOut_p).to(self.DEVICE)
            if self.USE_HALF_PRECISION:
                imOut_p = imOut_p.half()
            if self.GPU_INTERPOLATE:
                # GPU interpolation
                imOut_p = F.interpolate(imOut_p.unsqueeze(0), size=(25, 50), mode='bilinear', align_corners=False).squeeze(0)
            else:
                # CPU interpolation
                imOut_p = imOut_p.cpu()
                imOut_p = self.transform_cpu(imOut_p)
                imOut_p = imOut_p.to(self.DEVICE)
                if self.USE_HALF_PRECISION:
                    imOut_p = imOut_p.half()
            imOut_p = imOut_p / 255.0
            imOut_p = imOut_p.view(1,2,1,25,50)
            # Model A
            features_timestep_a = self.FE_a(imOut_p)
            y_timestep_a, self.hidden1_a, self.hidden2_a = self.SR_a(features_timestep_a, self.hidden1_a, self.hidden2_a)
            # Model B (independent)
            features_timestep_b = self.FE_b(imOut_p)
            y_timestep_b, self.hidden1_b, self.hidden2_b = self.SR_b(features_timestep_b, self.hidden1_b, self.hidden2_b)

            rescaled_COP = (y_timestep_a*10 + 5) * self.SHOE_SIZE_LOOKUP[3] # THESE RESCALINGS MIGHT BE WRONG, FROM OLD CODE
            rescaled_TOI = y_timestep_b *14 / 60 * 1000 # THESE RESCALINGS MIGHT BE WRONG, FROM OLD CODE, NEED TO CONFIRM
        return rescaled_TOI.cpu(), rescaled_COP.cpu() # TOI, COP



handshake = False

vision_system = VisionClass()
sender = SenderFlowControl(ser, vision_system.get_predictions)
total_preds = 0

try:
    while True: 
    ##NEW
        if handshake is False:
            # first handshake to establish comms
            shaker = HandShakeSender(ser)
            shaker.generate_handshake()
            handshake = shaker.start() # if it fails then an exception is raised

            time.sleep(2) # after our handshake, wait a few seconds
            start_time = time.time()
        
        sender.transmit_and_await_feedback(verbose = True)
        total_preds+=1

except KeyboardInterrupt:
    # since we find the ack after the next processed frame
    print(total_preds/ (time.time()-start_time))



