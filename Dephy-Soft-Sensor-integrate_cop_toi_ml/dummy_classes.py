import json
import struct
import serial
import numpy as np

NO_SLACK_CURRENT = 400


def NM_TO_NMM(torque):
    return (torque) * 1000

def A_TO_MA(torque):
    return (torque) * 1000

class ExoBoot:
    def __init__(self, *args, **kwargs):
        self.frequency = kwargs['frequency'] 
        self.user_mass = kwargs['user_mass']
        self.CURRENT_LIMIT = 4800 # change to whatever it should be 
        self.kt = 140
    def define_current_gains(self, *args, **kwargs):
        ...
    def set_controller(self, *args, **kwargs):
        ...
    def init_collins_profile(self, *args, **kwargs):
        ...
    def set_exo_current(self, *args, **kwargs):
        ...
    def read_data(self):
        ...
    def run_collins_profile(self, *args, **kwargs):
        ...
        #self.set_exo_current(*args)
        ...

    def fifth_degree_polynomial_modified(x, a, b, c, d, e, f):
        func = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
        return func if (x>3 and x<=60) and func > 0 else 0


    def run_stair_profile_heel(self, *args, **kwargs):
        with open("params/heel_params.json",'r') as f:
            params = json.load(f)
        torque_function = lambda x: ExoBoot.fifth_degree_polynomial_modified(x, *params)
        #TEMP FOR NOW
        percent_gait_current = 1
        #TEMP FOR NOW
        tau =  torque_function(percent_gait_current) * self.user_mass
        self.set_exo_current(max(NO_SLACK_CURRENT, A_TO_MA(self.ankle_torque_to_current(NM_TO_NMM(tau))))) 
        
    def run_stair_profile_toe(self, *args, **kwargs):
        with open("params/toe_params.json",'r') as f:
            params = json.load(f)
        torque_function = lambda x: ExoBoot.fifth_degree_polynomial_modified(x, *params)
        #TEMP FOR NOW
        percent_gait_current = 1
        #TEMP FOR NOW
        tau =  torque_function(percent_gait_current) * self.user_mass
        self.set_exo_current(max(NO_SLACK_CURRENT, A_TO_MA(self.ankle_torque_to_current(NM_TO_NMM(tau))))) 

    def read_sensors(self):
        ...

    def set_exo_current(self, currentCommand):
        if abs(currentCommand) < self.CURRENT_LIMIT:
            self.current_cmd = currentCommand
            #fxs.send_motor_command(self.dev_id, fxe.FX_CURRENT, currentCommand)
        else:
            print("exoBoot :: set_exo_current : CURRENT TOO HIGH, requested " + str(currentCommand) + " mA")
            self.current_cmd = CURRENT_LIMIT
            #fxs.send_motor_command(self.dev_id, fxe.FX_CURRENT, CURRENT_LIMIT)

    def ankle_torque_to_current(self, torque):
        # get the current based on the torque cmd and the system state
        current = torque / self.kt

        return current
    

class State: # contains the state of the visual perception
    def __init__(self):
        self.level_ground = True #default
        self.heel = False
        self.toe = False
    def update_state(self, new_state):
        if new_state == "level_ground":
            self.level_ground = True
            self.heel = False
            self.toe = False
        elif new_state == "heel":
            self.level_ground = False
            self.heel = True
            self.toe = False
        elif new_state == "toe":
            self.level_ground = False
            self.heel = False
            self.toe = True
        else:
            raise ValueError("Invalid state")


class OccipitalLobe: # just a quippy name for our VisionProcessor
    def __init__(self, electromechanical_delay_ms, fps = 60, res = '720p', mid_foot_mm = 263.2/2):
        self.fps = fps
        self.res = res
        self.EMD = electromechanical_delay_ms
        self.Mf = mid_foot_mm
        self.state = State()
        

        ## estabilish communication init
        ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=None) 
        handshaker = self.Communicator.HandShakeReceiver(ser)
        self.receiver = self.Communicator.ReceiverFlowControl(ser)

        # Start handshake
        print("Please run Jetson UART sending script")
        handshaker.wait_for_sender()

        
    ## Subclasses for communication protocol
    class Communicator:
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

    def communicate_with_jetson(self):
        TOI, COP = self.receiver.receive_predictions_and_send_ack()
        return TOI, COP

    
    def sense(self,verbose = False): # return heel or toe
        COP, TOI = self.communicate_with_jetson()
        if TOI <= self.EMD:
            #then form the prediction
            if COP <= self.Mf:
                self.state.update_state("heel")
                if verbose:
                    print("HEEL")
            elif COP > self.Mf:
                if verbose:
                    print("TOE")
                self.state.update_state("toe")
            else:
                raise Exception("Invalid")
        else: # if far from step or no step is detected
            if verbose:
                print("LG")
            self.state.update_state("level_ground")

    
