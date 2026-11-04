#C:/Users/sgfki/AppData/Local/Programs/Python/Python37 python
import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(thisdir)

from test_new_def_v2 import *

#load in dummy classes and overwrite the ones from test_new_def_v2
from dummy_classes import ExoBoot
from control_logic import HolisticExoSkeleton, OccipitalLobe


import time

ZEROING_CURRENT = 400 # mA
 
def zero_boots (leftExo,rightExo) : # this is now a dummy function

	# the gains need to be defined before this is called

	leftExo.set_controller(999)
	rightExo.set_controller(999)

	leftExo.set_exo_current(999)
	rightExo.set_exo_current(999)

	time.sleep(1) # wait a second

	"""
	# may not be necessary for new dephy api

	# leftExo.zero_encoders()
	# rightExo.zero_encoders()
	# print("zeroBoots : leftExo.ankle_ticks_offset = " + str(leftExo.ankle_ticks_offset) )
	# print("zeroBoots : rightExo.ankle_ticks_offset = " + str(rightExo.ankle_ticks_offset) )
	"""


	#leftExo.set_exo_current(0)
	#rightExo.set_exo_current(0)
	#leftExo.set_controller(fxe.FX_NONE)
	#rightExo.set_controller(fxe.FX_NONE)


def zero_boot (exo) :

	# the gains need to be defined before this is called

	exo.set_controller(fxe.FX_CURRENT)


	exo.set_exo_current(ZEROING_CURRENT)

	time.sleep(2) # wait a second



def main():
	idx = 0 # keep idx = 0 for left boot
	print('creating left boot')
	# dummy values
	ports = (None,None)
	baud_rate = 100000000
	user_mass = 75
	left_boot =  ExoBoot(ports[1], int(baud_rate), idx, log_level = 4, shouldLog = True, frequency = 100, user_mass = user_mass) # recent change the first line is now the baud rate
	idx +=1 
	right_boot = ExoBoot(ports[0], int(baud_rate), idx, log_level = 4, shouldLog = True, frequency = 100, user_mass = user_mass)

	# dummy values
	user_mass = 75
	rspg = 0
	opg = 23
	ppg = 38
	spg = 48
	ot = 0.056 
	npt = 0.025/3 

	if False : 
		print("At least one boot is missing")
	else :
		left_boot.define_current_gains(40,400,128)
		right_boot.define_current_gains(40,400,128)

		zero_boots(left_boot, right_boot)


		left_boot.init_collins_profile(mass = user_mass, ramp_start_percent_gait = rspg, onset_percent_gait = opg, peak_percent_gait = ppg, stop_percent_gait = spg, onset_torque = ot, normalized_peak_torque = npt, frequency = 400, user_mass = user_mass)	# initialize the Zhang/Collins profile
		right_boot.init_collins_profile(mass = user_mass, ramp_start_percent_gait = rspg, onset_percent_gait = opg, peak_percent_gait = ppg, stop_percent_gait = spg, onset_torque = ot, normalized_peak_torque = npt, frequency = 400, user_mass = user_mass)	# initialize the Zhang/Collins profile

		start_time = 0

		# DEFINE HOLISTIC EXO
		vision_system = OccipitalLobe(50) #assume a 50ms electromechanical delay for now
		whole_exo = HolisticExoSkeleton(left_boot, right_boot, vision_system)
		print('starting loop')
		try:
			while True:
				whole_exo.read_sensors()
				whole_exo.run_logic()

				time.sleep(1/left_boot.frequency)  # Lazy alternative to the waiting while loop but will run a bit slower.

		except KeyboardInterrupt:
			print("KeyboardInterrupt has been caught.")


	left_boot.set_exo_current(0)
	right_boot.set_exo_current(0)

	time.sleep(.3)

	#left_boot.set_controller(fxe.FX_NONE)
	#right_boot.set_controller(fxe.FX_NONE)

	time.sleep(.3)
	del left_boot
	time.sleep(.3)
	del right_boot
	# cleanupPlanStack()



main()
