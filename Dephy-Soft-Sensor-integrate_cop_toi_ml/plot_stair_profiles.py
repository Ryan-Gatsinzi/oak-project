import numpy as np
import json
import matplotlib.pyplot as plt

# It should be noted that the percent_gait was computed based on level ground walking and hence is not precise here, there has to be a new way to conduct percent gait for when ascending stairs

def fifth_degree_polynomial_modified(x, a, b, c, d, e, f):
    func = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    return func if (x>3 and x<=60) and func > 0 else 0

# easier since we based it only one a 5th order polynomial
def heel_stair_profile(perent_gait_current,user_mass = 70): 
    # the collins profile had hard coded coeffs, here we will just take them from our curve fits
    with open("/Users/michaelmurray/Documents/GitHub/stair_dynamics/torque_curves/params/heel_params.json",'r') as f:
        params = json.load(f)
    # [-1.0434579155542933e-08, 2.3127995260912266e-06, -0.0001752874886550902, 0.005132543736387709, -0.032088360641006966, 0.050814585640558324]
    torque_function = lambda x: fifth_degree_polynomial_modified(x, *params) # just to simplify the way it is called

    return torque_function(perent_gait_current) * user_mass

def toe_stair_profile(percent_gait_current, user_mass = 70): 
    with open("/Users/michaelmurray/Documents/GitHub/stair_dynamics/torque_curves/params/toe_params.json",'r') as f:
        params = json.load(f)
    torque_function = lambda x: fifth_degree_polynomial_modified(x, *params)
    return torque_function(percent_gait_current) * user_mass

def mid_stair_profile(percent_gait_current, user_mass = 70):
    with open("/Users/michaelmurray/Documents/GitHub/stair_dynamics/torque_curves/params/mid_params.json",'r') as f:
        params = json.load(f)
    torque_function = lambda x: fifth_degree_polynomial_modified(x, *params) 
    return torque_function(percent_gait_current) * user_mass

def run_entire_profile(type):
    # call the respective profile recurrently
    if type == "heel":
        profile = heel_stair_profile
    elif type == "toe":
        profile = toe_stair_profile
    elif type == "mid":
        profile = mid_stair_profile
        
    percent_gait = np.linspace(0, 100, 1000)
    tau = []
    for sample in percent_gait: # calling as if it would in real-time (just finer)
        tau.append(profile(sample))
    tau = np.array(tau)
    
    plt.plot(percent_gait, tau, label=f"Murray {type} Profile")
    plt.xlabel("Percent Gait (%)")
    plt.ylabel("Torque (Nm)")
    plt.title(f"Murray Profile [alpha] {type} Torque Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_entire_profile("heel")