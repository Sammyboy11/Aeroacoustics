import numpy as np
from matplotlib import pyplot as plt

# Constants
mu = 1.81*pow(10,-5)    # dynamic viscosity
rho = 1.225
nu = mu/rho

Vinf = 20

# XFOIL ouput
delta = 1                       # BL thickness
theta = 1                       # Momentum thickness
U_ratio = 1
Ue = Vinf *(U_ratio)            # Velocity at the boundary-layer edge
Cf = 1
Tw = 0.5*rho*Vinf*Vinf * Cf     # Shear stress

# Dependent variables
Rt = 0.11* pow(Ue*theta/nu,0.75)

phi_pp = np.zeros(9901)
w = 0
for omega in range (100,10001):
    k = omega*delta/Ue
    # Goody's equation for surface pressure spectrum
    phi_pp[w] = (Tw*Tw*delta/Ue) * (3*k*k) / ( pow(pow(k,0.75) + 0.5, 3.7) + pow(1.1*pow(Rt,-0.57)*k, 7) )
    w+=1

print(phi_pp)
