import numpy as np
from matplotlib import pyplot as plt

# Constants
mu = 1.81*pow(10,-5)    # dynamic viscosity
rho = 1.225
nu = mu/rho

Vinf = 20

# XFOIL ouput
theta = 0.002280                      # Momentum thickness
delta_star = 0.018379
shape_factor = 2.086
delta = delta_star * shape_factor                       # BL thickness

U_ratio = 0.93327
Ue = Vinf *(U_ratio)            # Velocity at the boundary-layer edge
Cf = 0.002623
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
x = np.linspace(100, 10001,9901)
print(x.shape)
plt.loglog(x, -10*np.log(phi_pp))
