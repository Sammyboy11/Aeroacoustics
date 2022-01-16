import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import scipy.special as sc
U_c = 1


# Constants
mu = 1.81*pow(10,-5)    # dynamic viscosity
rho = 1.225
nu = mu/rho
c = 0.2
Vinf = 20

# XFOIL ouput
theta = 0.013316                      # Momentum thickness
delta_star = 0.027783
shape_factor = 2.086
delta = delta_star * shape_factor                       # BL thickness

theta_2 = 0.004334                      # Momentum thickness
delta_star_2 = 0.007812
shape_factor_2 = 1.802
delta_2 = delta_star_2 * 8                      # BL thickness

U_ratio = 0.92030
Ue = Vinf *(U_ratio)            # Velocity at the boundary-layer edge
Cf = 0.000195
Tw = 0.5*rho*Vinf*Vinf * Cf     # Shear stress

U_ratio_2 = 0.92030
Ue = Vinf *(U_ratio)            # Velocity at the boundary-layer edge
Cf_2 = 0.002566
Tw_2 = 0.5*rho*Vinf*Vinf * Cf_2    # Shear stress

# Dependent variables
Rt = 0.11* pow(Ue*theta/nu,0.75)

Rt_2 = 0.11* pow(Ue*theta_2/nu,0.75)

phi_pp = np.zeros(9901)
phi_pp_2 = np.zeros(9901)
w = 0
for omega in range (100,10001):
    k = omega*delta/Ue
    # Goody's equation for surface pressure spectrum
    phi_pp[w] = (Tw*Tw*delta/Ue) * (3*k*k) / ( pow(pow(k,0.75) + 0.5, 3.7) + pow(1.1*pow(Rt,-0.57)*k, 7) )


    k_2 = omega*delta_2/Ue
    # Goody's equation for surface pressure spectrum
    phi_pp_2[w] = (Tw_2*Tw_2*delta_2/Ue) * (3*k_2*k_2) / ( pow(pow(k_2,0.75) + 0.5, 3.7) + pow(1.1*pow(Rt_2,-0.57)*k_2, 7) )
    w+=1

print(phi_pp)
x = np.linspace(100, 10001,9901)
print(x.shape)
plt.loglog(x, -10*np.log(phi_pp))
plt.show()
# =============================================================================
# #Corcos model
# =============================================================================
co = 0.625##corelation constant
l_y = co*U_c/omega #=3.4
real_phi = phi_pp*l_y/np.pi

# =============================================================================
# Defining parameters
# =============================================================================
# =============================================================================

c_0 = 334 ##speed of sound
M = Ue/c_0##Mach number
K_x = (omega*c)/(2*U_c)
al = Ue/U_c
beta = np.sqrt(1-M**2)
mu_bar = (omega*c)/(2*c_0*beta**2)
k_x = 1
k_y = 1
k_y_bar = k_y*c/2
k_x_bar = k_x*c/2
xr = 0
yr = 0
zr = 1
sigma = np.sqrt(xr**2 + (beta**2)*(yr**2 + zr**2))
kappa = np.sqrt(mu_bar**2-k_y_bar**2/beta**2)
eps = 1/np.sqrt(1+1/(4*kappa))
# =============================================================================
# Fresnel Integral
# =============================================================================

B = K_x - M*mu_bar + kappa
C = K_x - mu_bar*(xr/sigma - M)
i = 0+1j #imaginary
O = kappa - mu_bar*xr/sigma
H = ((1+i)*np.exp(i*(-4*i*kappa))*(1-O**2))/(2*np.sqrt(np.pi)*(al-1)*k_x_bar*np.sqrt(B))


Es1, Ec1 = sc.fresnel((2*B-2*C)*np.sqrt(2/np.pi))
E1 = Ec1 + i*Es1

Es2, Ec2 = sc.fresnel((2*B)*np.sqrt(2/np.pi))
E2 = Ec2 + i*Es2

Es3, Ec3 = sc.fresnel((4*kappa)*np.sqrt(2/np.pi))
E3 = Ec3 + i*Es3

Es4, Ec4 = sc.fresnel((2*O)*np.sqrt(2/np.pi))
E4 = Ec4 + i*Es4

E3_con = np.transpose(np.conjugate(E3))
#print(E3); print(E3_con)

print(mu_bar); print(kappa); print(O); print(sigma)

# =============================================================================
# G = (1+eps)*np.exp(i)*(2*kappa+O)*((np.sin(O-2*kappa))/(O-2*kappa)) \
#      + (1-eps)*np.exp(i*(-2*kappa+O))*((np.sin(O+2*kappa))/(O+2*kappa)) \
#      + (((1+eps)*(1-i))/(2*(O-2*kappa))) * (np.exp(4*i*k)*E3)\
#      -(((1-eps)*(1+i))/(2*(O+2*kappa))) * (np.exp(-4*i*k)*E3_con)\
#      +(np.exp(2*i*O)/2)*(np.sqrt(2*kappa/O)*E4)*(((1-eps)(1+i)/(O+2*kappa))-((1+eps)(1-i)/(O-2*kappa)))
#
# =============================================================================

G  = (1+eps)*np.exp(i*(2*kappa+O))*np.sin((O-2*kappa))\
    + (1-eps)*np.exp(i*(O-2*kappa))*np.sin((O+2*kappa))\
     + (1+eps)*(1-i)*np.exp(4*i*kappa)*E3/(2*(O-2*kappa))\
     - (1-eps)*(1+i)*np.exp(-4*i*kappa)*E3_con/(2*(O+2*kappa))\
     + 0.5*np.exp(2*i*O)*np.sqrt(2*kappa/O)*E4*((1-eps)*(1+i)/(O+2*kappa)-(1+eps)*(1-i)/(O-2*kappa))


I_1 = i*np.exp(2*i*C)*((1+i)*np.exp(-2*i*C)*np.sqrt(B/(B-C))*E1 - (1+i)*E2 + 1)/C

cI_2 = (1-(1+i)*E3)*np.exp(4*i*kappa);
cI_2 = np.real(cI_2) + i*np.imag(cI_2)*eps;

I_2 = H*cI_2 + H*(-np.exp(-2*i*O)+i*(O + k_x_bar + M*mu_bar-kappa)*G)

I = abs(I_1 + I_2)

# S_gg calculations

alpha = 0.625   # Defined
L = 0.08      # span length [m]
w = 0
S_pps = np.zeros(9901)
S_ppp = np.zeros(9901)
S_pp = np.zeros(9901)
for omega in range (100,10001):
    k_bar = omega*c/(2*c_0)
    # Goody's equation for surface pressure spectrum

    S_pps[w] = ((k_bar * zr)/(2*np.pi*sigma**2))**2  * 2 * L * (alpha * U_c / omega ) * phi_pp[w] * I**2
    S_ppp[w] = ((k_bar * zr)/(2*np.pi*sigma**2))**2  * 2 * L * (alpha * U_c / omega ) * phi_pp_2[w] * I**2
    S_pp[w] = S_pps[w] + S_ppp[w]
    w+=1
print('S_pp = ',S_pps)
print(omega)
x_2 = np.linspace(100, 10001,9901)


Pref = 2E-5 # Pa
#S_pp = S_pps + S_ppp

PSD = 10*np.log(S_pp)-20*np.log(Pref) # dB
PSDS = 10*np.log(S_pps)-20*np.log(Pref) # dB
PSDP = 10*np.log(S_ppp)-20*np.log(Pref) # dB

plt.plot(np.log(x_2), PSD)
plt.show()
plt.plot(np.log(x_2), PSDS)
plt.show()
plt.plot(np.log(x_2), PSDP)
plt.show()





 
