import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sc


# %% Constants
rho = 1.225                         # air density
mu  = 1.81*pow(10,-5)               # dynamic viscosity
nu  = mu/rho                        # kinematic viscosity
c   = 0.2                           # chord length of airfoil

U_inf = 20                          # freestream velocity
U_c  = 0.8 * U_inf                  # BL convective velocity at TE

omega_i = 100                        # start frequency
omega_f = 10000 + 1                 # end frequency


# %% XFOIL output parameters

# XFOIL data for Suction Side (SS)
theta =  0.011968                   # Momentum thickness
delta_star = 0.035869               # 
shape_factor = 2.997                # Shape factor
delta = delta_star * 8              # BL thickness

U_ratio = 0.92030
Ue = U_inf *(U_ratio)               # Velocity at the boundary-layer edge
Cf = 0.000195                       # Skin friction coefficient at TE
Tw = 0.5*rho*U_inf**2 * Cf          # Shear stress at TE due to BL

Rt = 0.11* pow(Ue*theta/nu,0.75)

# XFOIL data for Pressure Side (PS)
theta_2 = 0.004334                  # Momentum thickness
delta_star_2 = 0.007812             #
shape_factor_2 = 1.802              # Shape factor
delta_2 = delta_star_2 * 8          # BL thickness

U_ratio_2 = 0.92030             
Ue_2 = U_inf *(U_ratio_2)           # Velocity at the boundary-layer edge
Cf_2 = 0.002566                     # Skin friction coefficient at TE
Tw_2 = 0.5*rho*U_inf**2 * Cf_2      # Shear stress at TE due to BL

Rt_2 = 0.11* pow(Ue_2*theta_2/nu,0.75)


# %% defining phi_pp for SS and PS

phi_pp = np.zeros(omega_f - omega_i)         # SS
phi_pp_2 = np.zeros(omega_f - omega_i)       # PS
w = 0                                        # increment counter



for omega in range (omega_i, omega_f):
    # Goody's equation for surface pressure spectrum on SS
    k = omega*delta/Ue
    phi_pp[w] = (Tw*Tw*delta/Ue) * (3*k*k) / ( pow(pow(k,0.75) + 0.5, 3.7) + pow(1.1*pow(Rt,-0.57)*k, 7) )

    # Goody's equation for surface pressure spectrum on PS
    k_2 = omega*delta_2/Ue
    phi_pp_2[w] = (Tw_2*Tw_2*delta_2/Ue) * (3*k_2*k_2) / ( pow(pow(k_2,0.75) + 0.5, 3.7) + pow(1.1*pow(Rt_2,-0.57)*k_2, 7) )
    
    w+=1                                    # counter increment


# %% Plotting phi_pp
x_axis = np.linspace(omega_i, omega_f, omega_f - omega_i)
plt.figure(1)
plt.plot(np.log10((x_axis*delta)/Ue), 10*np.log10(phi_pp/(Tw*Tw*delta/(Ue))), label = 'SS')
plt.plot(np.log10((x_axis*delta_2)/Ue), 10*np.log10(phi_pp_2/(Tw_2*Tw_2*delta_2/Ue)), label = 'PS')
plt.xlabel("$\omega\delta/U_e$"); plt.ylabel("$10log(\Phi_{pp}*U_e/Tw^2\delta)$")
plt.legend()
plt.show()



# %% Diffraction theory for TE noise

alpha = 1.6                     # correlation length constant
L = 0.08                        # span length of airfoil [m]
c_0 = 334                       # speed of sound [m/s]
M = U_c /c_0   #U_inf/c_0                 # convection Mach number

w = 0
S_pp_SS = np.zeros(omega_f - omega_i)
S_pp_PS = np.zeros(omega_f - omega_i)
S_pp_Total = np.zeros(omega_f - omega_i)


for omega in range (omega_i, omega_f):
        
    # Radiation integral function calculation
    K_x = (omega*c)/(2*U_c)   #  omega/U_c or (omega*c)/(2*U_c)
    al = Ue/U_c
    beta = np.sqrt(1-M**2)
    mu_bar = (omega*c)/(2*c_0*beta**2)

    xr = 1
    yr = 0
    zr = 0.004
    sigma = np.sqrt(xr**2 + (beta**2)*(yr**2 + zr**2))

    k_bar = omega*c/(2*c_0)
    k_x = omega/U_c
    k_y = omega*yr/(c_0*sigma)
    k_y_bar = k_y*c/2
    k_x_bar = k_x*c/2
    kappa = np.sqrt(mu_bar**2 - (k_y_bar**2/beta**2))
    eps = 1/np.sqrt(1+(1/(4*kappa)))

    i = 0+1j                            #imaginary
    B = K_x - M * mu_bar + kappa
    C = K_x - mu_bar*(xr/sigma - M)
    O = kappa - mu_bar*xr/sigma
    
    Theta = np.sqrt((k_x_bar + mu_bar * M + kappa)/(al * k_x_bar + mu_bar * M + kappa))
    H = (1+i)*np.exp(-4*i*kappa)*(1-Theta**2)/(2*np.sqrt(np.pi)*(al-1)*k_x_bar*np.sqrt(B))
    
    Es1, Ec1 = sc.fresnel((2*B-2*C)*np.sqrt(2/np.pi))
    E1 = Ec1 + i*Es1

    Es2, Ec2 = sc.fresnel((2*B)*np.sqrt(2/np.pi))
    E2 = Ec2 + i*Es2

    Es3, Ec3 = sc.fresnel((4*kappa)*np.sqrt(2/np.pi))
    E3 = Ec3 + i*Es3
    E3_con = np.transpose(np.conjugate(E3))

    Es4, Ec4 = sc.fresnel((2*O)*np.sqrt(2/np.pi))
    E4 = Ec4 + i*Es4
    
    G  = (1+eps)*np.exp(i*(2*kappa+O))*np.sin((O-2*kappa))/(O-2*kappa) \
         + (1-eps)*np.exp(i*(O-2*kappa))*np.sin((O+2*kappa))/(O+2*kappa) \
         + (1+eps)*(1-i)*np.exp(4*i*kappa)*E3/(2*(O-2*kappa)) \
         - (1-eps)*(1+i)*np.exp(-4*i*kappa)*E3_con/(2*(O+2*kappa)) \
         + 0.5*np.exp(2*i*O)*np.sqrt(2*kappa/O)*E4*(((1-eps)*(1+i)/(O+2*kappa))-((1+eps)*(1-i)/(O-2*kappa)))


    I_1 = i*np.exp(2*i*C)*((1+i)*np.exp(-2*i*C)*np.sqrt(B/(B-C))*E1 - (1+i)*E2 + 1)/C

    cI_2 = (1-(1+i)*E3)*np.exp(4*i*kappa);
    cI_2 = np.real(cI_2) + i*np.imag(cI_2)*eps;

    I_2 = H*cI_2 + H*(-np.exp(-2*i*O)+i*(O + k_x_bar + M*mu_bar - k_bar)*G)

    I = abs(I_1 + I_2)
    
    # Amiet's TE theory: Power spectral density
    S_pp_SS[w] = ((k_bar * zr)/(2*np.pi*sigma**2))**2  * 2 * L * (alpha * U_c / omega ) * phi_pp[w] * I**2
    S_pp_PS[w] = ((k_bar * zr)/(2*np.pi*sigma**2))**2  * 2 * L * (alpha * U_c / omega ) * phi_pp_2[w] * I**2
    S_pp_Total[w] = S_pp_SS[w] + S_pp_PS[w]
    
    w+=1


# %% Calculating SPL

Pref = 2E-5                                         # reference pressure

SPL = 10*np.log10(S_pp_Total)-20*np.log10(Pref)     # dB
SPL_SS = 10*np.log10(S_pp_SS)-20*np.log10(Pref)     # dB
SPL_PS = 10*np.log10(S_pp_PS)-20*np.log10(Pref)     # dB

Sf = pow(SPL/10,10)
for omega in range (omega_i, omega_f):
    E = Sf
OASPL = 10*np.log10(E)

# %% Plotting PSD
plt.plot(np.log10(x_axis), SPL_SS, label = 'SS')
plt.plot(np.log10(x_axis), SPL_PS, label = 'PS')
plt.legend()

plt.show()
plt.plot(np.log10(x_axis), SPL, label = 'SS + PS')
plt.legend()
plt.show()
