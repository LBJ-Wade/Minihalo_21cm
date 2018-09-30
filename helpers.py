import numpy as np

# Define relevant constants
omega_b = 0.0492
omega_cdm = 0.266
omega_M = omega_b + omega_cdm
omega_R = 5.43e-5
omega_L = 1. - omega_b - omega_cdm - omega_R

h_little = 0.67
H0 = 2.2348e-4 # units Mpc^-1

t_star = 0.068 # K
A10 = 2.85e-15 # s^-1
f12 = 0.416

hbar = 6.582e-16 # ev * s
kbolt = 8.617e-5 # ev / K

nu_21 = 1.4204e9 # Hz
clight = 2.998e8 # m/s

kpc_to_m = 3.086e19 # 1 kpc = 3.086e19 m

sigma_thomp = 6.65e-25 # cm^2

def n_hyd(z):
    return 8.6e-6 * omega_b * h_little**2. * (1+z)**3 # cm^-3

def hubble(z):
    return H0 * np.sqrt(omega_L + omega_R*(1.+z)**4. + omega_M*(1.+z)**3.)


def T_CMB(z):
    return 2.778 * (1. + z)

lambda_dict = {'100': 1e-10, '1000': 1e-7, '10000': 1e-4}
lambda_dict_show = {'100': 1, '1000': 1, '10000': 1}
