import numpy as np
import os
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from helpers import *

class Minihalos(object):
    def __init__(self, M, fpbh=1, example_plots=False):
        self.e_min = 0.2
        self.e_max = 200.
        self.M = M
        if example_plots:
            self.lamb = lambda_dict_show['{:.0f}'.format(M)]
        else:
            self.lamb = lambda_dict['{:.0f}'.format(M)]
        
        self.n_pbh = 1.256e-2 * (omega_cdm / 1e-9) * fpbh / M * 1e4 # Mpc^-3
        
        loadTK = np.loadtxt('input_files/recfast_LCDM.dat')
        self.tk_0_highr = interp1d(loadTK[:,0], loadTK[:,-1])
        load21 = np.loadtxt('input_files/tb_file_Xi_60_Tmin_5.0000e+04_Rfmp_15_chiUV_2.00e+56_Nalpha_4.00e+03.dat')
        self.btb_0_highr = interp1d(load21[:,0], load21[:,1])
        return


    def J_tilde(self, r, z):
        prefactor = self.phi_alpha(z,r)*clight*1e2*n_hyd(z)*self.solve_xH(r,z) / \
                    (4. * np.pi * hubble(z) * nu_21) * 1e3 * kpc_to_m * 1e2
        elist = np.logspace(np.log10(self.e_min), np.log10(self.e_max), 100)
        integral = np.trapz(self.sigma_E(elist) * self.dN_dE(elist, r, z), elist)
        return prefactor * integral

    def phi_alpha(self, z, r):
        xe = 1. - self.solve_xH(r, z)
        return 0.48 * (1. - xe**0.27)**1.52


    def sigma_E(self, E):
        p = np.zeros_like(E)
        p[E < 0.25] = -2.65
        p[E >= 0.25] = -3.30
        return 4.25e-21 * (E / 0.25)**p # cm^2

    def A_MLamb(self):
        Ledd = 8.614e46 * self.M # keV / s, M in SM
        int_term = np.log(self.e_max / self.e_min)
        return self.lamb * Ledd / int_term

    def dN_dE(self, E, r, z): #units 1 / cm^2 / s / keV
        tau = 0.
        dtau = 0.
        prefactor = self.A_MLamb() / (4.*np.pi*r**2.) * (1./3.086e19/1e2)**2.
        term1 = np.exp(-tau) / E**2.
        term2 = -dtau * np.exp(-tau) / E
        return prefactor * (term1 + term2)
    
    def NumPh(self, E, r, z, xh): # units 1/cm^2/s
        tau = self.tau(E, r, z, xh)
#        tau = 0.
        return self.A_MLamb() / (4.*np.pi*r**2. *E) * np.exp(-tau) * (1./3.086e19/1e2)**2.
        
    def Gam_r(self, r, z, xh):
        elist = np.logspace(np.log10(self.e_min), np.log10(self.e_max), 60)
        integrand = self.sigma_E(elist) * self.NumPh(elist, r, z, xh) / elist
        return np.trapz(integrand, elist)

    def solve_xH(self, r, z):
        alphaH = 2.6e-13 # cm^3/s
        soln = fsolve(lambda x: alphaH * n_hyd(z)**2 * (1.-x)**2. - self.Gam_r(r,z,x)*x*n_hyd(z), 0.5)
        return soln

    def tau(self, E, r, z, xH): # Figure out way to solve for this
        approx_step = self.sigma_E(E) * xH * n_hyd(z) * r * kpc_to_m * 1e2
        return approx_step

    def H_Pbh(self, r, z):
        xhVal = self.solve_xH(r, z)
        prefact = n_hyd(z) * self.f_fitting(1. - xhVal) * xhVal
        elist = np.logspace(np.log10(self.e_min), np.log10(self.e_max), 100)
        term2 = np.trapz(self.sigma_E(elist) * self.NumPh(elist, r, z, xhVal), elist)
        return term2 * prefact * 1e3 # eV / cm^3 / s

    def f_fitting(self, xe):
        if xe < 1e-4:
            return 0.15
        return 0.9771*(1. - (1. - xe**0.2663)**1.3163)

    def H_compton(self, r, z, tk):
        clight_cm = clight * 1e2
        xhVal = self.solve_xH(r, z)
        ne = (1. - xhVal) * n_hyd(z) # cm^-3
        prefact = 4.*np.pi**2.*sigma_thomp*kbolt**5.*T_CMB(z)**4.*ne*(T_CMB(z) - tk) / (15.*5.11e5) / hbar**3. # ev/cm/s^3
        return prefact / clight_cm**2. # should be eV / cm^3 / s

    def H_exp(self, r, z, tk):
        xhVal = self.solve_xH(r, z)
        return -3.*hubble(z)*kbolt*tk*n_hyd(z)*(2. - xhVal) / kpc_to_m / 1e3 * clight # eV/cm^3/s

    def solve_Tk(self, r, z):
        soln = fsolve(lambda x: self.H_exp(r,z,x) + self.H_compton(r,z,x) + self.H_Pbh(r,z), 1e2)
        return soln + self.solve_xH(r,z)*self.tk_0_highr(z)

    def y_k(self, r, z, tk):
        return t_star / (tk * A10) * (self.deexcit_CH(r,z,tk) + \
               self.deexcit_Ce(r,z,tk) + self.deexcit_Cp(r,z,tk))
    
    def deexcit_CH(self, r, z, tk):
        prefact = 3.1e-11 * n_hyd(z) * self.solve_xH(r,z)
        return prefact * tk**0.357 * np.exp(-32./tk)

    def deexcit_Ce(self, r, z, tk):
        return self.gamma_e(tk) * n_hyd(z) * (1. - self.solve_xH(r, z))

    def deexcit_Cp(self, r, z, tk):
        return 3.2*(1. - self.solve_xH(r, z)) * self.deexcit_CH(r, z, tk)

    def gamma_e(self, Tk):
        if Tk > 1e4:
            Tk = 1e4
        if Tk < 0.1: # This is a safety net that I shouldnt, hopefully
            print 'Tk falling below 0.1??? Consider killing run.'
            Tk = 0.1
        lnVal = -9.607 + 0.5 * np.log(Tk)*np.exp(-np.log(Tk)**4.5/1800.)
        return np.exp(lnVal) # units 1/cm^3/s

    def y_alpha(self, r, z, tk):
        prefactor = 16. * np.pi**2. * t_star * f12 / (27. * A10 * 5.11e5 * clight * 1e2) * 0.0917  # e^2 = 0.0917 in these units??
        term1 = self.J_tilde(r, z) / tk
        return prefactor * term1

    def T_spin(self, r, z):
        tk = self.solve_Tk(r,z)
        yk = self.y_k(r,z,tk)
        y_alp = self.y_alpha(r,z,tk)
        numer = t_star + T_CMB(z) + tk * (yk + y_alp)
        return numer / (1. + yk + y_alp)

    def T_21(self, r, z):
        if type(r) != np.array:
            return 27. * self.solve_xH(r,z) * (1. - T_CMB(z)/self.T_spin(r, z))*(omega_b*h_little**2./0.023)*\
                    ((1.+z)*0.15/(10.*omega_M*h_little**2.))**.5 - self.btb_0_highr(z)*self.solve_xH(r,z)
        soln = np.zeros_like(r)
        for i, rr in enumerate(r):
            xhV = self.solve_xH(rr,z)
            soln[i] = 27. * xhV * (1. - T_CMB(z)/self.T_spin(rr, z))*(omega_b*h_little**2./0.023)*\
                    ((1.+z)*0.15/(10.*omega_M*h_little**2.))**.5
        return soln - self.btb_0_highr(z)*self.solve_xH(r,z)

    def mean_T21(self, z):
        rtable = np.logspace(-3, 5, 200)
        t21 = np.zeros_like(rtable)
        deltNu = np.zeros_like(rtable)
        for i,r in enumerate(rtable):
            t21[i] = self.T_21(r, z)
            deltNu[i] = self.delta_nu_eff(r, z)
        tilde_21 = np.trapz(2. * np.pi * deltNu * t21 * rtable, rtable)
        preterm = (1.+z)**2./nu_21/hubble(z)*self.n_pbh/1e6
        return preterm * tilde_21

    def delta_nu_eff(self, r, z):
        tk = self.solve_Tk(r, z)
        return np.sqrt(2.*kbolt*tk / 0.938e9) / (1.+z) * nu_21 / (1. + z)


