import numpy as np
import os
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.special import gammaincc
from helpers import *

class Minihalos(object):
    def __init__(self, M, fpbh=1, bernal_plots=False, density_evol=False, ADAF=False):
        self.e_min = 0.2
        self.e_max = 200.
        self.density_evol = density_evol
        self.ADAF = ADAF
        self.M = M
        if bernal_plots and not ADAF:
            self.lamb = lambda_dict_show['{:.0f}'.format(M)]
        elif not bernal_plots and not ADAF:
            self.lamb = lambda_dict_bernal['{:.0f}'.format(M)]
#            self.lamb = lambda_dict['{:.0f}'.format(M)]
        elif ADAF:
            self.lamb = lambda_poulin['{:.0f}'.format(M)]
    
        self.fpbh = fpbh
        self.n_pbh = 1.256e-2 * (omega_cdm / 1e-9) * fpbh / M * 1e4 # Mpc^-3
        loadTK = np.loadtxt('input_files/recfast_LCDM.dat')
        self.tk_0_highr = interp1d(loadTK[:,0], loadTK[:,-1])
        self.xe_highr_highZ = interp1d(loadTK[:,0], loadTK[:,1])
        xe_lowZ = np.loadtxt('input_files/ThermalHistory_Xi_30_Tmin_1.000e+04_Rfmp_15_chiUV_2.00e+56_Nalpha_4.00e+03.dat')
        self.xe_highr_lowZ = interp1d(xe_lowZ[:,0], xe_lowZ[:,1])
        
        self.load_files()
        return
        
    def xH_0_highr(self, z):
        if z > 29:
            return 1. - self.xe_highr_highZ(z)
        else:
            return self.xe_highr_lowZ(z)
        
    def file_tags(self):
        tag = '_Minihalos_Mass_{:.1e}_fpbh_{:.2e}_lamb_{:.1e}_DensityProf_'.format(self.M, self.fpbh, self.lamb)
        if self.density_evol:
            tag += 'True_'
        else:
            tag += 'False_'
        tag += 'ADAF_'
        if self.ADAF:
            tag += 'True_'
        else:
            tag += 'False_'
        tag += '.dat'
        return tag
    
    def load_files(self):
        veff = np.loadtxt('loads/veff.dat')
        self.v_effective = interp1d(veff[:,0], veff[:,1], kind='linear', fill_value='extrapolate', bounds_error=False)
        return

    def J_tilde(self, r, z):
        if self.density_evol:
            v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            self.density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
#            print 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
#            exit()
        else:
            self.density_r = n_hyd(z)
        
        energydep = self.phi_alpha(z,r)
        xH = self.solve_xH(r,z)
        prefactor = energydep*clight*1e2*self.density_r*xH  / \
                    (4. * np.pi * hubble(z) * nu_21) * 1e3 * kpc_to_m * 1e2
        
        elist = np.logspace(np.log10(self.e_min), np.log10(self.e_max), 100)
        
        integral = np.trapz(self.sigma_E(elist)* self.NumPh(elist, r, z, xH),  elist)#* self.dN_dE(elist, r, z), elist)
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
    
    def Mdot_HB(self, z):
        rho_inf = 200. * ((1. + z) / 1e3)**3. * 0.938 # GeV / cm^3
        mass_term = 4.* np.pi * (self.M * 6.67e-11 * 2e30)**2.
        units = 1e6*1e-3
        return rho_inf * self.lamb * mass_term / self.v_effective(z)**3. * units # keV / s
    
    def Luminosity(self, z):
        Mdot = self.Mdot_HB(z)
        epsilon = self.epsilon_acc(Mdot)
        wmin = np.sqrt(10. / self.M)*1e-3
        Ts = 2e2
        spec_norm = gammaincc(0., wmin / Ts)
        return epsilon * Mdot
    
    def spectrum(self, E):
        wmin = np.sqrt(10. / self.M) * 1e-3
        Ts = 2e2
        sols = E**-1. * np.exp(-E / Ts)
        sols = sols[E > wmin]
        return sols

    def diff_spectrum(self, E):
        wmin = np.sqrt(10. / self.M) * 1e-3
        Ts = 2e2
        sol = - np.exp(- E / Ts) / E * (1. / E + 1. / Ts)
        sol = sol[E > wmin]
        return sol

    def dN_dE(self, E, r, z): #units 1 / cm^2 / s / keV
        tau = self.tau(E, r, z, self.solve_xH(r, z))
        if not self.ADAF:
            pfact = np.zeros_like(E)
            pfact[E < 0.25] = -2.65
            pfact[E >= 0.25] = -3.30
            term2 = - pfact / E * tau
            prefactor = self.A_MLamb() / (4.*np.pi*r**2.) * (1./3.086e19/1e2)**2.
            term1 = -1. / E
            return prefactor * (term1 + term2) * np.exp(-tau) / E
        else:
            prefactor = self.Luminosity(z) / (4. * np.pi * r**2.) * np.exp(-tau)
            dtau = -pfact * (E / 0.25) * tau
            return prefactor * (self.diff_spectrum(E) - dtau * self.spectrum(E))

    
    def NumPh(self, E, r, z, xh): # units 1/cm^2/s
        tau = self.tau(E, r, z, xh)
        if not self.ADAF:
            return self.A_MLamb() / (4.*np.pi*r**2. * E) * np.exp(-tau) * (1./3.086e19/1e2)**2.
        else:
            return self.Luminosity(z) / (4.*np.pi*r**2.) * self.spectrum(E) * np.exp(-tau) * (1./3.086e19/1e2)**2.
    
    def Gam_r(self, r, z, xh):
        elist = np.logspace(np.log10(self.e_min), np.log10(self.e_max), 60)
        integrand = self.sigma_E(elist) * self.NumPh(elist, r, z, xh) / elist
        val = np.trapz(integrand, elist)
        return val

    def solve_xH(self, r, z):
        if self.density_evol:
            #v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            v_b = self.v_effective(z) / 2.998e5
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            if r > r_b:
                density_r = n_hyd(z)
            else:
                density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
        else:
            density_r = n_hyd(z)

        alphaH = 2.6e-13 # cm^3/s
        soln = fsolve(lambda x: alphaH * density_r**2 * (1. - x)**2. - self.Gam_r(r,z,x)*x*density_r, 0.5)
        
        # Not sure about this implimentation...
        if soln > self.xH_0_highr(z):
            return self.xH_0_highr(z)
        return soln

    def tau(self, E, r, z, xH, use_tau=True): # Figure out way to solve for this
        if not use_tau:
            return 0.
        if self.density_evol:
            #v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            v_b = self.v_effective(z) / 2.998e5
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            if r > r_b:
                density_r = n_hyd(z)
            else:
                density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
        else:
            density_r = n_hyd(z)
        approx_step = self.sigma_E(E) * xH * density_r * r * kpc_to_m * 1e2
        return approx_step

    def H_Pbh(self, r, z):
        if self.density_evol:
            #v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            v_b = self.v_effective(z) / 2.998e5
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            if r > r_b:
                density_r = n_hyd(z)
            else:
                density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
        else:
            density_r = n_hyd(z)
        
        xhVal = self.solve_xH(r, z)
     
        energydep =  self.f_fitting(1. - xhVal)
        prefact = density_r * xhVal * energydep
        elist = np.logspace(np.log10(self.e_min), np.log10(self.e_max), 100)
        term2 = np.trapz(self.sigma_E(elist) * self.NumPh(elist, r, z, xhVal), elist)
        return term2 * prefact * 1e3 # eV / cm^3 / s

    def f_fitting(self, xe):
        if xe < 1e-4:
            return 0.15
        return 0.9771*(1. - (1. - xe**0.2663)**1.3163)
    
    def epsilon_acc(self, mdot):
        Ledd = 8.614e46 * self.M # keV /s
        if (10. * mdot / Ledd) < 9.4e-5:
            ep_0 = 0.12
            pp = 0.59
        elif (10. * mdot / Ledd) > 5e-3:
            ep_0 = 0.50
            pp = 4.53
        else:
            ep_0 = 0.026
            pp = 0.27
        return ep_0 * (10. * mdot / Ledd) ** pp
    

    def H_compton(self, r, z, tk):
        if self.density_evol:
            #v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            v_b = self.v_effective(z) / 2.998e5
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            if r > r_b:
                density_r = n_hyd(z)
            else:
                density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
        else:
            density_r = n_hyd(z)
        
        clight_cm = clight * 1e2
        xhVal = self.solve_xH(r, z)
        ne = (1. - xhVal) * density_r # cm^-3
        prefact = 4.*np.pi**2.*sigma_thomp*kbolt**5.*T_CMB(z)**4.*ne*(T_CMB(z) - tk) / (15.*5.11e5) / hbar**3. # ev/cm/s^3
        val = prefact / clight_cm**2. # should be eV / cm^3 / s
        return val

    def H_exp(self, r, z, tk):
        if self.density_evol:
            #v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            v_b = self.v_effective(z) / 2.998e5
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            if r > r_b:
                density_r = n_hyd(z)
            else:
                density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
        else:
            density_r = n_hyd(z)
        
        xhVal = self.solve_xH(r, z)
        return -3.*hubble(z)*kbolt*tk*density_r*(2. - xhVal) / kpc_to_m / 1e3 * clight # eV/cm^3/s

    def solve_Tk(self, r, z):
        soln = fsolve(lambda x: self.H_exp(r,z,x) + self.H_compton(r,z,x) + self.H_Pbh(r,z), 1e2)
        return soln + self.solve_xH(r,z)*self.tk_0_highr(z)

    def y_k(self, r, z, tk):
        return t_star / (tk * A10) * (self.deexcit_CH(r,z,tk) + \
               self.deexcit_Ce(r,z,tk) + self.deexcit_Cp(r,z,tk))
    
    def deexcit_CH(self, r, z, tk):
        if self.density_evol:
            #v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            v_b = self.v_effective(z) / 2.998e5
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            if r > r_b:
                density_r = n_hyd(z)
            else:
                density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
        else:
            density_r = n_hyd(z)
        
        prefact = 3.1e-11 * density_r * self.solve_xH(r,z)
        return prefact * tk**0.357 * np.exp(-32./tk)

    def deexcit_Ce(self, r, z, tk):
        if self.density_evol:
            #v_b = np.sqrt(self.tk_0_highr(z) * 8.615e-5 / 0.938e9)
            v_b = self.v_effective(z) / 2.998e5
            r_b = 6.67e-11*2e30*self.M / v_b**2. / (2.998e8)**2. / 3.086e19
            if r > r_b:
                density_r = n_hyd(z)
            else:
                density_r = 0.164*n_hyd(z)*(r / r_b)**(-3. / 2.)
        else:
            density_r = n_hyd(z)
        
        return self.gamma_e(tk) * density_r * (1. - self.solve_xH(r, z))

    def deexcit_Cp(self, r, z, tk):
        return 3.2*(1. - self.solve_xH(r, z)) * self.deexcit_CH(r, z, tk)

    def gamma_e(self, Tk):
        if Tk > 1e4:
            Tk = 1e4
        if Tk < 1.: # This is a safety net that I shouldnt, hopefully
            print 'Tk falling below 0.1??? Consider killing run.'
            Tk = 1.
        lnVal = -9.607 + 0.5 * np.log10(Tk)*np.exp(-np.log10(Tk)**4.5/1800.)
        
        return np.exp(lnVal) # units 1/cm^3/s

    def y_alpha(self, r, z, tk):
        prefactor = 16. * np.pi**2. * t_star * f12 / (27. * A10 * 5.11e5 * clight * 1e2) * 0.0917
        term1 = self.J_tilde(r, z) / tk
        return prefactor * term1

    def T_spin(self, r, z):
        tk = self.solve_Tk(r,z)
        yk = self.y_k(r,z,tk)
        y_alp = self.y_alpha(r,z,tk)
        numer = t_star + T_CMB(z) + tk * (yk + y_alp)
        return numer / (1. + yk + y_alp)

    def T_21(self, r, z, endpoint=None):
        if endpoint is None:
            return 27. * self.solve_xH(r,z) * (1. - T_CMB(z)/self.T_spin(r, z))*(omega_b*h_little**2./0.023)*\
                    ((1.+z)*0.15/(10.*omega_M*h_little**2.))**0.5
        if type(r) != np.array:
            print T_CMB(z)/self.T_spin(r, z), endpoint, self.solve_xH(r,z)
            return 27. * self.solve_xH(r,z) * (1. - T_CMB(z)/self.T_spin(r, z))*(omega_b*h_little**2./0.023)*\
                    ((1.+z)*0.15/(10.*omega_M*h_little**2.))**0.5 - endpoint*self.solve_xH(r,z)
        soln = np.zeros_like(r)
        for i, rr in enumerate(r):
            xhV = self.solve_xH(rr,z)
            soln[i] = 27. * xhV * (1. - T_CMB(z)/self.T_spin(rr, z))*(omega_b*h_little**2./0.023)*\
                    ((1.+z)*0.15/(10.*omega_M*h_little**2.))**0.5 - xhV
        return soln

    def mean_T21(self, z):
        vB = self.v_effective(z)
#        t_cmb = 2.725 * (1. + z)
#        kboltz = 8.617e-5
#        vB = np.sqrt((1. + 1. - self.xH_0_highr(z)) * t_cmb * kboltz / 0.938e9) * 2.998e5
        rB = 6.67e-11*2e30*self.M / vB**2. / 1e6 / 3.086e19 # kpc
        print rB, 1.2e-4 * self.M * 1e3 / (1. + z) * 1e-3
        exit()
        rtable = np.logspace(np.log10(rB) - 4, np.log10(rB), 100)
        deltNu = np.zeros_like(rtable)
        for i,r in enumerate(rtable):
            deltNu[i] = self.delta_nu_eff(r, z)
        
        supress = self.xH_0_highr(z)
        t21  = 27. * supress * np.sqrt((1. + z)/10.)
        tilde_21 = np.trapz(2. * np.pi * deltNu * t21 * rtable, rtable)
        preterm = self.n_pbh / 1e6 / hubble(z)
      
        return preterm * tilde_21 * supress
#        rtable = np.logspace(-3, 6, 100)
#        t21 = np.zeros_like(rtable)
#        deltNu = np.zeros_like(rtable)
#        endpoint = self.T_21(1e10, z)
#        for i,r in enumerate(rtable):
#            t21[i] = self.T_21(r, z, endpoint=endpoint)
#            deltNu[i] = self.delta_nu_eff(r, z)
#        if np.abs(t21[-1]) < 1:
#            for i in range(len(rtable)):
#
#                if np.abs(t21[-1-i]) > 1:
#                    break_indx = len(rtable) - 1 - i
#                    break
#        else:
#            break_indx = len(rtable) - 1
#        rtable = rtable[:break_indx + 1]
#        deltNu = deltNu[:break_indx + 1]
#        t21 = t21[:break_indx + 1]
#        tilde_21 = np.trapz(2. * np.pi * deltNu * t21 * rtable, rtable)
#        preterm = self.n_pbh / 1e6 / hubble(z)
#        supress = self.xH_0_highr(z)
#        return preterm * tilde_21 * supress

    def delta_nu_eff(self, r, z):
        tk = self.solve_Tk(r, z)
        return np.sqrt(2.*kbolt*tk / 0.938e9)

