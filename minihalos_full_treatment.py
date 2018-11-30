import numpy as np
import os
from scipy.optimize import fsolve, root
from scipy.interpolate import interp1d
from scipy.special import gammaincc
from helpers import *


class Minihalos(object):
    def __init__(self, M, fpbh=1, lamb=0.01):
        self.M = M
        self.e_min = 0.2
        self.e_max = 2e5
        self.n_pbh = 1.256e-2 * (omega_cdm / 1e-9) * fpbh / M * 1e4 # Mpc^-3
        loadTK = np.loadtxt('input_files/recfast_LCDM.dat')
        self.tk_0_highr = interp1d(loadTK[:,0], loadTK[:,-1], kind='linear')
        self.load_files()
        self.lamb = lamb
        self.xe_highr_highZ = interp1d(loadTK[:,0], loadTK[:,1])
        xe_lowZ = np.loadtxt('input_files/ThermalHistory_Xi_30_Tmin_1.000e+04_Rfmp_15_chiUV_2.00e+56_Nalpha_4.00e+03.dat')
        self.xe_highr_lowZ = interp1d(xe_lowZ[:,0], xe_lowZ[:,1], kind='linear')

    def load_files(self):
        veff = np.loadtxt('loads/veff.dat')
        self.v_effective = interp1d(veff[:,0], veff[:,1], kind='linear', fill_value='extrapolate', bounds_error=False)
        return
    
    def file_tag(self):
        tag = '_Mass_{:.2e}_fpbh_{:.2e}_lamb_{:.2e}'.format(self.M, self.fpbh, self.lamb)
        return tag
    
    def load_profile(self, file, z):
        try:
            profile_pbh = np.loadtxt(file)
        except FileNotFoundError:
            print 'File with profile not found! Exiting...'
            exit()
        rho_inf = 200. * ((1. + z) / 1e3)**3. * 0.938 # GeV / cm^3
        #t_cmb = 2.725 * (1. + z)
        t_infty = self.tk_0_highr(z)
        kboltz = 8.617e-5
        vB = np.sqrt((1. + 1. - self.xH_0_highr(z)) * t_infty * kboltz / 0.938e9) * 2.998e5
        rB = 6.67e-11*2e30*self.M / vB**2. / 1e6 / 3.086e19 # kpc
        self.density_prof = interp1d(np.log10(profile_pbh[:,0]*rB), np.log10(profile_pbh[:,1] * rho_inf),
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
        self.temp_prof = interp1d(np.log10(profile_pbh[:,0]*rB), np.log10(profile_pbh[:,3] * t_infty),
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
        self.xH_prof = interp1d(np.log10(profile_pbh[:,0]*rB), np.log10(profile_pbh[:,4]),
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
        print 'Min/Max radius [kpc]: ', profile_pbh[-1,0]*rB, profile_pbh[0,0]*rB
        return
        
        
    def xH_0_highr(self, z):
        if z > 20:
            return 1. - self.xe_highr_highZ(z)
        else:
            return self.xe_highr_lowZ(z)

    def Mdot_HB(self, z):
        rho_inf = 200. * ((1. + z) / 1e3)**3. * 0.938 # GeV / cm^3
        mass_term = 4.* np.pi * (self.M * 6.67e-11 * 2e30)**2.
        return rho_inf * self.lamb * mass_term / self.v_effective(z)**3. / 1e3 # GeV / s

    def Luminosity(self, z):
        Mdot = self.Mdot_HB(z)
        epsilon = self.epsilon_acc(Mdot*1e6)
        wmin = np.sqrt(10. / self.M)*1e-3
        Ts = 200.
        spec_norm = gammaincc(0., wmin / Ts)
        return epsilon * Mdot

    def spectrum(self, E):
        wmin = np.sqrt(10. / self.M) * 1e-3
        Ts = 200.
        sols = E**-1. * np.exp(-E / Ts)
        sols = sols[E > wmin]
        return sols

    def Gam_r(self, r, z, xh):
        elist = np.logspace(np.log10(13.7), np.log10(self.e_max), 60)
        integrand = self.sigma_E(elist / 1e3) * self.NumPh(elist, r, z, xh) / elist

        integrand[elist > 13.7] *= (1. + elist[elist > 13.7] / 13.7 * 0.33)

        val = np.trapz(integrand, elist)
        return val
        
    def sigma_E(self, E):
        p = np.zeros_like(E)
        p[E < 0.25] = -2.65
        p[E >= 0.25] = -3.30
        return 4.25e-21 * (E / 0.25)**p # cm^2
        
    def NumPh(self, E, r, z, xh): # units 1/cm^2/s
        tau = 0.
        return self.Luminosity(z) / (4.*np.pi*r**2.) * self.spectrum(E) * np.exp(-tau) * (1./3.086e19/1e2)**2. * 1e6
    
        
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
        
    def f_fitting(self, xe):
        if xe < 1e-4:
            return 0.15
        return 0.9771*(1. - (1. - xe**0.2663)**1.3163)
        

    def init_conditions(self, z, r0):
        xev = 1. - self.xH_0_highr(z)
        rho_inf = 200. * ((1+z) / 1e3)**3. * 0.938 # GeV / cm^3
        #t_cmb = 2.725 * (1. + z)
        kboltz = 8.617e-5
        vB = np.sqrt((1. + 1. - self.xH_0_highr(z)) * self.tk_0_highr(z) * kboltz / 0.938e9) * 2.998e5
        rB = 6.67e-11*2e30*self.M / vB**2. / 1e6 / 3.086e19 # kpc
        xx = r0 / rB
        lambda_c = self.Mdot_HB(z) / (4. * np.pi * rho_inf *  rB**2. * vB) / 1e5 / (3.086e21)**2.
        #return [0., -30, 0., np.log10(xev)] # rho, v, T, xe
        return [1., -lambda_c / xx**2., 1., 1. - xev]

    def solve_step(self, z, init_cs, guess, radius2):
        soln = fsolve(self.equations, guess, args=(init_cs, radius2, z), maxfev=10000)
    
        return soln

    def solve_radius_runner(self, z):
    
        step_size = 1e-1
        r_0 = 2.
        r_current = r_0 - step_size
        i_conds = self.init_conditions(z, r_0)
        
        sve_array = [[10.**r_0, i_conds[0], i_conds[1], i_conds[2], i_conds[3]]]

        perturb = np.asarray([1.02, 1.02, 1.02, 0.98])
        new_guess = np.array([sve_array[-1][1:]])*perturb
    
        pass_threshold = 1e-1
        pass_test = False
        while r_current > -4:
            radius2 = [10.**r_current, 10.**r_0]
            pass_test = False
            
            
            while not pass_test:
                perturb_base = np.array([1., 1., 1., 0.])
                perturb = np.zeros(4)
                for i in range(len(perturb)):
                    perturb[i] = 1.
                    if i < 3:
                        perturb[i] += np.random.rand(1) * 1e-1
                    else:
                        perturb[i] -= np.random.rand(1) * 1e-2
            
                new_guess = perturb * np.asarray(sve_array[-1][1:])
                
                new_radius = self.solve_step(z, sve_array[-1][1:], new_guess, radius2)
                
                check = (new_radius - sve_array[-1][1:])
                test_posi = []
                if check[0] >= 0:
                    test_posi.append(True)
                else:
                    test_posi.append(False)
                if check[1] <= 0:
                    test_posi.append(True)
                else:
                    test_posi.append(False)
                if check[2] >= 0:
                    test_posi.append(True)
                else:
                    test_posi.append(False)
                if check[3] <= 0:
                    test_posi.append(True)
                else:
                    test_posi.append(False)
                if np.all(np.asarray(test_posi)) == True:
                    pass_test = True
#                print 'CHECK: ', new_radius
#                print check, np.asarray(test_posi)

#            test1 = np.sum(np.abs(self.equations(new_radius, sve_array[-1][1:], radius2, z)))
#            test2 = np.abs(new_radius) - np.asarray(sve_array[-1][1:])

            sve_array.append([10.**r_current, new_radius[0], new_radius[1], new_radius[2], new_radius[3]])
            print sve_array[-1]

            if np.isnan(new_guess[0]):
                exit()
            r_0 = r_current
            r_current = r_current - step_size

        return sve_array


    def equations(self, x, init_cs, radius2, z):
        rho_i, vel_i, tk_i, xH_i = x
        rho_i_m1, vel_i_m1, tk_i_m1, xH_i_m1 = init_cs

        xe_i = 1. - xH_i
        xe_i_m1 = 1. - xH_i_m1

        r_i, r_im1 = radius2
        rad_diff = r_i - r_im1
        rho_inf = 200. * ((1+z) / 1e3)**3. * 0.938 # GeV / cm^3
        tk_inf = self.tk_0_highr(z)
        
        rho_cmb = 0.260 * (1. + z)**4. * 1e-9 # GeV / cm^3
        t_cmb = 2.725 * (1. + z)
        kboltz = 8.617e-5
        
        #vB = self.v_effective(z)
        #vB = np.sqrt((1. + 1. - self.xH_0_highr(z)) * t_cmb * kboltz / 0.938e9) * 2.998e5
        vB = np.sqrt((1. + 1. - self.xH_0_highr(z)) * tk_inf * kboltz / 0.938e9) * 2.998e5
        rB = 6.67e-11*2e30*self.M / vB**2. / 1e6 / 3.086e19 # kpc
        tB = 6.67e-11*2e30*self.M / vB**3. / 1e9 # s
#        xe = 1. - self.xH_0_highr(z)
        lambda_c = self.Mdot_HB(z) / (4. * np.pi * rho_inf *  rB**2. * vB) / 1e5 / (3.086e21)**2.
    
        beta = 4. * xe_i * 6.65e-25 * 6.67e-11*2e30*self.M / (0.938 * vB**3.) * rho_cmb * 2.998e10 / (1e3)**3
#        beta = 0.
        gamma = 2. * 0.938 * beta / (5.11e-4 * (1. + xe_i))

        eddyL = self.Mdot_HB(z) / (1.26e38 * self.M * 6.242e11 / 1e9)
#        print self.epsilon_acc(self.Mdot_HB(z)*1e6) * eddyL, lambda_c, beta, gamma, self.lamb
#        exit()

        e_integ = np.logspace(np.log10(13.7), np.log10(self.e_max), 50)
        h_pbh = self.f_fitting(xe_i) * rho_i * rho_inf / (0.938) * (1. - xe_i) * \
            np.trapz(self.NumPh(e_integ, r_i, z, 1. - xe_i) * self.sigma_E(e_integ / 1e3) * (e_integ - 13.7) / e_integ, e_integ)
        h_pbh = 0.
        h_compton = 4.*6.65e-25*rho_cmb * rho_inf * rho_i * xe_i / (5.11e-4 * 0.938) * (t_cmb - tk_i) * 2.998e10 * kboltz
        
        eq1 = rho_i * np.abs(vel_i) * r_i**2. - lambda_c
       
        eq2 = vel_i * (vel_i - vel_i_m1) / rad_diff  + tk_inf*kboltz / (0.938e9 * vB**2.) * (2.998e5)**2. / rho_i * \
            (rho_i * (1. + xe_i) * tk_i - rho_i_m1 * (1. + xe_i_m1) * tk_i_m1) / rad_diff + \
            6.67e-11 * 2e30 * self.M / (rB * vB**2. * r_i**2.) / (1e3)**2. / (3.086e19)
        
        eq3 = vel_i  * np.abs(rho_i)**(2./3)  * (tk_i * (1. + xe_i) / np.abs(rho_i)**(2./3) -
            tk_i_m1 * (1. + xe_i_m1) / np.abs(rho_i_m1)**(2./3)) / rad_diff - \
            2. *0.938 / (3. * rho_inf * rho_i) * rB / (vB * tk_inf * kboltz) * 3.086e16 * (h_pbh + h_compton)

        alph_H = 2.6e-13 * (np.abs(tk_i) * tk_inf / 1e4)**-0.85 # neglecting Tk dependence cm^3/s
        eq4 = -vel_i * (xH_i - xH_i_m1) / rad_diff - self.Gam_r(r_i, z, xH_i) * rB / vB * xH_i * 3.086e16 + \
            alph_H*rho_inf* rB / (0.938 * vB) * (1. - xH_i)**2. * 3.086e16


        penalty = np.array([0., 0., 0., 0.])
        if xe_i < 0. or xe_i > 1:
            penalty += np.ones_like(penalty) * np.abs(xe_i) * 1e10
        if vel_i > 0.:
            penalty += np.ones_like(penalty) * np.abs(vel_i) * 1e10
        if tk_i < 0.:
            penalty += np.ones_like(penalty) * np.abs(tk_i) * 1e10
        return [eq1 + penalty[0], eq2 + penalty[1], eq3 + penalty[2], eq4 + penalty[3]]

    def y_k(self, r, z, tk):
        return t_star / (tk * A10) * (self.deexcit_CH(r,z,tk) + \
               self.deexcit_Ce(r,z,tk) + self.deexcit_Cp(r,z,tk))
    
    def deexcit_CH(self, r, z, tk):
        density_r = 10.**self.density_prof(np.log10(r))
        xH = 10.**self.xH_prof(np.log10(r))
        prefact = 3.1e-11 * density_r * xH
        return prefact * tk**0.357 * np.exp(-32./tk)

    def deexcit_Ce(self, r, z, tk):
        density_r = 10.**self.density_prof(np.log10(r))
        xH = 10.**self.xH_prof(np.log10(r))
        return self.gamma_e(tk) * density_r * (1. - xH)

    def deexcit_Cp(self, r, z, tk):
        xH = 10.**self.xH_prof(np.log10(r))
        return 3.2*(1. - xH) * self.deexcit_CH(r, z, tk)

    def gamma_e(self, Tk):
        if Tk > 1e4:
            Tk = 1e4
        if Tk < 1.: # This is a safety net that I shouldnt, hopefully
            print 'Tk falling below 0.1??? Consider killing run.'
            Tk = 1.
        lnVal = -9.607 + 0.5 * np.log10(Tk)*np.exp(-np.log10(Tk)**4.5/1800.)
        
        return np.exp(lnVal) # units 1/cm^3/s

    def J_tilde(self, r, z):
    
        density_r = 10.**self.density_prof(np.log10(r))
        energydep = self.phi_alpha(z,r)
        xH = 10.**self.xH_prof(np.log10(r))
        prefactor = energydep*clight*1e2*density_r*xH  / \
                    (4. * np.pi * hubble(z) * nu_21) * 1e3 * kpc_to_m * 1e2
        
        elist = np.logspace(np.log10(self.e_min), np.log10(self.e_max), 100)
        
        integral = np.trapz(self.sigma_E(elist)* self.NumPh(elist, r, z, xH),  elist) / 1e3
        return prefactor * integral


    def phi_alpha(self, z, r):
        xe = 1. - 10.**self.xH_prof(np.log10(r))
        return 0.48 * (1. - xe**0.27)**1.52

    def y_alpha(self, r, z, tk):
        prefactor = 16. * np.pi**2. * t_star * f12 / (27. * A10 * 5.11e5 * clight * 1e2) * 0.0917
        term1 = self.J_tilde(r, z) / tk
        return prefactor * term1

    def T_spin(self, r, z):
        tk = 10.**self.temp_prof(np.log10(r))
        
        yk = self.y_k(r, z, tk)
        y_alp = self.y_alpha(r,z,tk)
        numer = t_star + T_CMB(z) + tk * (yk + y_alp)
        return numer / (1. + yk + y_alp)

    def T_21(self, r, z):
        xH = 10.**self.xH_prof(np.log10(r))
        Tspin = self.T_spin(r, z)
        return 27. * xH * (1. - T_CMB(z)/Tspin)*(omega_b*h_little**2./0.023)*\
                    ((1.+z)*0.15/(10.*omega_M*h_little**2.))**0.5

    def mean_T21(self, z):
        rtable = np.logspace(-7, -1, 50)
        deltNu = np.zeros_like(rtable)
        t21vals = np.zeros_like(rtable)
        for i,r in enumerate(rtable):
            deltNu[i] = self.delta_nu_eff(r, z)
            t21vals[i] = self.T_21(r, z)

        deltNu = deltNu[t21vals > 0]
        rtable = rtable[t21vals > 0]
        t21vals = t21vals[t21vals > 0]

        tilde_21 = np.trapz(2. * np.pi * deltNu * t21vals * rtable, rtable)
        preterm = self.n_pbh / 1e6 / hubble(z)
        return preterm * tilde_21


    def delta_nu_eff(self, r, z):
        tk = 10.**self.temp_prof(np.log10(r))
        return np.sqrt(2.*kbolt*tk / 0.938e9)




