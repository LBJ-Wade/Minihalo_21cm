from minihalos import *
import time

SM = 100
SM_2 = 1e4
redshift=30
Sve = True
mHalo = Minihalos(SM)

mHalo = Minihalos(SM, example_plots=True)
mHalo_21 = Minihalos(SM_2, fpbh=1e-7, example_plots=False)

radius = 1e-3
r21 = 1e0
print 'XH for M\lambda = 100 and z = 30. Radius = 1e-3 kpc: ', mHalo.solve_xH(radius, redshift)
print 'Tau for M\lambda = 100 and z = 30. Radius 1e2 kpc: ', mHalo.tau(.2, 1e2, redshift, mHalo.solve_xH(1e2, redshift))
tk = mHalo.solve_Tk(radius, redshift)
print 'Tk for M\lambda = 100 and z = 30. Radius = 1e-3 kpc: ', tk
print 'yk for M\lambda = 100 and z = 30. Radius = 1e-3 kpc: ', mHalo.y_k(radius, redshift, tk)
print 'y_alpha for M\lambda = 100 and z = 30. Radius = 1e-3 kpc: ', mHalo.y_alpha(radius, redshift, tk)
print 'Ts for M\lambda = 100 and z = 30. Radius = 1e-3 kpc: ', mHalo.T_spin(radius, redshift)
print 'T_21 for M\lambda = 100 and z = 30. Radius = 1 kpc: ', mHalo.T_21(r21, redshift)
print 'T_21 for M = 100 (REAL LAMBDA) and z = 30. Radius = 1 kpc: ', mHalo_21.T_21(r21, redshift)
print 'Tk for M\lambda = 100 and z = 30. Radius = 1e4 kpc: ', mHalo.solve_Tk(1e4, redshift)

tstart = time.time()
print 'Sky averaged T_21 for M\lambda = 100, z = 30, f_pbh = 1: ', mHalo_21.mean_T21(redshift)

tend = time.time()
print tend - tstart
exit()

if Sve:
    radius_scan = np.logspace(-3, 5, 100)
    xh_store = np.zeros_like(radius_scan)
    tau_store = np.zeros_like(radius_scan)
    tk_store = np.zeros_like(radius_scan)
    yk_store = np.zeros_like(radius_scan)
    yalph_store = np.zeros_like(radius_scan)
    ts_store = np.zeros_like(radius_scan)
    t21_store = np.zeros_like(radius_scan)
    for i,rr in enumerate(radius_scan):
        xh_store[i] = mHalo.solve_xH(rr, redshift)
        tau_store[i] = mHalo.tau(.2, rr, redshift, mHalo.solve_xH(rr, redshift))
        tk_store[i] = mHalo.solve_Tk(rr, redshift)
        yk_store[i] = mHalo.y_k(rr, redshift, tk_store[i])
        yalph_store[i] = mHalo.y_alpha(rr, redshift, tk_store[i])
        ts_store[i] = mHalo.T_spin(rr, redshift)
        t21_store[i] = mHalo_21.T_21(rr, redshift)

    redshift_list = np.logspace(1, 2, 40)
    t21_global = np.zeros_like(redshift_list)
    for i, z in enumerate(redshift_list):
        t21_global[i] = mHalo_21.mean_T21(z)

    np.savetxt('Store_Tests/xH_M_{:.0e}_redshift_{:.0e}_radius_scan.dat'.format(SM, redshift), np.column_stack((radius_scan, xh_store)))
    np.savetxt('Store_Tests/Tau_M_{:.0e}_redshift_{:.0e}_radius_scan.dat'.format(SM, redshift), np.column_stack((radius_scan, tau_store)))
    np.savetxt('Store_Tests/tk_M_{:.0e}_redshift_{:.0e}_radius_scan.dat'.format(SM, redshift), np.column_stack((radius_scan, tk_store)))
    np.savetxt('Store_Tests/yk_M_{:.0e}_redshift_{:.0e}_radius_scan.dat'.format(SM, redshift), np.column_stack((radius_scan, yk_store)))
    np.savetxt('Store_Tests/yalpha_M_{:.0e}_redshift_{:.0e}_radius_scan.dat'.format(SM, redshift), np.column_stack((radius_scan, yalph_store)))
    np.savetxt('Store_Tests/TSpin_M_{:.0e}_redshift_{:.0e}_radius_scan.dat'.format(SM, redshift), np.column_stack((radius_scan, ts_store)))
    np.savetxt('Store_Tests/T21_M_{:.0e}_redshift_{:.0e}_radius_scan.dat'.format(SM, redshift), np.column_stack((radius_scan, t21_store)))

    np.savetxt('Store_Tests/Global21_M_{:.0e}.dat'.format(SM, redshift), np.column_stack((redshift_list, t21_global)))

