from minihalos import *

# Mass of monochromatic PBH distribution
Mass = 1e2 # 1e2, 1e4
# Fraction of PBH accounting for DM
fpbh = 1. #1e-7 / 0.266
# Treat baryonic density distribution as uniform
density_evol = False
# ADAF = Poulin et al PBH accretion, otherwise Bernal et al
ADAF = False

# Create data for plots detailing xH, Tk, yk, ...
step_plots = False

# If calculating global 21cm contribution, redshifts of interest
z_max = 40
z_min = 8
z_number = 15


# Use value of \lambda used in Bernal et al's plots, ie \lambda = 1
bernal_plots_lambda = True
if ADAF:
    bernal_plots_lambda = False
redshift = 30 # redshift of example in step_plots


MiniHalos = Minihalos(Mass, fpbh=fpbh, bernal_plots=bernal_plots_lambda, density_evol=density_evol, ADAF=ADAF)
output_tag = 'outputs/'

if step_plots:
    if not bernal_plots_lambda:
        radius_scan = np.logspace(-6, 2, 100)
    else:
        radius_scan = np.logspace(-3, 6, 100)

    xh_store = np.zeros_like(radius_scan)
    tau_store = np.zeros_like(radius_scan)
    tk_store = np.zeros_like(radius_scan)
    yk_store = np.zeros_like(radius_scan)
    yalph_store = np.zeros_like(radius_scan)
    ts_store = np.zeros_like(radius_scan)
    t21_store = np.zeros_like(radius_scan)

    endpoint = MiniHalos.T_21(1e10, redshift)
    for i,rr in enumerate(radius_scan):
        xh_store[i] = MiniHalos.solve_xH(rr, redshift)
        tau_store[i] = MiniHalos.tau(.2, rr, redshift, MiniHalos.solve_xH(rr, redshift))
        tk_store[i] = MiniHalos.solve_Tk(rr, redshift)
        yk_store[i] = MiniHalos.y_k(rr, redshift, tk_store[i])
        yalph_store[i] = MiniHalos.y_alpha(rr, redshift, tk_store[i])
        ts_store[i] = MiniHalos.T_spin(rr, redshift)
        t21_store[i] = MiniHalos.T_21(rr, redshift, endpoint=endpoint)

    miniH_tag = MiniHalos.file_tags()
    np.savetxt(output_tag + 'XH_z_{:.0f}_'.format(redshift) + miniH_tag, np.column_stack((radius_scan, xh_store)))
    np.savetxt(output_tag + 'Tau_E_0.2keV_z_{:.0f}_'.format(redshift) + miniH_tag, np.column_stack((radius_scan, tau_store)))
    np.savetxt(output_tag + 'Tk_z_{:.0f}_'.format(redshift) + miniH_tag, np.column_stack((radius_scan, tk_store)))
    np.savetxt(output_tag + 'Yk_z_{:.0f}_'.format(redshift) + miniH_tag, np.column_stack((radius_scan, yk_store)))
    np.savetxt(output_tag + 'Yalpha_z_{:.0f}_'.format(redshift) + miniH_tag, np.column_stack((radius_scan, yalph_store)))
    np.savetxt(output_tag + 'Ts_z_{:.0f}_'.format(redshift) + miniH_tag, np.column_stack((radius_scan, ts_store)))
    np.savetxt(output_tag + 'T21_z_{:.0f}_'.format(redshift) + miniH_tag, np.column_stack((radius_scan, t21_store)))

if not step_plots:
    redshift_list = np.logspace(np.log10(z_min), np.log10(z_max), z_number)
    t21_global = np.zeros_like(redshift_list)
    
    for i, z in enumerate(redshift_list):
        t21_global[i] = MiniHalos.mean_T21(z)
        print z, t21_global[i]

    fileName = output_tag + 'T21_Global_' + MiniHalos.file_tags()
    np.savetxt(fileName, np.column_stack((redshift_list, t21_global)))



