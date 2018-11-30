from minihalos_full_treatment import *

Mass = 1e2
fpbh = 1
lamb = 1e-2
redshift = 30

miniH = Minihalos(Mass, fpbh, lamb)

compute_profile = True

fileName = 'outputs/SphericalTreatment_Mass_{:.2e}_fpbh_{:.2e}_lamb_{:.2e}_redshift_{:.0f}.dat'.format(Mass, fpbh, lamb, redshift)

if compute_profile:
    print 'Computing Profile...'
    soln_r =  np.asarray(miniH.solve_radius_runner(redshift))
    print 'Success!'
    np.savetxt(fileName, soln_r)
    exit()
else:
    miniH.load_profile(fileName, redshift)

print 'Compute T_21...'

r_list = np.logspace(-7, -1, 30)
for i,r in enumerate(r_list):
    print r, miniH.T_21(r, redshift)

print miniH.mean_T21(redshift)
