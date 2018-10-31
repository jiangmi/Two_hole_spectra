import math
import numpy as np
M_PI = math.pi

Mc = 3
ed = 0
eps = np.arange(4.0, 4.01, 0.5) #[3.5]#,3.5,4.5]

# Note: tpd and tpp are only amplitude signs are considered separately in hamiltonian.py
# Slater Koster integrals and the overlaps between px and d_x^2-y^2 is sqrt(3) bigger than between px and d_3x^2-r^2 
# These two are proportional to tpd,sigma, whereas the hopping between py and d_xy is proportional to tpd,pi

# Eskes thesis Table 3.1 defines:
# tpd = sqrt(3)/2*pds ~ (1.3)
# tpp = 1/2*(ppp-pps) ~ (0.65)

# IMPORTANT: keep all hoppings below positive to avoid confusion
#            hopping signs are considered in dispersion separately
Norb = 9
if Norb==3 or Norb==7:
    #tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    tpds = [0.4] #np.arange(0.25, 0.4, 0.01)
    tpps = [0.55]
elif Norb==9:    
    pdss = [1.5]
    pdps = [0.7]
    pdss = [0.00001]
    pdps = [0.00001]
    pps = 1.7
    ppp = 0.3
    #pps = 0.00001
    #ppp = 0.00001

eta = 0.1
Lanczos_maxiter = 800

# restriction on variational space
VS_only_up_up = 0
VS_only_up_dn = 0

basis_change_type = 'all_states' # 'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_find_lowpeak = 0
if if_find_lowpeak==1:
    peak_mode = 'lowest_peak' # 'lowest_peak' or 'highest_peak'
    if_write_lowpeak_ep_tpd = 1
if_write_Aw = 0
if_savefig_Aw = 1

if_get_ground_state = 0
if if_get_ground_state==1:
    Neval = 1
if_compute_Aw_dd_total = 0
if_compute_Aw_pp = 0
if_compute_Aw_dp = 0
if_compute_Aw_Cu_dx2y2_O = 1

if Norb==3:
    Cu_orbs = ['dx2y2']
else:
    Cu_orbs = ['dx2y2','dxy','dxz','dyz','d3z2r2']
    #Cu_orbs = ['dx2y2','d3z2r2']
    
if Norb==3 or Norb==7:
    O1_orbs  = ['px']
    O2_orbs  = ['py']
elif Norb==9:
    O1_orbs  = ['px1','py1']
    O2_orbs  = ['px2','py2']
O_orbs = O1_orbs + O2_orbs
# sort the list to facilliate the setup of interaction matrix elements
Cu_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
O_orbs.sort()
print "Cu_orbs = ", Cu_orbs
print "O1_orbs = ",  O1_orbs
print "O2_orbs = ",  O2_orbs
orbs = Cu_orbs + O1_orbs + O2_orbs
#assert(len(orbs)==Norb)
# ======================================================================
# Below for interaction matrix
Upps = [0]
if Norb==3:
    Udd = 8.84  # A+4*B+3*C
if Norb==7 or Norb==9:
    interaction_sym = ['ALL']#,'3B1']#,'1A2','3A2','1B1','3B1','1E','3E']#,'1B2','3B2']
    print "turn on interactions for symmetries = ",interaction_sym
    
    if interaction_sym == ['ALL']:
        symmetries = ['1A1','3B1']
    else:
        symmetries = ['1A1']
    print "compute A(w) for symmetries = ",symmetries
    
    A = 6.5
    B = 0.15
    C = 0.58
    #A = 50
    #B = 0
    #C = 0
    E_1S = A+14*B+7*C
    E_1G = A+4*B+2*C
    E_1D = A-3*B+2*C
    E_3P = A+7*B
    E_3F = A-8*B
    print "E_1S = ", E_1S      
    print "E_1G = ", E_1G     
    print "E_1D = ", E_1D 
    print "E_3P = ", E_3P
    print "E_3F = ", E_3F
    