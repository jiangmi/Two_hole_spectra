import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import scipy.linalg
import math
import matplotlib.pyplot as plt
import timeit
import sys
sys.path.append('../../')
import rspace
from hamiltonian import Parameters

M_PI = math.pi 

# Arash's tight binding parameters:
prm = Parameters(2.10, 2.10, 0.67, -4.73, -3.06, 0, 0, 0, 0.04)
rspace.M=5

def calc_spec_weight_spsolve(wmin,wmax):
    wvals = []
    gf0 = []
    gf1 = []
    gf2 = []

    rhs = rspace.create_rhs_matrix()
    w = wmin
    dw = prm.eta/2.0
    static_eom = rspace.create_static_eom_matrix(prm)
    static_eom = static_eom.tocsc()
    static_eom.sort_indices()
    sps.linalg.dsolve.use_solver( useUmfpack = True,\
                                  assumeSortedIndices = True )
    while w < wmax:
        eom = rspace.create_eom_matrix(static_eom,w)
        x = sps.linalg.spsolve(eom.tocsc(),rhs).toarray() 
        gf0.append(x[rspace.get_index(0,0,0),0])
        gf1.append(x[rspace.get_index(0,0,1),1])
        gf2.append(x[rspace.get_index(0,0,2),2])
        wvals.append(w)
        w += dw
        print (w-wmin)/(wmax-wmin)

    # plt.plot(wvals,-np.imag(gf0)/M_PI,'-r',label='s')
    # plt.plot(wvals,-np.imag(gf1)/M_PI,'-b',label='px')
    # plt.plot(wvals,-np.imag(gf2)/M_PI,'-g',label='py')
    # plt.legend(loc='upper left')
    # plt.show()

def calc_spec_weight_bicg(wmin,wmax):
    wvals = []
    gf0 = []
    gf1 = []
    gf2 = []
    rhs0 = rspace.create_rhs_vec(0)
    rhs1 = rspace.create_rhs_vec(1)
    rhs2 = rspace.create_rhs_vec(2)
    w = wmin
    dw = prm.eta/2.0
    static_eom = rspace.create_static_eom_matrix(prm)
    sps.linalg.dsolve.use_solver( useUmfpack = True,\
                                  assumeSortedIndices = True )
    while w < wmax:
        eom = rspace.create_eom_matrix(static_eom,w)
        x0, info = sps.linalg.bicg(eom,rhs0) 
        x1, info = sps.linalg.bicg(eom,rhs1) 
        x2, info = sps.linalg.bicg(eom,rhs2) 
        gf0.append(x0[rspace.get_index(0,0,0)])
        gf1.append(x1[rspace.get_index(0,0,1)])
        gf2.append(x2[rspace.get_index(0,0,2)])
        wvals.append(w)
        w += dw
    #     print (w-wmin)/(wmax-wmin)
    # plt.plot(wvals,-np.imag(gf0)/M_PI,'-r',label='s')
    # plt.plot(wvals,-np.imag(gf1)/M_PI,'-b',label='px')
    # plt.plot(wvals,-np.imag(gf2)/M_PI,'-g',label='py')
    # plt.legend()
    # plt.show()

def calc_spec_weight_pardiso(wmin,wmax):
    wvals = []
    gf0 = []
    gf1 = []
    gf2 = []

    rhs0 = rspace.create_rhs_vec(0)
    rhs1 = rspace.create_rhs_vec(1)
    rhs2 = rspace.create_rhs_vec(2)
    rhs = np.asfortranarray(np.vstack((rhs0,rhs1,rhs2)).T)
    w = wmin
    dw = prm.eta/2.0
    static_eom = rspace.create_static_eom_matrix(prm)
    # static_eom.sort_indices()
    ia = static_eom.indptr+1  # fortran indexing starts at 1
    ja = static_eom.indices+1 # fortran indexing starts at 1
    a = static_eom.data

    pt = np.empty(64, dtype='i8', order='F')
    iparm = np.empty(64, dtype='i4', order='F')
    dparm = np.empty(64, dtype='f8', order='F')
    iparm[2] = 1 # number of processors

    error = pardiso.init(pt, iparm, dparm)
    if error != 0:
        print 'Error in pardiso.init(), ERROR={:d}'.format(error)
    x, error = pardiso.sym_fact(pt,a,ia,ja,iparm,rhs,dparm)
    if error != 0:
        print 'Error in pardiso.sum_fact(), ERROR={:d}'.format(error)
    
    while w < wmax:
        eom = rspace.create_eom_matrix(static_eom,w)
        a = eom.data
        x, error =  pardiso.solve(pt,a,ia,ja,iparm,rhs,dparm)
        if error != 0:
            print 'Error in pardiso.solve(), ERROR={:d}'.format(error)
        gf0.append(x[rspace.get_index(0,0,0),0])
        gf1.append(x[rspace.get_index(0,0,1),1])
        gf2.append(x[rspace.get_index(0,0,2),2])
        wvals.append(w)
        w += dw
    #     print (w-wmin)/(wmax-wmin)
    # plt.plot(wvals,-np.imag(gf0)/M_PI,'-r',label='s')
    # plt.plot(wvals,-np.imag(gf1)/M_PI,'-b',label='px')
    # plt.plot(wvals,-np.imag(gf2)/M_PI,'-g',label='py')
    # plt.legend(loc='upper left')
    # plt.show()

calc_spec_weight_spsolve(-12,2)
# calc_spec_weight_pardiso(-12,2)


t1 = timeit.Timer('calc_spec_weight_spsolve(-15,-12)',\
                  setup='from __main__ import calc_spec_weight_spsolve')
t2 = timeit.Timer('calc_spec_weight_bicg(-15,-12)',\
                  setup='from __main__ import calc_spec_weight_bicg')
t3 = timeit.Timer('calc_spec_weight_pardiso(-15,-12)',\
                  setup='from __main__ import calc_spec_weight_pardiso')

t4 = timeit.Timer('calc_spec_weight_spsolve(-10,-9)',\
                  setup='from __main__ import calc_spec_weight_spsolve')
t5 = timeit.Timer('calc_spec_weight_bicg(-10,-9)',\
                  setup='from __main__ import calc_spec_weight_bicg')
t6 = timeit.Timer('calc_spec_weight_pardiso(-10,-9)',\
                  setup='from __main__ import calc_spec_weight_pardiso')

