'''
Functions for constructing individual parts of the Hamiltonian. The 
matrices still need to be multiplied with the appropriate coupling 
constants t_pd, t_pp, etc..
'''
import parameters as pam
import lattice as lat
import variational_space as vs 
import numpy as np
import scipy.sparse as sps

directions_to_vecs = {'UR': (1,1),\
                      'UL': (-1,1),\
                      'DL': (-1,-1),\
                      'DR': (1,-1),\
                      'L': (-1,0),\
                      'R': (1,0),\
                      'U': (0,1),\
                      'D': (0,-1)}
tpp_nn_hop_dir = ['UR','UL','DL','DR']

def set_tpd_tpp(Norb,tpd,tpp,pds,pdp,pps,ppp):
    # dxz and dyz has no tpd hopping
    if pam.Norb==3:
        tpd_nn_hop_dir = {'dx2y2' : ['L','R','U','D'],\
                          'px'    : ['L','R'],\
                          'py'    : ['U','D']}
    elif pam.Norb==7:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'px'    : ['L','R'],\
                          'py'    : ['U','D']}
    elif pam.Norb==9:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D']}
    if pam.Norb==3:
        if_tpd_nn_hop = {'dx2y2' : 1,\
                         'px'    : 1,\
                         'py'    : 1}
    elif pam.Norb==7:
        if_tpd_nn_hop = {'d3z2r2': 1,\
                         'dx2y2' : 1,\
                         'dxy'   : 0,\
                         'dxz'   : 0,\
                         'dyz'   : 0,\
                         'px'    : 1,\
                         'py'    : 1}
    elif pam.Norb==9:
        if_tpd_nn_hop = {'d3z2r2': 1,\
                         'dx2y2' : 1,\
                         'dxy'   : 1,\
                         'dxz'   : 0,\
                         'dyz'   : 0,\
                         'px1'   : 1,\
                         'py1'   : 1,\
                         'px2'   : 1,\
                         'py2'   : 1}
    # hole language: sign convention followed from Fig 1 in H.Eskes's PRB 1990 paper
    #                or PRB 2016: Characterizing the three-orbital Hubbard model...
    if pam.Norb==3:
        tpd_nn_hop_fac = {('dx2y2','L','px'):   tpd,\
                          ('dx2y2','R','px'):  -tpd,\
                          ('dx2y2','U','py'):   tpd,\
                          ('dx2y2','D','py'):  -tpd,\
                          # below just inverse dir of the above one by one
                          ('px','R','dx2y2'):   tpd,\
                          ('px','L','dx2y2'):  -tpd,\
                          ('py','D','dx2y2'):   tpd,\
                          ('py','U','dx2y2'):  -tpd}
    elif pam.Norb==7:
        # d3z2r2 has +,-,+ sign structure so that it is negative in x-y plane
        tpd_nn_hop_fac = {('d3z2r2','L','px'): -tpd/np.sqrt(3),\
                          ('d3z2r2','R','px'):  tpd/np.sqrt(3),\
                          ('d3z2r2','U','py'):  tpd/np.sqrt(3),\
                          ('d3z2r2','D','py'): -tpd/np.sqrt(3),\
                          ('dx2y2','L','px'):   tpd,\
                          ('dx2y2','R','px'):  -tpd,\
                          ('dx2y2','U','py'):   tpd,\
                          ('dx2y2','D','py'):  -tpd,\
                          # below just inverse dir of the above one by one
                          ('px','R','d3z2r2'): -tpd/np.sqrt(3),\
                          ('px','L','d3z2r2'):  tpd/np.sqrt(3),\
                          ('py','D','d3z2r2'):  tpd/np.sqrt(3),\
                          ('py','U','d3z2r2'): -tpd/np.sqrt(3),\
                          ('px','R','dx2y2'):   tpd,\
                          ('px','L','dx2y2'):  -tpd,\
                          ('py','D','dx2y2'):   tpd,\
                          ('py','U','dx2y2'):  -tpd}
    elif pam.Norb==9:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'):  pds/2.0,\
                          ('d3z2r2','R','px1'): -pds/2.0,\
                          ('d3z2r2','U','py2'): -pds/2.0,\
                          ('d3z2r2','D','py2'):  pds/2.0,\
                          ('dx2y2','L','px1'):   pds*c,\
                          ('dx2y2','R','px1'):  -pds*c,\
                          ('dx2y2','U','py2'):   pds*c,\
                          ('dx2y2','D','py2'):  -pds*c,\
                          ('dxy','L','py1'):  -pdp,\
                          ('dxy','R','py1'):   pdp,\
                          ('dxy','U','px2'):   pdp,\
                          ('dxy','D','px2'):  -pdp,\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'):  pds/2.0,\
                          ('px1','L','d3z2r2'): -pds/2.0,\
                          ('py2','D','d3z2r2'): -pds/2.0,\
                          ('py2','U','d3z2r2'):  pds/2.0,\
                          ('px1','R','dx2y2'):   pds*c,\
                          ('px1','L','dx2y2'):  -pds*c,\
                          ('py2','D','dx2y2'):   pds*c,\
                          ('py2','U','dx2y2'):  -pds*c,\
                          ('py1','R','dxy'):  -pdp,\
                          ('py1','L','dxy'):   pdp,\
                          ('px2','D','dxy'):   pdp,\
                          ('px2','U','dxy'):  -pdp}
    ########################## tpp below ##############################
    if pam.Norb==3 or pam.Norb==7:
        tpp_nn_hop_fac = {('UR','px','py'): -tpp,\
                                 ('UL','px','py'):  tpp,\
                                 ('DL','px','py'): -tpp,\
                                 ('DR','px','py'):  tpp}
    elif pam.Norb==9:
        tpp_nn_hop_fac = {('UR','px1','px2'):  0.5*(ppp-pps),\
                          ('UL','px1','px2'):  0.5*(ppp-pps),\
                          ('DL','px1','px2'):  0.5*(ppp-pps),\
                          ('DR','px1','px2'):  0.5*(ppp-pps),\
                          ('UR','py1','py2'):  0.5*(ppp-pps),\
                          ('UL','py1','py2'):  0.5*(ppp-pps),\
                          ('DL','py1','py2'):  0.5*(ppp-pps),\
                          ('DR','py1','py2'):  0.5*(ppp-pps),\
                          ('UR','px1','py2'): -0.5*(ppp+pps),\
                          ('UL','px1','py2'):  0.5*(ppp+pps),\
                          ('DL','px1','py2'): -0.5*(ppp+pps),\
                          ('DR','px1','py2'):  0.5*(ppp+pps),\
                          ('UR','px2','py1'): -0.5*(ppp+pps),\
                          ('UL','px2','py1'):  0.5*(ppp+pps),\
                          ('DL','px2','py1'): -0.5*(ppp+pps),\
                          ('DR','px2','py1'):  0.5*(ppp+pps)}
        
    return tpd_nn_hop_dir, if_tpd_nn_hop, tpd_nn_hop_fac, tpp_nn_hop_fac

def get_interaction_mat(sym):
    '''
    Get d-d Coulomb and exchange interaction matrix
    total_spin based on lat.spin_int: up:1 and dn:0
    
    Rotating by 90 degrees, x goes to y and indeed y goes to -x so that this basically interchanges 
    spatial wave functions of two holes and can introduce - sign (for example (dxz, dyz)).
    But one has to look at what such a rotation does to the Slater determinant state of two holes.
    Remember the triplet state is (|up,down> +|down,up>)/sqrt2 so in interchanging the holes 
    the spin part remains positive so the spatial part must be negative. 
    For the singlet state interchanging the electrons the spin part changes sign so the spatial part can stay unchanged.
    
    Triplets cannot occur for two holes in the same spatial wave function while only singlets only can
    But both singlets and triplets can occur if the two holes have orthogonal spatial wave functions 
    and they will differ in energy by the exchange terms
    
    ee denotes xz,xz or xz,yz depends on which integral <ab|1/r_12|cd> is nonzero, see handwritten notes
    
    AorB_sym = +-1 is used to label if the state is (e1e1+e2e2)/sqrt(2) or (e1e1-e2e2)/sqrt(2)
    For syms (in fact, all syms except 1A1 and 1B1) without the above two states, AorB_sym is set to be 0
    '''
    A = pam.A
    B = pam.B
    C = pam.C
    
    if sym=='1AB1':
        fac = np.sqrt(6)
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 4,\
                       ('d3z2r2','dx2y2') : 5}
        interaction_mat = [[A+4.*B+3.*C,  4.*B+C,       4.*B+C,           B+C,           B+C,       0], \
                           [4.*B+C,       A+4.*B+3.*C,  C,             3.*B+C,        3.*B+C,       0], \
                           [4.*B+C,       C,            A+4.*B+3.*C,   3.*B+C,        3.*B+C,       0], \
                           [B+C,          3.*B+C,       3.*B+C,        A+4.*B+3.*C,   3.*B+C,       B*fac], \
                           [B+C,          3.*B+C,       3.*B+C,        3.*B+C,        A+4.*B+3.*C, -B*fac], \
                           [0,            0,            0,              B*fac,         -B*fac,      A+2.*C]]
    if sym=='1A1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 1
        fac = np.sqrt(2)
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 3}
        interaction_mat = [[A+4.*B+3.*C,  4.*B+C,       4.*B+C,        fac*(B+C)], \
                           [4.*B+C,       A+4.*B+3.*C,  C,             fac*(3.*B+C)], \
                           [4.*B+C,       C,            A+4.*B+3.*C,   fac*(3.*B+C)], \
                           [fac*(B+C),    fac*(3.*B+C), fac*(3.*B+C),  A+7.*B+4.*C]]
    if sym=='1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dx2y2'): 0,\
                       ('dxz','dxz')     : 1,\
                       ('dyz','dyz')     : 1}
        interaction_mat = [[A+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A+B+2.*C]]
    if sym=='1A2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0}
        interaction_mat = [[A+4.*B+2.*C]]
    if sym=='3A2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0,\
                       ('dxz','dyz')  : 1}
        interaction_mat = [[A+4.*B,   6.*B], \
                           [6.*B,     A-5.*B]]
    if sym=='3B1':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('d3z2r2','dx2y2'): 0}
        interaction_mat = [[A-8.*B]]
    if sym=='1B2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxy'): 0,\
                       ('dxz','dyz')   : 1}
        interaction_mat = [[A+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A+B+2.*C]]
    if sym=='3B2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = -1
        state_order = {('d3z2r2','dxy'): 0}
        interaction_mat = [[A-8.*B]]
    if sym=='1E':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}    
        interaction_mat = [[A+3.*B+2.*C,  0,           -B*fac,      0,          0,        -B*fac], \
                           [0,            A+3.*B+2.*C,  0,          B*fac,     -B*fac,     0], \
                           [-B*fac,       0,            A+B+2.*C,   0,          0,        -3.*B], \
                           [0,            B*fac,        0,          A+B+2.*C,   3.*B,      0 ], \
                           [0,           -B*fac,        0,          3.*B,       A+B+2.*C,  0], \
                           [-B*fac,       0,           -3.*B,       0,          0,         A+B+2.*C]]
    if sym=='3E':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}        
        interaction_mat = [[A+B,         0,         -3.*B*fac,    0,          0,        -3.*B*fac], \
                           [0,           A+B,        0,           3.*B*fac,  -3.*B*fac,  0], \
                           [-3.*B*fac,   0,          A-5.*B,      0,          0,         3.*B], \
                           [0,           3.*B*fac,   0,           A-5.*B,    -3.*B,      0 ], \
                           [0,          -3.*B*fac,   0,          -3.*B,       A-5.*B,    0], \
                           [-3.*B*fac,   0,          3.*B,        0,          0,         A-5.*B]]
        
    return state_order, interaction_mat, Stot, Sz_set, AorB_sym

def get_Aw_dd_state(sym):
    '''
    Currently useless. States are obtained in create_interaction_matrix
    Starting vector. If only ground state, Use something randomly initialized for lanczos
    If computing spectral function, use desired state
        
    Get the state list for calculating A(w) in spectral_weight_lanczos.py
    There are two types of list: one for G_dd and one for G_d8. See appendix of Eskes's PRB 1990
    dd_states are subset of d8_states
    '''                    
    if sym=='1A1':
        dd_states = [('up','dx2y2','dn','dx2y2')]
        d8_states = [('up','d3z2r2','dn','d3z2r2'), ('up','dx2y2','dn','dx2y2'),\
                     ('up','dxy','dn','dxy'), ('up','dxz','dn','dxz'),('up','dyz','dn','dyz')]
        coef_frac_parentage = {('up','dx2y2','dn','dx2y2'): 1.0}
    if sym=='1A2':
        # singlet states should have opposite spin
        dd_states = [('up','dx2y2','dn','dxy'),('up','dxy','dn','dx2y2')]
        d8_states = [('up','dx2y2','dn','dxy'),('dn','dx2y2','up','dxy')]
        coef_frac_parentage = {('up','dx2y2','dn','dxy'): 0.5, \
                               ('up','dxy','dn','dx2y2'): 0.5}
    if sym=='3A2':
        # triplet states can have opposite or same spin
        dd_states = [('up','dx2y2','up','dxy'),('up','dx2y2','dn','dxy')]
        d8_states = [('up','dx2y2','up','dxy'),('up','dx2y2','dn','dxy'),\
                     ('dn','dx2y2','up','dxy'),('dn','dx2y2','dn','dxy'),\
                     ('up','dxz','dn','dxz'),('up','dyz','dn','dyz')]
        coef_frac_parentage = {('up','dx2y2','up','dxy'): 1.0,\
                               ('up','dx2y2','dn','dxy'): 0.5}
    if sym=='1B1':
        dd_states = [('up','dx2y2','dn','d3z2r2')]
        d8_states = [('up','d3z2r2','dn','dx2y2'),('up','dx2y2','dn','d3z2r2'),\
                     ('up','dxz','dn','dyz'),('up','dyz','dn','dxz')]
        coef_frac_parentage = {('up','dx2y2','dn','d3z2r2'): 0.5}
    if sym=='3B1':
        dd_states = [('up','d3z2r2','up','dx2y2'),('up','dx2y2','dn','d3z2r2')]
        d8_states = [('up','d3z2r2','up','dx2y2'),('up','d3z2r2','dn','dx2y2'),\
                     ('up','dx2y2','dn','d3z2r2'),('dn','d3z2r2','dn','dx2y2')]
        coef_frac_parentage = {('up','d3z2r2','up','dx2y2'): 1.0,\
                               ('up','dx2y2','dn','d3z2r2'): 0.5}
    if sym=='1B2':
        dd_states = []
        d8_states = [('up','d3z2r2','dn','dxy'),('dn','d3z2r2','up','dxy'),\
                     ('up','dxz','dn','dyz'),('dn','dxz','up','dyz')]
        coef_frac_parentage = {}
    if sym=='3B2':
        dd_states = []
        d8_states = [('up','d3z2r2','up','dxy'),('up','d3z2r2','dn','dxy'),\
                     ('dn','d3z2r2','up','dxy'),('dn','d3z2r2','dn','dxy')]
        coef_frac_parentage = {}
    if sym=='1E':
        dd_states = [('up','dx2y2','dn','dxz'),('up','dx2y2','dn','dyz')]
        d8_states = [('up','d3z2r2','dn','dxz'),('dn','d3z2r2','up','dxz'),\
                     ('up','d3z2r2','dn','dyz'),('dn','d3z2r2','up','dyz'),\
                     ('up','dx2y2','dn','dxz'),('dn','dx2y2','up','dxz'),\
                     ('up','dx2y2','dn','dyz'),('dn','dx2y2','up','dyz'),\
                     ('up','dxy','dn','dxz'),('dn','dxy','up','dxz'),\
                     ('up','dxy','dn','dyz'),('dn','dxy','up','dyz')]
        coef_frac_parentage = {('up','dx2y2','dn','dxz'): 1.0,\
                               ('up','dx2y2','dn','dyz'): 1.0}
    if sym=='3E':
        dd_states = [('up','dx2y2','up','dxz'),('up','dx2y2','dn','dxz'),\
                     ('up','dx2y2','up','dyz'),('up','dx2y2','dn','dyz')]
        d8_states = [('up','d3z2r2','up','dxz'),('up','d3z2r2','dn','dxz'),\
                     ('dn','d3z2r2','up','dxz'),('dn','d3z2r2','dn','dxz'),\
                     ('up','d3z2r2','up','dyz'),('up','d3z2r2','dn','dyz'),\
                     ('dn','d3z2r2','up','dyz'),('dn','d3z2r2','dn','dyz'),\
                     ('up','dx2y2','up','dxz'),('up','dx2y2','dn','dxz'),\
                     ('dn','dx2y2','up','dxz'),('dn','dx2y2','dn','dxz'),\
                     ('up','dx2y2','up','dyz'),('up','dx2y2','dn','dyz'),\
                     ('dn','dx2y2','up','dyz'),('dn','dx2y2','dn','dyz'),\
                     ('up','dxy','up','dxz'),('up','dxy','dn','dxz'),\
                     ('dn','dxy','up','dxz'),('dn','dxy','dn','dxz'),\
                     ('up','dxy','up','dyz'),('up','dxy','dn','dyz'),\
                     ('dn','dxy','up','dyz'),('dn','dxy','dn','dyz')]
        coef_frac_parentage = {('up','dx2y2','up','dxz'): 2.0,\
                               ('up','dx2y2','dn','dxz'): 1.0,\
                               ('up','dx2y2','up','dyz'): 2.0,\
                               ('up','dx2y2','dn','dyz'): 1.0}
        
    return dd_states, d8_states, coef_frac_parentage

def get_Aw_pp_state():
    '''
    Starting vector. If only ground state, Use something randomly initialized for lanczos
    If computing spectral function, use desired state
    
    Get the state list for calculating A(w) for two holes locating on p-orbitals
    '''                    
    if pam.Norb==3 or pam.Norb==7:
        pp_states = [('up','px','dn','px'),('up','px','dn','py'),('up','px','up','py'),\
                     ('up','dx2y2','up','px'),('up','dx2y2','dn','px')]
    elif pam.Norb==9:
        pp_states = [('up','px1','dn','px1'),('up','px1','dn','py1'),('up','px1','dn','px2'),('up','px1','dn','py2'),\
                                             ('up','px1','up','py1'),('up','px1','up','px2'),('up','px1','up','py2')]
        
    return pp_states

def create_phase_dict(kx,ky,VS):
    '''
    Create a dictionary containing the phase values 
    exp(-kx*Rx*1j/2.0-ky*Ry*1j/2.0) for all relevant (Rx,Ry). Used
    to avoid recalculating these values.
    
    directions_to_vecs assumes the unit cell length is 2, so need /2.0

    Parameters
    ----------
    kx, ky: momentum in x, y direction.
    VS: VariationalSpace class from the module variationalSpace. Only
        VS.Mc is used to determine which (Rx,Ry) values are needed.

    Returns
    -------
    phase: dictionary, keys are tuples (Rx,Ry) and items are the
        corresponding phase values exp(-kx*Rx*1j/2.0-ky*Ry*1j/2.0).
        Additional keys are 'momentum' and 'Mc'.
    '''
    Mc = VS.Mc
    phase = {}
    phase['Mc'] = VS.Mc
    phase['momentum'] = (kx,ky)
    for Rx in range(-Mc-4,Mc+5):
        for Ry in range(-Mc-4,Mc+5):
            phase[(Rx,Ry)] = np.exp(-(kx*Rx+ky*Ry)*1j/2.0)
    return phase

def set_matrix_element(row,col,data,new_state,col_index,VS,element):
    '''
    Helper function that is used to set elements of a matrix using the
    sps coo format.

    Parameters
    ----------
    row: python list containing row indices
    col: python list containing column indices
    data: python list containing non-zero matrix elements
    col_index: column index that is to be appended to col
    new_state: new state corresponding to the row index that is to be
        appended.
    VS: VariationalSpace class from the module variationalSpace
    element: (complex) matrix element that is to be appended to data.

    Returns
    -------
    None, but apppends values to row, col, data.
    '''
    row_index = VS.get_index(new_state)
    if row_index != None:
        data.append(element)
        row.append(row_index)
        col.append(col_index)

def create_tpd_nn_matrix(phase,VS,tpd_nn_hop_dir, if_tpd_nn_hop, tpd_nn_hop_fac):
    '''
    Create nearest neighbor (NN) pd hopping part of the Hamiltonian

    Parameters
    ----------
    phase: dictionary containing the phase values exp(-kx*Rx*1j/2.0-ky*Ry*1j/2.0).
        Created with create_phase_dict.
    VS: VariationalSpace class from the module variationalSpace
    
    Returns
    -------
    matrix: (sps coo format) t_pd hopping part of the Hamiltonian without 
        the prefactor t_pd.
    
    Note from the sps documentation
    -------------------------------
    By default when converting to CSR or CSC format, duplicate (i,j)
    entries will be summed together
    
    IMPORTANT
    -------------------------------
    See note_on_set_hopping_term.jpg for how to set up the hopping elements
    The convention is to assign the phase to hole1 hopping 
    When hole1 hops, there is an additional phase 
    It seems hole2 can hop to another orb, but actually not, 
    j'+n-1=j-n only change hole2's coordinate, not the orb
    For multiorbital cases, should make sure that hole2 does not change orb
    '''    
    print "start create_tpd_nn_matrix"
    print "=========================="
    
    dim = VS.dim
    tpd_orbs = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in xrange(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # double check which cost some time, might not necessary
        assert VS.get_uid(start_state) == VS.lookup_tbl[i]
        
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        x1, y1 = start_state['hole1_coord']
        x2, y2 = start_state['hole2_coord']
        
        #print s1,s2,orb1,orb2,x1, y1,x2, y2

        # some d-orbitals might have no tpd
        if if_tpd_nn_hop[orb1] == 1:
            # hole 1 hops (coordninates need to be shifted -> phase change)
            for dir_ in tpd_nn_hop_dir[orb1]:
                vx, vy = directions_to_vecs[dir_]

                # recall that x1, y1 are kept in (1,0), (0,1), (0,0)
                x1_new, y1_new = x1 + vx, y1 + vy

                # it is possible that up hole hops to neighboring unit cell
                # so need to get the position of reference site(Cu) after hopping 
                orb, Rx_new, Ry_new = lat.get_unit_cell_rep(x1_new,y1_new)

                if orb == ['NotOnSublattice']:
                    continue

                # get the coordinates relative to the new unit cell origin
                x1_shift = x1_new - Rx_new
                y1_shift = y1_new - Ry_new

                # dn hole needs change coor corresponding to the new unit cell of up hole
                # but not change orb. see above and note_on_set_hopping_term.jpg
                x2_shift = x2 - Rx_new
                y2_shift = y2 - Ry_new

                orbs1_new, _, _ = lat.get_unit_cell_rep(x1_shift,y1_shift)
                orbs2_new, _, _ = lat.get_unit_cell_rep(x2_shift,y2_shift)

                if orbs1_new == ['NotOnSublattice'] or orbs2_new == ['NotOnSublattice']:
                    continue

                # consider t_pd for all cases
                # recall that when up hole hops, dn hole should not change orb
                for o1 in orbs1_new:
                    if if_tpd_nn_hop[o1] == 0:
                        continue
                    # consider Pauli principle
                    if s1==s2 and o1==orb2 and x1_shift==x2_shift and y1_shift==y2_shift:
                        continue
                        
                    #if pam.if_project_out_two_holes_on_different_Cu == 1:
                    #    if (o1 in pam.Cu_orbs and orb2 in pam.Cu_orbs and (x1_shift,y1_shift)!=(x2_shift,y2_shift)):
                    #        continue
                                    
                    if vs.check_in_vs_condition(x1_shift,y1_shift,x2_shift,y2_shift):
                        tmp_state = vs.create_state(s1,o1,  x1_shift,y1_shift,\
                                                    s2,orb2,x2_shift,y2_shift)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = tuple([orb1, dir_, o1])
                        if o12 in tpd_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS,\
                                               tpd_nn_hop_fac[o12]*phase[(Rx_new,Ry_new)]*ph)

        # hole 2 hops; some d-orbitals might have no tpd
        if if_tpd_nn_hop[orb2] == 1:
            for dir_ in tpd_nn_hop_dir[orb2]:
                vx, vy = directions_to_vecs[dir_]
                x2_new, y2_new = x2 + vx, y2 + vy
                orbs2_new, _, _ = lat.get_unit_cell_rep(x2_new, y2_new)

                if orbs2_new == ['NotOnSublattice']:
                    continue
                for o2 in orbs2_new:
                    if if_tpd_nn_hop[o2] == 0:
                        continue
                    # consider Pauli principle
                    if s1==s2 and orb1==o2 and x1==x2_new and y1==y2_new:
                        continue
                        
                    #if pam.if_project_out_two_holes_on_different_Cu == 1:
                    #    if (orb1 in pam.Cu_orbs and o2 in pam.Cu_orbs and (x1,y1)!=(x2_new,y2_new)):
                    #        continue
                            
                    if vs.check_in_vs_condition(x1,y1,x2_new,y2_new):
                        tmp_state = vs.create_state(s1,orb1,x1,y1,s2,o2,x2_new,y2_new)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = tuple([orb2, dir_, o2])
                        if o12 in tpd_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS, tpd_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out

def create_tpp_nn_matrix(phase,VS,tpp_nn_hop_fac): 
    '''
    similar to comments in create_tpd_nn_matrix
    '''   
    print "start create_tpp_nn_matrix"
    print "=========================="
    
    dim = VS.dim
    data = []
    row = []
    col = []
    for i in xrange(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # below check has been done in create_tpd_nn_matrix so here not necessary
        #assert VS.get_uid(start_state) == VS.lookup_tbl[i]
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        x1, y1 = start_state['hole1_coord']
        x2, y2 = start_state['hole2_coord']
        
        # only p-orbitals has t_pp 
        if orb1 in pam.O_orbs: 
            # hole 1 hops (coordninates need to be shifted -> phase change)
            for dir_ in tpp_nn_hop_dir:
                vx, vy = directions_to_vecs[dir_]
                x1_new, y1_new = x1 + vx, y1 + vy

                # get the position of reference site(Cu) in the same unit cell 
                orbs1_new, Rx_new, Ry_new = lat.get_unit_cell_rep(x1_new,y1_new)
                if orbs1_new == ['NotOnSublattice'] or orbs1_new == pam.Cu_orbs:
                    continue

                x1_shift = x1_new - Rx_new
                y1_shift = y1_new - Ry_new
                x2_shift = x2 - Rx_new
                y2_shift = y2 - Ry_new

                # consider t_pp for all cases
                for o1 in orbs1_new:
                    if s1==s2 and o1==orb2 and x1_shift==x2_shift and y1_shift==y2_shift:
                        continue
                        
                    #if pam.if_project_out_two_holes_on_different_Cu == 1:
                    #    if (o1 in pam.Cu_orbs and orb2 in pam.Cu_orbs and (x1_shift,y1_shift)!=(x2_shift,y2_shift)):
                    #        continue
                            
                    if vs.check_in_vs_condition(x1_shift,y1_shift,x2_shift,y2_shift):
                        tmp_state = vs.create_state(s1,o1,  x1_shift,y1_shift,\
                                                    s2,orb2,x2_shift,y2_shift)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = sorted([orb1, dir_, o1])
                        o12 = tuple(o12)
                        set_matrix_element(row,col,data,new_state,i,VS,\
                                           tpp_nn_hop_fac[o12]*phase[(Rx_new,Ry_new)]*ph)
                    
        # hole 2 hops, only p-orbitals has t_pp 
        if orb2 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy = directions_to_vecs[dir_]
                x2_new, y2_new = x2 + vx, y2 + vy
                
                orbs2_new, _, _ = lat.get_unit_cell_rep(x2_new, y2_new)
                if orbs2_new == ['NotOnSublattice'] or orbs2_new == pam.Cu_orbs:
                    continue
                    
                for o2 in orbs2_new:
                    # consider Pauli principle
                    if s1==s2 and orb1==o2 and x1==x2_new and y1==y2_new:
                        continue
                        
                    #if pam.if_project_out_two_holes_on_different_Cu == 1:
                    #    if (orb1 in pam.Cu_orbs and o2 in pam.Cu_orbs and (x1,y1)!=(x2_new,y2_new)):
                    #        continue
                            
                    if vs.check_in_vs_condition(x1,y1,x2_new,y2_new):
                        tmp_state = vs.create_state(s1,orb1,x1,y1,s2,o2,x2_new,y2_new)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = sorted([orb2, dir_, o2])
                        o12 = tuple(o12)
                        set_matrix_element(row,col,data,new_state,i,VS, tpp_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out

def create_edep_diag_matrix(VS,ep):
    '''
    Create diagonal part of the site energies. Assume ed = 0!
    '''    
    print "start create_edep_diag_matrix"
    print "============================="
    dim = VS.dim
    data = []
    row = []
    col = []

    for i in xrange(0,dim):
        diag_el = 0.
        state = VS.get_state(VS.lookup_tbl[i])

        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']

        if orb1 in pam.Cu_orbs: 
            diag_el += pam.ed
        elif orb1 in pam.O_orbs:
            diag_el += ep
        if orb2 in pam.Cu_orbs: 
            diag_el += pam.ed
        elif orb2 in pam.O_orbs:
            diag_el += ep
            
        # apply a large energy cutoff to configurations with
        # two holes on different Cu to effectively forbid those configuration
        x1, y1 = state['hole1_coord']
        x2, y2 = state['hole2_coord']
        if pam.if_project_out_two_holes_on_different_Cu == 1:
            if orb1 in pam.Cu_orbs and orb2 in pam.Cu_orbs and (x1,y1)!=(x2,y2): 
                diag_el = 100.

        data.append(diag_el); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    return out

def get_double_occu_list(VS):
    '''
    Get the list of states that two holes are both d or p-orbitals
    used in the following create_interaction_matrix routine
    
    for d_doulbe list, there are 5 orbitals, the number of double-occupied:
    up-up:  C^2_NCu = NCu*(NCu-1)/2
    dn-dn:  C^2_NCu = NCu*(NCu-1)/2
    up-dn:  NCu^2
    dn-up:  same as up-dn in case of double_occu, so only keep up-dn
    so the total number is NCu*(2*NCu-1)
    
    for p_doulbe list (9-orbital case), there are 2 orbitals, the number of double-occupied:
    up-up:  1
    dn-dn:  1
    up-dn:  4
    dn-up:  same as up-dn in case of double_occu, so only keep up-dn
    so the total number is 6*2 (*2 because two O sites in unit celll)
    '''
    dim = VS.dim
    d_list = []
    p_list = []
    
    NCu = len(pam.Cu_orbs)
    Nd = NCu*(NCu-1) + NCu**2
    if pam.Norb==3 or pam.Norb==7:
        Np = 2
    elif pam.Norb==9:
        Np = 12
    
    for i in xrange(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        x1, y1 = state['hole1_coord']
        x2, y2 = state['hole2_coord']

        # seems no need to enforce s1!=s2
        # depends on requirement in following create_interaction_matrix 
        if (x1,y1) == (x2,y2):
            if orb1 in pam.Cu_orbs and orb2 in pam.Cu_orbs:
                d_list.append(i)
                #print "d_double: idx", i, s1,orb1,x1,y1,s2,orb2,x2,y2
            elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
                p_list.append(i)
                #print "p_double: ", s1,orb1,x1,y1,s2,orb2,x2,y2

    #print "len(p_list)", len(p_list)
    #assert len(d_list)==Nd
    #assert len(p_list)==Np
    
    return d_list, p_list

def create_interaction_matrix(VS,sym,d_double,p_double,S_val, Sz_val, AorB_sym, Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets
    
    Cannot directly use the table in thesis or PRB paper since 
    those matrix elements are between singlets/triplets,
    need transform the basis in create_singlet_triplet_basis_change_matrix
    '''    
    #print "start create_interaction_matrix"
    
    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []

    # Create Coulomb-exchange matrix for d-orbital multiplets
    state_order, interaction_mat, Stot, Sz_set, AorB = get_interaction_mat(sym)
    sym_orbs = state_order.keys()
    print "orbitals in sym ", sym, "= ", sym_orbs

    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o12 = sorted([o1,o2])
        o12 = tuple(o12)

        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]

        # continue only if (o1,o2) is within desired sym
        if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
            continue
            
        if (o1==o2=='dxz' or o1==o2=='dyz') and AorB_sym[i]!=AorB:
            continue

        # get the corresponding index in sym for setting up matrix element
        idx1 = state_order[o12]
        for j in d_double:
            state = VS.get_state(VS.lookup_tbl[j])
            o3 = state['hole1_orb']
            o4 = state['hole2_orb']
            o34 = sorted([o3,o4])
            o34 = tuple(o34)
            S34  = S_val[j]
            Sz34 = Sz_val[j]
            
            if (o3==o4=='dxz' or o3==o4=='dyz') and AorB_sym[j]!=AorB:
                continue

            # only same total spin S and Sz state have nonzero matrix element
            if o34 in sym_orbs and S34==S12 and Sz34==Sz12:
                idx2 = state_order[o34]

                #print S12,Sz12,o1,o2," ",S34,Sz34,o3,o4," ", interaction_mat[idx1][idx2]
                #print idx1, idx2

                val = interaction_mat[idx1][idx2]
                data.append(val); row.append(i); col.append(j)

        # get index for desired dd states
        # Note: for transformed basis of singlet/triplet
        # index can differ from that in original basis
        if 'dx2y2' in o12:
            dd_state_indices.append(i)
            print "dd_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2
                 
    # Create Upp matrix for p-orbital multiplets
    for i in p_double:
        data.append(Upp); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out, dd_state_indices

def create_interaction_matrix_Norb3(VS,d_double,p_double,Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets
    
    Cannot directly use the table in thesis or PRB paper since 
    those matrix elements are between singlets/triplets,
    need transform the basis in create_singlet_triplet_basis_change_matrix
    '''    
    #print "start create_interaction_matrix"
    
    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []
    pp_state_indices = []
    dp_state_indices = []

    for i in d_double:
        data.append(pam.Udd); row.append(i); col.append(i)
        
        # get index for desired dd states
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']

        dd_state_indices.append(i)
        print "dd_state_indices", i, ", state: ", s1,o1,s2,o2
   
    for i in p_double:
        data.append(Upp); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out, dd_state_indices

def get_pp_state_indices(VS):
    '''
    Get the list of index for desired pp and dp states for computing A(w)
    '''   
    dim = VS.dim
    pp_state_indices = []
    
    for i in xrange(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        x1, y1 = state['hole1_coord']
        x2, y2 = state['hole2_coord']

        if o1 in pam.O_orbs and o2 in pam.O_orbs and (x1,y1) in [(1,0),(0,1)] and (x2,y2) in [(1,0),(0,1)]:
            pp_state_indices.append(i)
            print "pp_state_indices", i, ", state: ", s1,o1,x1,y1,s2,o2,x2,y2
    
    return pp_state_indices

def get_dp_state_indices(VS):
    '''
    Get the list of index for desired pp and dp states for computing A(w)
    '''   
    dim = VS.dim
    dp_state_indices = []
    
    for i in xrange(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        x1, y1 = state['hole1_coord']
        x2, y2 = state['hole2_coord']
            
        if o1 in pam.Cu_orbs and o2 in pam.O_orbs and (x1,y1)==(0,0) and (x2,y2) in [(1,0),(0,1)]:
            dp_state_indices.append(i)
            print "dp_state_indices", i, ", state: ", s1,o1,x1,y1,s2,o2,x2,y2
    
    return dp_state_indices

def get_Cu_dx2y2_O_indices(VS):
    '''
    Get the list of index for states with one hole on Cu and the other on neighboring O 
    with dx2-y2 symmetry (1/sqrt(4)) * (px1-py2-px3+py4)
    '''   
    dim = VS.dim
    Cu_dx2y2_O_indices = []
    
    state = vs.create_state('up','dx2y2',0,0,'dn','px',1,0)
    canonical_state,_ = vs.make_state_canonical(state)
    idx = VS.get_index(canonical_state)
    Cu_dx2y2_O_indices.append(idx)
    print "Cu_dx2y2_O_indices", idx
    
    state = vs.create_state('up','dx2y2',0,0,'dn','py',0,1)
    canonical_state,_ = vs.make_state_canonical(state)
    idx = VS.get_index(canonical_state)
    Cu_dx2y2_O_indices.append(idx)
    print "Cu_dx2y2_O_indices", idx
    
    '''
    state = vs.create_state('up','dx2y2',0,0,'dn','px',-1,0)
    canonical_state,_ = vs.make_state_canonical(state)
    Cu_dx2y2_O_indices.append(VS.get_index(canonical_state))
    
    state = vs.create_state('up','dx2y2',0,0,'dn','py',0,-1)
    canonical_state,_ = vs.make_state_canonical(state)
    Cu_dx2y2_O_indices.append(VS.get_index(canonical_state))
    '''
    return Cu_dx2y2_O_indices

def check_dense_matrix_hermitian(matrix):
    '''
    Check if dense matrix is Hermitian. Returns True or False.
    '''
    dim = matrix.shape[0]
    out = True
    for row in range(0,dim):
        for col in range(0,dim):
            #if row==38 and col==85:
            #    print row, col, matrix[row,col], matrix[col,row]
            
            # sparse matrix has many zeros
            if abs(matrix[row,col])<1.e-10:
                continue
                
            if abs(matrix[row,col]-np.conjugate(matrix[col,row]))>1.e-10:
                print row, col, matrix[row,col], matrix[col,row]
                out = False
                break
    return out

def check_spin_group(row,col,data,VS):
    '''
    check if hoppings or interaction matrix occur within groups of (up,up), (dn,dn), and (up,dn) 
    since (up,up) state cannot hop to a (up,dn) or (dn,dn) state
    '''
    out = True
    dim = len(data)
    
    for i in range(0,dim):
        irow = row[i]
        icol = col[i]
        
        rstate = VS.get_state(VS.lookup_tbl[irow])
        rs1 = rstate['hole1_spin']
        rs2 = rstate['hole2_spin']
        cstate = VS.get_state(VS.lookup_tbl[icol])
        cs1 = cstate['hole1_spin']
        cs2 = cstate['hole2_spin']
        
        rs = sorted([rs1,rs2])
        cs = sorted([cs1,cs2])
                
        if rs!=cs:
            print 'Error:'+str(rs)+' hops to '+str(cs)
            out = False
            break
    return out

def compare_matrices(m1,m2):
    '''
    Check if two matrices are the same. Returns True or False
    '''
    dim = m1.shape[0]
    if m2.shape[0] != dim:
        return False
    else:
        out = True
        for row in range(0,dim):
            for col in range(0,dim):
                if m1[row,col] != m2[row,col]:
                    out = False
                    break
        return out
        
if __name__ == '__main__':
    pass
