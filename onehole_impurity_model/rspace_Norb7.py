import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import math

M_PI = math.pi
M = None

def get_index(i,j,orb):
    '''
    set up the index of p-orbitals; 
    impurity Cu's are labeled as first 5 indices separately
    Every state |i,j,orb> in the real space basis is associated
    with a unique index which is used for storing the eom of in 
    matrix format.

    Parameters
    ----------
    i, j : integers, location of unit cell R_{i,j}
    orb:   0  px; 1 py

    Returns
    -------
    index: a unique integer that identifies the state
           +M to ensure the index is positive
           see Mirko's email on 6/29/18 for the index values
    '''
    #print 'i=',i,'j=',j,'orb=',orb,'index=',orb+2*(j+M)+2*(2*M+1)*(i+M)
    # final 2 accounts for two d-orbitals because only two d-orb have tpd
    
    # debug:
    '''
    M = 16
    idx = []
    for i in range(-M,M+1):
        for j in range(-M,M+1):   
            for orb in range(0,2):
                idx.append(orb+2*(j+M)+2*(2*M+1)*(i+M)+5)
    idx.sort()
    print 'all index',min(idx),max(idx),len(idx)
    '''
    # offset = 5 for Norb=7 and =3 for Norb=3
    offset = 5
    return orb+2*(j+M)+2*(2*M+1)*(i+M)+offset 

def create_static_eom_matrix(prm):
    '''The static eom matrix for the real-space GFs does not depend on
    omega. The diagonal entries are -i*eta +Esite. From this the
    eom matrix can be created by adding (-omega) to the diagonal. This
    is convenient because the static_eom_matrix only needs to be
    created once.

    Parameters
    ----------
    prm: parameter class of the Hamiltonian

    Returns
    -------
    static_eom_matrix: scipy sparse matrix with sorted indices
                       in csr format.
    
    Note
    ----
    Different from Mirko's convention to label the unit cell
    from small to large number as going left or down direction
    Here using more conventional going up or right increase index
    
    It seems to be important for PARDISO that only the non-zero 
    matrix elements are stored. This is true for the off-diagonal 
    elements, but I am not sure if it is also true for the diagonal
    elements. Therefore I am explicitly checking each off-diagonal 
    element before storing it. Note that when setting, e.g. tspx to
    zero for consistency checks a lot of matrix elements will be zero.

    The matrix type of the eom is complex and structurally symmetric.
    This is important when solving the eom with pardiso. Furthermore
    for structurally symmetric matrices pardiso expects the diagonal
    elements to be present.
    '''

    # unpack parameters
    Norb = prm.Norb
    tpd = prm.tpd
    tpp = prm.tpp
    eps_d = prm.eps_d
    eps_p = prm.eps_p
    eta = prm.eta
    
    row = []
    col = []
    data = []
    
    # first set up the impurity Cu's hoppings separately
    # only the tpd part of H
    #===================================================
    # orb = d3z2r2
    print 'set tpd for d3z2r2'
    row_index = 0
    # diagonal
    row.append(row_index)
    col.append(row_index)
    data.append(-eta*(1j)+eps_d)
    # hop left
    val = -tpd/np.sqrt(3)
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(1,0,0))
        data.append(val)
        row.append(get_index(1,0,0))
        col.append(row_index)
        data.append(val)
    # hop right
    val = tpd/np.sqrt(3)
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(0,0,0))
        data.append(val)
        row.append(get_index(0,0,0))
        col.append(row_index)
        data.append(val)
    # hop up
    val = tpd/np.sqrt(3)
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(0,0,1))
        data.append(val)
        row.append(get_index(0,0,1))
        col.append(row_index)
        data.append(val)
    # hop down
    val = -tpd/np.sqrt(3)
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(0,1,1))
        data.append(val)
        row.append(get_index(0,1,1))
        col.append(row_index)
        data.append(val)
    #-------------------------------------------------       
    # orb = dx2y2
    print 'set tpd for dx2y2'
    row_index = 1
        
    # diagonal
    row.append(row_index)
    col.append(row_index)
    data.append(-eta*(1j)+eps_d)
    # hop left
    val = tpd
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(1,0,0))
        data.append(val)
        row.append(get_index(1,0,0))
        col.append(row_index)
        data.append(val)
    # hop right
    val = -tpd
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(0,0,0))
        data.append(val)
        row.append(get_index(0,0,0))
        col.append(row_index)
        data.append(val)
    # hop up
    val = tpd
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(0,0,1))
        data.append(val)
        row.append(get_index(0,0,1))
        col.append(row_index)
        data.append(val)
    # hop down
    val = -tpd
    if val != 0.0:
        row.append(row_index)
        col.append(get_index(0,1,1))
        data.append(val)
        row.append(get_index(0,1,1))
        col.append(row_index)
        data.append(val)
        
    #==================================================================
    # Then set up the p-orbtials' hoppings, namely tpp
    for i in range(-M,M+1):
        for j in range(-M,M+1):   
            # orb = px
            row_index = get_index(i,j,0)
            # diagonal
            row.append(row_index)
            col.append(row_index)
            data.append(-eta*(1j)+eps_p)
            # hop up left, must be in the same unit cell
            val = tpp
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,1))
                data.append(val)
            # hop down left
            if j < M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j+1,1))
                    data.append(val)
            # hop down right
            if i > -M and j < M:
                val = tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i-1,j+1,1))
                    data.append(val)
            # hop up right
            if i > -M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i-1,j,1))
                    data.append(val)
            #----------------------------------------------------   
            # orb = py
            row_index = get_index(i,j,1)
            # diagonal
            row.append(row_index)
            col.append(row_index)
            data.append(-eta*(1j)+eps_p)
            # hop down right
            val = tpp
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,0))
                data.append(val)
            # hop up right
            if j > -M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j-1,0))
                    data.append(val)
            # hop up left
            if i < M and j > -M:
                val = tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i+1,j-1,0))
                    data.append(val)
            # hop down left
            if i < M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i+1,j,0))
                    data.append(val)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    dim  = get_index(M,M,1)+1
    print 'dim(H) = ', dim
    static_eom_matrix = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    static_eom_matrix = static_eom_matrix.tocsr()
    static_eom_matrix.sort_indices()
    return static_eom_matrix

def create_eom_matrix(static_eom_matrix, w):
    '''
    Parameters
    ----------
    static_eom_matrix: output from create_static_eom_matrix()
    w: the energy w.
    
    Returns
    -------
    eom_matrix: scipy sparse matrix with sorted indices in csr format

    Note
    ----
    scipy.sparse.linalg.setdiag() is very slow and therefore not used.

    '''
    dim = static_eom_matrix.shape[0]
    diag = -w*sps.identity(dim,dtype='complex') # matrix format is dia
    # Note that csc + dia = csc and csr + dia = csr format
    return static_eom_matrix+diag

def create_rhs_vec(sigma):
    '''
    Parameters
    ----------
    sigma: the orbital of the bra state. In my notes this is usually
           called sigma'.
    
    Returns
    -------
    rhs: numpy array with the right hand side of the eom.
    
    Note
    ----
    The vector format of the rhs is needed for pardiso and iterative
    methods.

    '''
    dim = get_index(M,M,6)+1
    rhs = np.zeros(dim,dtype='complex')
    rhs[get_index(0,0,sigma)] = -1.0
    # dim = get_index(M,M,2)+1    
    # row = np.array([get_index(0,0,sigma)])
    # col = np.array([0])
    # data = np.array([-1.0],dtype='complex')
    # rhs = sps.coo_matrix((data,(row,col)),shape=(dim,1))
    # rhs = rhs.toarray()
    return rhs

def create_rhs_matrix():
    '''
    Returns
    -------
    rhs: scipy sparse array in csc format with the right hand side
         for sigma = d3z2r2,dx2y2,dxy,dxz,dyz, px and py.
    
    Note
    ----
    The sparse matrix format of the rhs is needed for 
    scipy.sparse.spsolve()

    '''
    dim = get_index(M,M,1)+1
    row = np.array([0,1,2,3,4,get_index(0,0,0),get_index(0,0,1),get_index(10,10,0)])
    col = np.array([0,1,2,3,4,5,6,7])
    data = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],dtype='complex') 
    rhs = sps.coo_matrix((data,(row,col)),shape=(dim,8))
    return rhs.tocsc()
    
