import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import math

M_PI = math.pi
M = None

def get_index(i,j,sigma):
    '''
    Every state |i,j,sigma> in the real space basis is associated
    with a unique index which is used for storing the eom of in 
    matrix format.

    Parameters
    ----------
    i, j : integers, location of unit cell R_{i,j}
    sigma: type of orbital can take one of three values:
           0-4  d3z2r2,dx2y2,dxy,dxz,dyz
           5 px orbital
           6 py orbital

    Returns
    -------
    index: a unique integer that identifies the state

    '''
    return (j+M)*7+(i+M)*(2*M+1)*7+sigma

def create_static_eom_matrix(prm):
    '''The static eom matrix for the real-space GFs does not depend on
    omega. The diagonal entries are -i*eta +eps_{s,p}. From this the
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
    tpd = prm.tpd
    tpp = prm.tpp
    eps_d = prm.eps_d
    eps_p = prm.eps_p
    eta = prm.eta
    
    row = []
    col = []
    data = []

    for i in range(-M,M+1):
        for j in range(-M,M+1):
            #=========================================
            # orb = d3z2r2
            row_index = get_index(i,j,0)
            # diagonal
            row.append(row_index)
            col.append(row_index)
            data.append(-eta*(1j)+eps_d)
            # hop left
            if i < M:
                val = tpd/np.sqrt(3)
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i+1,j,5))
                    data.append(val)
            # hop right
            val = -tpd/np.sqrt(3)
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,5))
                data.append(val)
            # hop up
            val = -tpd/np.sqrt(3)
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,6))
                data.append(val)
            # hop down
            if j < M:
                val = tpd/np.sqrt(3)
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j+1,6))
                    data.append(val)
            #=========================================        
            # orb = dx2y2
            row_index = get_index(i,j,1)
            # diagonal
            row.append(row_index)
            col.append(row_index)
            data.append(-eta*(1j)+eps_d)
            # hop left
            if i < M:
                val = tpd
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i+1,j,5))
                    data.append(val)
            # hop right
            val = -tpd
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,5))
                data.append(val)
            # hop up
            val = tpd
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,6))
                data.append(val)
            # hop down
            if j < M:
                val = -tpd
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j+1,6))
                    data.append(val)
            #=========================================    
            # orb = px
            row_index = get_index(i,j,5)
            # diagonal
            row.append(row_index)
            col.append(row_index)
            data.append(-eta*(1j)+eps_p)
            # hop left
            val = -tpd/np.sqrt(3)
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,0))
                data.append(val)
                
            val = -tpd
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,1))
                data.append(val)
            # hop right
            if i > -M:
                val = tpd/np.sqrt(3)
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i-1,j,0))
                    data.append(val)
                    
                val = tpd
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i-1,j,1))
                    data.append(val)
            # ----------tpp below-------------------
            # hop up left
            val = tpp
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,6))
                data.append(val)
            # hop down left
            if j < M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j+1,6))
                    data.append(val)
            # hop down right
            if i > -M and j < M:
                val = tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i-1,j+1,6))
                    data.append(val)
            # hop up right
            if i > -M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i-1,j,6))
                    data.append(val)
            #=========================================    
            # orb = py
            row_index = get_index(i,j,6)
            # diagonal
            row.append(row_index)
            col.append(row_index)
            data.append(-eta*(1j)+eps_p)
            # hop up
            if j > -M:
                val = tpd
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j-1,0))
                    data.append(val)
                val = -tpd
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j-1,1))
                    data.append(val)
            # hop down
            val = -tpd
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,0))
                data.append(val)
            val = tpd
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,1))
                data.append(val)
            # ----------tpp below-------------------
            # hop down right
            val = tpp
            if val != 0.0:
                row.append(row_index)
                col.append(get_index(i,j,5))
                data.append(val)
            # hop up right
            if j > -M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i,j-1,5))
                    data.append(val)
            # hop up left
            if i < M and j > -M:
                val = tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i+1,j-1,5))
                    data.append(val)
            # hop down left
            if i < M:
                val = -tpp
                if val != 0.0:
                    row.append(row_index)
                    col.append(get_index(i+1,j,5))
                    data.append(val)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    dim = get_index(M,M,6)+1 # dimension of the matrix
    
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
    dim = get_index(M,M,6)+1
    row = np.array([get_index(0,0,0), get_index(0,0,1), get_index(0,0,2), \
                    get_index(0,0,3), get_index(0,0,4), get_index(0,0,5),get_index(0,0,6)])
    col = np.array([0,1,2,3,4,5,6])
    data = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],dtype='complex') 
    rhs = sps.coo_matrix((data,(row,col)),shape=(dim,7))
    return rhs.tocsc()
    
