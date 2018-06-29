import numpy as np
import math
import sys

M_PI = math.pi

def create_h0_matrix(prm,kx,ky):
    
    # unpack parameters
    tspx = prm.tspx
    tspy = prm.tspy
    tpp = prm.tpp
    eps_s = prm.eps_s
    eps_p = prm.eps_p

    h0_matrix = np.zeros((3,3),dtype='complex')
    # first row
    h0_matrix[0,0] = eps_s
    h0_matrix[0,1] = tspx*(-1 + np.exp(-kx*(1j)))
    h0_matrix[0,2] = tspy*(-1 + np.exp(-ky*(1j)))
    # second row
    h0_matrix[1,0] = np.conj(h0_matrix[0,1])
    h0_matrix[1,1] = eps_p
    h0_matrix[1,2] = tpp*(-1 + np.exp(-ky*(1j)) + np.exp(kx*(1j)) - \
                          np.exp((kx-ky)*(1j)))
    # third row
    h0_matrix[2,0] = np.conj(h0_matrix[0,2])
    h0_matrix[2,1] = np.conj(h0_matrix[1,2])
    h0_matrix[2,2] = eps_p
    return h0_matrix

def calc_dispersion(prm,kx,ky):
    h0_matrix = create_h0_matrix(prm,kx,ky)
    eig, vec = np.linalg.eigh(h0_matrix)
    return eig

def calc_rspace_gf_with_FT(prm,x,y,w):
    Nlat = 50
    diag = (w+prm.eta*(1j))*np.identity(3,dtype='complex')
    gf = np.zeros((3,3),dtype='complex')
    kvals = np.array(range(0,Nlat),dtype='float')
    kvals = kvals*2.0*M_PI/Nlat

    for kx in np.nditer(kvals):
        for ky in np.nditer(kvals):
            phase = np.exp((kx*x+ky*y)*(1j))
            h0_matrix = create_h0_matrix(prm,kx,ky)
            h0_matrix = diag - h0_matrix
            inv = np.linalg.inv(h0_matrix)
            gf += phase*inv 
    return gf/(Nlat*Nlat)

def calc_kspace_gf(prm,kx,ky,w):
    diag = (w+prm.eta*(1j))*np.identity(3,dtype='complex')
    gf = np.zeros((3,3),dtype='complex')
    matrix = diag - create_h0_matrix(prm,kx,ky)
    G0 = np.linalg.inv(matrix)
    return G0
    

