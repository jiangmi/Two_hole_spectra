'''
Functions pertaining to the CuO2 lattice. Primitive Vectors are tuples
 of integers, i.e. they are given in units of a/2 (a is lattice constant). 
'''
import parameters as pam

if pam.Norb==3:
    orb_pos = {'dx2y2': (0,0),\
               'px':    (1,0),\
               'py':    (0,1)}
    # below two used for get_uid and get_state in VS
    orb_int = {'dx2y2': 0,\
               'px':    1,\
               'py':    2} 
    int_orb = {0: 'dx2y2',\
               1: 'px',\
               2: 'py'}
elif pam.Norb==7:
    orb_pos = {'d3z2r2': (0,0),\
               'dx2y2':  (0,0),\
               'dxy':    (0,0),\
               'dxz':    (0,0),\
               'dyz':    (0,0),\
               'px':     (1,0),\
               'py':     (0,1)}
    # below two used for get_uid and get_state in VS
    orb_int = {'d3z2r2': 0,\
               'dx2y2':  1,\
               'dxy':    2,\
               'dxz':    3,\
               'dyz':    4,\
               'px':     5,\
               'py':     6} 
    int_orb = {0: 'd3z2r2',\
               1: 'dx2y2',\
               2: 'dxy',\
               3: 'dxz',\
               4: 'dyz',\
               5: 'px',\
               6: 'py'}
elif pam.Norb==9:
    orb_pos = {'d3z2r2': (0,0),\
               'dx2y2':  (0,0),\
               'dxy':    (0,0),\
               'dxz':    (0,0),\
               'dyz':    (0,0),\
               'px1':    (1,0),\
               'py1':    (1,0),\
               'px2':    (0,1),\
               'py2':    (0,1)}
    # below two used for get_uid and get_state in VS
    orb_int = {'d3z2r2': 0,\
               'dx2y2':  1,\
               'dxy':    2,\
               'dxz':    3,\
               'dyz':    4,\
               'px1':    5,\
               'py1':    6,\
               'px2':    7,\
               'py2':    8} 
    int_orb = {0: 'd3z2r2',\
               1: 'dx2y2',\
               2: 'dxy',\
               3: 'dxz',\
               4: 'dyz',\
               5: 'px1',\
               6: 'py1',\
               7: 'px2',\
               8: 'py2'} 
# below two used for get_uid and get_state in VS
spin_int = {'up': 1,\
            'dn': 0}
int_spin = {1: 'up',\
            0: 'dn'} 

def get_unit_cell_rep(x,y):
    '''
    Given a vector (x,y) return the unit cell origin.

    Parameters
    -----------
    x,y: (integer) x and y component of vector pointing to a lattice site.
    
    Returns
    -------
    orbital: One of the following strings 'dx2y2', 
            'Ox1', 'Ox2', 'Oy1', 'Oy2', 'NotOnSublattice'
    Rx: (integer) x-component of unit cell vector
    Ry: (integer) y-component of unit cell vector
    '''

    # Note that x, y can be negative
    if abs(x) % 2 == 0 and abs(y) % 2 == 0:
        return pam.Cu_orbs, x, y
    elif abs(x) % 2 == 1 and abs(y) % 2 == 0:
        return pam.O1_orbs, x-1, y
    elif abs(x) % 2 == 0 and abs(y) % 2 == 1:
        return pam.O2_orbs, x, y-1
    else:
        return ['NotOnSublattice'], None, None
