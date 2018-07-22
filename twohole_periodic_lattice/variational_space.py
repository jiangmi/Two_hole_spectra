'''
Contains a class for the variational space for the cuprate layer
and functions used to represent states as dictionaries.
Distance (L1-norm) between any two particles must not exceed a cutoff
denoted by Mc. 
'''

import parameters as pam
import lattice as lat
import bisect
import numpy as np

def create_state(spin1,orb1,x1,y1,spin2,orb2,x2,y2):
    '''
    Creates a dictionary representing a state

    Parameters
    ----------
    spin1, spin2   : string of spin
    orb_up, orb_dn : string of orb
    x_up, y_up: integer coordinates of hole1
                Must be (1,0), (0,1), (0,0)
    x_dn, y_dn: integer coordinates of hole2

    Note
    ----
    It is possible to return a state that is not in the 
    variational space because the hole-hole Manhattan distance
    exceeds Mc.
    '''
    assert(check_in_vs_condition(x1,y1,x2,y2))
    assert not (((x1,y1))==(x2,y2) and spin1==spin2 and orb1==orb2)
    assert (x1,y1) in [(0,0),(1,0),(0,1)]
    orb, _, _ = lat.get_unit_cell_rep(x1,y1)
    assert orb1 in orb
    orb, _, _ = lat.get_unit_cell_rep(x2,y2)
    assert orb2 in orb
    
    state = {'hole1_spin' :spin1,\
             'hole1_orb'  :orb1,\
             'hole1_coord':(x1,y1),\
             'hole2_spin' :spin2,\
             'hole2_orb'  :orb2,\
             'hole2_coord':(x2,y2)}
    return state

def make_state_canonical(state):
    '''
    1. There are a few cases to avoid having duplicate states where 
    the holes are indistinguishable. 
    
    The sign change due to anticommuting creation operators should be 
    taken into account so that phase below has a negative sign
    
    see Mirko's notes VS_state.pdf for the real meaning of states!
    =============================================================
    Case 1: 
    when hole2 is on left of hole 1, need to shift hole2's coordinates
    to origin unit cell so that all states' first hole locates in origin
    
    Orders the hole coordinates in such a way that the coordinates 
    of the left creation operator are lexicographically
    smaller than those of the right.
    =======================================================
    Case 2: 
    If two holes locate on the same sites
    a) same spin state: 
      up, dxy,    (0,0), up, dx2-y2, (0,0)
    = up, dx2-y2, (0,0), up, dxy,    (0,0)
    need sort orbital order
    b) opposite spin state:
    only keep spin1 up state
    
    For phase, see note_on_two_holes_same_spin.jpg
    Note the /2.0, same as create_phase_dict in hamiltonian.py
    
    2. Besides, see emails with Mirko on Mar.1, 2018:
    Suppose Tpd|state_i> = |state_j> = phase*|tmp_state_j> = phase*ph*|canonical_state_j>, then 
    tpd = <state_j | Tpd | state_i> 
        = conj(phase*ph)* <canonical_state_j | Tpp | state_i>
    
    so <canonical_state_j | Tpp | state_i> = tpd/conj(phase*ph)
                                           = tpd*phase*ph
    
    Because conj(phase) = 1/phase, *phase and /phase in setting tpd and tpp seem to give same results
    But need to change * or / in both tpd and tpp functions
    
    Similar for tpp
    '''
    s1 = state['hole1_spin']
    s2 = state['hole2_spin']
    orb1 = state['hole1_orb']
    orb2 = state['hole2_orb']
    x1, y1 = state['hole1_coord']
    x2, y2 = state['hole2_coord']
        
    canonical_state = state
    phase = 1.0
        
    # see Mirko's notes VS_state.pdf; note that phase depends on Dx,Dy instead of dx,dy
    if (x2,y2)<(x1,y1):
        dx = x2-x1
        dy = y2-y1
        (ux,uy) = lat.orb_pos[orb2]
        canonical_state = create_state(s2,orb2,ux,uy,s1,orb1,ux-dx,uy-dy)
        kx = pam.kx
        ky = pam.ky
        
        _, Rx2, Ry2 = lat.get_unit_cell_rep(x2,y2)
        _, Rx1, Ry1 = lat.get_unit_cell_rep(x1,y1)
        Dx = Rx2-Rx1
        Dy = Ry2-Ry1
        phase = -np.exp(-(kx*Dx+ky*Dy)*1j/2.0)
        
    elif (x1,y1)==(x2,y2):           
        if s1==s2:
            o12 = list(sorted([orb1,orb2]))
            if o12[0]==orb2:
                canonical_state = create_state(s2,orb2,x1,y1,s1,orb1,x1,y1)
                phase = -1.0  
        elif s1=='dn' and s2=='up':
            canonical_state = create_state('up',orb2,x1,y1,'dn',orb1,x1,y1)
            phase = -1.0

    return canonical_state, phase
    
def calc_manhattan_dist(x1,y1,x2,y2):
    '''
    Calculate the Manhattan distance (L1-norm) between two vectors
    (x1,y1) and (x2,y2).
    '''
    out = abs(x1-x2) + abs(y1-y2)
    return out

def check_in_vs_condition(x1,y1,x2,y2):
    '''
    Restrictions: the distance between two holes should be less than cutoff Mc
    '''     
    if calc_manhattan_dist(x1,y1,x2,y2) > pam.Mc:
        return False
    else:
        return True
    
class VariationalSpace:
    '''
    Distance (L1-norm) between any two particles must not exceed a
    cutoff denoted by Mc. 

    Attributes
    ----------
    Mc: Cutoff for the hole-hole 
    lookup_tbl: sorted python list containing the unique identifiers 
        (uid) for all the states in the variational space. A uid is an
        integer which can be mapped to a state (see docsting of get_uid
        and get_state).
    dim: number of states in the variational space, i.e. length of
        lookup_tbl
    filter_func: a function that is passed to create additional 
        restrictions on the variational space. Default is None, 
        which means that no additional restrictions are implemented. 
        filter_func takes exactly one parameter which is a dictionary representing a state.

    Methods
    -------
    __init__
    create_lookup_table
    get_uid
    get_state
    get_index
    '''

    def __init__(self,Mc,filter_func=None):
        self.Mc = Mc
        if filter_func == None:
            self.filter_func = lambda x: True
        else:
            self.filter_func = filter_func
        self.lookup_tbl = self.create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print "VS.dim = ", self.dim
        #self.print_VS()

    def print_VS(self):
        for i in xrange(0,self.dim):
            state = self.get_state(self.lookup_tbl[i])
            ts1 = state['hole1_spin']
            ts2 = state['hole2_spin']
            torb1 = state['hole1_orb']
            torb2 = state['hole2_orb']
            tx1, ty1 = state['hole1_coord']
            tx2, ty2 = state['hole2_coord']
            #if ts1=='up' and ts2=='dn':
            print i, ts1,torb1,tx1,ty1,ts2,torb2,tx2,ty2
                
    def create_lookup_tbl(self):
        '''
        Create a sorted lookup table containing the unique identifiers 
        (uid) of all the states in the variational space.

        Returns
        -------
        lookup_tbl: sorted python list.
        '''
        Mc = self.Mc
        # Loops are over all states for which the Manhattan distance 
        # between any two particles does not exceed Mc.
        lookup_tbl = []

        for s1 in ['up','dn']:
            for orb1 in pam.orbs:
                ux, uy = lat.orb_pos[orb1]

                # Manhattan distance between any two holes does not exceed Mc
                for vx in range(-Mc+ux,Mc+ux+1):
                    B = Mc - abs(vx-ux)
                    for vy in range(-B+uy,B+uy+1):
                        orb2s, _, _ = lat.get_unit_cell_rep(vx,vy)

                        if orb2s==['NotOnSublattice']:
                            continue

                        for orb2 in orb2s:
                            for s2 in ['up','dn']:   
                                # try screen out same spin states
                                #if s1==s2:
                                #    continue
                                
                                # consider Pauli principle
                                if s1==s2 and orb1==orb2 and ux==vx and uy==vy:
                                    continue 
                                    
                                #if pam.if_project_out_two_holes_on_different_Cu == 1:
                                #    if (orb1 in pam.Cu_orbs and orb2 in pam.Cu_orbs and (ux,uy)!=(vx,vy)):
                                #        continue
                                
                                #if s1=='dn' and s2=='dn':
                                #    print "candiate state: ", s1,orb1,ux,uy,s2,orb2,vx,vy
                                
                                if check_in_vs_condition(ux,uy,vx,vy):
                                    state = create_state(s1,orb1,ux,uy,s2,orb2,vx,vy)
                                    canonical_state,_ = make_state_canonical(state)

                                    #if pam.if_project_out_two_holes_on_different_Cu == 1:
                                    #    if torb1 in pam.Cu_orbs and torb2 in pam.Cu_orbs and (tx1, ty1)!=(tx2, ty2):
                                    #        continue

                                if self.filter_func(canonical_state):
                                    uid = self.get_uid(canonical_state)
                                    lookup_tbl.append(uid)
 
        lookup_tbl = list(set(lookup_tbl)) # remove duplicates
        lookup_tbl.sort()
        #print "\n lookup_tbl:\n", lookup_tbl
        return lookup_tbl
            
    def check_in_vs(self,state):
        '''
        Check if a given state obeys the restrictions of the variational
        space.

        Parameters
        ----------
        state: dictionary created by one of the functions which create 
            states.
        Mc: integer cutoff for the Manhattan distance.

        Returns
        -------
        Boolean: True or False
        '''
        assert(self.filter_func(state) in [True,False])
        if self.filter_func(state) == False:
            return False

        x1, y1 = state['hole1_coord']
        x2, y2 = state['hole2_coord']
  
        if check_in_vs_condition(x1,y1,x2,y2):
            return True
        else:
            return False

    def get_uid(self,state):
        '''
        Every state in the variational space is associated with a unique
        identifier (uid) which is an integer number.
        must use the number of all possible orbitals, namely N=7 !!!

        Parameters
        ----------
        state: dictionary created by one of the functions which create 
            states.

        Returns
        -------
        uid (integer) or None if the state is not in the variational space.
        '''
        if not self.check_in_vs(state):
            return None

        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4

        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        i1 = lat.spin_int[s1]
        i2 = lat.spin_int[s2]
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        o1 = lat.orb_int[orb1]
        o2 = lat.orb_int[orb2]
        x1, y1 = state['hole1_coord']
        x2, y2 = state['hole2_coord']

        uid = i1 + 2*i2 + 4*(o1 + N*o2 + N*N*( (y1+1) + (x1+1)*3 + (y2+s)*9 + (x2+s)*B1*9 ) )
        
        # check if uid maps back to the original state, namely uid's uniqueness
        tstate = self.get_state(uid)
        ts1 = tstate['hole1_spin']
        ts2 = tstate['hole2_spin']
        torb1 = tstate['hole1_orb']
        torb2 = tstate['hole2_orb']
        tx1, ty1 = tstate['hole1_coord']
        tx2, ty2 = tstate['hole2_coord']
        assert((s1,orb1,x1,y1,s2,orb2,x2,y2)==(ts1,torb1,tx1,ty1,ts2,torb2,tx2,ty2))
        
        return uid

    def get_state(self,uid):
        '''
        Given a unique identifier, return the corresponding state. See 
        get_uid.

        must use the number of all possible orbitals, namely N=7 !!!
        '''
        N = pam.Norb
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4

        x2 = uid/(B1*9*N*N*4) - s
        uid_ = uid % (B1*9*N*N*4)
        y2 = uid_/(9*N*N*4) - s
        uid_ = uid_ % (9*N*N*4)
        x1 = uid_/(3*N*N*4) - 1
        uid_ = uid_ % (3*N*N*4)
        y1 = uid_/(N*N*4) - 1
        uid_ = uid_ % (N*N*4)
        o2 = uid_/(4*N)
        uid_ = uid_ % (4*N)
        o1 = uid_/4
        uid_ = uid_ % 4
        i2 = uid_/2
        i1 = uid_%2
        
        orb2 = lat.int_orb[o2]
        orb1 = lat.int_orb[o1]
        s2 = lat.int_spin[i2]
        s1 = lat.int_spin[i1]

        state = create_state(s1,orb1,x1,y1,s2,orb2,x2,y2)
        return state

    def get_index(self,state):
        '''
        Return the index under which the state is stored in the lookup
        table.  These indices are consecutive and can be used to
        index, e.g. the Hamiltonian matrix

        Parameters
        ----------
        state: dictionary representing a state

        Returns
        -------
        index: integer such that lookup_tbl[index] = get_uid(state,Mc).
            If the state is not in the variational space None is 
            returned.

        '''
        uid = self.get_uid(state)
        if uid == None:
            return None
        else:
            index = bisect.bisect_left(self.lookup_tbl,uid)
            if self.lookup_tbl[index] == uid:
                return index
            else:
                return None


