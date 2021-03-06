import variational_space as vs
import lattice as lat
import bisect
import numpy as np
import scipy.sparse as sps

def count_VS(VS):
    '''
    Get statistics in VS
    '''
    count_upup = 0
    count_updn = 0
    count_dnup = 0
    count_dndn = 0
    
    for i in xrange(0,VS.dim):
        state = VS.get_state(VS.lookup_tbl[i])
        
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']

        if s1=='up' and s2=='up':
            count_upup += 1
        elif s1=='up' and s2=='dn':
            count_updn += 1
        elif s1=='dn' and s2=='up':
            count_dnup += 1
        elif s1=='dn' and s2=='dn':
            count_dndn += 1

    print 'No. of states with count_upup, count_updn, count_dnup, count_dndn:',\
         count_upup, count_updn, count_dnup, count_dndn
    
    assert(count_upup==count_dndn)
    return count_upup, count_updn, count_dnup, count_dndn

def find_singlet_triplet_partner(state,VS):
    '''
    For a given state find its partner state to form a singlet/triplet.
    Applies to general opposite-spin state, not nesessarily in d_double

    Parameters
    ----------
    phase: dictionary containing the phase values exp(-kx*Rx*1j/2.0-ky*Ry*1j/2.0).
        Created with hamiltonian.create_phase_dict.
    VS: VS: VariationalSpace class from the module variational_space

    Returns
    -------
    index: index of the singlet/triplet partner state in the VS
    phase: phase factor with which the partner state needs to be multiplied.
    '''
    s1 = state['hole1_spin']
    s2 = state['hole2_spin']
    orb1 = state['hole1_orb']
    orb2 = state['hole2_orb']
    x1, y1 = state['hole1_coord']
    x2, y2 = state['hole2_coord']
    
    tmp_state = vs.create_state(s2,orb1,x1,y1,s1,orb2,x2,y2)
    partner_state, phase = vs.make_state_canonical(tmp_state)
        
    return VS.get_index(partner_state), phase

def create_singlet_triplet_basis_change_matrix(VS,d_double):
    '''
    Create a matrix representing the basis change to singlets/triplets. The
    columns of the output matrix are the new basis vectors. 
    The Hamiltonian transforms as U_dagger*H*U. 

    Parameters
    ----------
    phase: dictionary containing the phase factors created with
        hamiltonian.create_phase_dict.
    VS: VariationalSpace class from the module variational_space. 

    Returns
    -------
    U: matrix representing the basis change to singlets/triplets in
        sps.coo format.
    '''
    print "start create_singlet_triplet_basis_change_matrix"
    data = []
    row = []
    col = []
    
    #count_upup, count_updn, count_dnup, count_dndn = count_VS(VS)
    #print count_upup, count_updn, count_dnup, count_dndn
    
    count_singlet = 0
    count_triplet = 0
    
    # store index of partner state to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []
    
    # denote if the new state is singlet (0) or triplet (1)
    S_val    = np.zeros(VS.dim, dtype=int)
    Sz_val   = np.zeros(VS.dim, dtype=int)
    AorB_sym = np.zeros(VS.dim, dtype=int)
    
    for i in range(0,VS.dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        x1, y1 = start_state['hole1_coord']
        x2, y2 = start_state['hole2_coord']

        if i not in count_list:
            j, ph = find_singlet_triplet_partner(start_state,VS)
            #print "partner states:", i,j
            #print "state i = ", s1, orb1, s2, orb2
            #print "state j = ",'up',orb2,'dn',orb1
            count_list.append(j)
                
            if j==i:
                if s1==s2:
                    # must be triplet
                    data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                    S_val[i] = 1
                    if s1=='up':
                        Sz_val[i] = 1
                    elif s1=='dn':
                        Sz_val[i] = -1
                    count_triplet += 1
                else:
                    # only possible other states for j=i 
                    assert(s1=='up' and s2=='dn' and orb1==orb2 and (x1,y1)==(x2,y2))
                    
                    # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                    # instead of e1e1 and e2e2
                    if orb1!='dxz' and orb1!='dyz':
                        data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                        S_val[i]  = 0
                        Sz_val[i] = 0
                        count_singlet += 1
                    elif orb1==orb2=='dxz':  # no need to consider e2='dyz' case
                        # find e2e2 state:
                        for e2 in d_double:
                            state = VS.get_state(VS.lookup_tbl[e2])
                            orb1 = state['hole1_orb']
                            orb2 = state['hole2_orb']
                            if orb1==orb2=='dyz':
                                data.append(1.0);  row.append(i);  col.append(i)
                                data.append(1.0);  row.append(e2); col.append(i)
                                AorB_sym[i]  = 1
                                S_val[i]  = 0
                                Sz_val[i] = 0
                                count_singlet += 1
                                data.append(1.0);  row.append(i);  col.append(e2)
                                data.append(-1.0); row.append(e2); col.append(e2)
                                AorB_sym[e2] = -1
                                S_val[e2]  = 0
                                Sz_val[e2] = 0
                                count_singlet += 1
            else:
                # append matrix elements for singlet states
                # convention: original state col i stores singlet and 
                #             partner state col j stores triplet
                data.append(1.0);  row.append(i); col.append(i)
                data.append(-ph);   row.append(j); col.append(i)
                S_val[i]  = 0
                Sz_val[i] = 0

                # append matrix elements for triplet states
                data.append(1.0);  row.append(i); col.append(j)
                data.append(ph);  row.append(j); col.append(j)
                S_val[j]  = 1
                Sz_val[j] = 0

                count_singlet += 1
                count_triplet += 1

    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), S_val, Sz_val, AorB_sym

def find_singlet_triplet_partner_d_double(state,VS):
    '''
    For a given state find its partner state to form a singlet/triplet.
    Right now only applied for d_double states

    Parameters
    ----------
    phase: dictionary containing the phase values exp(-kx*Rx*1j/2.0-ky*Ry*1j/2.0).
        Created with hamiltonian.create_phase_dict.
    VS: VS: VariationalSpace class from the module variational_space

    Returns
    -------
    index: index of the singlet/triplet partner state in the VS
    phase: phase factor with which the partner state needs to be multiplied.
    '''
    s1 = state['hole1_spin']
    s2 = state['hole2_spin']
    orb1 = state['hole1_orb']
    orb2 = state['hole2_orb']
    x1, y1 = state['hole1_coord']
    x2, y2 = state['hole2_coord']

    partner_state = vs.create_state('up',orb2,x2,y2,'dn',orb1,x1,y1)
    phase = -1.0
        
    return VS.get_index(partner_state), phase

def create_singlet_triplet_basis_change_matrix_d_double(VS,d_double):
    '''
    Similar to above create_singlet_triplet_basis_change_matrix but only applies
    basis change for d_double states
    '''
    print "start create_singlet_triplet_basis_change_matrix_d_double"
    data = []
    row = []
    col = []
    
    count_singlet = 0
    count_triplet = 0
    indices = range(0,VS.dim)
    
    # store index of partner state in d_double to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []
    
    # denote if the new state is singlet (0) or triplet (1)
    S_val  = np.zeros(VS.dim, dtype=int)
    Sz_val = np.zeros(VS.dim, dtype=int)
    AorB_sym = np.zeros(VS.dim, dtype=int)
    
    # first set the matrix to be identity matrix (for states not d_double)
    for i in indices:
        if i not in d_double:
            data.append(np.sqrt(2.0)); row.append(i); col.append(i)
        
    for i in d_double:
        start_state = VS.get_state(VS.lookup_tbl[i])
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']

        if s1 == s2:
            # must be triplet
            # see case 2 of make_state_canonical in vs.py, namely
            # for same spin states, always order the orbitals
            S_val[i] = 1
            data.append(np.sqrt(2.0));  row.append(i); col.append(i)
            if s1=='up':
                Sz_val[i] = 1
            elif s1=='dn':
                Sz_val[i] = -1
            count_triplet += 1
            
        elif s1=='dn' and s2=='up':
            print 'Error: d_double cannot have states with s1=dn, s2=up !'
            break
            
        elif s1=='up' and s2=='dn':
            if orb1==orb2:
                # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                # instead of e1e1 and e2e2
                if orb1!='dxz' and orb1!='dyz':
                    data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                    S_val[i]  = 0
                    Sz_val[i] = 0
                    count_singlet += 1
                elif orb1=='dxz':  # no need to consider e2='dyz' case
                    # find e2e2 state:
                    for e2 in d_double:
                        state = VS.get_state(VS.lookup_tbl[e2])
                        orb1 = state['hole1_orb']
                        orb2 = state['hole2_orb']
                        if orb1==orb2=='dyz':
                            data.append(1.0);  row.append(i);  col.append(i)
                            data.append(1.0);  row.append(e2); col.append(i)
                            AorB_sym[i]  = 1
                            S_val[i]  = 0
                            Sz_val[i] = 0
                            count_singlet += 1
                            data.append(1.0);  row.append(i);  col.append(e2)
                            data.append(-1.0); row.append(e2); col.append(e2)
                            AorB_sym[e2] = -1
                            S_val[e2]  = 0
                            Sz_val[e2] = 0
                            count_singlet += 1
            else:
                if i not in count_list:
                    j, ph = find_singlet_triplet_partner_d_double(start_state,VS)

                    # append matrix elements for singlet states
                    # convention: original state col i stores singlet and 
                    #             partner state col j stores triplet
                    data.append(1.0);  row.append(i); col.append(i)
                    data.append(ph);   row.append(j); col.append(i)
                    S_val[i]  = 0
                    Sz_val[i] = 0

                    #print "partner states:", i,j
                    #print "state i = ", s1, orb1, s2, orb2
                    #print "state j = ",'up',orb2,'dn',orb1

                    # append matrix elements for triplet states
                    data.append(1.0);  row.append(i); col.append(j)
                    data.append(-ph);  row.append(j); col.append(j)
                    S_val[j]  = 1
                    Sz_val[j] = 0

                    count_list.append(j)

                    count_singlet += 1
                    count_triplet += 1

    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), S_val, Sz_val, AorB_sym
