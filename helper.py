import numpy as np
import quimb as quimb

def print_state(dense_state):
    """
    Prints a @dense_state with kets. Compatible with quimb states.
    """
    # quimb to_dense adaptation
    if type( dense_state ) == quimb.core.qarray:
        dense_state = [ pop[0] for pop in dense_state]    
    
    N = int(np.log2(len(dense_state)))
    
    binaries = [bin(i)[2:] for i in range(2**N)]
    binaries = ['0'*(N-len(a)) + a for a in binaries] #Pad with 0s
    
    ket = []
    for i, c in enumerate(dense_state):
        if not np.isclose(np.abs(c), 0.):
            if np.isclose(np.imag(c), 0.):
                if np.isclose(np.real(c), 1.):
                    ket.append('|{}>'.format(binaries[i]))
                else:
                    ket.append('{:.3f}|{}>'.format(np.real(c), binaries[i]))  
            else:
                ket.append('{:.3f}|{}>'.format(c, binaries[i]))
    print(' + '.join(ket))
 

