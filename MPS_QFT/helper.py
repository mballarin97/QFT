#Import necessary packages
import numpy as np
import quimb as quimb
from ncon import ncon
from numpy import linalg as LA

def print_state(dense_state):
    """
    Prints a *dense_state* with kets. Compatible with quimb states.
    
    Parameters
    ----------
    dense_state: array_like 
        Dense representation of a quantum state
        
    Returns
    -------
    None: None
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


# +
#---MANUAL IMPLEMENTATION OF QFT---
#Helper functions
def right_contract(states):
    """
    Given the N right-most *states* of a MPS, computes their contraction with themselves, as in the Example
    
     Parameters
    ----------
    states: list of tensors
        N right-most *states* of a MPS
    
    Returns
    -------
    Matrix: order 2 tensor
        Returns a order 2 tensor.

    Examples
    --------
    >>>  - o  -  o  -  o  -  o 
    >>>    |     |     |     | 
    >>>  o.H - o.H - o.H - o.H 
    """
    #Add np.conjugate (when we will use complex numbers)
    
    N = len(states)
    
    #Numbering of indices follows "diagonals from top-left to bottom-right". See:
    #-2 -  o -3- o -6- o -9-  o
    #     1|     |4    |7     |10
    #-1 -  o -2- o -5- o -8-  o
    
    top_indices = 3 * np.arange(1,N)
    bottom_indices = top_indices - 1
    
    top_indices = np.insert(top_indices, 0, -2)
    bottom_indices = np.insert(bottom_indices, 0, -1)
    
    middle_indices = 3 * np.arange(1, N+1) - 2
    
    top_connections = [[top_indices[i], middle_indices[i], top_indices[i+1]] for i in range(N-1)] + [[top_indices[-1], middle_indices[-1]]]
    bottom_connections = [[bottom_indices[i], middle_indices[i], bottom_indices[i+1]] for i in range(N-1)] + [[bottom_indices[-1], middle_indices[-1]]]

    #print(top_connections, bottom_connections)
    
    conj_states = [np.conjugate(s) for s in states]
    return ncon(states + conj_states, top_connections + bottom_connections)

def left_contract(states):
    """
    Given the N left-most *states* of a MPS, computes their contraction with themselves, as in the Example.
    
    Parameters
    ----------
    states: list of tensors
        N left-most *states* of a MPS
    
    Returns
    -------
    Matrix: order 2 tensor
        Returns a order 2 tensor.

    Examples
    --------
    >>> o  -  o  -  o  -  o   -
    >>> |     |     |     |
    >>> o.H - o.H - o.H - o.H -
    """
    
    #Convention is the following (idk if it's the most efficient...)
    #e.g. for N=4
    # U1 -2- U2 -5- U3 -8- U4 - (-1)
    # |1     |4     |7     |10
    # U1 -3- U2 -6- U3 -9- U4 - (-2)
    #Basically, the contracted indices follow the order of "diagonals" from the bottom-left to the top-right.
    #Start with the first connection, which links U1 to U1.H. Now the "top-right" is between U1 and U2.
    #Now restart from the next connection at the bottom, which links U1 and U2. Going to the "top-right" we see U2 - U2.H, and then
    #U2-U3, and so on.
    
    N = len(states) #Number of tensors to be contracted
    
    if (N == 1):
        state = states[0]
        return ncon([state, np.conjugate(state)], [[1, -1], [1, -2]]) 
    
    #Read indices "by row"
    bottom_indices = 3 * (np.arange(N)+1)
    top_indices = bottom_indices - 1
    middle_indices = bottom_indices - 2

    #Free indices
    top_indices[-1] = -1
    bottom_indices[-1] = -2 

    top_connections = [[1,2]] + [[top_indices[i], middle_indices[i+1], top_indices[i+1]] for i in range(N-1)]
    bottom_connections = [[1,3]] + [[bottom_indices[i], middle_indices[i+1], bottom_indices[i+1]] for i in range(N-1)]

    conj_states = [np.conjugate(s) for s in states]
    return ncon(states + conj_states, top_connections + bottom_connections) 
    #Here we work with Python lists, so the + is not elementwise addition, but concatenation of arrays!
    


# +
#---CONVERT BETWEEN MPS AND DENSE---
def to_full_MPS(dense_state, N, d=2):
    """
    Converts a *dense_state* of a *N*-body system made by *d*-dimensional sites into a Matrix Product State 
    in left-canonical form, with sufficiently sized bonds so that exactness is maintained.
    
    Parameters
    ----------
    dense_state : ndarray of shape (d^N,)
        Input dense state, such that the (i,j,k...) entry in dense_state.reshape([d]*N) is the (i,j,k...) coefficient 
        of the state in the computational basis.
    N : integer > 0
        Number of particles/sites
    d : integer > 0, optional
        Local dimension of each particle/site. For a qubit, d=2.
    
    Returns
    -------
    MPS : list of *N* tensors
        List of *N* tensors containing the left-canonical MPS. The first and last tensors are of order 2 (matrices), while
        all the others are of order 3. We can see the sketch in the Example.
        The index ordering convention is from left-to-right. 
        For instance, the "left" index of U2 is the first, the "bottom" one is the second, and the "right" one is the third.

    Examples
    --------
    >>> U1 - U2 - U3 - ... - UN 
    >>> |    |    |          |
    """
    
    assert N > 0, "Number of sites must be > 0"
    assert d > 0, "Local dimension must be > 0"
    assert len(dense_state.flatten()) == d**N, "The dense_state must be of dimension d**N"
    
    state_tensor = dense_state.reshape([d] * N) #Reshape into a tensor of order N
    MPS = []
    
    last_svd_dim = 1
    for i in range(N-1):
        U, S, Vh = LA.svd(state_tensor.reshape(last_svd_dim*d, d**(N-(i+1))), full_matrices=False)
        
        state_tensor = (np.diag(S) @ Vh)
        
        if i > 0: #first does not need reshaping
            U = U.reshape(last_svd_dim, d, -1) #reshape to allow the contraction
        
        last_svd_dim = len(S)
        MPS.append(U.copy())
        
        
    MPS.append(state_tensor)
        
    return MPS

def to_dense(MPS):
    """
    Given a list of N tensors *MPS* [U1, U2, ..., UN] , representing a Matrix Product State, perform the contraction in the Examples,
    leading to a single tensor of order N, representing a dense state.
    
    The index ordering convention is from left-to-right. 
    For instance, the "left" index of U2 is the first, the "bottom" one is the second, and the "right" one is the third.
    
    Parameters
    ----------
    MPS : list of ndarrays
        List of tensors. First and last should be of order 2, all the others of order 3. 
        The last dimension of MPS[i] should be the same of the first dimension of MPS[i+1], for all i.
    
    Returns
    -------
    dense_state : ndarray of shape ([d] * N)
        N-order tensor representing the dense state.

    Examples
    --------
    >>> U1 - U2 - ... - UN
    >>>  |    |          |
    """
    
    #TODO add assertions
    
    N = len(MPS)
    first_indices  = [-1, 1]
    middle_indices = [[i, -(i+1), i+1] for i in range(1,N-1)]
    last_indices   = [N-1, -N]
    connect_list = [first_indices, *middle_indices, last_indices]
    
    return ncon(MPS, connect_list)

#Now, let's fix a maximum bond dimension chi:

def to_approx_MPS(dense_state, N, d=2, chi=2):
    """
    Converts a *dense_state* of a *N*-body system made by *d*-dimensional sites into a Matrix Product State 
    in left-canonical form, with the size of links bounded by *chi*.
    
    
    
    Parameters
    ----------
    dense_state : ndarray of shape (d^N,)
        Input dense state, such that the (i,j,k...) entry in dense_state.reshape([d]*N) is the (i,j,k...) coefficient 
        of the state in the computational basis.
    N : integer > 0
        Number of particles/sites
    d : integer > 0, optional
        Local dimension of each particle/site. For a qubit, d=2.
    chi : integer > 0, optional
        Maximum bond dimension
    
    Returns
    -------
    MPS: List of *N* tensors containing the left-canonical MPS
        List of *N* tensors containing the left-canonical MPS. The first and last tensors are of order 2 (matrices), while
        all the others are of order 3. The shapes are not fixed, but they are (a_i, d, a_{i+1}), with a_i, a_{i+1} <= chi 
        for the order 3 tensors, and (d, a_1) or (a_{N-1}, d) for the order 2 tensors at the boundaries. We can see the representation
        in the first example.
        The index ordering convention is from left-to-right. 
        For instance, the "left" index of U2 is the first, the "bottom" one is the second, and the "right" one is the third.
    
    Examples
    --------
    >>> U1 - U2 - U3 - ... - UN 
    >>> |    |    |          |
    # For d=2, N=7 and chi=5, the tensor network is as follows:
    >>> U1 -2- U2 -4- U3 -5- U4 -5- U5 -4- U6 -2- U7
    >>> |      |      |      |      |      |      |
    # where -x- denotes the bounds' dimension (all the "bottom-facing" indices are of dimension d=2). Thus, the shapes
    # of the returned tensors are as follows:
    >>>    U1       U2          U3         U4        U5         U6        U7
    >>> [(2, 2), (2, 2, 4), (4, 2, 5), (5, 2, 5), (5, 2, 4), (4, 2, 2), (2, 2)]
    """
    
    assert N > 0, "Number of sites must be > 0"
    assert d > 0, "Local dimension must be > 0"
    assert len(dense_state.flatten()) == d**N, "The dense_state must be of dimension d**N"
    
    state_tensor = dense_state.reshape([d] * N) #Reshape into a tensor of order N
    MPS = []
    
    last_svd_dim = 1
    for i in range(N-1):
        U, S, Vh = LA.svd(state_tensor.reshape(last_svd_dim * d, d**(N-(i+1))), full_matrices=False)
        
        #Truncation
        U  = U[...,:chi]    #shape is (d...d,chi)
        Vh = Vh[:chi, ...]  #shape is (chi, d...d)
        state_tensor = (np.diag(S[:chi]) @ Vh)
        
        if i > 0: #first does not need reshaping
            U = U.reshape(min(last_svd_dim, chi), d, -1) #reshape to allow the contraction
            
        MPS.append(U.copy())
        
        last_svd_dim = len(S) if len(S) < chi else chi
        
    MPS.append(state_tensor)
        
    return MPS


# -

def get_figsize(wf=0.5, hf=(5.**0.5-1.0)/2.0, ):
    """
    Parameters
    ----------
    wf [float]:  width fraction in columnwidth units, optional
    hf [float]:  height fraction in columnwidth units, optional
        Set by default to golden ratio.
    columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth , optional
    Returns
    -------
    fig_size: list of float
        fig_size [width, height] that should be given to matplotlib
    """
    columnwidth = 510.0 #! The width of the Latex paper should be put here [OK]
    
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]


def Fidelity(psi, phi):
    """
    Returns the fidelity bewteen two quantum states *psi*, *phi* defined as |<*psi*|*phi*>|^2
    
    Parameters
    ----------
    psi: complex np.array or quimb.core.qarray
        Quantum state
    phi: complex np.array or quimb.core.qarray
        Quantum state
        
    Returns
    -------
    Fidelity: double real
        Fidelity of the two quantum states
    """

    # Enforcing normalization
    psi /= np.sqrt( np.abs(np.sum( psi.conj()*psi ))   )
    phi /= np.sqrt(  np.abs(np.sum( phi.conj()*phi )) )
    
    return np.abs( np.dot(psi.conj(), phi) )**2


def reverse_qiskit(qiskit_dense, N):
    """
        Reverse the ordering of a qiskit state, in which the 0-th qubit is the leftmost 
        (for example in |1000> the 0-th qubit is in the state 1), to match the more usual notation
        in which the 0-th qubit is the rightmost.
        
        Parameter
        ---------
        qiskit_dense: array_like
            Dense qiskit state vector
        N: int
            Number of qubits which forms *qiskit_dense*
            
        Returns
        -------
        qiskit_reordered: array_like
            Reordered dense state_vector
            
        Examples
        --------
        >>> qiskit_result = [0, 0, 1, 0] # |01>, but usually recognized as |10>
        >>> reverse_qiskit(qiskit_result, 2)
        [0, 1, 0, 0] # |01> with the usual representation
    """
    assert 2**N==len(qiskit_dense), f'qiskit_dense is not the statevector representing {N} qubits, since its length is length is {len(qiskit_dense)} while it should be {2**N}'
        
    indexing = np.array( [ int( '{:0{width}b}'.format(i, width=N)[::-1], 2) for i in range(2**N) ] )
    qiskit_reordered = qiskit_dense[indexing]
    
    return qiskit_reordered


def qiskit_get_statevect(qc):
    """
        Returns the statevector of the qiskit quantum circuit *qc* in the reversed order
        
        Parameters
        ----------
        qc: Quantum circuit
            Quantum circuit of which we want the statevector
            
        Returns
        -------
        st: array_like
            Statevector of the quantum circuit after the application of the reverse operation on the qubit's ordering
    """
    N = qc.num_qubits
    st = reverse_qiskit(execute(qc, Aer.get_backend('statevector_simulator')).result().get_statevector(), N)
    
    return st
