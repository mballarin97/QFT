# +
#Import necessary packages
import numpy as np
import quimb as quimb
from ncon import ncon
from numpy import linalg as LA
from copy import deepcopy

from helper import right_contract, left_contract
from helper import to_full_MPS, to_dense, to_approx_MPS

# -

#---MANUAL MPS GATES APPLICATION---
#2QUBIT without bond dimension
def apply_two_qubit_gate_full(gate_matrix, pos, state):
    """
    Apply a 2-qubit gate specified by the unitary matrix @gate_matrix to qubits at sites (@pos, @pos+1)
    for a MPS @state.
    """
    
    N = len(state)
    state = deepcopy(state) #avoid modifying the original list
    
    gate = np.array(gate_matrix.reshape(2,2,2,2)) #Reshape gate matrix to 4-order tensor

    assert pos < N-1, f"Trying to apply 2-qubit gate to sites ({pos},{pos+1}), but {pos+1} does not exist"

    left_tens = np.array(state[pos])
    right_tens = np.array(state[pos+1])

    #Check boundaries
    #A tensor is at a boundary if it has order 2
    left_is_boundary = len(left_tens.shape) == 2
    right_is_boundary = len(right_tens.shape) == 2
    
    
    #---Contraction with gate---#
    gate_contraction_list = [[-2,-3,1,3], [-1,1,2], [2,3,-4]]

    if left_is_boundary:
        gate_contraction_list[1].pop(0) #Remove free index of left_tens
    if right_is_boundary:
        gate_contraction_list[2].pop(-1) #Remove free index of right_tens

    gate_contracted = ncon([gate, left_tens, right_tens], gate_contraction_list)

    
    #---Set gate_contracted to center of orthogonality---#

    #Compute density matrices of left/right branches
    if not left_is_boundary:
        rho_left = left_contract(state[:pos])

        #Then we diagonalize it
        eigenval, eigenvec = LA.eigh(rho_left)

        #Mask out 0 eigenvalues to avoid dividing by 0
        mask = eigenval > 0
        eigenval = eigenval[mask]
        eigenvec = eigenvec[:, mask]

        sqrt_eigenval = np.sqrt(abs(eigenval))
        X = eigenvec @ np.diag(sqrt_eigenval) @ np.conjugate(eigenvec.T)
        Xinv = eigenvec @ np.diag(1./sqrt_eigenval) @ np.conjugate(eigenvec.T)

        #Absorb X and Xinv into the tensors
        gate_contraction_indices = [1, -2, -3, -4]

        if right_is_boundary:
            gate_contraction_indices.pop(-1)

        #print(X.shape)
        gate_contracted = ncon([X, gate_contracted], [[-1,1], gate_contraction_indices])

        left_boundary = state[pos-1]
        left_connection_indices = [-1, -2, 1]

        if len(left_boundary.shape) == 2: #if this qubit is at the left boundary
            left_connection_indices = [-1, 1]

        state[pos-1] = ncon([left_boundary, Xinv], [left_connection_indices, [1,-3]])

    if not right_is_boundary:
        rho_right = right_contract(state[pos+2:])

        #Then we diagonalize it
        eigenval, eigenvec = LA.eigh(rho_right)

        #Mask out 0 eigenvalues to avoid dividing by 0
        mask = eigenval > 0
        eigenval = eigenval[mask]
        eigenvec = eigenvec[:, mask]

        sqrt_eigenval = np.sqrt(abs(eigenval))
        X = eigenvec @ np.diag(sqrt_eigenval) @ np.conjugate(eigenvec.T)
        Xinv = eigenvec @ np.diag(1./sqrt_eigenval) @ np.conjugate(eigenvec.T)

        #Absorb X and Xinv into the tensors
        gate_contraction_indices = [-1, -2, -3, 1]

        if left_is_boundary:
            gate_contraction_indices = [-1, -2, 1]

        gate_contracted = ncon([gate_contracted, X], [gate_contraction_indices, [1, -4]])

        right_boundary = state[pos+2]
        right_connection_indices = [1, -2, -3]

        if len(right_boundary.shape) == 2: #if this qubit is at the left boundary
            right_connection_indices = [1, -2]

        state[pos+2] = ncon([Xinv, right_boundary], [[-1, 1], right_connection_indices])
    
    #---Split to MPS---#
    #Dimensions for the SVD
    chi_left  = 1
    chi_right = 1

    if left_is_boundary:
        d = left_tens.shape[0]
    else:
        chi_left = left_tens.shape[0]
        d = left_tens.shape[1]

    if not right_is_boundary:
        chi_right = right_tens.shape[-1]

    #---Perform SVD---#
    U, S, Vh = LA.svd(gate_contracted.reshape(chi_left * d, d * chi_right), full_matrices=False)

    if left_is_boundary:
        state[pos] = U
    else:
        state[pos] = U.reshape(chi_left, d, -1)    

    right_tens = np.diag(S) @ Vh
    if right_is_boundary:
        state[pos+1] = right_tens
    else:
        state[pos+1] = right_tens.reshape(-1, d, chi_right)
    
    return state

#Max bond dimension definition
def max_bond_dimension(mps):
    """
    Return the maximum bond dimension of a mps state.
    """
    
    if len(mps) == 1:
        return None
    
    start = mps[0].shape[-1]
    end   = mps[-1].shape[0]
    
    middle = 1
    if len(mps) > 2:
        middle = np.max([a.shape[-1] for a in mps[1:-1]])
        
    #print(start, middle, end)
    
    return max(start, end, middle)

#2 qubit gate with bond dimension
def apply_two_qubit_gate(gate_matrix, pos, state, chi=2):
    """
    Apply a 2-qubit gate specified by the unitary matrix @gate_matrix to qubits at sites (@pos, @pos+1)
    for a MPS @state and contract the resulting tensor back with maximum bond dimension @chi.
    @chi cannot be less than the maximum bond dimension of the initial mps @state.
    """
    
    N = len(state)
    state = deepcopy(state) #avoid modifying the original list
    
    gate = np.array(gate_matrix.reshape(2,2,2,2)) #Reshape gate matrix to 4-order tensor

    assert pos < N-1, f"Trying to apply 2-qubit gate to sites ({pos},{pos+1}), but {pos+1} does not exist"
    assert chi >= max_bond_dimension(state), "The initial MPS state cannot have a bond dimension higher than @chi."

    left_tens = np.array(state[pos])
    right_tens = np.array(state[pos+1])

    #Check boundaries
    #A tensor is at a boundary if it has order 2
    left_is_boundary = len(left_tens.shape) == 2
    right_is_boundary = len(right_tens.shape) == 2
    
    
    #---Contraction with gate---#
    gate_contraction_list = [[-2,-3,1,3], [-1,1,2], [2,3,-4]]

    if left_is_boundary:
        gate_contraction_list[1].pop(0) #Remove free index of left_tens
    if right_is_boundary:
        gate_contraction_list[2].pop(-1) #Remove free index of right_tens

    gate_contracted = ncon([gate, left_tens, right_tens], gate_contraction_list)
    
    #---Set gate_contracted to center of orthogonality---#

    #Compute density matrices of left/right branches
    if not left_is_boundary:
        rho_left = left_contract(state[:pos])

        #Then we diagonalize it
        eigenval, eigenvec = LA.eigh(rho_left)

        #Mask out 0 eigenvalues to avoid dividing by 0
        mask = eigenval > 0
        eigenval = eigenval[mask]
        eigenvec = eigenvec[:, mask]

        sqrt_eigenval = np.sqrt(abs(eigenval))
        X = eigenvec @ np.diag(sqrt_eigenval) @ np.conjugate(eigenvec.T)
        Xinv = eigenvec @ np.diag(1./sqrt_eigenval) @ np.conjugate(eigenvec.T)

        #Absorb X and Xinv into the tensors
        gate_contraction_indices = [1, -2, -3, -4]

        if right_is_boundary:
            gate_contraction_indices.pop(-1)

        #print(X.shape)
        gate_contracted = ncon([X, gate_contracted], [[-1,1], gate_contraction_indices])

        left_boundary = state[pos-1]
        left_connection_indices = [-1, -2, 1]

        if len(left_boundary.shape) == 2: #if this qubit is at the left boundary
            left_connection_indices = [-1, 1]

        state[pos-1] = ncon([left_boundary, Xinv], [left_connection_indices, [1,-3]])

    if not right_is_boundary:
        rho_right = right_contract(state[pos+2:])

        #Then we diagonalize it
        eigenval, eigenvec = LA.eigh(rho_right)

        #Mask out 0 eigenvalues to avoid dividing by 0
        mask = eigenval > 0
        eigenval = eigenval[mask]
        eigenvec = eigenvec[:, mask]

        sqrt_eigenval = np.sqrt(abs(eigenval))
        X = eigenvec @ np.diag(sqrt_eigenval) @ np.conjugate(eigenvec.T)
        Xinv = eigenvec @ np.diag(1./sqrt_eigenval) @ np.conjugate(eigenvec.T)

        #Absorb X and Xinv into the tensors
        gate_contraction_indices = [-1, -2, -3, 1]

        if left_is_boundary:
            gate_contraction_indices = [-1, -2, 1]

        gate_contracted = ncon([gate_contracted, X], [gate_contraction_indices, [1, -4]])

        right_boundary = state[pos+2]
        right_connection_indices = [1, -2, -3]

        if len(right_boundary.shape) == 2: #if this qubit is at the left boundary
            right_connection_indices = [1, -2]

        state[pos+2] = ncon([Xinv, right_boundary], [[-1, 1], right_connection_indices])
    
    #---Split to MPS---#
    #Dimensions for the SVD
    chi_left  = 1
    chi_right = 1

    if left_is_boundary:
        d = left_tens.shape[0]
    else:
        chi_left = left_tens.shape[0]
        d = left_tens.shape[1]

    if not right_is_boundary:
        chi_right = right_tens.shape[-1]

    #---Perform SVD---#
    U, S, Vh = LA.svd(gate_contracted.reshape(chi_left * d, d * chi_right), full_matrices=False)
    
    #Truncation
    U  = U[...,:chi]    
    Vh = Vh[:chi, ...]

    if left_is_boundary:
        state[pos] = U
    else:
        state[pos] = U.reshape(min(chi_left, chi), d, -1)    

    right_tens = np.diag(S[:chi]) @ Vh
    if right_is_boundary:
        state[pos+1] = right_tens
    else:
        state[pos+1] = right_tens.reshape(-1, d, min(chi_right, chi))
    
    return state

#1 qubit gate
def apply_one_qubit_gate(gate, pos, state):
    """
    Applies a one qubit gate @gate to the site @pos of the MPS @state.
    """
    
    state = deepcopy(state)
    
    contraction_indices = [-1 , 1, -3]
    
    if (pos == 0): #site at the left boundary
        contraction_indices.pop(0)
    if (pos == len(state)-1): #site at the right boundary
        contraction_indices.pop(-1)
        
    # Apply gate to (q1)
    temp = ncon( [state[pos], gate],
                     [ contraction_indices, [1, -2] ] )
    state[pos] = temp
    
    return state
