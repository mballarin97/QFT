# +
#Import necessary packages
import numpy as np
import quimb as quimb
from ncon import ncon
from numpy import linalg as LA
from copy import deepcopy

from helper import right_contract, left_contract
from helper import to_full_MPS, to_dense, to_approx_MPS


# +
def left_compress(left, right):
    """
    Perform a change of gauge in the bond between the tensors @left and @right, making @left a (semi)-unitary tensor.
    """

    left_dim = left.shape[:-1]
    left = left.reshape(np.prod(left_dim), -1)
    
    q, r = LA.qr(left) #QR decompose the left tensor. 
    
    left = q.reshape(*left_dim, -1) #Set left to q (and reshape back)
    
    right_dim = right.shape[1:]
    right = right.reshape(-1, np.prod(right_dim))
    right = (r @ right).reshape(-1, *right_dim) #Absorb R into right.
    
    return (left, right)

def right_compress(left, right):
    """
    Perform a change of gauge in the bond between the tensors @left and @right, making @right a (semi)-unitary tensor.
    """
    
    right_dim = right.shape[1:]

    right = right.reshape(-1, np.prod(right_dim)).T 
    #We want (left) - (right) => (left) - R - Q =  
    #So we need R before Q. That is why we transpose: (right).T = Q R => (right) = R.T Q.T (and rename R.T => R and Q.T => Q)

    q, r = LA.qr(right)
    q = q.T
    r = r.T
    
    right = q.reshape(-1, *right_dim) #set (right) to Q (and reshape back)
    
    left_dim = left.shape[:-1]
    left = left.reshape(np.prod(left_dim), -1)
    left = (left @ r).reshape(*left_dim, -1) #absorb R into (left)
    
    return (left, right)

def right_canonize(sites):
    """Apply a gauge transformation to all bonds between @sites, so that all sites beside the first (leftmost one) are
       set to (semi)-unitary tensors."""
    
    N = len(sites)
    sites = sites.copy()
    
    for i in range(N-1):
        left  = sites[N-2-i]
        right = sites[N-1-i]
        
        left, right = right_compress(left, right)

        sites[N-1-i] = right
        sites[N-2-i] = left
        
    return sites

def left_canonize(sites):
    """Apply a gauge transformation to all bonds between @sites, so that all sites beside the last (rightmost one) are
       set to (semi)-unitary tensors."""
    
    N = len(sites)
    sites = sites.copy()
    
    for i in range(N-1):
        left  = sites[i]
        right = sites[i+1]
        
        left, right = left_compress(left, right)

        sites[i] = left
        sites[i+1] = right
        
    return sites


# -

def tensor_trace(MPS):
    """
    Contract an MPS with itself. (TensorTrace)
    """
    
    #Use the same schema from left_contract for the first N-1 sites
    N = len(MPS)-1 
    bottom_indices = 3 * (np.arange(N)+1)
    top_indices = bottom_indices - 1
    middle_indices = bottom_indices - 2

    top_connections = [[1,2]] + [[top_indices[i], middle_indices[i+1], top_indices[i+1]] for i in range(N-1)]
    bottom_connections = [[1,3]] + [[bottom_indices[i], middle_indices[i+1], bottom_indices[i+1]] for i in range(N-1)]

    #Add the contraction with the N-th site
    last_index_bot = bottom_connections[-1][-1]
    last_index_top = top_connections[-1][-1]
    new_contraction_index = last_index_bot + 1
    full_bottom = bottom_connections + [[last_index_bot, new_contraction_index]]
    full_top = top_connections + [[last_index_top, new_contraction_index]]

    MPSconj = [np.conjugate(a) for a in MPS]
    tensor_trace = ncon(MPS + MPSconj, full_top + full_bottom)
    
    return tensor_trace


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
    
    #---Set site at @pos to center of orthogonality (mixed canonical form)---#
    if pos > 0: #left-canonization
        state = left_canonize(state[:pos]) + state[pos:]
    if pos < N-1: #right-canonization
        state = state[:pos] + right_canonize(state[pos:])

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
    
    #---Set site at @pos to center of orthogonality (mixed canonical form)---#
    if pos > 0: #left-canonization
        state = left_canonize(state[:pos+1]) + state[pos+1:]
    if pos < N-1: #right-canonization
        state = state[:pos] + right_canonize(state[pos:])
        
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
