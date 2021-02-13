import numpy as np
from helper import left_contract, right_contract

def check_left_canonization(states):
    """
    Check if *states* is in left-canonical form.
    
    Parameters
    ----------
    states : list of N ndarrays
        The N leftmost tensors in the MPS-OBC representation of a state of an n-body. 
        *states*[0] should be an order-2 tensor, and all the others should be order-3 tensors, such that
        *states*[i].shape[-1] == *states*[i+1].shape[0] (i.e. they can be contracted "in line").
    
    Returns
    -------
    flag : Boolean
        True if *states* are in a left-canonical form, False otherwise.
    """
    
    N = len(states)
    
    for k in range(1,N):
        rho = left_contract(states[:k])
        
        if not np.allclose(rho, np.eye(len(rho))):
            return False
        
    return True    

def check_right_canonization(states):
    """
    Check if *states* are in right-canonical form.
    
    Parameters
    ----------
    states : list of N ndarrays
        The N rightmost tensors in the MPS-OBC representation of a state of an n-body. 
        *states*[-1] should be an order-2 tensor, and all the others should be order-3 tensors, such that
        *states*[i].shape[-1] == *states*[i+1].shape[0] (i.e. they can be contracted "in line").
    
    Returns
    -------
    flag : Boolean
        True if *states* are in a right-canonical form, False otherwise.
    """
    
    N = len(states)
    
    for k in range(1,N):
        rho = right_contract(states[-k:])
        
        if not np.allclose(rho, np.eye(len(rho))):
            return False
        
    return True    
