# Test the conversion between Dense and MPS representation
# Tested functions: helper.to_full_MPS, helper.to_approx_MPS, helper.to_dense

#Allow imports from parent folder
import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from numpy import linalg as LA
import quimb #For gates

from helper import to_full_MPS, to_approx_MPS, to_dense
from manual import tensor_trace, max_bond_dimension
from checks import check_left_canonization

#Physical dimension of a site (2 for qubits)
d = 2

#Range of system's sizes to be tested
Nmin = 2
Nmax = 10

chi_min = 2
chi_max = 10

def new_random_state(d, n):
    """
    Generate a random (normalized) state in dense representation for a system of *n* *d*-dimensional sites.
    
    Parameters
    ----------
    n: int
        Number of degrees of freedom
    d: int
        Local dimension of the single degree of freedom
        
    Returns
    -------
    random_state: numpy array shape (d^n)
        Normalized dense random state of *n* degrees of freedom of dimension *d*
    """
    random_state =  np.random.rand(d ** n)
    random_state =  random_state + np.random.rand(d ** n) * 1j #Add also some random imaginary values
    random_state /= LA.norm(random_state) #Normalize
    
    return random_state

def test_random_states():
    """Generate random states of n qubits, convert them to MPS and back to dense, and see if the results match with the originals."""
    
    for n in range(Nmin, Nmax + 1):
        random_state =  new_random_state(d, n)
        
        MPS = to_full_MPS(random_state, n, d)
        
        reconstructed = to_dense(MPS).flatten()
        
        fidelity = np.abs(np.dot(random_state, np.conjugate(reconstructed))) ** 2
        
        assert np.isclose(fidelity, 1.), "Error in reconstruction"

def test_left_canonical():
    """Test if MPS are in left-canonical form"""
    
    for n in range(Nmin, Nmax + 1):
        random_state = new_random_state(d, n)
        
        MPS_full = to_full_MPS(random_state, n, d)
        
        assert check_left_canonization(MPS_full), "Newly created MPS is not left-canonical"
        
        #Check that tracing the whole MPS == tracing just the rightmost site (i.e. the rightmost site is the center of orthogonality)
        global_trace = tensor_trace(MPS_full)
        local_trace  = np.trace(MPS_full[-1] @ np.conjugate(MPS_full[-1].T))
        
        assert np.isclose(global_trace, local_trace), "Mismatch in global/local traces"
        
        for chi in range(chi_min, chi_max):
            MPS_approx = to_approx_MPS(random_state, n, d, chi=chi)
            
            assert check_left_canonization(MPS_approx), "Newly created MPS (approx) is not left-canonical"
            
            global_trace = tensor_trace(MPS_approx)
            local_trace  = np.trace(MPS_approx[-1] @ np.conjugate(MPS_approx[-1].T))
            
            assert np.isclose(global_trace, local_trace), "Mismatch in global/local traces (for approx MPS)"


def test_with_quimb_mps():
    """Convert a random state with the manual algorithm and with quimb. Check if they are compatible with each other"""
    
    #left_canonization does not fix completely the gauge, so the two representation may differ by unitary transformations in the bonds
    #however, the trace at the center of orthogonality should be the same, because it is == to the trace of the entire network
    
    for n in range(Nmin, Nmax + 1):
        random_state = new_random_state(d, n)
        
        ket = quimb.qu(random_state, qtype='ket')
        MPS_quimb = quimb.tensor.MatrixProductState.from_dense(ket, dims=[d]*n)
        MPS_quimb.left_canonize()
        
        MPS_manual = to_full_MPS(random_state, n, d)
        
        manual_trace = np.trace(MPS_manual[-1] @ np.conjugate(MPS_manual[-1].T))
        quimb_trace = np.trace(MPS_quimb[-1].data @ np.conjugate(MPS_quimb[-1].data.T))
        
        assert np.isclose(manual_trace, quimb_trace), "Traces differ with those of Quimb"

def test_bond_dimension():
    """The maximum bond dimension should be d**(np.floor(N/2))"""
    
    for n in range(Nmin, Nmax + 1):
        random_state = new_random_state(d, n)
        
        MPS_manual = to_full_MPS(random_state, n, d)
        
        max_bond = max_bond_dimension(MPS_manual)
        
        assert max_bond == d**(np.floor(n/2)), "Error in max bond dimension"

def test_approx_error():
    """For a sufficiently high chi, the approximation error should go to 0"""
    
    for n in range(Nmin, Nmax + 1):
        random_state = new_random_state(d, n)
        
        chi_high = int(d ** (np.floor(n/2)))
        
        MPS_approx = to_approx_MPS(random_state, n, d, chi=chi_high)
        
        reconstructed = to_dense(MPS_approx).flatten()
        
        assert np.isclose(LA.norm(reconstructed - random_state), 0.), "Error in approximation with sufficiently high chi"

def test_ghz_state():
    """Test with the ghz state, for which a chi=2 dimension should suffice for full precision"""
    
    for n in range(Nmin, Nmax + 1):
        ghz = np.zeros(d**n, dtype=float)
        ghz[0] = 1
        ghz[-1] = 1
        ghz = ghz / np.sqrt(2)
        
        MPS_approx = to_approx_MPS(ghz, n, d, chi=2)
        reconstructed = to_dense(MPS_approx).flatten()
        
        assert np.isclose(LA.norm(reconstructed - ghz), 0.), "Error in ghz representation"
