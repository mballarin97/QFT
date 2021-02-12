#Test for canonizing MPS
#Tested functions: manual.left_canonize, manual.right_canonize

#Allow imports from parent folder
import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import quimb

from manual import left_canonize, right_canonize
from checks import check_left_canonization, check_right_canonization
from test_mps import new_random_state

from helper import to_full_MPS, to_approx_MPS

#Physical dimension of a site (2 for qubits)
d = 2

#Range of system's sizes to be tested
Nmin = 2
Nmax = 10

chi_min = 2
chi_max = 10

def test_left_canonize():
    """Test if left_canonize works as expected"""
    
    for n in range(Nmin, Nmax+1):
        random_state = new_random_state(d, n)
        
        rand_mps = []
        MPS_full = to_full_MPS(random_state, n, d) #just for the shapes
        
        for shape in [a.shape for a in MPS_full]: #Generate an unnormalized mps (since MPS_full would be already in left-canonical form by construction)
            rand_mps.append(np.random.rand(*shape)) 
        
        MPS_leftcanonical = left_canonize(rand_mps)
        
        assert check_left_canonization(MPS_leftcanonical[:-1]), "left_canonize error"
        
def test_right_canonize():
    """Test if right_canonize works as expected"""

    for n in range(Nmin, Nmax+1):
        random_state = new_random_state(d, n)

        MPS_full = to_full_MPS(random_state, n, d) #this is in left-canonical form
        
        MPS_rightcanonical = right_canonize(MPS_full)
        
        assert check_right_canonization(MPS_rightcanonical[1:]), "right_canonize error"
        
def test_mixed_canonize():
    """Test if left_canonize and right_canonize work as expected when used together"""
    
    for n in range(Nmin, Nmax+1):
        random_state = new_random_state(d, n)
        rand_mps = []
        MPS_full = to_full_MPS(random_state, n, d) #just for the shapes
        
        for shape in [a.shape for a in MPS_full]: #Generate an unnormalized mps (since MPS_full would be already in left-canonical form by construction)
            rand_mps.append(np.random.rand(*shape)) 
            
        MPS_mixedcanonical = rand_mps
        for pos in range(n): 
            #set the site at pos as center of orthogonality
            if pos > 0:
                MPS_mixedcanonical = left_canonize(MPS_mixedcanonical[:pos+1]) + MPS_mixedcanonical[pos+1:]
            if pos < n-1:
                MPS_mixedcanonical = MPS_mixedcanonical[:pos] + right_canonize(MPS_mixedcanonical[pos:])

            if pos > 0:
                assert check_left_canonization(MPS_mixedcanonical[:pos+1]), "Error in mixed left"
            if pos < n-1:
                assert check_right_canonization(MPS_mixedcanonical[pos:]), "Error in mixed right"
            
    
    