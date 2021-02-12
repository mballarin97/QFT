#Test for the QFT results
#Tested functions: circuit.qft_circuit_swap_full, circuit.qft_circuit_swap_approx

#Allow imports from parent folder
import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from circuit import qft_circuit_swap_full, qft_circuit_swap_approx
from helper import to_full_MPS, to_approx_MPS, to_dense

import numpy as np
from numpy import linalg as LA
import cirq

from test_mps import new_random_state

#Physical dimension of a site (2 for qubits)
d = 2

#Range of system's sizes to be tested
Nmin = 2
Nmax = 10

def cphase_and_swap_cirq(ctrl, target, phase):
    """
    Combine the CPHASE gate with the SWAP operation, used for QFT algorithm.
    Input
        crtl: 
        target:
        phase:
    Return
        cirq blabla
    """
    yield cirq.CZ(ctrl, target) ** phase
    yield cirq.SWAP(ctrl, target)


def qft_circuit_swap_cirq(qubits, circuit=[]):
    """
    Build a circuit implementing the QFT algorithm on the given @qubits. 
    The order of @qubits is preserved by SWAP operations.
    Implemented using only local operations, i.e. gates acting on neighbouring qubits.
    Adapted from: https://github.com/quantumlib/Cirq/blob/master/examples/quantum_fourier_transform.py and extended to
    n generic qubits through recursion.
    """
    n = len(qubits)
    assert n > 0, "Number of qubits must be > 0"
    
    if (n == 1):
        circuit.append(cirq.H(qubits[0]))
        return cirq.Circuit(circuit, strategy=cirq.InsertStrategy.EARLIEST)
    else:
        circuit.append(cirq.H(qubits[0]))
        circuit.extend(cphase_and_swap_cirq(qubits[i], qubits[i+1], 1/2**(i+1)) for i in range(n-1))
        return qft_circuit_swap_cirq(qubits[:n-1], circuit)
    
def test_random_states():
    """Apply the QFT to random states and compare the (exact) result with that of Cirq"""
    
    for n in range(Nmin, Nmax + 1):
        random_state = new_random_state(d, n)
        
        #Manual
        MPS_random = to_full_MPS(random_state, n, d)
        manual_qft = to_dense(qft_circuit_swap_full(MPS_random, n)).flatten()
        
        #Cirq
        qubits = cirq.LineQubit.range(n)
        circuit = qft_circuit_swap_cirq(qubits, [])
        
        simulator = cirq.Simulator()
        cirq_qft = simulator.simulate(circuit, initial_state = random_state).final_state_vector
        
        assert np.isclose(LA.norm(manual_qft - cirq_qft), 0., atol=1e-6), "QFT mismatch"
    

def test_ghz():
    """Apply the QFT to ghz states, for which a (approx9 chi=2 should still give the exact result"""
    
    for n in range(Nmin, Nmax + 1):
        ghz = np.zeros(d**n, dtype=float)
        ghz[0] = 1
        ghz[-1] = 1
        ghz = ghz / np.sqrt(2)
        
        #Manual
        MPS_ghz = to_approx_MPS(ghz, n, d, chi=2)
        manual_qft = to_dense(qft_circuit_swap_approx(MPS_ghz, n, chi=2)).flatten()
        
        #Cirq
        qubits = cirq.LineQubit.range(n)
        circuit = qft_circuit_swap_cirq(qubits, [])
        
        simulator = cirq.Simulator()
        cirq_qft = simulator.simulate(circuit, initial_state = ghz).final_state_vector

        assert np.isclose(LA.norm(manual_qft - cirq_qft), 0., atol=1e-6), "Approx QFT mismatch"
        
        