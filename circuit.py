# Import necessary packages
import quimb as quimb
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import library as lb

from gates import cphase_swap_qiskit, cphase_swap_quimb
from manual import apply_two_qubit_gate_full, max_bond_dimension, apply_two_qubit_gate, apply_one_qubit_gate

# +
#--- QFT - MANUAL ---
def qft_circuit_swap_full(state, N, verbosity=False):
    """
    Computes the QFT of a MPS @state with @N qubits.
    """
    
    H = np.array(quimb.hadamard())
    for pos in range(N):
        if verbosity: print("H(0)")
        state = apply_one_qubit_gate(H, 0, state)
        
        for i in range(N-pos-1):
            if verbosity: print(f"CZS(({i},{i+1}), {1/2**(i+1)})")
            state = apply_two_qubit_gate_full(cphase_swap_quimb(1/2**(i+1)), i, state) #to (i, i+1)
    
    return state

def qft_circuit_swap_approx(state, N, verbosity=False, chi=2):
    """
    Computes the QFT of a MPS @state with @N qubits.
    """
    
    H = np.array(quimb.hadamard())
    for pos in range(N):
        if verbosity: print("H(0)")
        state = apply_one_qubit_gate(H, 0, state)
        
        for i in range(N-pos-1):
            if verbosity: print(f"CZS(({i},{i+1}), {1/2**(i+1)})")
            state = apply_two_qubit_gate(cphase_swap_quimb(1/2**(i+1)), i, state, chi) #to (i, i+1)
    
    return state


# -

#---QFT - QISKIT---
def qft_circuit_qiskit(circuit, n):
    """
    Apply the rotations needed for the QFT
        Parameters:
            circuit : quantum circuit
            n       : int
                number of qubits in @circuit
    
    """
    if n == 0:
        return circuit
    elif n==1:
        circuit.h(0)
        return circuit
    
    circuit.h(0)
    for i in range(n-1):
        cphase_swap_qiskit(circuit, i, i+1, 1/2**(i+1))
        
    return qft_circuit_qiskit(circuit, n-1)

#---MPS-QISKIT INTERFACE---
class circ_data():
    """ Class to contain the metadata of a qiskit quantum circuit

    """
    def __init__(self, circ):  
        self.data = [ info for info in circ.data]
        # Gates applied in order
        self.gates = [ qub[0].name for qub in self.data ]
        self.gates_params = [ qub[0].params for qub in self.data ]
        # Indices of application of the gates
        self.indeces = [ self._to_index( qub[1])  for qub in self.data  ]
        # Number of qubits in the circuit
        self.n_qub = circ.num_qubits
        
        
    def _to_index(self, x):
        return [ y.index for y in x ]

def MPS_circ(qc, gates, init_state = None, chi=None):
    """ Function to transform a qiskit circuit @qc to a quimb MPS circuit, using the gates @gates
        Parameters:
            qc         : QuantumCircuit
            init_state : String containing the state in computational basis, i.e. '000' or '0011'
                         If None starts from empty state '0'*num_qubits
            gates      : dict qiskit_gate_name: quimb_gate
            chi        : maximum bond dimension. If None it is automatically chosen.
    """
    data = circ_data(qc)
    
    if init_state == None:
        init_state = '0' * data.n_qub
    
    MPS = quimb.tensor.MPS_computational_state(init_state, tags = 'psi0')
    
    for gate_name, qub_idx, params in zip( data.gates, data.indeces, data.gates_params):
        qubits = tuple(qub_idx)
        if len(params)==0:
            # Non parametric gates
            MPS.gate_( gates[ gate_name ], qubits, tags=gate_name, max_bond=chi, contract='split-gate') 
        else:
            # Parametric gates
            MPS.gate_( gates[ gate_name ](*params), qubits, tags=gate_name, max_bond=chi, contract='split-gate') 
    
    return MPS
