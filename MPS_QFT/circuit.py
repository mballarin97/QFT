# Import necessary packages
import quimb as quimb
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import library as lb

from gates import cphase_swap_qiskit, cphase_swap_quimb, GATES, cphase_and_swap_cirq
from manual import apply_two_qubit_gate_full, max_bond_dimension, apply_two_qubit_gate, apply_one_qubit_gate

# +
#--- QFT - MANUAL ---
def qft_circuit_swap_full(state, N, verbosity=False):
    """
    Computes the QFT of a MPS *state* with *N* qubits.
    
    Parameters
    ----------
    state: list of ndarray
        list of tensors of the MPS
    N: int
        number of qubits in *state*
    verbosity: bool, optional
        Default to False. If True prints applied gates
        
    Returns
    -------
    state: list of ndarray
        MPS where we have applied the QFT
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
    Computes the QFT of a MPS *state* with *N* qubits and a bond dimension *chi*.
    
    Parameters
    ----------
    state: list of ndarray
        list of tensors of the MPS
    N: int
        number of qubits in *state*
    verbosity: bool, optional
        Default to False. If True prints applied gates
    chi: int, optional
        Bond dimension
        
    Returns
    -------
    state: list of ndarray
        MPS where we have applied the QFT
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
    Apply the QFT to a qiskit quantum circuit *circuit* in a recursive way
    
    Parameters
    ----------
        circuit : quantum circuit
            quantum circuit where we want to apply the QFT
        n       : int
            number of qubits in *circuit*
    
    Returns
    -------
        None: None
            Acts in place
    """
    if n == 0:
        return circuit
    elif n==1:
        circuit.h(0)
        return circuit
    
    circuit.h(0)
    for i in range(n-1):
        cphase_swap_qiskit(circuit, i, i+1, np.pi*1/2**(i+1))
        
    return qft_circuit_qiskit(circuit, n-1)

# ---QFT - CIRQ---
def qft_circuit_swap_cirq(qubits, circuit=[]):
    """
    Build a circuit implementing the QFT algorithm on the given *qubits*. 
    The order of *qubits* is preserved by SWAP operations.
    Implemented using only local operations, i.e. gates acting on neighbouring qubits.
    Adapted from: https://github.com/quantumlib/Cirq/blob/master/examples/quantum_fourier_transform.py and extended to
    n generic qubits through recursion.
    
    Parameters
    ----------
    qubits: cirq.LineQubit
        qubits
    circuit: list
        list of directives to create the quantum circuit
        
    Returns
    -------
    circuit: cirq.Circuit
        quantum circuit where we have applied the QFT
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


# ---QFT - QUIMB---
def qft_circuit_quimb(mps, n, chi=2):
    """
        Apply the QFT to the matrix product state *mps* with a bond dimension of *chi*
        in a recursive way
        
        Parameters
        ----------
            mps : quimb.tensor.tensor_1d.MatrixProductState
                Matrix product state to which we want to apply the QFT
            n       : int
                number of qubits in *mps*
            chi : int, optional
                Bond dimension of the MPS. Default to 2
                
        Returns
        -------
            mps: quimb.tensor.tensor_1d.MatrixProductState
                Matrix product state to which we have applied the QFT
    
    """
    H = quimb.hadamard()
    if n == 0:
        return mps
    elif n==1:
        mps.gate_(H, 0, tags='H', contract=True) 
        return mps
    
    mps.gate_(H, 0, tags='H', contract=True)
    for i in range(n-1):
        mps.gate_(cphase_swap_quimb(1/2**(i+1)), (i, i+1), tags='Cphase_swap', max_bond=chi, contract='swap+split')
        
    return qft_circuit_quimb(mps, n-1)


#---MPS-QISKIT INTERFACE---
class circ_data():
    """ Class to contain the metadata of a qiskit quantum circuit
    
        Attributes
        ----------
        data: list
            data of a quantum circuit
        gates: list of string
            names of the gate applied in the correct order
        gates_params: list of lists
            gate parameters. if no parameter is present it returns an empty list []
        indeces: list of tuples of ints
            indeces to which the gates are applyed
        n_qub: int
            number of qubits in the circuit
            

    """
    def __init__(self, circ):
        """
            Initialize the class by registering the gates applied in the circuit in @self.gates, the gate parameters
            in @self.gate_params (if no parameter is present it returns an empty list []), the indeces to which the 
            gates are applyed @self.indeces and the number of qubits in the circuit @self.n_qub
        """
        self.data = [ info for info in circ.data]
        # Gates applied in order
        self.gates = [ qub[0].name for qub in self.data ]
        self.gates_params = [ qub[0].params for qub in self.data ]
        # Indices of application of the gates
        self.indeces = [ self._to_index( qub[1])  for qub in self.data  ]
        # Number of qubits in the circuit
        self.n_qub = circ.num_qubits
        
        
    def _to_index(self, x):
        """
            Unpack indeces in @x
        """
        return [ y.index for y in x ]

def MPS_circ(qc, gates = None, init_state = None, chi=None, verbosity=False):
    """ Function to transform a qiskit circuit *qc* to a quimb MPS circuit, using the gates *gates*
        
        Parameters
        ----------
            qc         : QuantumCircuit
                Qiskit quantum circuit to apply to the MPS
            gates      : dictionary or None, optional
                dictionary containing the gate used in the circuit in the format qiskit_gate_name: quimb_gate
                If None a standard set is used
            init_state : String, None or quimb.tensor.tensor_1d.MatrixProductState, optional
                If String containing the state in computational basis, i.e. '000' or '0011'
                If None starts from empty state '0'x(num_qubits)
                If MPS in quimb format the initial state is the MPS
            chi        : int, optional
                maximum bond dimension. If None it is automatically chosen.
            verbosity  : bool, optional
                if True prints when the state is initialized. Default to false.
                
        Returns
        -------
            MPS        : quimb.tensor.tensor_1d.MatrixProductState
                quimb MPS where we have applied the quantum circuit *qc* starting from *init_state*
    """
    data = circ_data(qc)
    
    if gates == None:
        gate_list = GATES()
        gates = gate_list.gate_dict
    
    if init_state == None:
        init_state = '0' * data.n_qub
        
    if type(init_state) == quimb.tensor.tensor_1d.MatrixProductState:
        if verbosity: print('--- State initialized ---')
        MPS = init_state
    else:
        MPS = quimb.tensor.MPS_computational_state(init_state, tags = 'psi0')
    
    
    for gate_name, qub_idx, params in zip( data.gates, data.indeces, data.gates_params):
        qubits = tuple(qub_idx)
        identity = np.eye(2**len(qubits))
        if len(qubits)==1:
            if len(params)==0:
                MPS.gate_( gates[ gate_name ], qubits, tags=gate_name, max_bond=chi, contract=True)
            else:
                MPS.gate_( gates[ gate_name ](*params), qubits, tags=gate_name, max_bond=chi, contract=True)
        elif len(qubits)==2:   
            if len(params)==0:
                MPS.gate_( gates[ gate_name ], qubits, tags=gate_name, max_bond=chi, contract='swap+split') 
            else:
                # Parametric gates
                MPS.gate_( gates[ gate_name ](*params), qubits, tags=gate_name, max_bond=chi, contract='swap+split') 

    return MPS


def GHZ_qiskit(circ):
    """
    Generates a GHZ state in the quantum circuit *circ* composed by n qubits. Acts in place.
    
    Parameters
    ----------
    circ: Quantum Circuit
        The quantum circuit in the state '00...0' where to build the GHZ
        
    Returns
    -------
    None: None
        Acts in place.
    """
    n = circ.num_qubits
    circ.h(0)
    for i in range(1, n):
        circ.cx( 0, i)
