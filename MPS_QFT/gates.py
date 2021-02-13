# Import necessary packages
import quimb as quimb
import cirq as cirq
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import library as lb
# +
#---------QUIMB-----------
# Controlled phase shift
def CPHASE(phi, dtype=complex, sparse=False):
    """Construct a controlled phase gate for two qubits.

    Parameters
    ----------
    phi : double
        Angle of rotation.
    sparse : bool, optional
        Whether to construct a sparse operator.

    Returns
    -------
    C : immutable operator
        The controlled two-qubit gate operator.
    """
    ph = quimb.gen.operators.qu([[1., 0.],
             [0., np.exp(1.0j * phi )]], dtype=dtype, sparse=sparse)
    kws = {'dtype': dtype, 'sparse': sparse}
    
    op = ((quimb.gen.operators.qu([1, 0], qtype='dop', **kws) & quimb.gen.operators.eye(2, **kws)) +
          (quimb.gen.operators.qu([0, 1], qtype='dop', **kws) & ph))
    quimb.gen.operators.make_immutable(op)
    return op

#controlled phase + swap gate
def cphase_swap_quimb(phase):
    cphase = np.array(quimb.controlled('z')) ** phase
    swap = np.array(quimb.swap())
    
    return swap @ cphase



# -

class GATES():
    """
        Class to contain the various gates in quimb format
        
        Attributes
        ----------
        H: matrix shape(2, 2)
            Hadamard gate
        NOT: matrix shape(2, 2)
            Not gate
        CNOT: matrix shape(4, 4)
            controlled not gate
        SWAP: matrix shape(4, 4)
            swap gate
        CPHASE(phi): function
            returns the controlled phase matrix with phase *phi*
        gate_dict: dictionary
            dictionary of the format qiskit_gate_name : quimb_gate
        
    """
    def __init__(self):
        """
            Initialize the various gates from quimb and create a dictionary with the format
            qiskit_gate_name : quimb_gate
        """
        # Controlled NOT
        self.CNOT = quimb.controlled('not')
        # Hadamard
        self.H = quimb.gen.operators.hadamard()
        # NOT
        self.NOT = quimb.gen.operators.pauli('x')
        # SWAP
        self.SWAP = quimb.gen.operators.swap()
        # CPHASE
        self.CPHASE = CPHASE
        
        self.gate_dict = {'h': self.H, 
                         'cx': self.CNOT,
                         'cp': self.CPHASE,
                         'swap': self.SWAP,
                         'x' : self.NOT}


#---QISKIT---
#Controlled phase + swap gate
def cphase_swap_qiskit(circuit, control, target, phase):
    """
        Apply to a quantum circuit *circuit* the cphase and swap gate. Acts in place.
        
        Parameters
        ----------
        circuit: Quantum Circuit
            Qiskit quantum circuit 
        control: int
            Index of the control qubit for the controlled phase
        target: int 
            Index of the target qubit for the controlled phase
        phase: double
            Phase to apply in the controlled phase in radiants
        
        Returns
        -------
        None: None
            Acts in place
    """
    circuit.cp(phase, control, target)
    circuit.swap(control, target)


# ------- CIRQ -------
def cphase_and_swap_cirq(ctrl, target, phase):
    """
    Combine the CPHASE gate with the SWAP operation, used for QFT algorithm.
    Input
        crtl: cirq.devices.line_qubit.LineQubit
            control qubit
        target: cirq.devices.line_qubit.LineQubit
            target qubit
        phase: double
            phase to apply in radiants
    Return
        cirq gate
    """
    yield cirq.CZ(ctrl, target) ** phase
    yield cirq.SWAP(ctrl, target)
