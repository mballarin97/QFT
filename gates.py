#---------QUIMB-----------
# Controlled phase shift
import quimb as quimb
import numpy as np

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
             [0., np.exp(1.0j * phi * np.pi)]], dtype=dtype, sparse=sparse)
    kws = {'dtype': dtype, 'sparse': sparse}
    
    op = ((quimb.gen.operators.qu([1, 0], qtype='dop', **kws) & quimb.gen.operators.eye(2, **kws)) +
          (quimb.gen.operators.qu([0, 1], qtype='dop', **kws) & ph))
    quimb.gen.operators.make_immutable(op)
    return op



