# QFT
Final project for the course of Quantum Information and Computing for the Master's Degree in Physics of Data @ Università degli Studi di Padova. 

The main results are shown in \(3\) Jupyter Notebooks, saved in the R-Markdown (`.Rmd`) format. To use them instead of the `.ipynb` files install jupytext as follows:
```
pip install jupytext
jupyter notebook --generate-config
```
The notebooks are the following:
1. `01_cirq-Qiskit_QFT.Rmd` contains implementations for the Quantum Fourier Transform (QFT) circuit in the `Cirq` and `Qiskit` libraries for quantum computing.
2. `02_ManualMPS_QFT.Rmd` shows how to convert quantum states from a dense representation to the Matrix Product States with Open Boundary Conditions (MPS-OBC) representation "manually", i.e. without using external libraries for tensor networks. Functions for applying 1 or 2-qubit quantum gates to an MPS are provided. This allows applying the QFT to MPS using Time Evolving Block Decimation (TEBD) methods. The result is then compared to that obtained from the tensor library `quimb`.
3. `03_Qiskit2MPS.Rmd` contains a custom interface that can convert simple quantum circuits programmed in `Qiskit` to their `Quimb` analogues, so that they can be immediately applied to MPS. We use again a QFT implementation as an example for such procedure.
4. `04_CorrectnessCheck.Rmd`. To check the correctness of the code, two different initial states are chosen, the GHZ and a randomone, with a fixed number of qubits N=7. We then apply the QFT and compare the results from `quimb` and our manual implementation.
5. `05_PerformanceCheck.Rmd`. We check the efficiency of the code, comparing `quimb`, `qiskit`, `cirq` and our implementation.

<!-- TODO: Add some more comments/docs in the notebooks. They serve as examples. -->

The actual code for the functions invoked in the notebooks can be imported from standalone `.py` files:
- `helper.py` contains the code for “manually” converting between dense and MPS representation for quantum states, for printing them, and some miscellaneous functions.
- `manual.py` contains the code for “manually” applying 1 or 2-qubit quantum gates to MPS, and for changing their gauge to left/right-canonical form.
- `gates.py` contains the code for commonly used quantum gates in several libraries (`Quimb` and `Qiskit`).
- `circuit.py` contains the implementations of the QFT with the “manual” functions, and for `Qiskit` and `Quimb`. It also includes the interface between `Qiskit` and `Quimb`.
- `checks.py` contains a few functions for testing.

<!-- TODO: Maybe we can improve the code subdivision -->

All the main functions can be automatically tested by running the command `pytest` in the main project folder.

Extensive documentation for all the code is available at `docs/Documentation/index.html` or at this [link](https://mballarin97.github.io/QFT/).
