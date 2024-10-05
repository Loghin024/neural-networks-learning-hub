# Linear Algebra Assignment - Neural Networks (NN2024)
 **Loghin Catalin** from 3A1

## Requirements
- use python lists as data structures for matrices and vectors
- no external libraries (e.g., NumPy)
- experiment with other values for the coefficients and free terms

## Folder Structure
project_root/ 
│ ├── data/ # directory for input files 
│     └── test_input.txt # test input file for parsing equations 
│ ├── src/ # Source code │ 
│   ├── init.py # init as a Python module 
│   ├── parser.py # parse input equations 
│   ├── matrix_ops.py # code for matrix/vector operations 
│   ├── cramer.py # code to solve using Cramer’s Rule 
│   ├── inversion.py # Code to solve using matrix inversion 
│ ├── tests/ # Unit tests 
│   ├── test_parser.py unittests for the parser  
│   ├── test_matrix_ops.py # Unit tests for matrix operations 
│   ├── test_cramer.py # Unit tests for Cramer’s Rule 
│   └── test_inversion.py # Unit tests for matrix inversion 
├── main.py # Main script to run the assignment  
├── .gitignore # Git ignore file  
└── README.md # Project documentation


## Implementation Details
1. **Parsing Equations**:
   - The `custom_parser.py` module reads equations from a text file, extracting coefficients of the variables (`x`, `y`, `z`) and constants from the right-hand side of the equations.
   
2. **Matrix and Vector Operations**:
   - The `matrix_ops.py` module implements core matrix and vector operations, including:
     - **Determinant**: Calculates the determinant of a 3x3 matrix.
     - **Trace**: Computes the trace (sum of the diagonal elements) of the matrix.
     - **Vector Norm**: Calculates the Euclidean norm of a vector.
     - **Transpose**: Returns the transpose of a matrix.
     - **Matrix-Vector Multiplication**: Multiplies a matrix by a vector.

3. **Solving Methods**:
   - **Cramer’s Rule** (`cramer.py`): Solves the system of equations using determinants for each variable (`x`, `y`, `z`).
   - **Matrix Inversion** (`inversion.py`): Solves the system by computing the inverse of the coefficient matrix and multiplying it by the constants vector.
