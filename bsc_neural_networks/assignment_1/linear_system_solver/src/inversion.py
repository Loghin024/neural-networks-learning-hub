#src/inversion.py

from .matrix_ops import determinant_3x3, transpose_of_matrix, matrix_vector_multiplication, determinant_2x2

def minor(A, row, col):
    """
    Get the minor of a matrix
    """
    return [[A[i][j] for j in range(len(A)) if j != col] for i in range(len(A)) if i != row]

def cofactor(A):
    """
    Get the cofactor of a matrix
    """
    return [[(-1)**(i+j)* determinant_2x2(minor(A, i, j)) for j in range(len(A))] for i in range(len(A))]

def inverse(A):
    """
    Calculate the inverse of a matrix
    """
    det_A = determinant_3x3(A)

    if det_A == 0:
        print("The matrix is not invertible\n The determinant is zero :(((")
        return None

    adj_A = transpose_of_matrix(cofactor(A))
    return [[adj_A[i][j] / det_A for j in range(len(A))] for i in range(len(A))]

def solve_inverse_system(A, B):
    """
    Solve a system of linear equations using matrix inversion
    """
    A_inv = inverse(A)

    if A_inv is None:
        return None

    return matrix_vector_multiplication(A_inv, B)