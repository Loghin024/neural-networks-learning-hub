#src/cramer.py

from .matrix_ops import determinant_3x3

def replace_column(A, column, vector):
    """
    Replace a column in a matrix with a vector
    """
    matrix = [row[:] for row in A]
    for i in range(len(matrix)):
        matrix[i][column] = vector[i]

    return matrix

def solve_system(A, B):
    """
    Solve a system of linear equations using Cramer's rule
    """
    determinant = determinant_3x3(A)

    if determinant == 0:
        print("The system of equations has no unique solution")
        return None

    Ax = replace_column(A, 0, B)
    Ay = replace_column(A, 1, B)
    Az = replace_column(A, 2, B)

    #get solutions
    x = determinant_3x3(Ax) / determinant
    y = determinant_3x3(Ay) / determinant
    z = determinant_3x3(Az) / determinant

    return x, y, z