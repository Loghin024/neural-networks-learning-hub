#src/matrix_ops.py

def determinant_3x3(matrix):
    """
    Calculate the determinant of a 3x3 matrix
    """
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h

def trace(matrix):
    """
    Calculate the trace of a matrix
    """
    return sum(matrix[i][i] for i in range(len(matrix)))

def vector_norm(vector):
    """
    Calculate the norm of a vector
    """
    return sum(b**2 for b in vector)**0.5

def transpose_of_matrix(matrix):
    """
    Calculate the transpose of a matrix
    """
    return [[matrix[j][i] for j in range(len(matrix[0]))] for i in range(len(matrix))]

def matrix_vector_multiplication(matrix, vector):
    """
    Multiply a matrix by a vector
    """
    return [sum(matrix[i][j] * vector[j] for j in range(len(matrix))) for i in range(len(vector))]