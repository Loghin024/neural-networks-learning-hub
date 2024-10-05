import unittest
from src.matrix_ops import determinant_3x3, trace, vector_norm, transpose_of_matrix, matrix_vector_multiplication


class TestMatrixOps(unittest.TestCase):

    def setUp(self):
        self.A = [
            [2, 3, -1],
            [1, -1, 4],
            [3, 1, 2]
        ]
        self.B = [5, 6, 7]

    def test_determinant(self):
        self.assertEqual(determinant_3x3(self.A), 14)

    def test_trace(self):
        self.assertEqual(trace(self.A), 3)

    def test_vector_norm(self):
        self.assertAlmostEqual(vector_norm(self.B), 10.4880884817, places=5)

    def test_transpose(self):
        expected_transpose = [
            [2, 1, 3],
            [3, -1, 1],
            [-1, 4, 2]
        ]
        self.assertEqual(transpose_of_matrix(self.A), expected_transpose)

    def test_matrix_vector_multiplication(self):
        expected_result = [21, 27, 35]
        self.assertEqual(matrix_vector_multiplication(self.A, self.B), expected_result)


if __name__ == '__main__':
    unittest.main()
