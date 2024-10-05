import unittest
from src.cramer import solve_system


class TestCramer(unittest.TestCase):

    def setUp(self):
        self.A = [
            [2, 3, -1],
            [1, -1, 4],
            [3, 1, 2]
        ]
        self.B = [5, 6, 7]

    def test_solve_cramers_rule(self):
        expected_solution = (5/14, 29/14, 27/14)
        self.assertEqual(solve_system(self.A, self.B), expected_solution)


if __name__ == '__main__':
    unittest.main()
