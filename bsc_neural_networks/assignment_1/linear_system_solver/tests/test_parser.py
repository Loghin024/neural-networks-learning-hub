import unittest
from src.custom_parser import parse_equation_system as parse_equations


class TestParser(unittest.TestCase):

    def test_parse_equations(self):

        with open('../data/test_input.txt', 'w') as f:
            f.write("2x + 3y - z = 5\n")
            f.write("x - y + 4z = 6\n")
            f.write("3x + y + 2z = 7\n")

        A, B = parse_equations('../data/test_input.txt')

        expected_A = [
            [2, 3, -1],
            [1, -1, 4],
            [3, 1, 2]
        ]
        expected_B = [5, 6, 7]

        self.assertEqual(A, expected_A)
        self.assertEqual(B, expected_B)


if __name__ == '__main__':
    unittest.main()
