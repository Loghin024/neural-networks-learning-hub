from src.custom_parser import parse_equation_system
from src.cramer import solve_system
from src.inversion import solve_inverse_system

if __name__ == '__main__':
    A, B = parse_equation_system('data/input.txt')

    print("Solving using Cramer's Rule:")
    print(solve_system(A, B))

    print("Solving using Inversion:")
    print(solve_inverse_system(A, B))

