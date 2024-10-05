from src.custom_parser import parse_equation_system

if __name__ == '__main__':
    A, B = parse_equation_system('data/input.txt')
    print(A)
    print(B)

