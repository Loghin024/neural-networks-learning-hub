# src/custom_parser.py

def parse_equation(line):
    """"
    Parse a string representing an equation in the form of: ax + by + cz = d
    """
    coefficients = []

    left_side, right_side = line.split('=')

    # get the coefficients of the left side
    eq = left_side.split()

    sign = '+'
    for term in eq:
        if term in ['+', '-']:
            sign = term
        else:
            term = sign + term

            if 'x' in term:
                term = term.replace('x', '')
                if term == '+' or term == '-':
                    term += '1'
            elif 'y' in term:
                term = term.replace('y', '')
                if term == '+' or term == '-':
                    term += '1'
            elif 'z' in term:
                term = term.replace('z', '')
                if term == '+' or term == '-':
                    term += '1'
            coefficients.append(int(term))
            # print(term)
    # get the coefficients of the right side
    coefficients.append(int(right_side))

    # coefficients = [int(eq[i]) for i in range(0, len(eq) - 1)]
    return coefficients + [int(right_side)]


def parse_equation_system(path_to_file):
    """"
Parse a file containing a system of equations in the form of: ax + by + cz = d
    """
    with open(path_to_file, 'r') as f:
        lines = f.readlines()

    A = []
    B = []

    for line in lines:
        coefficients = parse_equation(line)
        A.append(coefficients[:-2])
        B.append(coefficients[-1])

    return A, B
