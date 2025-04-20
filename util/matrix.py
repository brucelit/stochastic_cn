import sympy as sp
import re


def matrix_to_sympy(matrix):
    """
    Convert a matrix with string expressions to a SymPy Matrix.

    Parameters:
    - matrix: List of lists containing string expressions

    Returns:
    - SymPy Matrix
    - Dictionary of created symbols
    """
    # Create a dictionary to store all symbols
    symbols_dict = {}

    # Helper function to create symbols on demand
    def get_symbol(name):
        if name not in symbols_dict:
            symbols_dict[name] = sp.Symbol(name)
        return symbols_dict[name]

    # Process the matrix
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    # Create a result matrix
    result = sp.zeros(rows, cols)

    # Fill in the result matrix
    for i in range(rows):
        for j in range(cols):
            cell = matrix[i][j]

            # If the cell is a simple number
            if cell == '0':
                result[i, j] = 0
                continue
            elif cell == '1':
                result[i, j] = 1
                continue

            # Find and create all variables in the expression
            var_pattern = r'([oi]\d+)'
            variables = re.findall(var_pattern, cell)

            for var in variables:
                get_symbol(var)

            # Create a namespace with all symbols for evaluation
            namespace = {var: get_symbol(var) for var in variables}

            try:
                # Use sympy.sympify to convert the string expression to a SymPy expression
                result[i, j] = sp.sympify(cell, locals=namespace)
            except Exception as e:
                print(f"Error parsing expression '{cell}' at position [{i}][{j}]: {e}")
                result[i, j] = 0

    return result, symbols_dict


# Example usage:
if __name__ == "__main__":
    example_matrix = [
        ['0', '1', '0', '0', '0', '0', '0'],
        ['0', '0', '1', '0', '0', '0', '0'],
        ['0', '0', '0', 'o0/(o0+o1)', 'o1/(o0+o1)', '0', '0'],
        ['0', '0', '0', 'o0/(o0+o1)', 'o1/(o0+o1)', '0', '0'],
        ['0', '0', '0', '0', '0', '1', '0'],
        ['0', '0', '0', '0', '0', '0', '1'],
        ['0', '0', '0', '0', '0', '0', '0']
    ]

    sympy_matrix, symbols = matrix_to_sympy(example_matrix)

    print("Created symbols:")
    for name, symbol in symbols.items():
        print(f"{name}: {symbol}")

    print("\nSymPy matrix:")
    print(sympy_matrix)