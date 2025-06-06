import numba
import numpy as np
import re

# Define operator indices
PLUS_IDX = -1
MINUS_IDX = -2
PROD_IDX = -3
DIV_IDX = -4
UNARY_MINUS_IDX = -5


@numba.njit("float64(int16[::1], float64[::1], float64[::1])", inline='always', cache=True)
def calculate_inverse_poland_expression_numba(inverse_poland_expression, constants_dict, var_lst):
    calculate_stack = np.zeros(len(inverse_poland_expression), dtype=np.float64)
    stack_ptr = int(0)
    len_var_lst = len(var_lst)

    for idx in inverse_poland_expression:
        if idx == PLUS_IDX:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 + p1
            stack_ptr -= 1
        elif idx == MINUS_IDX:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 - p1
            stack_ptr -= 1
        elif idx == PROD_IDX:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 * p1
            stack_ptr -= 1
        elif idx == DIV_IDX:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 / p1
            stack_ptr -= 1
        elif idx == UNARY_MINUS_IDX:
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 1] = -p1
        else:
            if idx < len_var_lst:
                val = var_lst[idx]
                calculate_stack[stack_ptr] = val
            else:
                constant_idx = idx - len_var_lst
                calculate_stack[stack_ptr] = constants_dict[constant_idx]
            stack_ptr += 1

    return calculate_stack[0]


def tokenize_expression(exp):
    """Tokenize expression handling negative numbers properly"""
    tokens = []
    i = 0

    while i < len(exp):
        # Skip whitespace
        while i < len(exp) and exp[i] == ' ':
            i += 1
        if i >= len(exp):
            break

        # Handle numbers (including negative numbers)
        if exp[i].isdigit() or (exp[i] == '-' and i + 1 < len(exp) and exp[i + 1].isdigit() and
                                (i == 0 or exp[i - 1] in '(+*/,-')):
            num = ""
            if exp[i] == '-':
                num += exp[i]
                i += 1
            while i < len(exp) and (exp[i].isdigit() or exp[i] == '.'):
                num += exp[i]
                i += 1
            tokens.append(num)
        # Handle variables
        elif exp[i].isalpha():
            var = ""
            while i < len(exp) and (exp[i].isalnum() or exp[i] == '_'):
                var += exp[i]
                i += 1
            tokens.append(var)
        # Handle operators and parentheses
        elif exp[i] in '+-*/()':
            tokens.append(exp[i])
            i += 1
        else:
            i += 1

    return tokens


def get_inverse_poland_expression(exp):
    """Improved RPN conversion with better negative number handling"""
    if exp is None:
        return None
    tokens = tokenize_expression(exp)
    operator_stack = []
    output = []

    # Operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    for i, token in enumerate(tokens):
        if token not in '+-*/()':
            # Numbers and variables
            output.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            if operator_stack:
                operator_stack.pop()  # Remove the '('
        elif token in '+-*/':
            # Check if it's a unary minus
            if (token == '-' and
                    (i == 0 or tokens[i - 1] in '(+-*/')):
                # This is a unary minus - handle by making next number negative
                # We already handled this in tokenization
                continue
            else:
                # Binary operator
                while (operator_stack and
                       operator_stack[-1] != '(' and
                       operator_stack[-1] in precedence and
                       precedence[operator_stack[-1]] >= precedence[token]):
                    output.append(operator_stack.pop())
                operator_stack.append(token)

    # Pop remaining operators
    while operator_stack:
        output.append(operator_stack.pop())

    return output


def parse_expression_to_numeric_indices(expression, var_names):
    """Improved parsing with better constants handling"""
    # Get the RPN tokens
    rpn_tokens = get_inverse_poland_expression(expression)
    print("RPN tokens:", rpn_tokens)

    # Create a mapping from variable names to indices
    var_map = {name: idx for idx, name in enumerate(var_names)}

    # Collect constants first - handle negative numbers properly
    constants = []
    for token in rpn_tokens:
        if token not in ['+', '-', '*', '/'] and token not in var_map:
            try:
                const_value = float(token)
                if const_value not in constants:
                    constants.append(const_value)
            except ValueError:
                pass

    print("Constants found:", constants)

    # Create constants mapping using string representation
    const_map = {}
    for i, const in enumerate(constants):
        # Handle both string representations
        const_map[str(const)] = i
        if const == int(const):
            const_map[str(int(const))] = i

    print("Constants map:", const_map)

    # Convert tokens to numeric indices
    numeric_indices = []
    for token in rpn_tokens:
        if token == '+':
            numeric_indices.append(PLUS_IDX)
        elif token == '-':
            numeric_indices.append(MINUS_IDX)
        elif token == '*':
            numeric_indices.append(PROD_IDX)
        elif token == '/':
            numeric_indices.append(DIV_IDX)
        elif token in var_map:
            numeric_indices.append(var_map[token])
        else:
            # Handle constants
            try:
                const_value = float(token)
                # Find the index in constants array
                const_idx = constants.index(const_value)
                numeric_indices.append(len(var_names) + const_idx)
            except (ValueError, IndexError):
                raise ValueError(f"Unknown token: {token}")

    return np.array(numeric_indices, dtype=np.int16), np.array(constants, dtype=np.float64)


def evaluate_expression(expression, var_values):
    """Improved evaluation function"""
    print("Expression:", expression)
    print("Variable values:", var_values)

    # Extract all variable names from the expression
    var_names = sorted(var_values.keys())

    # Convert expression to numeric indices and extract constants
    numeric_indices, constants = parse_expression_to_numeric_indices(expression, var_names)

    print("Numeric indices:", numeric_indices)
    print("Constants:", constants)

    # Convert var_values to a numpy array in the correct order
    var_array = np.array([var_values[name] for name in var_names], dtype=np.float64)
    print("Variable array:", var_array)

    # Evaluate using the numba function
    result = calculate_inverse_poland_expression_numba(numeric_indices, constants, var_array)

    return result


def clean_expression(expr):
    """Clean expression by removing unnecessary multiplications by 1"""
    # Remove *1 and 1* patterns but be careful with variables
    cleaned = re.sub(r'\*1(?![0-9])', '', expr)  # Remove *1 not followed by digit
    cleaned = re.sub(r'(?<![0-9])1\*', '', cleaned)  # Remove 1* not preceded by digit
    return cleaned


def preprocess_expression(expr):
    """Simple preprocessing to handle -1* patterns"""
    # Replace -1* with (0-1)*
    expr = re.sub(r'\(-1\*', '(0-1*', expr)
    expr = re.sub(r'^-1\*', '0-1*', expr)
    expr = re.sub(r'([+\-*/\(])-1\*', r'\g<1>0-1*', expr)

    # Remove unnecessary *1 multiplications
    expr = re.sub(r'\*1(?![0-9])', '', expr)
    expr = re.sub(r'(?<![0-9])1\*(?![0-9])', '', expr)

    return expr


def evaluate_expression_simple(expression, var_values):
    """Simple evaluation using preprocessing"""
    print("Original expression:", expression)

    # Preprocess to handle negative numbers
    processed_expr = preprocess_expression(expression)
    print("Processed expression:", processed_expr)

    # Use the original (fixed) parsing logic
    var_names = sorted(var_values.keys())

    # Get RPN using your original function (without the reversal bug)
    rpn_tokens = get_inverse_poland_expression(processed_expr)
    print("RPN tokens:", rpn_tokens)

    # Create mappings
    var_map = {name: idx for idx, name in enumerate(var_names)}

    # Collect and map constants
    constants = []
    for token in rpn_tokens:
        if token not in ['+', '-', '*', '/'] and token not in var_map:
            try:
                const_value = float(token)
                if const_value not in constants:
                    constants.append(const_value)
            except ValueError:
                pass

    # Convert to indices
    numeric_indices = []
    for token in rpn_tokens:
        if token == '+':
            numeric_indices.append(PLUS_IDX)
        elif token == '-':
            numeric_indices.append(MINUS_IDX)
        elif token == '*':
            numeric_indices.append(PROD_IDX)
        elif token == '/':
            numeric_indices.append(DIV_IDX)
        elif token in var_map:
            numeric_indices.append(var_map[token])
        else:
            try:
                const_value = float(token)
                const_idx = constants.index(const_value)
                numeric_indices.append(len(var_names) + const_idx)
            except (ValueError, IndexError):
                raise ValueError(f"Unknown token: {token}")

    # Convert to numpy arrays
    numeric_indices = np.array(numeric_indices, dtype=np.int16)
    constants = np.array(constants, dtype=np.float64)
    var_array = np.array([var_values[name] for name in var_names], dtype=np.float64)

    print("Numeric indices:", numeric_indices)
    print("Constants:", constants)

    # Evaluate
    result = calculate_inverse_poland_expression_numba(numeric_indices, constants, var_array)
    return result


# Test the improved version
if __name__ == "__main__":
    # Define your complex expression
    expression = "((-1*w24/(w24+w25)*1*w17/(w16+w17)*1*w23/(w21+w22+w23) - 1*w25/(w24+w25)*1*w17/(w16+w17)*1*w23/(w21+w22+w23))/(1*w21/(w21+w22+w23)*1*w32/(w31+w32) - 1))+((-1*w24/(w24+w25)*1*w17/(w16+w17)*1*w21/(w21+w22+w23)*1*w31/(w31+w32) - 1*w25/(w24+w25)*1*w17/(w16+w17)*1*w21/(w21+w22+w23)*1*w31/(w31+w32))/(1*w21/(w21+w22+w23)*1*w32/(w31+w32) - 1))"

    # Define variable values
    var_values = {
        'w15': 1.0, 'w16': 1.0, 'w17': 1.0, 'w18': 1.0, 'w19': 1.0, 'w20': 1.0,
        'w21': 1.0, 'w22': 1.0, 'w23': 1.0, 'w24': 1.0, 'w25': 1.0, 'w26': 1.0,
        'w27': 1.0, 'w28': 1.0, 'w29': 1.0, 'w30': 1.0, 'w31': 1.0, 'w32': 1.0
    }

    print("=== IMPROVED VERSION ===")

    # Test with a simpler expression first
    simple_expr = "-1*w24"
    print(f"\nTesting simple expression: {simple_expr}")
    try:
        simple_result = evaluate_expression_simple(simple_expr, {'w24': 1.0})
        print(f"Simple result: {simple_result}")
        print(f"Expected: -1.0")
        print(f"Match: {abs(simple_result - (-1.0)) < 1e-10}")
    except Exception as e:
        print("Simple expression error:", e)

    # Now test the complex expression
    print(f"\nTesting complex expression...")
    try:
        result = evaluate_expression_simple(expression, var_values)
        print("Complex result:", result)
        print("Expected result: 0.3")
        print("Match:", abs(result - 0.3) < 1e-10)
    except Exception as e:
        print("Error:", e)
        import traceback

        traceback.print_exc()