import csv
import time

import numba
import scipy
import numpy as np

from collections import defaultdict
from obj.symbolic_causal_net import import_symbolic_causal_net_from_xml, Semantics, \
    project_binding_sequence_to_activities
from util.stochastic_language import import_slang, compute_markov_abstraction
from util.symbolic_conversion import get_inverse_poland_expression, clean_expression
from itertools import chain



@numba.njit("float64(int16[::1], float64[::1], float64[::1])", inline='always', cache=True)
def calculate_inverse_poland_expression_numba(inverse_poland_expression, constants_dict, var_lst):
    calculate_stack = np.zeros(len(inverse_poland_expression), dtype=np.float64) # preallocate the stack
    stack_ptr = int(0)
    len_var_lst = len(var_lst)
    plus_idx = -1
    minus_idx = -2
    prod_idx = -3
    div_idx = -4
    for idx in inverse_poland_expression:
        if idx == plus_idx:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 + p1
            stack_ptr -= 1
        elif idx == minus_idx:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 - p1
            stack_ptr -= 1
        elif idx == prod_idx:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 * p1
            stack_ptr -= 1
        elif idx == div_idx:
            p2 = calculate_stack[stack_ptr - 2]
            p1 = calculate_stack[stack_ptr - 1]
            calculate_stack[stack_ptr - 2] = p2 / p1
            stack_ptr -= 1
        else:
            if idx < len_var_lst:
                val = var_lst[idx]
                calculate_stack[stack_ptr] = val
            else:
                constant_idx = idx - len_var_lst
                calculate_stack[stack_ptr] = constants_dict[constant_idx]
            stack_ptr += 1

    return calculate_stack[0]

def get_obj_func(markovian_slang, symbolic_cn, k):
    param_mapping = symbolic_cn.assign_parameterized_weights()

    # Create the semantics
    semantics = Semantics(symbolic_cn)

    # Generate all valid binding sequences
    sampled_activity_sequences = semantics.generate_activity_sequences()

    print("sampled_activity_sequences num: ", len(sampled_activity_sequences))

    total_freq4sub_trace_dict = defaultdict(lambda:"")
    covered_sub_trace = set()

    total_freq = ""

    # update the sub_trace_probabilities
    for activity_sequence, prob in sampled_activity_sequences.items():
        temp_prob = clean_expression(prob[:-1])
        # activity_sequence = project_binding_sequence_to_activities(valid_sequence)
        for i in range(len(activity_sequence) - k + 1):
            total_freq4sub_trace_dict[tuple(activity_sequence[i:i+k])] += temp_prob
            total_freq4sub_trace_dict[tuple(activity_sequence[i:i + k])] += "+"
            covered_sub_trace.add(tuple(activity_sequence[i:i+k]))

        temp_prob += "*"
        temp_prob += str(len(activity_sequence) - k + 1)
        total_freq += temp_prob
        total_freq += "+"
    total_freq = total_freq[:-1]

    covered_trace = sum(float(markovian_slang[trace]) for trace in covered_sub_trace if trace in markovian_slang.keys())
    print(f"The stochastic discovery covers {covered_trace:.2f} of the traces from the log.")

    obj2add = []
    for trace, prob in markovian_slang.items():
        if trace in covered_sub_trace:
            obj2add.append([clean_expression(total_freq4sub_trace_dict[trace]), prob])
        else:
            obj2add.append(["0", prob])
    obj2add.append([clean_expression(total_freq),0])

    # Generate the objective function
    inverse_obj2add = [
        (get_inverse_poland_expression(trace_symbolic_prob), trace_real_prob)
        for trace_symbolic_prob, trace_real_prob in obj2add
    ]

    # get the transition to weight mapping
    var_name2idx_map = {}
    var_idx2name_map = {}
    var_idx = 0
    var_lst = []
    for para_name in param_mapping.keys():
        var_lst.append(1)
        var_name2idx_map[para_name] = var_idx
        var_idx2name_map[var_idx] = para_name
        var_idx += 1

    assert len(var_name2idx_map) == max(var_name2idx_map.values()) + 1, "IDs must be continuously assigned"

    operator_indexes = {'+': -1, '-': -2, '*': -3, '/': -4}

    constant_symbols = {*chain(*(inverse_poland for inverse_poland, _ in inverse_obj2add))}
    constant_symbols = constant_symbols - var_name2idx_map.keys() - operator_indexes.keys()
    constant_symbols = list(constant_symbols)  # Put them on a list to order them

    constant_indexes = {symbol: len(var_name2idx_map) + idx for idx, symbol in enumerate(constant_symbols)}
    constants_lookup = np.array([float(symbol) for symbol in constant_symbols])

    symbol_to_idx = {**var_name2idx_map, **constant_indexes, **operator_indexes}
    inverse_obj2add = [
        ([symbol_to_idx[symbol] for symbol in inverse_poland], trace_prob)
        for inverse_poland, trace_prob in inverse_obj2add
    ]

    # Pack it into data types that are more friendly to numba
    # The most important is packing the poland expressions into a numpy array
    inverse_poland_exprs = [np.array(inverse_poland_exprs, dtype=np.int16)
                            for inverse_poland_exprs, _ in inverse_obj2add]
    inverse_poland_exprs = numba.typed.List(inverse_poland_exprs)
    trace_probs = [trace_prob for _, trace_prob in inverse_obj2add]
    trace_probs = np.array(trace_probs, dtype=np.float64)

    # Capture the variables
    def _uemsc_objective_function(x):
        return uemsc_objective_function(inverse_poland_exprs, trace_probs, constants_lookup, x)

    return _uemsc_objective_function, var_lst, var_idx2name_map, param_mapping


@numba.njit()
def uemsc_objective_function(inverse_poland_exprs, trace_probs, constants_lookup, x):
    result_lst = []
    total_markov_trace_prob = 0
    for idx, inverse_poland_expr in enumerate(inverse_poland_exprs):
        temp_lst = []
        trace_prob = trace_probs[idx]
        markov_trace_prob = calculate_inverse_poland_expression_numba(inverse_poland_expr, constants_lookup, x)
        temp_lst.append(markov_trace_prob)
        temp_lst.append(trace_prob)
        result_lst.append(temp_lst)
    total_freq = result_lst[-1][0]
    obj_func = 0
    temp_trace_prob = 0
    for i in range(len(result_lst)-1):
        total_markov_trace_prob += result_lst[i][0]
        obj_func += max(result_lst[i][1] - result_lst[i][0]/total_freq,0)
        temp_trace_prob += result_lst[i][0]
    return obj_func


# def optimize_with_basin_hopping(var_lst, end_var_lst, obj_func):
#     """
#     This function is used to optimize the objective function with basin hopping method,
#     Regarding basin hopping global optimiser, refer to https://en.wikipedia.org/wiki/Basin-hopping
#     :param var:
#     :param obj_func:
#     :return: the variable list that maximize uemsc^k measure
#     """
#     # add constraint such that every var is between 0 and 1
#     bds = [(0.0001, 1) for _ in range(len(var_lst))]
#     # for weights relevant to the artificial end, set it to the lower bound
#     for idx in end_var_lst:
#         bds[idx] = (0.0001, 0.0002)
#     print(bds)
#     # define the method and bound
#     minimizer_kwargs = {"bounds": bds}
#     # solve problem
#     res = scipy.optimize.basinhopping(obj_func,
#                                       var_lst,
#                                       minimizer_kwargs=minimizer_kwargs)
#     print("res of optimization: ", res.fun)
#     return res.x


def optimize_with_differential_evolution(var_lst, end_var_lst, obj_func):
    """
    This function is used to optimize the objective function with basin hopping method,
    Regarding basin hopping global optimiser, refer to https://en.wikipedia.org/wiki/Basin-hopping
    :param var:
    :param obj_func:
    :return: the variable list that maximize er or uemsc-based measure
    """
    # add constraint such that every var is between 0 and 1
    bds = [(0.0001, 10) for i in range(len(var_lst))]
    # for weights relevant to the artificial end, set it to the lower bound
    for idx in end_var_lst:
        bds[idx] = (0.0001, 0.0002)

    # solve problem
    res = scipy.optimize.differential_evolution(obj_func, bds)
    print("res of optimization: ", res.fun)
    return res

def optimize_with_k_th_uemsc(slang, symbolic_cn, k):
    log_markov_result = compute_markov_abstraction(slang, k)

    sum = 0
    for trace, v in log_markov_result.items():
        # print(f"log trace: {trace}, with probability: {v}")
        sum += v

    # get the objective function
    objective_function, var_lst, var_idx2name_map, param_mapping = get_obj_func(log_markov_result, symbolic_cn, k)

    end_var_lst = []
    for k2,v2 in var_idx2name_map.items():
        if param_mapping[v2][0] == "ARTIFICIAL_END":
            end_var_lst.append(k2)

    # run the optimization with basin hopping method
    result = optimize_with_differential_evolution(var_lst, end_var_lst, objective_function)
    param_result = result.x
    # return param_result

    count = 0
    # print("param result: ", param_result)
    # for i in range(len(param_result)):
    #     if param_result[i] < 0.9:
    #         count += 1
    #     print(f"wight: {param_result[i]}, Weight: {param_mapping[var_idx2name_map[i]]}")
    # print(f"Number of weights < 0.9: {count}")


def run_optimization_with_timing(slang, symbolic_cn, k_values):
    """
    Run optimization for multiple k values, track timing and results

    Args:
        slang: The stochastic language object
        symbolic_cn: The symbolic causal net
        k_values: List of k values to test

    Returns:
        List of dictionaries containing results for each k
    """
    results = []

    for k in k_values:
        print(f"Running optimization for k={k}")
        start_time = time.time()

        # Get the Markov abstraction for the given k
        log_markov_result = compute_markov_abstraction(slang, k)

        # Get the objective function
        objective_function, var_lst, var_idx2name_map, param_mapping = get_obj_func(log_markov_result, symbolic_cn, k)

        # Get the end variable list
        end_var_lst = []
        for k2, v2 in var_idx2name_map.items():
            if param_mapping[v2][0] == "ARTIFICIAL_END":
                end_var_lst.append(k2)

        # Run the optimization
        optimization_result = optimize_with_differential_evolution(var_lst, end_var_lst, objective_function)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Store results
        result_dict = {
            'k': k,
            'runtime_seconds': elapsed_time,
            'fun_value': optimization_result.fun,
            'success': optimization_result.success,
            'n_iter': optimization_result.nit,
            'params': optimization_result.x.tolist()  # Convert numpy array to list for CSV storage
        }
        results.append(result_dict)

        print(f"Completed k={k} in {elapsed_time:.2f} seconds")

    return results


def save_results_to_csv(results, filename="optimization_results.csv"):
    """
    Save optimization results to a CSV file

    Args:
        results: List of result dictionaries
        filename: Output CSV filename
    """
    # Extract all possible fields from results
    fieldnames = set()
    for result in results:
        for key in result.keys():
            if key != 'params':  # Handle parameters separately
                fieldnames.add(key)

    fieldnames = sorted(list(fieldnames))

    # Find the maximum number of parameters
    max_params = 0
    for result in results:
        if len(result['params']) > max_params:
            max_params = len(result['params'])

    # Add parameter fieldnames
    for i in range(max_params):
        fieldnames.append(f'param_{i}')

    # Write results to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row_dict = {k: v for k, v in result.items() if k != 'params'}

            # Add parameters to the row dictionary
            for i, param in enumerate(result['params']):
                row_dict[f'param_{i}'] = param

            writer.writerow(row_dict)

    print(f"Results saved to {filename}")

if __name__ == "__main__":

    # log_path = '../data/abcd.slang'
    # slang = import_slang(log_path)
    #
    # model_path = '../data/abcd.cnet'

    log_path = '../data/bpi19.slang'
    slang = import_slang(log_path)

    model_path = '../data/bpi19_hm.cnet'

    symbolic_cn = import_symbolic_causal_net_from_xml(model_path)
    symbolic_cn.assign_parameterized_weights()
    # Define k values to test
    k_values = [2, 3,4,5,6]

    # Run optimizations with timing
    results = run_optimization_with_timing(slang, symbolic_cn, k_values)

    # Save results to CSV
    save_results_to_csv(results, "bpi19_optimization_k_results.csv")
    # binding_weights = optimize_with_k_th_uemsc(slang, symbolic_cn, k)
    # scn = symbolic_cn.construct_scn(binding_weights)

#    # Export the SCN to a file
