import logging
import math

import numpy as np
import numba
import scipy

from obj.stochastic_causal_net import StochasticCausalNet
from obj.symbolic_causal_net import import_symbolic_causal_net_from_xml
from obj.symbolic_reachability_graph import SymbolicReachabilityGraph
from util.stochastic_language import import_slang, compute_markov_abstraction
from util.symbolic_conversion import get_inverse_poland_expression, clean_expression
from itertools import chain

import warnings
warnings.filterwarnings('ignore')

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

def get_obj_func(markovian_slang, symbolic_cn):
    param_mapping = symbolic_cn.assign_parameterized_weights()
    # for key, value in param_mapping.items():
    #     print(f"parameter mapping: {key}, with value: {value}")

    # Create the reachability graph
    symbolic_rg = SymbolicReachabilityGraph(symbolic_cn)
    symbolic_rg.generate_reachability_graph()

    # Generate parameter incidence matrix
    sympy_matrix, symbols, state2symbolic_probability= symbolic_rg.get_parameter_incidence_matrix()

    # Iterate each state in the symbolic
    sub_trace_frequencies, total_freq = symbolic_rg.generate_markovian_frequency(markovian_slang, state2symbolic_probability,2)

    # for k, v in sub_trace_frequencies.items():
    #     print(f"Trace: {k}, Probability: {v}")
    #
    # print("total_freq:", total_freq)

    obj2add = []
    for k, v in sub_trace_frequencies.items():
        sub_obj = [clean_expression(v), markovian_slang[k]]
        obj2add.append(sub_obj)
    obj2add.append([clean_expression(total_freq),0])

    print("The total probability of the log: ", clean_expression(total_freq))
    covered_trace = sum(float(sublist[1]) for sublist in obj2add if sublist[0] != "0")
    if len(obj2add) == 0:
        logging.warning("No traces fit the model, the stochastic discovery will fail. "
                        "Please check the log and the model.")
    else:
        print(f"The stochastic discovery covers {covered_trace:.2f} of the traces from the log.")

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

    # The trace probabilities are packed into a numpy array
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

    total_trace = 0
    for idx, inverse_poland_expr in enumerate(inverse_poland_exprs):
        temp_lst = []
        trace_prob = trace_probs[idx]
        markov_trace_prob = calculate_inverse_poland_expression_numba(inverse_poland_expr, constants_lookup, x)
        temp_lst.append(markov_trace_prob)
        temp_lst.append(trace_prob)
        total_trace += trace_prob
        result_lst.append(temp_lst)
    total_freq = result_lst[-1][0]
    obj_func = 0
    for i in range(len(result_lst)-1):
        total_markov_trace_prob += result_lst[i][0]
        obj_func += max(result_lst[i][1] - result_lst[i][0]/total_freq,0)
    print("res of optimization: ", obj_func,"\n")
    return obj_func



def optimize_with_basin_hopping(var_lst, end_var_lst, obj_func):
    """
    This function is used to optimize the objective function with basin hopping method,
    Regarding basin hopping global optimiser, refer to https://en.wikipedia.org/wiki/Basin-hopping
    :param var:
    :param obj_func:
    :return: the variable list that maximize uemsc^k measure
    """
    # add constraint such that every var is between 0 and 1
    bds = [(0.0001, 1) for _ in range(len(var_lst))]

    # for weights relevant to the artificial end, set it to the lower bound
    for idx in end_var_lst:
        bds[idx] = (0.0001, 0.0002)
    print(bds)
    # define the method and bound
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bds}
    # solve problem
    res = scipy.optimize.basinhopping(obj_func,
                                      var_lst,
                                      minimizer_kwargs=minimizer_kwargs)
    print("res of optimization: ", res.fun)
    return res.x


def optimize_with_dual_differential_evolution(var_lst, end_var_lst, obj_func):
    """
    This function is used to optimize the objective function with basin hopping method,
    Regarding basin hopping global optimiser, refer to https://en.wikipedia.org/wiki/Basin-hopping
    :param var:
    :param obj_func:
    :return: the variable list that maximize er or uemsc-based measure
    """
    # add constraint such that every var is between 0 and 1
    bds = [(0.0001, 10) for i in range(len(var_lst))]
    print("end_var_lst: ", end_var_lst)

    # for weights relevant to the artificial end, set it to the lower bound
    for idx in end_var_lst:
        bds[idx] = (0.0001, 0.0002)
    print(bds)

    # solve problem
    res = scipy.optimize.differential_evolution(obj_func, bds)
    print("res of optimization: ", res.fun)
    return res.x


def optimize_with_k_th_uemsc(slang, symbolic_cn, k):
    markov_result = compute_markov_abstraction(slang, k)
    # for k, v in markov_result.items():
    #     # print(f"log trace of length: {k}, with probability: {v}")
    #     sum += v
    # print(f"Total probability of the log: {sum}")

    # get the objective function
    objective_function, var_lst, var_idx2name_map, param_mapping  = get_obj_func(markov_result, symbolic_cn)

    # get the idx of var that corresponds to "ARTIFICIAL_END"
    end_var_lst = []
    for k2,v2 in var_idx2name_map.items():
        if param_mapping[v2][0] == "ARTIFICIAL_END":
            print(f"var_idx2name_map: {k2}, param_mapping: {v2}")
            end_var_lst.append(k2)

    # run the optimization with basin hopping method
    param_result = optimize_with_basin_hopping(var_lst, end_var_lst, objective_function)
    # param_result = optimize_with_direct(var_lst, end_var_lst, objective_function)
    # param_result = optimize_with_dual_annealing(var_lst, end_var_lst, objective_function)
    # param_result = optimize_with_dual_differential_evolution(var_lst, end_var_lst, objective_function)
    # return param_result

    count = 0
    print("param result: ", param_result)
    for i in range(len(param_result)):
        if param_result[i] < 0.9:
            count += 1
        print(f"wight: {param_result[i]}, Weight: {param_mapping[var_idx2name_map[i]]}")
    print(f"Number of weights < 0.9: {count}")

    stochastic_cn = StochasticCausalNet(symbolic_cn.start_activity,symbolic_cn.end_activity)

    # Add start and end activities to the set of activities
    stochastic_cn.activities = symbolic_cn.activities.copy()

    # Input and output bindings for each activity
    # A binding is a set of activities that can be executed together
    # Each binding is stored as a frozenset to allow it to be used as a dictionary key
    stochastic_cn.input_bindings = symbolic_cn.input_bindings.copy()
    stochastic_cn.output_bindings = symbolic_cn.output_bindings.copy()

    # Weights for input and output bindings
    # Dictionary structure: {activity: {binding_frozenset: weight}}
    # Default weight is 1.0 for all bindings
    stochastic_cn.input_binding_weights = symbolic_cn.input_binding_weights.copy()
    stochastic_cn.output_binding_weights = symbolic_cn.output_binding_weights.copy()

    # Initialize empty bindings for start and end activities
    # Start activity has no input bindings
    stochastic_cn.input_bindings[stochastic_cn.start_activity].add(frozenset())
    # End activity has no output bindings
    stochastic_cn.output_bindings[stochastic_cn.end_activity].add(frozenset())

    # return binding_weights
    #   assign the weight value to the binding

    # scn = symbolic_cn.construct_scn(binding_weights)
    # scn.export_to_file("binding_scn.xml")


if __name__ == "__main__":

    # log_path = '../data/abcd.slang'
    # slang = import_slang(log_path)
    #
    # model_path = '../data/abcd.cnet'

    log_path = '../data/bcde1.slang'
    slang = import_slang(log_path)
    print(compute_markov_abstraction(slang, 2))
    # model_path = '../data/bcde.cnet'
    # symbolic_cn = import_symbolic_causal_net_from_xml(model_path)
    # symbolic_cn.assign_parameterized_weights()
    # k = 2
    #
    # optimize_with_k_th_uemsc(slang, symbolic_cn, k)
    # binding_weights = optimize_with_k_th_uemsc(slang, symbolic_cn, k)
    # scn = symbolic_cn.construct_scn(binding_weights)

#    # Export the SCN to a file
