import csv
import logging
import math
import time
import numpy as np
import numba
import scipy

from obj.stochastic_reachability_graph import StochasticReachabilityGraph
from obj.symbolic_causal_net import import_symbolic_causal_net_from_xml, convert_symbolic_to_stochastic
from obj.symbolic_reachability_graph import SymbolicReachabilityGraph
from util.log_util import get_stochastic_language
from util.stochastic_language import import_slang, compute_markov_abstraction
from util.symbolic_conversion import get_inverse_poland_expression
from itertools import chain
from pm4py.objects.log.importer.xes import importer as xes_importer
import warnings
warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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

def get_obj_func(markovian_slang, symbolic_cn,k):
    param_mapping = symbolic_cn.assign_parameterized_weights()
    # for key, value in param_mapping.items():
    #     print(f"parameter mapping: {key}, with value: {value}")

    # Create the reachability graph
    symbolic_rg = SymbolicReachabilityGraph(symbolic_cn)
    symbolic_rg.generate_reachability_graph()

    # Generate parameter incidence matrix
    sympy_matrix, symbols, state2symbolic_probability= symbolic_rg.get_parameter_incidence_matrix()

    # for state, prob in state2symbolic_probability.items():
    #     print(f"state: {state}, symbolic probability: {prob}")

    # Iterate each state in the symbolic
    sub_trace_frequencies, total_freq = symbolic_rg.generate_markovian_frequency(markovian_slang, state2symbolic_probability,k)

    obj2add = []
    trace_idx = 0
    for k, v in sub_trace_frequencies.items():
        sub_obj = [v, markovian_slang[k]]
        trace_idx+=1
        obj2add.append(sub_obj)
    obj2add.append([total_freq,0])

    # print("The total probability of the log: ", clean_expression(total_freq))
    covered_trace = sum(float(sublist[1]) for sublist in obj2add if sublist[0] != "0")
    if len(obj2add) == 0:
        logging.warning("No traces fit the model, the stochastic discovery will fail. "
                        "Please check the log and the model.")
    else:
        print(f"The stochastic discovery covers {covered_trace:.2f} of the traces from the log.")

    # # Generate the objective function
    inverse_obj2add = [
        (get_inverse_poland_expression(trace_symbolic_prob), trace_real_prob)
        for trace_symbolic_prob, trace_real_prob in obj2add
    ]


    # # get the transition to weight mapping
    var_name2idx_map = {}
    var_idx2name_map = {}
    var_idx = 0
    var_lst = []
    for para_name in param_mapping.keys():
        var_lst.append(1)
        var_name2idx_map[para_name] = var_idx
        var_idx2name_map[var_idx] = para_name
        var_idx += 1

    #
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

    for idx, inverse_poland_expr in enumerate(inverse_poland_exprs):
        temp_lst = []
        trace_prob = trace_probs[idx]
        markov_trace_prob = calculate_inverse_poland_expression_numba(inverse_poland_expr, constants_lookup, x)
        temp_lst.append(markov_trace_prob)
        temp_lst.append(trace_prob)
        result_lst.append(temp_lst)
    total_freq = result_lst[-1][0]
    obj_func = 0
    for i in range(len(result_lst)-1):
        total_markov_trace_prob += result_lst[i][0]
        obj_func += max(result_lst[i][1] - result_lst[i][0]/total_freq,0)
    # print("total markov: ", total_markov_trace_prob,
    #       "total_freq: ",  total_freq,
    #       "res of opt: ", obj_func)
    return obj_func



def optimize_with_basin_hopping(var_lst, obj_func):
    """
    This function is used to optimize the objective function with basin hopping method,
    Regarding basin hopping global optimiser, refer to https://en.wikipedia.org/wiki/Basin-hopping
    :param var:
    :param obj_func:
    :return: the variable list that maximize uemsc^k measure
    """
    # add constraint such that every var is between 0 and 1
    bds = [(0.0001, 10) for _ in range(len(var_lst))]

    # for weights relevant to the artificial end, set it to the lower bound
    # for idx in end_var_lst:
    #     bds[idx] = (0.0001, 0.0002)
    # define the method and bound
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bds}
    # solve problem
    res = scipy.optimize.basinhopping(obj_func,
                                      var_lst,minimizer_kwargs=minimizer_kwargs)
    # print("res of optimization: ", res.fun)
    # print("res of weights: ", res.x)
    return res.x


def optimize_with_k_th_uemsc(slang, symbolic_cn, k):
    markovian_slang = compute_markov_abstraction(slang, k)

    # get the objective function
    objective_function, var_lst, var_idx2name_map, param_mapping  = get_obj_func(markovian_slang, symbolic_cn,k)
    # run the optimization with basin hopping method
    param_result = optimize_with_basin_hopping(var_lst, objective_function)

    # save the results to a cnet file
    scn = convert_symbolic_to_stochastic(symbolic_cn,param_result,param_mapping, var_idx2name_map)
    # print_scn_info(scn)
    # export_to_sc_net(scn, "../data/weighted_bcde.cnet")

    activity_num = len(scn.activities)
    stochastic_rg = StochasticReachabilityGraph(scn)
    stochastic_rg.generate_reachability_graph(max_depth=50)
    uemsc = 1
    ER = 0
    model_trace_sum = 0
    log_trace_sum = 0
    j1 = 0
    total_cost = 0
    for k, v in slang.items():
        # get the prob of trace according to model
        model_trace_prob = stochastic_rg.get_trace_prob(stochastic_rg.semantics.initial_state(), k)
        uemsc -= max(v - model_trace_prob, 0)
        if model_trace_prob > 0:
            model_trace_sum += model_trace_prob
            log_trace_sum += v
            temp_sum = v + model_trace_prob
            j1 += v * math.log2(2 * v / temp_sum) + model_trace_prob * math.log2(2 * model_trace_prob / temp_sum)
            total_cost += -v * math.log2(model_trace_prob)
        else:
            total_cost += v * (1 + len(k)) * math.log2(1 + activity_num)
    if model_trace_sum != 0 and model_trace_sum !=1:
        ER += -model_trace_sum * math.log2(model_trace_sum) - (1 - model_trace_sum) * math.log2(1 - model_trace_sum)
    ER += total_cost
    temp_j = float(log_trace_sum)
    temp_jssc = (j1 + 1.0 - model_trace_sum + 1.0 - temp_j) / 2.0
    jssc = math.sqrt(temp_jssc)
    second_markovian_slang = compute_markov_abstraction(slang, 2)
    state2probability = stochastic_rg.get_state_probability_vector()
    sub_trace_frequencies, total_freq = stochastic_rg.generate_markovian_frequency(second_markovian_slang, state2probability, 2)
    second_uemsc = 1
    model_sub_trace_sum = 0
    for k, v in second_markovian_slang.items():
        model_sub_trace_sum += sub_trace_frequencies[k]
        second_uemsc -= max(v - sub_trace_frequencies[k], 0)
    print("second order uemsc: ", second_uemsc)

    third_markovian_slang = compute_markov_abstraction(slang, 3)
    state2probability = stochastic_rg.get_state_probability_vector()
    sub_trace_frequencies, total_freq = stochastic_rg.generate_markovian_frequency(third_markovian_slang,state2probability, 3)
    third_uemsc = 1
    model_sub_trace_sum = 0
    for k, v in third_markovian_slang.items():
        model_sub_trace_sum += sub_trace_frequencies[k]
        second_uemsc -= max(v - sub_trace_frequencies[k], 0)
    print("third order uemsc: ", third_uemsc)
    print("uemsc: ", float(uemsc))
    print("jssc: ", 1.0 - jssc)
    print("ER: ", ER)


if __name__ == "__main__":
    # Define file paths
    log_path = '../data/application.xes'
    model_path = '../data/application_hm.cnet'

    # Load the data
    log = xes_importer.apply(log_path)
    slang = get_stochastic_language(log)
    symbolic_cn = import_symbolic_causal_net_from_xml(model_path)
    k = 2
    start_time = time.time()
    optimize_with_k_th_uemsc(slang, symbolic_cn, k)
    print("length:", k, ' Optimization completed in {:.2f} seconds'.format(time.time() - start_time))
    # Save results to CSV
    # save_results_to_csv(results, "international_freq013_optimization_k_results.csv")