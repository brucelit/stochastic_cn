import math

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import scipy

from obj.stochastic_causal_net import StochasticCausalNet, State, Semantics, Obligation
from collections import deque, defaultdict
from typing import Tuple
from util.scn_importer import import_scn_from_xml
from util.stochastic_language import import_slang, compute_markov_abstraction


class StochasticReachabilityGraph:
    """
    Class for generating and analyzing the reachability graph of a Causal-net.
    """

    def __init__(self, scn: StochasticCausalNet):
        """
        Initialize the reachability graph.

        Args:
            semantics: CausalNetSemantics object
        """
        self.stochastic_causal_net = scn
        self.semantics = Semantics(scn)
        self.graph = nx.DiGraph()
        self.state_mapping = {}  # Maps state strings to indices
        self.transitions = []  # List of transitions (source_state, target_state, activity, probability)

    def _state_to_str(self, state):
        """
        Convert a State object to a string representation.

        Args:
            state: State object

        Returns:
            String representation of the state
        """

        if state.is_start:
            return "initial empty state"

        if state.is_final:
            return "final empty state"

        obligations = state.get_all_obligations()
        sorted_obligations = sorted(obligations, key=lambda o: (o.source, o.target))
        return ", ".join(str(obligation) for obligation in sorted_obligations)


    def _str_to_state(self, state_str: str) -> State:
        """
        Convert a string representation back to a State object.

        Args:
            state_str: String representation of state

        Returns:
            State object
        """
        if state_str == "initial empty state":
            init_state = State()
            init_state.is_start = True
            return State()

        if state_str == "final empty state":
            fin_state = State()
            fin_state.is_final = True
            return State()

        # Parse the string representation
        state = State()
        if state_str == "{}" or state_str == "":
            return state

        # Remove braces and split by comma
        obligations_str = state_str[1:-1].split(", ")
        for obligation_str in obligations_str:
            # Parse "source→target:count"
            parts = obligation_str.split(":")
            count = int(parts[1])
            source_target = parts[0].split("→")
            source = source_target[0]
            target = source_target[1]

            obligation = Obligation(source, target)
            state.obligations[obligation] = count

        return state

    def generate_reachability_graph(self, max_depth: int = 50) -> nx.DiGraph:
        """
        Generate the reachability graph for the causal net.

        Args:
            max_depth: Maximum depth for state exploration

        Returns:
            NetworkX DiGraph representing the reachability graph
        """
        # Start with initial state
        initial_state = self.semantics.initial_state()
        initial_state_str = self._state_to_str(initial_state)

        # Add initial state to the graph
        self.graph.add_node(initial_state_str, state=initial_state, label=initial_state_str)
        self.state_mapping[initial_state_str] = 0  # Initial state gets index 0

        # Use BFS to explore all reachable states
        queue = deque([(initial_state, initial_state_str, 0)])  # (state, state_str, depth)
        visited = {initial_state_str}

        state_index = 1  # Counter for state indices

        while queue:

            current_state, current_state_str, depth = queue.popleft()
            if current_state.is_final:
                continue

            # Stop if we've reached the maximum depth
            if depth >= max_depth:
                continue

            # Get all enabled bindings for current state
            enabled_bindings = self.semantics.get_enabled_bindings(current_state)

            # Process each enabled binding
            for binding, probability in enabled_bindings.items():
                edge_label = f"{binding.activity}\np={probability:.4f}"
                # Execute the binding to get the next state
                try:
                    next_state = self.semantics.execute_binding(binding, current_state)
                    next_state_str = self._state_to_str(next_state)

                    # Add the new state to the graph if not seen before
                    if next_state_str not in visited:
                        self.graph.add_node(next_state_str, state=next_state, label=next_state_str)
                        self.state_mapping[next_state_str] = state_index
                        state_index += 1
                        visited.add(next_state_str)
                        queue.append((next_state, next_state_str, depth + 1))

                    # Store transition information
                    self.transitions.append((
                        self.state_mapping[current_state_str],
                        self.state_mapping[next_state_str],
                        binding.activity,
                        probability
                    ))

                    # Add edge between states
                    self.graph.add_edge(current_state_str, next_state_str,
                                        binding=binding,
                                        activity=binding.activity,
                                        probability=probability,
                                        label=edge_label)

                except ValueError as e:
                    print(f"Error executing binding {binding}: {e}")
        return self.graph

    def get_sub_trace_freq(self, source_state, trace, trace_length):
        """
        Generate the probability of a sub-trace from a given state

        Args:
            source_state: The state to start from
            trace: The trace to consider
            trace_length: consider sub-trace of length k
        Returns:
            A list of all valid binding sequences
        """
        probability4sub_trace = []
        def dfs(current_state, current_probability, sub_trace, depth):

            # Base case: maximum depth reached
            if depth >= trace_length:
                probability4sub_trace.append(current_probability)
                return
            if current_state.is_final:
                return
            # Get the dictionary of all enabled bindings
            enabled_bindings = self.semantics.get_enabled_bindings(current_state)
            for binding, probability in enabled_bindings.items():
                # Create a new sequence by adding this binding
                if binding.activity == sub_trace[0]:
                    # Calculate the new state
                    new_state = self.semantics.execute_binding(binding, current_state)
                    new_probability = current_probability * probability
                    dfs(new_state, new_probability, sub_trace[1:], depth + 1)

        # Start the search from the initial state
        dfs(source_state, 1, trace, 0)
        # If no valid binding sequences were found, return "0"
        if len(probability4sub_trace) == 0:
            return 0

        elif len(probability4sub_trace) == 1:
            return probability4sub_trace[0]

        # If some valid binding sequences were found, return the sum
        else:
            state_prop = 0
            for sub_trace_prob in probability4sub_trace:
                state_prop += sub_trace_prob
            return state_prop


    def get_trace_prob(self, initial_state, trace):
        """
        Generate the probability of a sub-trace from a given state

        Args:
            source_state: The state to start from
            trace: The trace to consider
            trace_length: consider sub-trace of length k
        Returns:
            A list of all valid binding sequences
        """
        probability4trace = []
        def dfs(current_state, current_probability, trace):
            if len(trace) == 0 and current_state.is_final:
                probability4trace.append(current_probability)
                return
            elif len(trace) == 0 and not current_state.is_final:
                return
            elif len(trace) > 0 and current_state.is_final:
                return
            # Get the dictionary of all enabled bindings
            enabled_bindings = self.semantics.get_enabled_bindings(current_state)
            for binding, probability in enabled_bindings.items():
                # Create a new sequence by adding this binding
                if binding.activity == trace[0]:
                    # Calculate the new state
                    new_state = self.semantics.execute_binding(binding, current_state)
                    new_probability = current_probability * probability
                    dfs(new_state, new_probability, trace[1:])
        # Start the search from the initial state
        dfs(initial_state, 1, trace)
        # If no valid binding sequences were found, return "0"

        if len(probability4trace) == 1:
            return probability4trace[0]
        elif len(probability4trace) > 1:
            print("serious mistake")
        # If some valid binding sequences were found, return the sum
        else:
            return 0

    def get_all_sub_trace_freq(self, source_state: State, trace_length: int) -> float:
        """
        Calculate the total probability of all possible sub-traces of a given length from a state.

        Args:
            source_state: The state to start from
            trace_length: Length of sub-traces to consider

        Returns:
            Total probability of all sub-traces of the given length
        """
        probability4sub_trace = []

        def dfs(current_state, current_probability, depth):
            # Base case: maximum depth reached
            if depth >= trace_length:
                probability4sub_trace.append(current_probability)
                return

            # If we reach final state before completing the sub-trace, stop this path
            if current_state.is_final and depth < trace_length:
                return

            # Get all enabled bindings for the current state
            enabled_bindings = self.semantics.get_enabled_bindings(current_state)

            for binding, probability in enabled_bindings.items():
                # Calculate the new state
                new_state = self.semantics.execute_binding(binding, current_state)
                new_probability = current_probability * probability

                # Continue the search
                dfs(new_state, new_probability, depth + 1)

        # Start the search from the given state
        dfs(source_state, 1.0, 0)

        # If no valid binding sequences were found, return 0
        if len(probability4sub_trace) == 0:
            return 0.0
        elif len(probability4sub_trace) == 1:
            return probability4sub_trace[0]
        else:
            # Return the sum of all probabilities
            return sum(probability4sub_trace)


    def generate_markovian_frequency(self,
                                   markovian_slang,
                                   state2probability,
                                   k,
                                   max_depth: int = 60):
        """
        Generate Markovian frequencies for a set of sub-traces.

        Args:
            markovian_slang: Dictionary mapping sub-traces to initial values
            state2probability: Dictionary mapping state strings to probabilities
            k: Length of sub-traces to consider
            max_depth: Maximum depth for state exploration

        Returns:
            Dictionary mapping sub-traces to their frequencies
        """
        # Convert tuple keys to lists for easier handling
        sub_trace_probabilities = {trace: 0.0 for trace in markovian_slang}
        total_freq = 0.0

        # Start with initial state
        initial_state = self.semantics.initial_state()
        initial_state_str = self._state_to_str(initial_state)

        # Use BFS to explore all reachable states
        queue = deque([(initial_state, initial_state_str, 0)])  # (state, state_str, depth)
        visited = {initial_state_str}

        while queue:
            current_state, current_state_str, depth = queue.popleft()

            # Get the total frequency of all sub-traces of length k from current-state
            state_prob = state2probability.get(current_state_str)
            temp_total_freq = self.get_all_sub_trace_freq(current_state, k) * state_prob
            total_freq += temp_total_freq

            # Calculate probability for each trace in markovian_slang
            for trace in markovian_slang:
                sub_trace_prob = self.get_sub_trace_freq(current_state, trace, k)
                if sub_trace_prob > 0:
                    sub_trace_probabilities[trace] += sub_trace_prob * state_prob

            # Stop if we've reached the maximum depth
            if depth >= max_depth:
                continue

            # Get all enabled bindings for current state
            enabled_bindings = self.semantics.get_enabled_bindings(current_state)

            # Process each enabled binding
            for binding, probability in enabled_bindings.items():
                # Execute the binding to get the next state
                try:
                    next_state = self.semantics.execute_binding(binding, current_state)
                    next_state_str = self._state_to_str(next_state)

                    # Add the new state to the graph if not seen before
                    if next_state_str not in visited:
                        visited.add(next_state_str)
                        queue.append((next_state, next_state_str, depth + 1))

                except ValueError as e:
                    print(f"Error executing binding {binding}: {e}")

        # Normalize the probabilities if total_freq > 0
        if total_freq > 0:
            for trace in sub_trace_probabilities:
                sub_trace_probabilities[trace] /= total_freq

        return sub_trace_probabilities, total_freq


    def get_state_probability_vector(self):
        """
        Generate an incidence matrix of the Sc-net's state space
        Returns:
            state2probability： A dictionary mapping state strings to their probabilities
        """
        # Get the number of states
        num_states = len(self.state_mapping)
        # Create a matrix of empty strings
        # matrix = np.empty((num_states, num_states), dtype=object)
        # matrix.fill(0)
        matrix = [[0.0 for _ in range(num_states)] for _ in range(num_states)]
        # Fill the matrix with parameter information
        for source_idx, target_idx, activity, probability in self.transitions:
            matrix[source_idx][target_idx] = probability
            # matrix[target_idx][source_idx] = probability

        for i in range(num_states):
            matrix[i][i] -= 1.0

        # Create a numpy vector of length num_states
        vector = [0 for i in range(num_states)]
        vector[0] =-1
        # b = np.array(vector)
        reverse_mapping = {idx: state for state, idx in self.state_mapping.items()}

        probability_solution_vector = scipy.linalg.solve(matrix, vector,transposed=True)
        # probability_solution_vector = numpy.linalg.solve(matrix, vector)

        state2probability = {}
        for i in range(len(probability_solution_vector)):
            state2probability[reverse_mapping[i]] = probability_solution_vector[i]

        return state2probability


    def visualize(self, output_file: str = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize the reachability graph.

        Args:
            output_file: Optional file path to save the visualization
            figsize: Figure size as (width, height) tuple
        """
        plt.figure(figsize=figsize)
        # Use hierarchical layout for better visualization
        pos = nx.spring_layout(self.graph)

        # Create node labels
        node_labels = {node: data.get('label', node) for node, data in self.graph.nodes(data=True)}

        # Draw nodes - highlight initial and final states
        initial_node = self._state_to_str(self.semantics.initial_state())
        final_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree == 0]

        # Draw regular nodes
        regular_nodes = [node for node in self.graph.nodes()
                         if node != initial_node and node not in final_nodes]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=regular_nodes,
                               node_color='lightblue', node_size=1500, alpha=0.8)

        # Draw initial node
        if initial_node in self.graph.nodes():
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[initial_node],
                                   node_color='green', node_size=1500, alpha=0.8)

        # Draw final nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=final_nodes,
                               node_color='orange', node_size=1500, alpha=0.8)

        # Draw edges with labels
        nx.draw_networkx_edges(self.graph, pos, width=3.0, node_size=1000, arrowsize=15, alpha=0.7)

        # Create edge labels
        edge_labels = {(u, v): data.get('label') for u, v, data in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10, font_weight='bold')

        plt.title("Causal Net Reachability Graph")
        plt.axis('off')

        # if output_file:
        #     plt.savefig(output_file, bbox_inches='tight', dpi=300)
        #     print(f"Saved visualization to {output_file}")

        plt.show()


if __name__ == "__main__":
    log_path = '../data/domestic.slang'
    # Load the data
    slang = import_slang(log_path)
    # k = 6
    # markovian_slang = compute_markov_abstraction(slang, k)

    scn = import_scn_from_xml("../data/domestic_hm.cnet")
    activity_num = len(scn.activities)
    stochastic_rg = StochasticReachabilityGraph(scn)
    # visualize the reachability graph
    stochastic_rg.generate_reachability_graph(max_depth=30)
    stochastic_rg.visualize()

    uemsc = 1
    ER = 0
    model_trace_sum = 0
    log_trace_sum = 0
    j1 = 0
    total_cost = 0

    for k, v in slang.items():
        # get the prob of trace according to model
        model_trace_prob = stochastic_rg.get_trace_prob(stochastic_rg.semantics.initial_state(), k)
        # print(f"Trace: {len(k)}, Probability: {v}, Model Trace Probability: {model_trace_prob}")
        uemsc -= max(v - model_trace_prob, 0)
        if model_trace_prob >0:
            print("Trace: ", k, "Model Trace Probability: ", model_trace_prob)
            model_trace_sum += model_trace_prob
            log_trace_sum += v
            temp_sum = v + model_trace_prob
            j1 += v*math.log2(2*v/temp_sum) + model_trace_prob*math.log2(2*model_trace_prob/temp_sum)
            total_cost += -v * math.log2(model_trace_prob)
        else:
            total_cost += v * (1 + len(k)) * math.log2(1+activity_num)

    ER += -model_trace_sum*math.log2(model_trace_sum) -(1-model_trace_sum)*math.log2(1-model_trace_sum)
    ER += total_cost
    # state2probability = reachability_graph.get_state_probability_vector()
    # sub_trace_frequencies, total_freq = reachability_graph.generate_markovian_frequency(markovian_slang, state2probability, k)
    # print("model_trace_sum: ", model_trace_sum)
    # print("log_trace_sum: ", float(log_trace_sum), "type: ", type(log_trace_sum))
    temp_j = float(log_trace_sum)
    temp_jssc =(j1 + 1.0 - model_trace_sum + 1.0 - temp_j)/2.0
    jssc = math.sqrt(temp_jssc)

    second_markovian_slang = compute_markov_abstraction(slang, 2)
    state2probability = stochastic_rg.get_state_probability_vector()
    sub_trace_frequencies, total_freq = stochastic_rg.generate_markovian_frequency(second_markovian_slang, state2probability,2)
    second_uemsc = 1
    model_sub_trace_sum =0
    for k, v in sub_trace_frequencies.items():
        model_sub_trace_sum += v
        second_uemsc -= max(second_markovian_slang[k] - v, 0)
    print("model_sub_trace_sum: ", model_sub_trace_sum)
    print("second_uemsc: ", second_uemsc)

    third_markovian_slang = compute_markov_abstraction(slang, 3)
    state2probability = stochastic_rg.get_state_probability_vector()
    print("state 2 prob: ", state2probability)
    sub_trace_frequencies, total_freq = stochastic_rg.generate_markovian_frequency(third_markovian_slang, state2probability,3)
    third_uemsc = 1
    model_sub_trace_sum =0
    for k, v in sub_trace_frequencies.items():
        model_sub_trace_sum += v
        third_uemsc -= max(third_markovian_slang[k] - v, 0)
    print("model_sub_trace_sum: ", model_sub_trace_sum)
    print("third_uemsc: ", third_uemsc)
    print("uemsc: ", uemsc)
    print("jssc: ", 1.0 - jssc)
    print("ER: ", ER)

