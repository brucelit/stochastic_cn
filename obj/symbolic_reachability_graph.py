import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import re

from sympy import Matrix

from obj.symbolic_causal_net import SymbolicCausalNet, State, Semantics, Obligation, import_symbolic_causal_net_from_xml
from collections import deque, defaultdict
from typing import Tuple

class SymbolicReachabilityGraph:
    """
    Class for generating and analyzing the reachability graph of a Causal-net.
    """

    def __init__(self, scn: SymbolicCausalNet):
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
                if probability == "":
                    probability = str(1)
                edge_label = f"{binding.activity}\np={probability}"

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

                    # if current_state_str == "final empty state":
                    #     print("current state is final")
                    #     continue

                    # Add edge between states
                    self.graph.add_edge(current_state_str, next_state_str,
                                        binding=binding,
                                        label_pos=1,
                                        activity=binding.activity,
                                        probability=probability,
                                        label=edge_label)

                except ValueError as e:
                    print(f"Error executing binding {binding}: {e}")

        return self.graph

    def get_sub_trace_probability(self, source_state, trace, trace_length):
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

            # Get the dictionary of all enabled bindings
            enabled_bindings = self.semantics.get_enabled_bindings(current_state)
            for binding, probability in enabled_bindings.items():
                # Create a new sequence by adding this binding
                if binding.activity == sub_trace[0]:
                    # Calculate the new state
                    new_state = self.semantics.execute_binding(binding, current_state)
                    new_probability = current_probability
                    if probability != "":
                        if current_probability == "1":
                            new_probability = probability
                        else:
                            new_probability += str("*")
                            new_probability += probability

                    # If the binding leads to the end activity and the state is empty, we reached the end
                    # if binding.activity == self.stochastic_causal_net.end_activity and new_state.is_empty():
                    #     continue

                    # Continue the search
                    dfs(new_state, new_probability, sub_trace[1:], depth + 1)

        # Start the search from the initial state
        dfs(source_state, "1", trace, 0)

        # If no valid binding sequences were found, return "0"
        if len(probability4sub_trace) == 0:
            return "0"

        elif len(probability4sub_trace) == 1:
            return probability4sub_trace[0]

        # If some valid binding sequences were found, return the sum
        else:
            state_prop = ""
            for sub_trace_prob in probability4sub_trace:
                print("sub trace probability: ", sub_trace_prob)
                state_prop += str(sub_trace_prob)
                state_prop += str("+")
            return state_prop[:-1]

    def generate_markovian_probability(self,
                                       markovian_slang,
                                       state2symbolic_probability,
                                       k,
                                       max_depth: int = 50):
        """
        Generate the reachability graph for the causal net.

        Args:
            max_depth: Maximum depth for state exploration

        Returns:
            NetworkX DiGraph representing the reachability graph
        """
        sub_trace_probabilities = dict.fromkeys(markovian_slang, "")

        # Start with initial state
        initial_state = self.semantics.initial_state()
        initial_state_str = self._state_to_str(initial_state)

        # Add initial state to the graph
        self.state_mapping[initial_state_str] = 0  # Initial state gets index 0

        # Use BFS to explore all reachable states
        queue = deque([(initial_state, initial_state_str, 0)])  # (state, state_str, depth)
        visited = {initial_state_str}

        state_index = 1  # Counter for state indices

        while queue:
            current_state, current_state_str, depth = queue.popleft()

            for trace, probability in markovian_slang.items():
                sub_trace_prob = self.get_sub_trace_probability(current_state, trace, k)
                if sub_trace_prob == "0":
                    continue
                if str(state2symbolic_probability[current_state_str]) == "1":
                    sub_trace_probabilities[trace] += sub_trace_prob  + str("+")
                else:
                    if sub_trace_prob == "1":
                        sub_trace_probabilities[trace] += str(state2symbolic_probability[current_state_str]) + str("+")
                    else:
                        sub_trace_probabilities[trace] += str(state2symbolic_probability[current_state_str]) + "*" + sub_trace_prob + str("+")

            # Stop if we've reached the maximum depth
            if depth >= max_depth:
                continue

            # Get all enabled bindings for current state
            enabled_bindings = self.semantics.get_enabled_bindings(current_state)

            # Process each enabled binding
            for binding, probability in enabled_bindings.items():
                if probability == "":
                    probability = str(1)
                edge_label = f"{binding.activity}\np={probability}"

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
                                        label_pos=1,
                                        activity=binding.activity,
                                        probability=probability,
                                        label=edge_label)

                except ValueError as e:
                    print(f"Error executing binding {binding}: {e}")

        for key,v in sub_trace_probabilities.items():
            if v == "":
                sub_trace_probabilities[key] = "0"

        return sub_trace_probabilities


    def visualize(self, output_file: str = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize the reachability graph.

        Args:
            output_file: Optional file path to save the visualization
            figsize: Figure size as (width, height) tuple
        """
        plt.figure(figsize=figsize)

        # Use hierarchical layout for better visualization
        pos = nx.spectral_layout(self.graph)

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


    def get_parameter_incidence_matrix(self):
        """
        Generate an incidence matrix with parameter names instead of concrete probabilities.
        This is useful when working with symbolic parameters.

        Returns:
            pandas.DataFrame: The incidence matrix with parameter names
        """
        # Get the number of states
        num_states = len(self.state_mapping)

        # Create a matrix of empty strings
        matrix = np.empty((num_states, num_states), dtype=object)
        matrix.fill("0")

        # Fill the matrix with parameter information
        for source_idx, target_idx, activity, probability in self.transitions:
            if matrix[source_idx, target_idx] != '':
                matrix[source_idx, target_idx] = probability
            # else:
            #     matrix[source_idx, target_idx] = probability
        for i in range(num_states):
            if matrix[i][i] == '0':
                matrix[i][i] = "-1"
            else:
                matrix[i][i] = matrix[i][i] + "-1"

        sympy_matrix, symbols = matrix_to_sympy(matrix.T)
        # Create reverse mapping from index to state string
        reverse_mapping = {idx: state for state, idx in self.state_mapping.items()}

        # Create a DataFrame with state labels
        df = pd.DataFrame(
            matrix,
            index=[reverse_mapping[i] for i in range(num_states)],
            columns=[reverse_mapping[i] for i in range(num_states)]
        )
        # df.to_csv("../data/rtf_ic.csv")

        # Create a numpy vector of length num_states
        vector = [0 for i in range(num_states)]
        vector[0] = -1
        b = Matrix(vector)
        result = sympy_matrix.solve(b)

        state2symbolic_probability = {}
        for i in range(len(result)):
            state2symbolic_probability[reverse_mapping[i]] = result[i]
        return df, sympy_matrix, symbols,state2symbolic_probability

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

# Example usage
def example_reachability_graph():
    # Create a simple Weighted C-net
    symbolic_cn = import_symbolic_causal_net_from_xml("../data/abbc.cnet")
    symbolic_cn.assign_parameterized_weights()

    # Create the reachability graph
    symbolic_rg = SymbolicReachabilityGraph(symbolic_cn)
    symbolic_rg.generate_reachability_graph()

    # Generate parameter incidence matrix
    param_matrix, sympy_matrix, symbols,state2symbolic_probability  = symbolic_rg.get_parameter_incidence_matrix()
    # print("Parameter Incidence Matrix:", param_matrix)
    # print("SymPy Matrix:", sympy_matrix)
    # print("Symbols:", symbols)
    #
    # # Print state mapping
    # print("\nState Mapping:")
    # for state_str, idx in symbolic_rg.state_mapping.items():
    #     print(f"State {idx}: {state_str}")

    return symbolic_rg


if __name__ == "__main__":
    symbolic_rg = example_reachability_graph()
    symbolic_rg.visualize()