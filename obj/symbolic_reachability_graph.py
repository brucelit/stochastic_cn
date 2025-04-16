from collections import deque
import networkx as nx
import numpy as np
import pandas as pd

from symbolic_causal_net import SymbolicCausalNet, State, Semantics, Obligation, import_symbolic_causal_net_from_xml


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
        if state.is_empty():
            return "∅"

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
        if state_str == "∅":
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

                    if current_state_str == "∅":
                        continue

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

    def generate_incidence_matrix(self):
        """
        Generate the incidence matrix for the reachability graph.

        The incidence matrix shows the transitions between states:
        - Rows represent source states
        - Columns represent target states
        - Cells contain the activity and probability of the transition

        Returns:
            pandas.DataFrame containing the incidence matrix
        """
        # Get the number of states
        num_states = len(self.state_mapping)

        # Create a matrix of empty strings
        matrix = np.empty((num_states, num_states), dtype=object)
        matrix.fill('')

        # Fill the matrix with transition information
        for source_idx, target_idx, activity, probability in self.transitions:
            if matrix[source_idx, target_idx] == '':
                matrix[source_idx, target_idx] = probability
            else:
                matrix[source_idx, target_idx] += probability

        # Create reverse mapping from index to state string
        reverse_mapping = {idx: state for state, idx in self.state_mapping.items()}

        # Create a DataFrame with state labels
        df = pd.DataFrame(
            matrix,
            index=[reverse_mapping[i] for i in range(num_states)],
            columns=[reverse_mapping[i] for i in range(num_states)]
        )

        return df

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
            if matrix[source_idx, target_idx] == '':
                matrix[source_idx, target_idx] = probability
            else:
                matrix[source_idx, target_idx] = probability

        print("matrix", matrix)

        # Create reverse mapping from index to state string
        reverse_mapping = {idx: state for state, idx in self.state_mapping.items()}

        # Create a DataFrame with state labels
        df = pd.DataFrame(
            matrix,
            index=[reverse_mapping[i] for i in range(num_states)],
            columns=[reverse_mapping[i] for i in range(num_states)]
        )

        return df

# Example usage
def example_reachability_graph():
    # Create a simple Weighted C-net
    symbolic_cn = import_symbolic_causal_net_from_xml("../data/abbc.cnet")
    symbolic_cn.assign_parameterized_weights()

    # Create the reachability graph
    symbolic_rg = SymbolicReachabilityGraph(symbolic_cn)
    symbolic_rg.generate_reachability_graph()


    # Generate incidence matrix
    incidence_matrix = symbolic_rg.generate_incidence_matrix()
    print("\nIncidence Matrix (with activities and probabilities):")
    print(incidence_matrix)
    incidence_matrix.to_csv("../data/incidence.csv")

    # Generate parameter incidence matrix
    param_matrix = symbolic_rg.get_parameter_incidence_matrix()
    print("\nParameter Incidence Matrix:")
    print(param_matrix)

    # Print state mapping
    print("\nState Mapping:")
    for state_str, idx in symbolic_rg.state_mapping.items():
        print(f"State {idx}: {state_str}")

    return symbolic_rg


if __name__ == "__main__":
    example_reachability_graph()