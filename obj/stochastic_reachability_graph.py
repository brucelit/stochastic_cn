# """
# Causal Net Reachability Graph Generator
#
# This module constructs and visualizes the complete reachability graph (state space)
# for a weighted causal net based on the obligations semantics.
# """
# import networkx as nx
# import matplotlib.pyplot as plt
#
# from collections import deque
# from typing import Dict, Tuple, Any
# from stochastic_causal_net import StochasticCausalNet, State, Semantics, Obligation
# from util.scn_importer import import_scn_from_xml
#
#
# class ReachabilityGraph:
#     """
#     Class for generating and visualizing the reachability graph of a causal net.
#     """
#
#     def __init__(self, scn: StochasticCausalNet):
#         """
#         Initialize with a causal net.
#
#         Args:
#             causal_net: CausalNet object representing the process model
#         """
#         self.stochastic_causal_net = scn
#         self.semantics = Semantics(scn)
#         self.graph = nx.DiGraph()  # Directed graph to represent the reachability graph
#
#     def _state_to_str(self, state: State) -> str:
#         """
#         Convert a state to a string representation that can be used as a node identifier.
#
#         Args:
#             state: State object with obligations
#
#         Returns:
#             String representation of the state
#         """
#         if state.is_empty():
#             return "∅"
#
#         # Sort obligations for consistent representation
#         obligations = []
#         for obligation, count in sorted(state.obligations.items(),
#                                         key=lambda x: (x[0].source, x[0].target)):
#             obligations.append(f"{obligation.source}→{obligation.target}:{count}")
#
#         return "{" + ", ".join(obligations) + "}"
#
#     def _str_to_state(self, state_str: str) -> State:
#         """
#         Convert a string representation back to a State object.
#
#         Args:
#             state_str: String representation of state
#
#         Returns:
#             State object
#         """
#         if state_str == "∅":
#             return State()
#
#         # Parse the string representation
#         state = State()
#         if state_str == "{}" or state_str == "":
#             return state
#
#         # Remove braces and split by comma
#         obligations_str = state_str[1:-1].split(", ")
#         for obligation_str in obligations_str:
#             # Parse "source→target:count"
#             parts = obligation_str.split(":")
#             count = int(parts[1])
#             source_target = parts[0].split("→")
#             source = source_target[0]
#             target = source_target[1]
#
#             obligation = Obligation(source, target)
#             state.obligations[obligation] = count
#
#         return state
#
#     def generate_reachability_graph(self, max_depth: int = 50) -> nx.DiGraph:
#         """
#         Generate the reachability graph for the causal net.
#
#         Args:
#             max_depth: Maximum depth for state exploration
#
#         Returns:
#             NetworkX DiGraph representing the reachability graph
#         """
#         # Start with initial state
#         initial_state = self.semantics.initial_state()
#         initial_state_str = self._state_to_str(initial_state)
#
#
#         # Add initial state to the graph
#         self.graph.add_node(initial_state_str, state=initial_state, label=initial_state_str)
#
#         # Use BFS to explore all reachable states
#         queue = deque([(initial_state, initial_state_str, 0)])  # (state, state_str, depth)
#         visited = {initial_state_str}
#
#         while queue:
#             current_state, current_state_str, depth = queue.popleft()
#
#             # Stop if we've reached the maximum depth
#             if depth >= max_depth:
#                 continue
#
#             # Get all enabled bindings for current state
#             enabled_bindings = self.semantics.get_enabled_bindings(current_state)
#
#             # Process each enabled binding
#             for binding, probability in enabled_bindings.items():
#                 edge_label = f"{binding.activity}\np={probability:.2f}"
#
#                 # Execute the binding to get the next state
#                 try:
#                     next_state = self.semantics.execute_binding(binding, current_state)
#                     next_state_str = self._state_to_str(next_state)
#
#                     # Add the new state to the graph if not seen before
#                     if next_state_str not in visited:
#                         self.graph.add_node(next_state_str, state=next_state, label=next_state_str)
#                         visited.add(next_state_str)
#                         queue.append((next_state, next_state_str, depth + 1))
#
#                     if current_state_str == "∅":
#                         print("Warning: reach the final state already")
#                         continue
#
#                     # Add edge between states
#                     self.graph.add_edge(current_state_str, next_state_str,
#                                         binding=binding,
#                                         label_pos=1,
#                                         activity=binding.activity,
#                                         probability=probability,
#                                         label=edge_label)
#
#                 except ValueError as e:
#                     print(f"Error executing binding {binding}: {e}")
#         return self.graph
#
#     def visualize(self, output_file: str = None, fig_size: Tuple[int, int] = (12, 8)):
#         """
#         Visualize the reachability graph.
#
#         Args:
#             output_file: Optional file path to save the visualization
#             figsize: Figure size as (width, height) tuple
#         """
#         plt.figure(figsize=fig_size)
#
#         # Use hierarchical layout for better visualization
#         pos = nx.spectral_layout(self.graph)
#
#         # Create node labels
#         node_labels = {node: data.get('label', node) for node, data in self.graph.nodes(data=True)}
#
#         # Draw nodes - highlight initial and final states
#         initial_node = self._state_to_str(self.semantics.initial_state())
#         final_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree == 0]
#
#         # Draw regular nodes
#         regular_nodes = [node for node in self.graph.nodes()
#                          if node != initial_node and node not in final_nodes]
#         nx.draw_networkx_nodes(self.graph, pos, nodelist=regular_nodes,
#                                node_color='lightblue', node_size=1500, alpha=0.8)
#
#         # Draw initial node
#         if initial_node in self.graph.nodes():
#             nx.draw_networkx_nodes(self.graph, pos, nodelist=[initial_node],
#                                    node_color='green', node_size=1500, alpha=0.8)
#
#         # Draw final nodes
#         nx.draw_networkx_nodes(self.graph, pos, nodelist=final_nodes,
#                                node_color='orange', node_size=1500, alpha=0.8)
#
#         # Draw edges with labels
#         nx.draw_networkx_edges(self.graph, pos, width=3.0, node_size=1000,arrowsize=15, alpha=0.7)
#
#         # Create edge labels
#         edge_labels = {(u, v): data.get('label') for u, v, data in self.graph.edges(data=True)}
#         nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
#
#         # Draw node labels
#         nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10, font_weight='bold')
#
#         plt.title("Causal Net Reachability Graph")
#         plt.axis('off')
#
#         # if output_file:
#         #     plt.savefig(output_file, bbox_inches='tight', dpi=300)
#         #     print(f"Saved visualization to {output_file}")
#
#         plt.show()
#
#     def export_to_dot(self, output_file: str = None) -> str:
#         """
#         Export the reachability graph to DOT format for use with Graphviz.
#
#         Args:
#             output_file: Optional file path to save the DOT file
#
#         Returns:
#             DOT representation as a string
#         """
#         # Create a DOT representation
#         dot = ['digraph ReachabilityGraph {',
#                '  rankdir=LR;',
#                '  node [shape=ellipse, style=filled];']
#
#         # Initial node (green)
#         initial_node = self._state_to_str(self.semantics.initial_state())
#         dot.append(f'  "{initial_node}" [fillcolor=green, label="{initial_node}"];')
#
#         # Find final nodes (no outgoing edges)
#         final_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree == 0]
#         for node in final_nodes:
#             dot.append(f'  "{node}" [fillcolor=orange, label="{node}"];')
#
#         # Regular nodes (blue)
#         regular_nodes = [node for node in self.graph.nodes()
#                          if node != initial_node and node not in final_nodes]
#         for node in regular_nodes:
#             dot.append(f'  "{node}" [fillcolor=lightblue, label="{node}"];')
#
#         # Add edges
#         for u, v, data in self.graph.edges(data=True):
#             edge_label = data.get('label', '').replace('\n', '\\n')
#             dot.append(f'  "{u}" -> "{v}" [label="{edge_label}"];')
#
#         dot.append('}')
#         dot_content = '\n'.join(dot)
#
#         if output_file:
#             with open(output_file, 'w') as f:
#                 f.write(dot_content)
#             print(f"Exported DOT file to {output_file}")
#
#         return dot_content
#
#     def get_state_space_info(self) -> Dict[str, Any]:
#         """
#         Get information about the state space.
#
#         Returns:
#             Dictionary with state space information
#         """
#         info = {
#             'num_states': len(self.graph.nodes()),
#             'num_transitions': len(self.graph.edges()),
#             'initial_state': self._state_to_str(self.semantics.initial_state()),
#             'final_states': [node for node, out_degree in self.graph.out_degree() if out_degree == 0],
#             'deadlock_states': [node for node, out_degree in self.graph.out_degree()
#                                 if out_degree == 0 and node not in self._state_to_str(self.semantics.initial_state())]
#         }
#         return info
#
#
#
# def example_reachability_graph():
#     """
#     Create and visualize a reachability graph for an example causal net.
#     """
#     # Create a simple Weighted C-net
#     scn = import_scn_from_xml("../data/abbc.cnet")
#
#     # Create the reachability graph
#     reachability = ReachabilityGraph(scn)
#     reachability.generate_reachability_graph()
#
#     # Print information about the state space
#     state_space_info = reachability.get_state_space_info()
#     print("State Space Information:")
#     print(f"Number of States: {state_space_info['num_states']}")
#     print(f"Number of Transitions: {state_space_info['num_transitions']}")
#     print(f"Initial State: {state_space_info['initial_state']}")
#     print(f"Final States: {state_space_info['final_states']}")
#
#     # Visualize the reachability graph
#     reachability.visualize(output_file="causal_net_reachability.png")
#
#     # Export to DOT format
#     # reachability.export_to_dot(output_file="causal_net_reachability.dot")
#     # print("To visualize with Graphviz, run: dot -Tpng causal_net_reachability.dot -o reachability_graphviz.png")
#
#
# if __name__ == "__main__":
#     example_reachability_graph()