import xml.etree.ElementTree as ET
from typing import Dict, Set, List, Tuple
from obj.weighted_causal_net import WeightedCausalNet, print_cnet_info, project_binding_sequence_to_activities, \
    Semantics


def import_weighted_cnet_from_xml(xml_content: str, default_weight: float = 1.0) -> WeightedCausalNet:
    """
    Import a C-net model from an XML string and convert it to a Weighted C-net.

    Args:
        xml_content: XML string containing the C-net definition
        default_weight: Default weight to assign to all bindings (default: 1.0)

    Returns:
        A WeightedCausalNet object representing the imported model
    """
    # Parse the XML
    root = ET.parse(xml_content)

    # Extract node information
    nodes = {}  # id -> name mapping
    start_node_id = None
    end_node_id = None

    for node_elem in root.findall('.//node'):
        node_id = node_elem.get('id')
        name_elem = node_elem.find('name')

        # If 'name' tag is not found, try 'n' tag (different format)
        if name_elem is None:
            name_elem = node_elem.find('n')

        if name_elem is not None:
            node_name = name_elem.text
            nodes[node_id] = node_name

    # Find start and end nodes
    start_node_elem = root.find('.//startTaskNode')
    if start_node_elem is not None:
        start_node_id = start_node_elem.get('id')

    end_node_elem = root.find('.//endTaskNode')
    if end_node_elem is not None:
        end_node_id = end_node_elem.get('id')

    if start_node_id is None or end_node_id is None:
        raise ValueError("Start or end node not found in the XML")

    if start_node_id not in nodes or end_node_id not in nodes:
        raise ValueError("Start or end node ID not found in node definitions")

    # Create the WeightedCausalNet
    start_activity = nodes[start_node_id]
    end_activity = nodes[end_node_id]
    cnet = WeightedCausalNet(start_activity, end_activity)

    # Add all activities
    for node_id, node_name in nodes.items():
        if node_id != start_node_id and node_id != end_node_id:
            cnet.add_activity(node_name)

    # Process input bindings
    for input_node_elem in root.findall('.//inputNode'):
        node_id = input_node_elem.get('id')
        if node_id == start_node_id:
            continue  # Skip start node

        if node_id not in nodes:
            continue  # Skip if node not found

        node_name = nodes[node_id]

        for input_set_elem in input_node_elem.findall('./inputSet'):
            input_set = set()
            for input_node in input_set_elem.findall('./node'):
                input_id = input_node.get('id')
                if input_id in nodes:
                    input_name = nodes[input_id]
                    input_set.add(input_name)

            if input_set:
                try:
                    cnet.add_input_binding(node_name, input_set, default_weight)
                except ValueError as e:
                    print(f"Warning: {e}")

    # Process output bindings
    for output_node_elem in root.findall('.//outputNode'):
        node_id = output_node_elem.get('id')
        if node_id == end_node_id:
            continue  # Skip end node

        if node_id not in nodes:
            continue  # Skip if node not found

        node_name = nodes[node_id]

        for output_set_elem in output_node_elem.findall('./outputSet'):
            output_set = set()
            for output_node in output_set_elem.findall('./node'):
                output_id = output_node.get('id')
                if output_id in nodes:
                    output_name = nodes[output_id]
                    output_set.add(output_name)

            if output_set:
                try:
                    cnet.add_output_binding(node_name, output_set, default_weight)
                except ValueError as e:
                    print(f"Warning: {e}")

    return cnet


def import_weighted_cnet_from_file(filename: str, default_weight: float = 1.0) -> WeightedCausalNet:
    """
    Import a C-net model from an XML file and convert it to a Weighted C-net.

    Args:
        filename: Path to the XML file
        default_weight: Default weight to assign to all bindings (default: 1.0)

    Returns:
        A WeightedCausalNet object representing the imported model
    """
    with open(filename, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    return import_weighted_cnet_from_xml(xml_content, default_weight)


def analyze_imported_cnet(cnet: WeightedCausalNet):
    """
    Analyze an imported Weighted Causal Net and print information about it.

    Args:
        cnet: The imported WeightedCausalNet
    """
    # Print information about the C-net
    print_cnet_info(cnet)

    # Create the semantics
    semantics = Semantics(cnet)

    # Generate all valid binding sequences
    valid_sequences = semantics.generate_all_valid_binding_sequences()

    probability_sum = 0.0
    # Print all unique activity sequences
    print(f"\nFound {len(valid_sequences)} valid binding sequences:")
    for sequence,probability in valid_sequences.items():
        # Project to activity sequence
        activity_sequence = project_binding_sequence_to_activities(sequence)
        print("Resulting trace and probability: ",probability,activity_sequence)
        probability_sum += probability
    print("Sum of probabilities: ",probability_sum)


if __name__ == "__main__":
    # Import the Weighted Causal Net
    cnet = import_weighted_cnet_from_xml("../data/abbc.cnet")
    # Analyze the imported Causal Net
    analyze_imported_cnet(cnet)
