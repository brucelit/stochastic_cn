import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Set
from obj.stochastic_causal_net import StochasticCausalNet, print_scn_info


def export_to_cnet(causal_net, filename: str) -> None:
    """
    Export the causal net to a .cnet XML file.

    Args:
        filename: The output filename
    """
    # Create the root element
    root = ET.Element('cnet')

    # Add net element
    net_elem = ET.SubElement(root, 'net')
    net_elem.set('type', 'http://www.processmining.org')
    net_elem.set('id', '')

    # Add name element with current date/time
    current_time = datetime.now().strftime("%d/%m/%Y %I:%M:%S %p")
    name_elem = ET.SubElement(root, 'name')
    name_elem.text = f"Causal net (C-net) @ {current_time}"

    # Create an activity to node ID mapping
    activity_to_id = _create_activity_id_mapping(causal_net)

    # Add all activities as nodes
    for activity, node_id in activity_to_id.items():
        node = ET.SubElement(root, 'node')
        node.set('id', str(node_id))
        node.set('isInvisible', 'false')
        name_subelem = ET.SubElement(node, 'name')
        name_subelem.text = activity

    # Set start and end task nodes
    start_task = ET.SubElement(root, 'startTaskNode')
    start_task.set('id', str(activity_to_id[causal_net.start_activity]))

    end_task = ET.SubElement(root, 'endTaskNode')
    end_task.set('id', str(activity_to_id[causal_net.end_activity]))

    # Add input and output bindings
    _add_input_bindings(causal_net,root, activity_to_id)
    _add_output_bindings(causal_net, root, activity_to_id)

    # Add arcs
    arc_id = len(activity_to_id)  # Start arc IDs after node IDs
    _add_arcs(causal_net, root, activity_to_id, arc_id)

    # Format the XML with proper indentation
    xml_string = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Write to file with XML declaration
    with open(filename, 'w', encoding='ISO-8859-1') as f:
        # Replace the default XML declaration with the format from the example
        xml_declaration = '<?xml version="1.0" encoding="ISO-8859-1"?>'
        final_xml = pretty_xml.replace('<?xml version="1.0" ?>', xml_declaration)
        f.write(final_xml)

    print(f"Exported causal net to {filename}")


def _create_activity_id_mapping(causal_net) -> Dict[str, int]:
    """
    Create a mapping from activity names to node IDs.

    Returns:
        Dictionary mapping activity names to numeric IDs
    """
    activity_to_id = {}
    for i, activity in enumerate(sorted(causal_net.activities)):
        activity_to_id[activity] = i
    return activity_to_id


def _add_input_bindings(causal_net, root: ET.Element, activity_to_id: Dict[str, int]) -> None:
    """
    Add input bindings to the XML.

    Args:
        root: The root XML element
        activity_to_id: Mapping from activity names to node IDs
    """
    for activity in causal_net.activities:
        # Skip start activity as it has no input bindings
        if activity == causal_net.start_activity:
            continue

        input_node = ET.SubElement(root, 'inputNode')
        input_node.set('id', str(activity_to_id[activity]))

        # Add each input binding
        for binding in causal_net.input_bindings[activity]:
            input_set = ET.SubElement(input_node, 'inputSet')

            # Add each activity in the binding
            for source_activity in binding:
                node = ET.SubElement(input_set, 'node')
                node.set('id', str(activity_to_id[source_activity]))

                # Add weight if available
                weight = causal_net.input_binding_weights[activity][binding]
                if weight != 1.0:  # Only add if not default
                    node.set('weight', str(weight))


def _add_output_bindings(causal_net, root: ET.Element, activity_to_id: Dict[str, int]) -> None:
    """
    Add output bindings to the XML.

    Args:
        root: The root XML element
        activity_to_id: Mapping from activity names to node IDs
    """
    for activity in causal_net.activities:
        # Skip end activity as it has no output bindings
        if activity == causal_net.end_activity:
            continue

        output_node = ET.SubElement(root, 'outputNode')
        output_node.set('id', str(activity_to_id[activity]))

        # Add each output binding
        for binding in causal_net.output_bindings[activity]:
            output_set = ET.SubElement(output_node, 'outputSet')

            # Add each activity in the binding
            for target_activity in binding:
                node = ET.SubElement(output_set, 'node')
                node.set('id', str(activity_to_id[target_activity]))

                # Add weight if available
                weight = causal_net.output_binding_weights[activity][binding]
                if weight != 1.0:  # Only add if not default
                    node.set('weight', str(weight))


def _add_arcs(causal_net, root: ET.Element, activity_to_id: Dict[str, int], start_arc_id: int) -> None:
    """
    Add arcs to the XML.

    Args:
        root: The root XML element
        activity_to_id: Mapping from activity names to node IDs
        start_arc_id: Starting ID for arcs
    """
    arc_id = start_arc_id

    # Create a set to track added arcs to avoid duplicates
    added_arcs = set()

    # Add arcs for all input and output relationships
    for activity in causal_net.activities:
        # Add arcs for input bindings
        if activity != causal_net.start_activity:
            for binding in causal_net.input_bindings[activity]:
                for source_activity in binding:
                    arc_tuple = (source_activity, activity)
                    if arc_tuple not in added_arcs:
                        _add_arc(causal_net, root, activity_to_id, arc_id, source_activity, activity)
                        added_arcs.add(arc_tuple)
                        arc_id += 1

        # Add arcs for output bindings
        if activity != causal_net.end_activity:
            for binding in causal_net.output_bindings[activity]:
                for target_activity in binding:
                    arc_tuple = (activity, target_activity)
                    if arc_tuple not in added_arcs:
                        _add_arc(causal_net, root, activity_to_id, arc_id, activity, target_activity)
                        added_arcs.add(arc_tuple)
                        arc_id += 1


def _add_arc(causal_net, root: ET.Element, activity_to_id: Dict[str, int],
             arc_id: int, source_activity: str, target_activity: str) -> None:
    """
    Add a single arc to the XML.

    Args:
        root: The root XML element
        activity_to_id: Mapping from activity names to node IDs
        arc_id: ID for this arc
        source_activity: Source activity
        target_activity: Target activity
    """
    arc = ET.SubElement(root, 'arc')
    arc.set('id', str(arc_id))
    arc.set('source', str(activity_to_id[source_activity]))
    arc.set('target', str(activity_to_id[target_activity]))


def load_cnet_from_xml(filename: str) -> StochasticCausalNet:
    """
    Load a causal net from a .cnet XML file.

    Args:
        filename: The .cnet XML file to load

    Returns:
        A StochasticCausalNet object
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    # Find start and end activity nodes
    start_node_id = root.find('startTaskNode').get('id')
    end_node_id = root.find('endTaskNode').get('id')

    # Create mapping from node IDs to activity names
    id_to_activity = {}
    for node in root.findall('node'):
        node_id = node.get('id')
        activity_name = node.find('name').text
        id_to_activity[node_id] = activity_name

    start_activity = id_to_activity[start_node_id]
    end_activity = id_to_activity[end_node_id]

    # Create the causal net
    cnet = StochasticCausalNet(start_activity, end_activity)

    # Add all activities
    for activity in id_to_activity.values():
        if activity != start_activity and activity != end_activity:
            cnet.add_activity(activity)

    # Process input bindings
    for input_node in root.findall('inputNode'):
        target_id = input_node.get('id')
        target_activity = id_to_activity[target_id]

        for input_set in input_node.findall('inputSet'):
            # Collect all activities in this input binding
            binding = set()
            weights = {}

            for node in input_set.findall('node'):
                source_id = node.get('id')
                source_activity = id_to_activity[source_id]
                binding.add(source_activity)

                # Get weight if specified
                if 'weight' in node.attrib:
                    weights[source_activity] = float(node.get('weight'))

            # Calculate a combined weight for the binding
            # We'll use the average of weights if multiple are specified
            if weights:
                weight = sum(weights.values()) / len(weights)
            else:
                weight = 1.0

            # Add the input binding
            if binding:  # Skip empty bindings
                cnet.add_input_binding(target_activity, binding, weight)

    # Process output bindings
    for output_node in root.findall('outputNode'):
        source_id = output_node.get('id')
        source_activity = id_to_activity[source_id]

        for output_set in output_node.findall('outputSet'):
            # Collect all activities in this output binding
            binding = set()
            weights = {}

            for node in output_set.findall('node'):
                target_id = node.get('id')
                target_activity = id_to_activity[target_id]
                binding.add(target_activity)

                # Get weight if specified
                if 'weight' in node.attrib:
                    weights[target_activity] = float(node.get('weight'))

            # Calculate a combined weight for the binding
            if weights:
                weight = sum(weights.values()) / len(weights)
            else:
                weight = 1.0

            # Add the output binding
            if binding:  # Skip empty bindings
                cnet.add_output_binding(source_activity, binding, weight)

    return cnet


def example_causal_net():
    """Create an example causal net and export it to a .cnet file"""
    # Create a weighted causal net based on the abbc.cnet example
    cnet = StochasticCausalNet("ARTIFICIAL_START", "ARTIFICIAL_END")

    # Add activities
    cnet.add_activity("a")
    cnet.add_activity("b")
    cnet.add_activity("c")

    # Add output bindings
    cnet.add_output_binding("ARTIFICIAL_START", {"a"}, 1.24)
    cnet.add_output_binding("a", {"b"}, 0.24)
    cnet.add_output_binding("b", {"b"}, 1.24)
    cnet.add_output_binding("b", {"c"}, 1.24)
    cnet.add_output_binding("c", {"ARTIFICIAL_END"}, 1.24)

    # Add input bindings
    cnet.add_input_binding("a", {"ARTIFICIAL_START"}, 1.24)
    cnet.add_input_binding("b", {"a"}, 1.3)
    cnet.add_input_binding("b", {"b"}, 0.5)
    cnet.add_input_binding("c", {"b"}, 1.0)  # Default weight
    cnet.add_input_binding("ARTIFICIAL_END", {"c"}, 1.24)

    # Print info about the causal net
    print_scn_info(cnet)

    # Export to .cnet file
    export_to_cnet(cnet, "../data/abcd.cnet")

    # Test round-trip by loading it back
    # loaded_cnet = load_cnet_from_xml("../data/exported_abcd.cnet")
    # print("\n\nLoaded Causal Net:")
    # print_scn_info(loaded_cnet)


if __name__ == "__main__":
    example_causal_net()