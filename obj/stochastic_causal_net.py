from typing import Set, List
from collections import defaultdict, Counter


class Obligation:
    """
    An obligation in a Causal-net represents a commitment that an activity
    should be followed by another specific activity.
    """

    def __init__(self, source: str, target: str):
        """
        Initialize an obligation.

        Args:
            source: The source activity that created the obligation
            target: The target activity that should be executed to fulfill the obligation
        """
        self.source = source
        self.target = target

    def __eq__(self, other):
        if not isinstance(other, Obligation):
            return False
        return self.source == other.source and self.target == other.target

    def __hash__(self):
        return hash((self.source, self.target))

    def __repr__(self):
        return f"({self.source}->{self.target})"


class State:
    """
    A state in a Causal-net is represented by a multi-set (Counter) of obligations.
    """

    def __init__(self, obligations=None):
        """
        Initialize a state with a set of obligations.

        Args:
            obligations: A dictionary mapping obligations to their counts
        """
        self.obligations = Counter()
        if obligations:
            self.obligations.update(obligations)

    def add_obligation(self, obligation: Obligation, count: int = 1) -> None:
        """Add an obligation to the state."""
        self.obligations[obligation] += count

    def remove_obligation(self, obligation: Obligation, count: int = 1) -> None:
        """Remove an obligation from the state."""
        if obligation in self.obligations and self.obligations[obligation] >= count:
            self.obligations[obligation] -= count
            if self.obligations[obligation] == 0:
                del self.obligations[obligation]
        else:
            raise ValueError(f"Cannot remove obligation {obligation} as it's not pending")

    def has_obligation(self, obligation: Obligation) -> bool:
        """Check if an obligation exists in the state."""
        return obligation in self.obligations

    def is_empty(self) -> bool:
        """Check if the state has no obligations."""
        return len(self.obligations) == 0

    def get_all_obligations(self) -> List[Obligation]:
        """Get all obligations in the state (including duplicates)."""
        result = []
        for obligation, count in self.obligations.items():
            result.extend([obligation] * count)
        return result

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.obligations == other.obligations

    def __repr__(self):
        return str(dict(self.obligations))

    def copy(self) -> 'State':
        """Create a copy of this state."""
        return State(self.obligations.copy())


class Binding:
    """
    A binding represents a concrete choice of input/output activities
    for a specific activity execution.
    """

    def __init__(self, activity: str, input_set: Set[str], output_set: Set[str]):
        """
        Initialize a binding.

        Args:
            activity: The activity being executed
            input_set: The specific input binding chosen
            output_set: The specific output binding chosen
        """
        self.activity = activity
        self.input_set = frozenset(input_set)
        self.output_set = frozenset(output_set)

    def __repr__(self):
        return f"{self.input_set} -> {self.activity} -> {self.output_set}"


class BindingSequence:
    """
    A binding sequence is a sequence of bindings that represents a process execution.
    """

    def __init__(self):
        """Initialize an empty binding sequence."""
        self.bindings = []

    def add_binding(self, binding: Binding) -> None:
        """Add a binding to the sequence."""
        self.bindings.append(binding)

    def __repr__(self):
        return " → ".join(str(binding) for binding in self.bindings)


class StochasticCausalNet:
    """
    A Stochastic Causal-net (WC-net) implementation representing process models.

    A WC-net extends a C-net by adding weights to input and output bindings,
    which can be used to calculate probabilities for different execution paths.
    """

    def __init__(self, start_activity: str, end_activity: str):
        """
        Initialize a Stochastic Causal-net.

        Args:
            start_activity: The activity that starts the process
            end_activity: The activity that ends the process
        """
        self.activities = set()  # Set of all activities
        self.start_activity = start_activity
        self.end_activity = end_activity

        # Add start and end activities to the set of activities
        self.activities.add(start_activity)
        self.activities.add(end_activity)

        # Input and output bindings for each activity
        # A binding is a set of activities that can be executed together
        # Each binding is stored as a frozenset to allow it to be used as a dictionary key
        self.input_bindings = defaultdict(set)
        self.output_bindings = defaultdict(set)

        # Weights for input and output bindings
        # Dictionary structure: {activity: {binding_frozenset: weight}}
        # Default weight is 1.0 for all bindings
        self.input_binding_weights = defaultdict(dict)
        self.output_binding_weights = defaultdict(dict)

        # Initialize empty bindings for start and end activities
        # Start activity has no input bindings
        empty_set = frozenset()
        self.input_bindings[start_activity].add(empty_set)
        self.input_binding_weights[start_activity][empty_set] = 1.0

        # End activity has no output bindings
        self.output_bindings[end_activity].add(empty_set)
        self.output_binding_weights[end_activity][empty_set] = 1.0

    def add_activity(self, activity: str) -> None:
        """Add an activity to the WC-net."""
        self.activities.add(activity)

    def add_input_binding(self, activity: str, binding: Set[str], weight: float = 1.0) -> None:
        """
        Add an input binding for an activity with a weight.

        Args:
            activity: The target activity
            binding: A set of activities that can lead to the target activity
            weight: The weight of this binding (default: 1.0)
        """
        if activity == self.start_activity:
            raise ValueError("Start activity cannot have input bindings")

        if weight < 0:
            raise ValueError("Weight must be non-negative")

        # Ensure all activities in the binding exist
        for act in binding:
            if act not in self.activities:
                raise ValueError(f"Activity {act} in binding doesn't exist in the WC-net")

        binding_frozenset = frozenset(binding)
        self.input_bindings[activity].add(binding_frozenset)
        self.input_binding_weights[activity][binding_frozenset] = weight

    def add_output_binding(self, activity: str, binding: Set[str], weight: float = 1.0) -> None:
        """
        Add an output binding for an activity with a weight.

        Args:
            activity: The source activity
            binding: A set of activities that can follow the source activity
            weight: The weight of this binding (default: 1.0)
        """
        if activity == self.end_activity:
            raise ValueError("End activity cannot have output bindings")

        if weight < 0:
            raise ValueError("Weight must be non-negative")

        # Ensure all activities in the binding exist
        for act in binding:
            if act not in self.activities:
                raise ValueError(f"Activity {act} in binding doesn't exist in the WC-net")

        binding_frozenset = frozenset(binding)
        self.output_bindings[activity].add(binding_frozenset)
        self.output_binding_weights[activity][binding_frozenset] = weight

    def update_input_binding_weight(self, activity: str, binding: Set[str], weight: float) -> None:
        """
        Update the weight of an existing input binding.

        Args:
            activity: The target activity
            binding: The input binding to update
            weight: The new weight
        """
        if weight < 0:
            raise ValueError("Weight must be non-negative")

        binding_frozenset = frozenset(binding)
        if binding_frozenset not in self.input_bindings[activity]:
            raise ValueError(f"Input binding {binding} does not exist for activity {activity}")

        self.input_binding_weights[activity][binding_frozenset] = weight

    def update_output_binding_weight(self, activity: str, binding: Set[str], weight: float) -> None:
        """
        Update the weight of an existing output binding.

        Args:
            activity: The source activity
            binding: The output binding to update
            weight: The new weight
        """
        if weight < 0:
            raise ValueError("Weight must be non-negative")

        binding_frozenset = frozenset(binding)
        if binding_frozenset not in self.output_bindings[activity]:
            raise ValueError(f"Output binding {binding} does not exist for activity {activity}")

        self.output_binding_weights[activity][binding_frozenset] = weight

    def get_input_binding_weight(self, activity: str, binding: frozenset[str]) -> float:
        """
        Get the weight of an input binding.

        Args:
            activity: The target activity
            binding: The input binding

        Returns:
            The weight of the binding
        """
        if binding not in self.input_bindings[activity]:
            raise ValueError(f"Input binding {binding} does not exist for activity {activity}")

        return self.input_binding_weights[activity][binding]

    def get_output_binding_weight(self, activity: str, binding: frozenset[str]) -> float:
        """
        Get the weight of an output binding.

        Args:
            activity: The source activity
            binding: The output binding

        Returns:
            The weight of the binding
        """
        binding_frozenset = frozenset(binding)
        if binding_frozenset not in self.output_bindings[activity]:
            raise ValueError(f"Output binding {binding} does not exist for activity {activity}")

        return self.output_binding_weights[activity][binding_frozenset]

    def get_successors(self, activity: str) -> Set[str]:
        """Get all possible successor activities of a given activity."""
        successors = set()
        for binding in self.output_bindings[activity]:
            successors.update(binding)
        return successors

    def get_predecessors(self, activity: str) -> Set[str]:
        """Get all possible predecessor activities of a given activity."""
        predecessors = set()
        for binding in self.input_bindings[activity]:
            predecessors.update(binding)
        return predecessors

    def get_total_output_binding_weight(self, activity: str) -> float:
        """
        Calculate the total weight of all output bindings for an activity.

        Args:
            activity: The activity to calculate total output weight for

        Returns:
            The sum of weights of all output bindings
        """
        # No output bindings for end activity
        if activity == self.end_activity:
            return 0.0

        # Calculate the sum of weights for all output bindings
        total_weight = sum(self.output_binding_weights[activity][binding]
                           for binding in self.output_bindings[activity])

        return total_weight

    def get_total_input_binding_weight(self, activity: str) -> float:
        """
        Calculate the total weight of all input bindings for an activity.

        Args:
            activity: The activity to calculate total input weight for

        Returns:
            The sum of weights of all input bindings
        """
        # No input bindings for start activity
        if activity == self.start_activity:
            return 0.0

        # Calculate the sum of weights for all input bindings
        total_weight = sum(self.input_binding_weights[activity][binding]
                           for binding in self.input_bindings[activity])

        return total_weight


class Semantics:
    """
    The semantics of a Causal-net. This class implements the rules for valid
    binding sequences and state transitions.
    """

    def __init__(self, causal_net: StochasticCausalNet):
        """
        Initialize the semantics for a given Causal-net.

        Args:
            causal_net: The Causal-net to analyze
        """
        self.causal_net = causal_net

    def initial_state(self) -> State:
        """Get the initial state (empty state)."""
        return State()

    def execute_binding(self, binding: Binding, current_state: State) -> State:
        """
        Execute a binding and return the resulting state.

        Args:
            binding: The binding to execute
            current_state: The current state before execution

        Returns:
            The new state after executing the binding
        """
        # Create a copy of the current state
        new_state = current_state.copy()

        # For the initial activity, we don't need to check pending obligations
        if binding.activity != self.causal_net.start_activity:
            # Ensure the input set exactly matches a valid input binding from the model
            if binding.input_set not in self.causal_net.input_bindings[binding.activity]:
                raise ValueError(f"Input binding {binding.input_set} is not valid for activity {binding.activity}")

            # Remove obligations corresponding to the input binding
            # We must have exactly the obligations we need - no more, no less
            pending_inputs = {}
            for obligation in current_state.get_all_obligations():
                if obligation.target == binding.activity:
                    if obligation.source not in pending_inputs:
                        pending_inputs[obligation.source] = 0
                    pending_inputs[obligation.source] += 1

            # Check if all required obligations are pending
            for input_activity in binding.input_set:
                if input_activity not in pending_inputs or pending_inputs[input_activity] == 0:
                    raise ValueError(
                        f"Cannot execute binding: Obligation from {input_activity} to {binding.activity} is not pending")
                pending_inputs[input_activity] -= 1

            # Now remove the obligations
            for input_activity in binding.input_set:
                obligation = Obligation(input_activity, binding.activity)
                new_state.remove_obligation(obligation)

        # Ensure the output set is valid according to the model
        if binding.activity != self.causal_net.end_activity:
            if binding.output_set not in self.causal_net.output_bindings[binding.activity]:
                raise ValueError(f"Output binding {binding.output_set} is not valid for activity {binding.activity}")

            # Add obligations corresponding to the output binding
            for output_activity in binding.output_set:
                obligation = Obligation(binding.activity, output_activity)
                new_state.add_obligation(obligation)

        return new_state

    def is_enabled(self, binding: Binding, current_state: State) -> bool:
        """
        Check if a binding is enabled in a given state.

        Args:
            binding: The binding to check
            current_state: The current state

        Returns:
            True if the binding is enabled, False otherwise
        """
        # Special case for start activity
        if binding.activity == self.causal_net.start_activity:
            # Start activity can only be executed in an empty state
            return current_state.is_empty() and binding.input_set == frozenset()

        # Special case for end activity
        if binding.activity == self.causal_net.end_activity:
            # End activity can only have an empty output binding
            if binding.output_set != frozenset():
                return False

        # Verify the binding respects the model's defined input and output bindings
        if binding.input_set not in self.causal_net.input_bindings[binding.activity]:
            return False

        if binding.activity != self.causal_net.end_activity and binding.output_set not in \
                self.causal_net.output_bindings[binding.activity]:
            return False

        # Count the pending obligations for each source activity
        pending_inputs = {}
        for obligation in current_state.get_all_obligations():
            if obligation.target == binding.activity:
                if obligation.source not in pending_inputs:
                    pending_inputs[obligation.source] = 0
                pending_inputs[obligation.source] += 1

        # Check if all required obligations are present
        for input_activity in binding.input_set:
            if input_activity not in pending_inputs or pending_inputs[input_activity] == 0:
                return False

        return True

    def get_enabled_bindings(self, current_state: State):
        """
        Get all enabled bindings in a given state.

        Args:
            current_state: The current state

        Returns:
            A list of all enabled bindings
        """
        enabled_bindings = dict()

        # Special case for the start activity when state is empty
        if current_state.is_empty():
            print("current state is empty", current_state)
            for output_binding in self.causal_net.output_bindings[self.causal_net.start_activity]:
                binding = Binding(self.causal_net.start_activity, set(), output_binding)
                probability = self.causal_net.get_output_binding_weight(self.causal_net.start_activity, output_binding)
                enabled_bindings[binding] = probability
            return enabled_bindings

        # Count all pending obligations grouped by target activity
        pending_obligations = defaultdict(lambda: defaultdict(int))
        for obligation in current_state.get_all_obligations():
            pending_obligations[obligation.target][obligation.source] += 1

        enabled_input_bindings = []
        # For all activities with pending obligations
        for activity, sources in pending_obligations.items():
            # Get the set of source activities
            source_activities = set(sources.keys())
            # Check if any input binding is exactly satisfied by the pending obligations
            for input_binding in self.causal_net.input_bindings[activity]:
                # Skip if input binding doesn't match the source activities
                if not set(input_binding).issubset(source_activities):
                    continue

                # # Check if we have exactly the right number of obligations
                # match = True
                # for source in input_binding:
                #     if sources[source] != 1:  # We need exactly one obligation per source
                #         match = False
                #         break
                #
                # if not match:
                #     continue

                enabled_input_bindings.append(input_binding)
                # For matching input binding, consider all possible output bindings
                for output_binding in self.causal_net.output_bindings[activity]:
                    binding = Binding(activity, input_binding, output_binding)
                    if self.is_enabled(binding, current_state):
                        enabled_bindings[binding] = 1

        #compute the probability of each enabled binding
        for binding in enabled_bindings:
            input_weight = self.causal_net.get_input_binding_weight(binding.activity, binding.input_set)
            output_weight = self.causal_net.get_output_binding_weight(binding.activity, binding.output_set)
            enabled_bindings[binding] = input_weight * output_weight / (
                        len(enabled_input_bindings) * len(self.causal_net.output_bindings[binding.activity]))

        return enabled_bindings


    def is_valid_binding_sequence(self, binding_sequence: List[Binding]) -> bool:
        """
        Check if a binding sequence is valid according to the C-net semantics.

        A valid sequence:
        1. Starts with the start activity
        2. Ends with the end activity
        3. Only removes obligations that are pending
        4. Ends without any pending obligations

        Args:
            binding_sequence: The binding sequence to check

        Returns:
            True if the sequence is valid, False otherwise
        """
        if not binding_sequence:
            return False

        # Check if the sequence starts with the start activity
        if binding_sequence[0].activity != self.causal_net.start_activity:
            return False

        # Check if the sequence ends with the end activity
        if binding_sequence[-1].activity != self.causal_net.end_activity:
            return False

        # Replay the sequence and check if all transitions are valid
        current_state = self.initial_state()

        for binding in binding_sequence:
            try:
                # Check if the binding is enabled in the current state
                if not self.is_enabled(binding, current_state):
                    return False

                # Execute the binding and update the state
                current_state = self.execute_binding(binding, current_state)
            except ValueError:
                # If an exception is raised, the sequence is invalid
                return False

        # Check if there are any pending obligations left
        return current_state.is_empty()

    def generate_all_valid_binding_sequences(self, max_depth: int = 10):
        """
        Generate all valid binding sequences up to a certain depth.

        Args:
            max_depth: Maximum number of bindings in a sequence

        Returns:
            A list of all valid binding sequences
        """
        valid_sequences = dict()

        def dfs(current_sequence, current_state, current_probability, depth):
            # Base case: maximum depth reached
            if depth >= max_depth:
                return

            # Get the dictionary of all enabled bindings
            enabled_bindings = self.get_enabled_bindings(current_state)

            for binding, probability in enabled_bindings.items():
                # Create a new sequence by adding this binding
                new_sequence = current_sequence + [binding]

                # Calculate the new state
                new_state = self.execute_binding(binding, current_state)
                new_probability = current_probability * probability

                # If the binding leads to the end activity and the state is empty,
                # we've found a valid sequence
                if binding.activity == self.causal_net.end_activity and new_state.is_empty():
                    valid_sequences[tuple(new_sequence)] = new_probability
                    continue

                # Continue the search
                dfs(new_sequence, new_state, new_probability, depth + 1)

        # Start the search from the initial state
        dfs([], self.initial_state(), 1,0)

        return valid_sequences


def project_binding_sequence_to_activities(binding_sequence: List[Binding]) -> List[str]:
    """
    Project a binding sequence to a sequence of activities.

    Args:
        binding_sequence: The binding sequence to project

    Returns:
        A list of activity names representing the sequence
    """
    return [binding.activity for binding in binding_sequence if binding.activity != "ARTIFICIAL_END" and binding.activity != "ARTIFICIAL_START"]


def print_scn_info(scn: StochasticCausalNet):
    """
    Print information about a Stochastic CausalNet object for debugging.

    Args:
        scn: The StochasticCausalNet object to print
    """
    print("Stochastic Causal Net Information:")
    print(f"Start Activity: {scn.start_activity}")
    print(f"End Activity: {scn.end_activity}")
    print(f"Activities: {scn.activities}")

    print("\nInput Bindings with Weights:")
    for activity in sorted(scn.activities):
        if activity != scn.start_activity:
            bindings = []
            for binding in scn.input_bindings[activity]:
                weight = scn.input_binding_weights[activity][binding]
                bindings.append(f"{list(binding)} (weight: {weight})")
            print(f"  {activity}: {bindings}")

    print("\nOutput Bindings with Weights:")
    for activity in sorted(scn.activities):
        if activity != scn.end_activity:
            bindings = []
            for binding in scn.output_bindings[activity]:
                weight = scn.output_binding_weights[activity][binding]
                bindings.append(f"{list(binding)} (weight: {weight})")
            print(f"  {activity}: {bindings}")

    print("\nTotal Output Binding Weights:")
    for activity in sorted(scn.activities):
        if activity != scn.end_activity:
            total_weight = scn.get_total_output_binding_weight(activity)
            print(f"  {activity}: {total_weight}")



# Example usage
def example_usage():
    # Create a simple Stochastic C-net
    scn = StochasticCausalNet("start", "end")

    # Add activities
    scn.add_activity("A")
    scn.add_activity("B")
    scn.add_activity("C")

    # Add input and output bindings with weights
    scn.add_output_binding("start", {"A"}, 1.0)

    scn.add_input_binding("A", {"start"}, 1.0)
    scn.add_output_binding("A", {"B", "C"}, 1.0)

    scn.add_input_binding("B", {"A"}, 1.0)
    scn.add_output_binding("B", {"end"}, 1.0)

    scn.add_input_binding("C", {"A"}, 1.0)
    scn.add_output_binding("C", {"end"}, 1.0)

    # The end activity requires both B and C as inputs
    scn.add_input_binding("end", {"B", "C"}, 1.0)

    # Print information about the C-net
    print_scn_info(scn)

    # Create the semantics
    semantics = Semantics(scn)

    # Generate all valid binding sequences
    valid_sequences = semantics.generate_all_valid_binding_sequences()

    print(f"\nFound {len(valid_sequences)} valid binding sequences:")
    for sequence,probability in valid_sequences.items():
        # Project to activity sequence
        activity_sequence = project_binding_sequence_to_activities(sequence)
        print("Resulting trace and probability: ",probability,activity_sequence)


if __name__ == "__main__":
    example_usage()