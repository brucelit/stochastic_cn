from collections import defaultdict
from fractions import Fraction


def import_slang(file_path):
    stochastic_language = {}

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("# trace"):
            # Read the probability
            i += 2
            fraction_str = lines[i]
            numerator, denominator = [part.strip() for part in fraction_str.split("/")]
            probability = Fraction(int(numerator), int(denominator))
            # Read the number of events
            i += 2
            num_events = int(lines[i])
            # Read the events
            i += 1
            events = ['ARTIFICIAL_START']
            for _ in range(num_events):
                events.append(lines[i])
                i += 1
            events.append('ARTIFICIAL_END')
            trace = tuple(events)
            stochastic_language[trace] = probability
        else:
            i += 1

    return stochastic_language


def compute_markov_abstraction(language: dict, k: int) -> dict:
    """
    Compute the k-th order Markovian abstraction of a stochastic language.

    Args:
        language: The original stochastic language
        k: The length of sub-traces to extract

    Returns:
        A new StochasticLanguage object representing the k-th order abstraction
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Dictionary to store sub-trace probabilities
    subtrace_probabilities = defaultdict(Fraction)
    subtrace_count = 0

    # Process each trace
    for trace, trace_prob in language.items():
        # Skip traces shorter than k
        if len(trace) < k:
            print(f"Warning: Trace is shorter than k={k}, skipping.")
            continue

        # Generate all sub-traces of length k
        for i in range(len(trace) - k + 1):
            subtrace = tuple(trace[i:i + k])
            subtrace_probabilities[subtrace] += trace_prob/(len(trace) - k + 1)
            subtrace_count += 1

    # If we found no valid sub-traces
    if subtrace_count == 0:
        raise ValueError(f"No sub-traces of length {k} found in the language.")

    # Normalize probabilities (each sub-trace is counted once per occurrence)
    result = {}

    for subtrace, probability in subtrace_probabilities.items():
        # Create a new trace with the normalized probability
        result[tuple(subtrace)] = probability

    return result

# file_path = '../data/test.slang'
# slang = import_slang(file_path)
#
# sum = Fraction(0)
# for trace, prob in slang.items():
#     print(f"Trace: {trace}, Probability: {prob}")
#     sum += prob
# print("Sum of probabilities:", sum)
#
# markov_result = compute_markov_abstraction(slang, 3)
# print("Markov abstraction result:", markov_result)
# m_sum =0
# for k, v in markov_result.items():
#     m_sum += v
# print("Sum of probabilities:", m_sum)