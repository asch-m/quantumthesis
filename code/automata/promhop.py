"""
DFA Pair Classification Utilities

This module provides utilities to classify pairs of DFA states by whether they
can be mapped to each other via application of input sequences (Hopcroft-style
pair classification). It includes helpers to:
- generate n-tuples,
- populate and reconcile pair classes starting from seed pairs,
- check completeness of the classification, and
- extract keys by value.

Dependencies:
- aalpy (for Dfa/DfaState)
- z3-solver (optional; used only in the __main__ example)

Note: The aalpy.Dfa API is assumed to provide:
- automaton.states: List[DfaState]
- automaton.execute_sequence(start_state, input_sequence) -> affects current_state
- automaton.get_shortest_path(start_state, end_state) -> List[Symbol] or None

If your version differs, adapt the code accordingly.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from aalpy.automata import Dfa, DfaState  # type: ignore

logger = logging.getLogger(__name__)


def get_all_n_tuples(input_list: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    """
    Generate all possible n-tuples (combinations) from the input list.

    Args:
        input_list: Sequence of items to generate tuples from.
        n: Size of each tuple.

    Returns:
        A list of all possible n-tuples.

    Raises:
        ValueError: If n < 0 or n > len(input_list).
    """
    if n < 0 or n > len(input_list):
        raise ValueError("n must be between 0 and len(input_list).")
    return list(itertools.combinations(input_list, n))


def _index_of_state(automaton: Dfa, state: DfaState) -> Optional[int]:
    """
    Helper to retrieve the index of a state within automaton.states.

    Returns:
        The index if found, otherwise None.
    """
    for i, s in enumerate(automaton.states):
        if s is state:
            return i
    return None


def populate_pair_classes(
    seed_pair: Tuple[int, int],
    pair_classes: Dict[Tuple[int, int], int],
    automaton: Dfa,
) -> Dict[Tuple[int, int], int]:
    """
    Populate the Hopcroft Table 'pair_classes' by classifying pairs by wether 
    they can be mapped to each other via application of input sequences. This method identifies all pairs that can be mapped starting from a seed pair.
    Args:
        seed_pair: The seed pair to start from, given as (i, j) indices into automaton.states.
        pair_classes: Dictionary from pair (i, j) to class label (int). Will be updated in-place.
        automaton: The DFA under consideration.

    Returns:
        The updated pair_classes dictionary (same object provided).

    Raises:
        ValueError: If seed_pair indices are invalid or identical.
    """
    i, j = seed_pair
    if i == j:
        raise ValueError("seed_pair must reference two distinct state indices.")
    if not (0 <= i < len(automaton.states)) or not (0 <= j < len(automaton.states)):
        raise ValueError("seed_pair indices out of range for automaton.states.")

    seed_label = (max(pair_classes.values()) + 1) if pair_classes else 0
    pair_classes[tuple(sorted(seed_pair))] = seed_label

    num_states = len(automaton.states)
    original_current = getattr(automaton, "current_state", None)

    try:
        for k in range(num_states):
            if k == i:
                continue

            start_state = automaton.states[i]
            target_state = automaton.states[k]
            suffix = automaton.get_shortest_path(start_state, target_state)

            if suffix is None:
                logger.debug("No path from %d to %d; skipping. This should not be happening because" \
                "we assume a group automaton as input.", i, k)
                continue

            # Execute from the second seed state with the same suffix.
            automaton.execute_sequence(automaton.states[j], suffix)
            mapped_state = getattr(automaton, "current_state", None)

            if mapped_state is None:
                logger.warning("Automaton did not set current_state after execution; skipping.")
                continue

            mapped_idx = _index_of_state(automaton, mapped_state)
            if mapped_idx is None:
                logger.warning("Mapped state not found in automaton.states; skipping.")
                continue

            candidate_pair = tuple(sorted((k, mapped_idx)))
            if candidate_pair[0] == candidate_pair[1]:
                # Mapped to the same state; not a pair to classify.
                continue

            # Assign or reconcile labels.
            existing = pair_classes.get(candidate_pair)
            if existing is None:
                pair_classes[candidate_pair] = seed_label
                logger.debug("Classified pair %s -> label %d", candidate_pair, seed_label)
            else:
                # Favor the smaller label; propagate across cluster.
                new_label = min(existing, seed_label)
                if existing != new_label:
                    logger.debug(
                        "Reconciling labels for %s: %d -> %d", candidate_pair, existing, new_label
                    )
                pair_classes[candidate_pair] = new_label

                if seed_label != new_label:
                    # Propagate the smaller label across the seed cluster.
                    for other_pair, lab in list(pair_classes.items()):
                        if lab == seed_label:
                            pair_classes[other_pair] = new_label
                    seed_label = new_label

    finally:
        # Restore original current_state to avoid side effects outside this function.
        if original_current is not None:
            automaton.current_state = original_current

    return pair_classes


def check_pair_classification_complete(
    pair_classes: Dict[Tuple[int, int], int],
    num_states: int,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Check if the pair classification is complete, i.e., all unordered pairs of
    distinct states in a DFA have been classified.

    Args:
        pair_classes: The dictionary holding the pair classification table.
        num_states: Number of states in the automaton under consideration.

    Returns:
        A tuple (is_complete, missing_pairs), where:
            - is_complete: True iff all pairs are classified.
            - missing_pairs: List of unordered pairs (i, j) that are not classified.
    """
    expected = math.comb(num_states, 2)
    missing: List[Tuple[int, int]] = []

    for i in range(num_states):
        for j in range(i + 1, num_states):
            if (i, j) not in pair_classes:
                missing.append((i, j))

    is_complete = len(pair_classes) == expected and not missing

    if is_complete:
        logger.info("Pair classification complete.")
    else:
        logger.info(
            "Pair classification incomplete: %d missing of %d.",
            len(missing),
            expected - len(pair_classes),
        )

    return is_complete, missing


def get_keys_from_value(
    d: Dict[Tuple[int, int], int],
    value: int,
) -> List[Tuple[int, int]]:
    """
    Return the keys whose value equals the given value.

    Args:
        d: Dictionary mapping pairs to class labels.
        value: Label to filter by.

    Returns:
        List of keys (pairs) whose associated value equals `value`.
    """
    return [key for key, val in d.items() if val == value]


if __name__ == "__main__":
    # Example usage
    # Demonstrates a DFA-induced promise problem (same as promisehopcroft.ipynb, take a look for more information).
    # A unary 6-cycle DFA is constructed; we classify pair clusters and test satisfiability with z3.

    logging.basicConfig(level=logging.INFO)

    num_states = 6
    states = [DfaState(state_id=f"{i}") for i in range(num_states)]
    input_alphabet = ["s"]

    # Only the initial state is accepting.
    states[0].is_accepting = True

    # Define transitions to form a 6-cycle.
    states[0].transitions["s"] = states[1]
    states[1].transitions["s"] = states[2]
    states[2].transitions["s"] = states[3]
    states[3].transitions["s"] = states[4]
    states[4].transitions["s"] = states[5]
    states[5].transitions["s"] = states[0]

    cyclic6 = Dfa(states[0], states)

    # Populate the Hopcroft-like table.
    # In a unary cyclic DFA, input preserves distance, so expect distance-based clusters.
    cyclic_pair_classes: Dict[Tuple[int, int], int] = {}

    for seed in [(0, 1), (0, 2), (0, 3)]:
        populate_pair_classes(seed, cyclic_pair_classes, cyclic6)

    complete, missing_pairs = check_pair_classification_complete(
        cyclic_pair_classes, num_states
    )
    logger.info("Complete: %s, Missing: %s", complete, missing_pairs)

    # Optional: Check if pairs within a cluster can be made indistinguishable
    # using equality constraints, under a known inequality condition.
    try:
        from z3 import Bool, Solver, sat  # type: ignore

        eq_clusters = [0, 1]
        ineq_pair = (0, 3)

        indices = list(range(num_states))
        bool_vars = [Bool(f"x_{i}") for i in indices]

        for k in eq_clusters:
            solver = Solver()

            # Add inequality constraint for the known accepting and rejecting states.
            solver.add(bool_vars[ineq_pair[0]] != bool_vars[ineq_pair[1]])

            # Add equality constraints for all pairs in the cluster.
            eqps = get_keys_from_value(cyclic_pair_classes, k)
            for i, j in eqps:
                solver.add(bool_vars[i] == bool_vars[j])

            if solver.check() == sat:
                model = solver.model()
                logger.info("Cluster %d: SAT with model: %s", k, model)
            else:
                logger.info("Cluster %d: UNSAT", k)

    except ImportError:
        logger.info("z3-solver not installed; skipping satisfiability check.")
