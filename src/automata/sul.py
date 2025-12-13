"""System Under Learning (SUL) adapter for agent behavior learning."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ACTION ABSTRACTION MAPPING
# ============================================================================
# Maps concrete actions to abstract semantic categories.
# This enables generalization by learning patterns at the abstract level.

ACTION_ABSTRACTIONS = {
    # Protocol actions (unchanged - already abstract)
    "TASK_ASSIGN": "TASK_ASSIGN",
    "TASK_COMPLETE": "TASK_COMPLETE",
    "REQUEST_CONTEXT": "REQUEST_CONTEXT",
    "PROVIDE_CONTEXT": "PROVIDE_CONTEXT",
    
    # Step 1: Initial processing/setup → STEP_1
    # Both workflows start with an initial processing step
    "validate_product_data": "STEP_1",       # Product: validate first
    "generate_marketing_copy": "STEP_1",     # Marketing: generate first
    
    # Step 2: Secondary processing → STEP_2
    # Both workflows have a secondary transformation step
    "generate_listing": "STEP_2",            # Product: generate listing
    "review_copy": "STEP_2",                 # Marketing: review copy
    
    # Step 3: Finalization/publishing → STEP_3
    # Both workflows end with a finalization step
    "confirm_upload": "STEP_3",              # Product: confirm upload
    "publish_campaign": "STEP_3",            # Marketing: publish campaign
    
    # Agent coordination actions
    "handoff": "HANDOFF",
    "recovery": "RECOVERY",
    
    # Fallback
    "unknown": "UNKNOWN",
}

# Abstract workflow pattern (what L* should learn):
# TASK_ASSIGN → STEP_1 → STEP_2 → STEP_3 → TASK_COMPLETE
# Both product and marketing workflows follow this pattern!

# Reverse mapping for de-abstraction (if needed)
ABSTRACT_TO_CONCRETE = {
    "TASK_ASSIGN": ["TASK_ASSIGN"],
    "TASK_COMPLETE": ["TASK_COMPLETE"],
    "REQUEST_CONTEXT": ["REQUEST_CONTEXT"],
    "PROVIDE_CONTEXT": ["PROVIDE_CONTEXT"],
    "STEP_1": ["validate_product_data", "generate_marketing_copy"],
    "STEP_2": ["generate_listing", "review_copy"],
    "STEP_3": ["confirm_upload", "publish_campaign"],
    "HANDOFF": ["handoff"],
    "RECOVERY": ["recovery"],
    "UNKNOWN": ["unknown"],
}

# Abstract alphabet (used for L* learning)
ABSTRACT_ALPHABET = [
    "TASK_ASSIGN",
    "TASK_COMPLETE",
    "REQUEST_CONTEXT",
    "PROVIDE_CONTEXT",
    "STEP_1",
    "STEP_2",
    "STEP_3",
    "HANDOFF",
    "RECOVERY",
]


def abstract_action(action: str) -> str:
    """Map a concrete action to its abstract category."""
    return ACTION_ABSTRACTIONS.get(action, "UNKNOWN")


def abstract_sequence(sequence: List[str]) -> List[str]:
    """Map a sequence of concrete actions to abstract categories."""
    return [abstract_action(action) for action in sequence]


@dataclass
class EventSequence:
    """A sequence of agent events for learning."""
    
    agent_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.events)
    
    def get_input_sequence(self) -> List[str]:
        """Get sequence of input symbols (message types/triggers)."""
        inputs = []
        for event in self.events:
            action_type = event.get("action_type", "unknown")
            inputs.append(action_type)
        return inputs
    
    def get_output_sequence(self) -> List[str]:
        """Get sequence of output symbols (next action types).
        
        For sequence learning, the output for input[i] is input[i+1].
        """
        inputs = self.get_input_sequence()
        outputs = []
        
        # Output is the NEXT action
        for i in range(len(inputs) - 1):
            outputs.append(inputs[i+1])
            
        # Last action has "TERMINAL" output
        outputs.append("TERMINAL")
        
        return outputs


class AgentBehaviorSUL:
    """System Under Learning for agent behavior.
    
    Implements the interface required by AALpy for L* learning.
    Maps agent event logs to input/output sequences for automata learning.
    
    Learning Model:
        Input: Current Action Type (e.g., "validate_product_data")
        Output: Next Action Type (e.g., "generate_listing")
        
    This allows the automaton to learn valid workflow sequences.
    """
    
    # Default input alphabet (message types / triggers)
    DEFAULT_INPUT_ALPHABET = [
        "TASK_ASSIGN",
        "TASK_COMPLETE", 
        "REQUEST_CONTEXT",
        "PROVIDE_CONTEXT",
        "validate_product_data",
        "generate_listing",
        "confirm_upload",
        "generate_marketing_copy",
        "handoff",
        "failure",
        "recovery",
    ]
    
    # Default output alphabet (next actions)
    # Includes all inputs + special terminals
    DEFAULT_OUTPUT_ALPHABET = DEFAULT_INPUT_ALPHABET + [
        "TERMINAL",
        "unknown",
    ]
    
    def __init__(
        self,
        event_sequences: List[EventSequence],
        input_alphabet: Optional[List[str]] = None,
        output_alphabet: Optional[List[str]] = None,
    ):
        """Initialize the SUL.
        
        Args:
            event_sequences: List of event sequences from agent logs.
            input_alphabet: Custom input alphabet. Uses default if None.
            output_alphabet: Custom output alphabet. Uses default if None.
        """
        self.event_sequences = event_sequences
        self.input_alphabet = input_alphabet or self.DEFAULT_INPUT_ALPHABET
        # Ensure output alphabet covers all inputs (since output is next input)
        self.output_alphabet = output_alphabet or (self.input_alphabet + ["TERMINAL", "unknown"])
        
        # Build transition table from events
        self._transitions: Dict[Tuple[int, str], Tuple[int, str]] = {}
        self._current_state = 0
        self._num_states = 1
        
        self._build_transitions()
    
    def _build_transitions(self) -> None:
        """Build transition table from event sequences."""
        # Track state transitions across all sequences
        # State key: "s{state_id}_{prev_action}" -> effectively 1-history lookback
        state_map: Dict[str, int] = {"initial": 0}
        next_state_id = 1
        
        for seq in self.event_sequences:
            current_state = 0
            inputs = seq.get_input_sequence()
            outputs = seq.get_output_sequence()
            
            for inp, out in zip(inputs, outputs):
                # Normalize input/output to alphabet
                inp_norm = inp if inp in self.input_alphabet else "unknown"
                out_norm = out if out in self.output_alphabet else "unknown"
                
                # Create state key based on path
                # For Mealy machine, state represents "history context"
                # Simple unique path approach:
                state_key = f"s{current_state}_{inp_norm}"
                
                if state_key not in state_map:
                    state_map[state_key] = next_state_id
                    next_state_id += 1
                
                next_state = state_map[state_key]
                
                # Record transition: (current_state, input) -> (next_state, output)
                # Where output is effectively the *predicted next input*
                self._transitions[(current_state, inp_norm)] = (next_state, out_norm)
                current_state = next_state
        
        self._num_states = next_state_id
        logger.info(f"Built SUL with {self._num_states} states and {len(self._transitions)} transitions")
    
    def pre(self) -> None:
        """Reset SUL to initial state."""
        self._current_state = 0
    
    def post(self) -> None:
        """Cleanup after query (no-op for this SUL)."""
        pass
    
    def step(self, input_symbol: str) -> str:
        """Execute one step and return output.
        
        Args:
            input_symbol: Input symbol from alphabet.
            
        Returns:
            Output symbol.
        """
        # Normalize input
        inp = input_symbol if input_symbol in self.input_alphabet else "unknown"
        
        # Look up transition
        key = (self._current_state, inp)
        if key in self._transitions:
            next_state, output = self._transitions[key]
            self._current_state = next_state
            return output
        else:
            # Default: stay in current state, output unknown
            return "unknown"
    
    def query(self, input_sequence: List[str]) -> List[str]:
        """Execute a sequence of inputs and return outputs.
        
        Args:
            input_sequence: Sequence of input symbols.
            
        Returns:
            Sequence of output symbols.
        """
        self.pre()
        outputs = []
        for inp in input_sequence:
            out = self.step(inp)
            outputs.append(out)
        self.post()
        return outputs
    
    def get_alphabet(self) -> Tuple[List[str], List[str]]:
        """Get input and output alphabets.
        
        Returns:
            Tuple of (input_alphabet, output_alphabet).
        """
        return self.input_alphabet, self.output_alphabet
    
    @property
    def num_states(self) -> int:
        """Get number of states in the learned model."""
        return self._num_states
    
    @property
    def num_transitions(self) -> int:
        """Get number of transitions."""
        return len(self._transitions)


def create_sul_from_events(
    events: List[Dict[str, Any]],
    agent_id: Optional[str] = None,
    extract_alphabet: bool = True,
) -> AgentBehaviorSUL:
    """Create SUL from a flat list of events.
    
    Args:
        events: List of event dictionaries from event store.
        agent_id: Optional agent ID to filter events.
        extract_alphabet: If True, extract input/output alphabets from events.
        
    Returns:
        AgentBehaviorSUL instance.
    """
    # Filter by agent if specified
    if agent_id:
        events = [e for e in events if e.get("agent_id") == agent_id]
    
    # Extract alphabets from events if requested
    input_alphabet = None
    output_alphabet = None
    if extract_alphabet:
        input_alphabet, output_alphabet = extract_alphabet_from_events(events)
        # Ensure we have at least the default alphabet
        if not input_alphabet:
            input_alphabet = None  # Will use default
    
    # Group events by thread_id to create sequences
    sequences_by_thread: Dict[str, List[Dict[str, Any]]] = {}
    
    for event in events:
        thread_id = event.get("thread_id", "default")
        if thread_id not in sequences_by_thread:
            sequences_by_thread[thread_id] = []
        sequences_by_thread[thread_id].append(event)
    
    # Sort events within each sequence by timestamp
    event_sequences = []
    for thread_id, thread_events in sequences_by_thread.items():
        # Sort by timestamp if available
        sorted_events = sorted(
            thread_events,
            key=lambda e: e.get("timestamp", ""),
        )
        
        seq = EventSequence(
            agent_id=agent_id or "unknown",
            events=sorted_events,
        )
        event_sequences.append(seq)
    
    return AgentBehaviorSUL(
        event_sequences,
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
    )


def extract_alphabet_from_events(events: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract input/output alphabets from events.
    
    Args:
        events: List of event dictionaries.
        
    Returns:
        Tuple of (input_alphabet, output_alphabet).
    """
    inputs = set()
    outputs = set()
    
    for event in events:
        action_type = event.get("action_type", "")
        if action_type:
            inputs.add(action_type)
        
        output_data = event.get("output_data", {})
        if isinstance(output_data, dict):
            status = output_data.get("status", "")
            if status:
                outputs.add(status)
    
    # Sort for reproducibility
    input_list = sorted(list(inputs))
    output_list = sorted(list(outputs))
    
    # Ensure output alphabet includes all inputs (since output is next input in sequence model)
    output_list = sorted(list(set(output_list) | set(input_list) | {"TERMINAL", "unknown"}))
    
    return input_list, output_list


# ============================================================================
# ABSTRACTION SUL - Enables True L* Generalization
# ============================================================================

class AbstractionSUL:
    """SUL that operates at the abstract action level for generalization.
    
    Instead of memorizing exact sequences, this SUL:
    1. Maps concrete actions to abstract categories
    2. Learns transitions between abstract states
    3. Generates responses based on abstract workflow position
    
    This enables generalization to OOD inputs that share the same abstract pattern.
    """
    
    def __init__(
        self,
        event_sequences: List[EventSequence],
        abstraction_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize the abstraction SUL.
        
        Args:
            event_sequences: List of event sequences from agent logs.
            abstraction_map: Custom abstraction mapping. Uses ACTION_ABSTRACTIONS if None.
        """
        self.event_sequences = event_sequences
        self.abstraction_map = abstraction_map or ACTION_ABSTRACTIONS
        
        # Use abstract alphabet
        self.input_alphabet = ABSTRACT_ALPHABET.copy()
        self.output_alphabet = ABSTRACT_ALPHABET + ["TERMINAL", "UNKNOWN"]
        
        # Build abstract transition table
        self._transitions: Dict[Tuple[int, str], Tuple[int, str]] = {}
        self._transition_counts: Dict[Tuple[int, str, str], int] = {}  # For probability
        self._current_state = 0
        self._num_states = 1
        
        self._build_abstract_transitions()
    
    def _abstract(self, action: str) -> str:
        """Map concrete action to abstract category."""
        return self.abstraction_map.get(action, "UNKNOWN")
    
    def _build_abstract_transitions(self) -> None:
        """Build transition table from abstract event sequences.
        
        Key difference from AgentBehaviorSUL:
        - Groups similar sequences by abstract pattern
        - Uses workflow position (abstract state) not exact history
        - Learns: TASK_ASSIGN → PROCESS → GENERATE → PUBLISH → TASK_COMPLETE
        """
        # Track abstract state transitions
        # State 0 = initial, State N = after N abstract workflow steps
        state_map: Dict[str, int] = {"initial": 0}
        next_state_id = 1
        
        for seq in self.event_sequences:
            current_state = 0
            inputs = seq.get_input_sequence()
            outputs = seq.get_output_sequence()
            
            for inp, out in zip(inputs, outputs):
                # Abstract both input and output
                abstract_inp = self._abstract(inp)
                abstract_out = self._abstract(out)
                
                # Skip UNKNOWN inputs
                if abstract_inp == "UNKNOWN":
                    continue
                
                # Create state key based on abstract workflow position
                # This groups all concrete actions that map to same abstract step
                state_key = f"s{current_state}_{abstract_inp}"
                
                if state_key not in state_map:
                    state_map[state_key] = next_state_id
                    next_state_id += 1
                
                next_state = state_map[state_key]
                
                # Count transitions for probability-based selection
                trans_key = (current_state, abstract_inp, abstract_out)
                self._transition_counts[trans_key] = self._transition_counts.get(trans_key, 0) + 1
                
                # Record most common transition
                existing_key = (current_state, abstract_inp)
                if existing_key in self._transitions:
                    # Check if new output is more common
                    existing_out = self._transitions[existing_key][1]
                    existing_count = self._transition_counts.get((current_state, abstract_inp, existing_out), 0)
                    new_count = self._transition_counts[trans_key]
                    if new_count > existing_count:
                        self._transitions[existing_key] = (next_state, abstract_out)
                else:
                    self._transitions[existing_key] = (next_state, abstract_out)
                
                current_state = next_state
        
        self._num_states = next_state_id
        logger.info(f"Built AbstractionSUL with {self._num_states} states and {len(self._transitions)} transitions")
    
    def pre(self) -> None:
        """Reset SUL to initial state."""
        self._current_state = 0
    
    def post(self) -> None:
        """Cleanup after query."""
        pass
    
    def step(self, input_symbol: str) -> str:
        """Execute one step with abstract input and return abstract output.
        
        Args:
            input_symbol: Input symbol (can be concrete or abstract).
            
        Returns:
            Abstract output symbol.
        """
        # Abstract the input if it's concrete
        abstract_inp = self._abstract(input_symbol) if input_symbol in self.abstraction_map else input_symbol
        
        # Normalize to alphabet
        if abstract_inp not in self.input_alphabet:
            abstract_inp = "UNKNOWN"
        
        # Look up transition
        key = (self._current_state, abstract_inp)
        if key in self._transitions:
            next_state, output = self._transitions[key]
            self._current_state = next_state
            return output
        else:
            # No transition - return UNKNOWN
            return "UNKNOWN"
    
    def query(self, input_sequence: List[str]) -> List[str]:
        """Execute a sequence of inputs and return outputs."""
        self.pre()
        outputs = []
        for inp in input_sequence:
            out = self.step(inp)
            outputs.append(out)
        self.post()
        return outputs
    
    @property
    def num_states(self) -> int:
        return self._num_states
    
    @property
    def num_transitions(self) -> int:
        return len(self._transitions)


def create_abstraction_sul_from_events(
    events: List[Dict[str, Any]],
    agent_id: Optional[str] = None,
) -> AbstractionSUL:
    """Create AbstractionSUL from a flat list of events.
    
    Args:
        events: List of event dictionaries from event store.
        agent_id: Optional agent ID to filter events.
        
    Returns:
        AbstractionSUL instance.
    """
    # Filter by agent if specified
    if agent_id:
        events = [e for e in events if e.get("agent_id") == agent_id]
    
    # Group events by thread_id to create sequences
    sequences_by_thread: Dict[str, List[Dict[str, Any]]] = {}
    
    for event in events:
        thread_id = event.get("thread_id", "default")
        if thread_id not in sequences_by_thread:
            sequences_by_thread[thread_id] = []
        sequences_by_thread[thread_id].append(event)
    
    # Sort events within each sequence by timestamp
    event_sequences = []
    for thread_id, thread_events in sequences_by_thread.items():
        sorted_events = sorted(
            thread_events,
            key=lambda e: e.get("timestamp", ""),
        )
        
        seq = EventSequence(
            agent_id=agent_id or "unknown",
            events=sorted_events,
        )
        event_sequences.append(seq)
    
    return AbstractionSUL(event_sequences)
