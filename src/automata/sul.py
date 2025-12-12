"""System Under Learning (SUL) adapter for agent behavior learning."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
        """Get sequence of output symbols (agent actions/responses)."""
        outputs = []
        for event in self.events:
            # Extract output from event
            output_data = event.get("output_data", {})
            if isinstance(output_data, dict):
                status = output_data.get("status", "unknown")
                outputs.append(status)
            else:
                outputs.append("unknown")
        return outputs


class AgentBehaviorSUL:
    """System Under Learning for agent behavior.
    
    Implements the interface required by AALpy for L* learning.
    Maps agent event logs to input/output sequences for automata learning.
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
    ]
    
    # Default output alphabet (agent actions / responses)
    DEFAULT_OUTPUT_ALPHABET = [
        "pending",
        "in_progress",
        "completed",
        "failed",
        "validated",
        "generated",
        "confirmed",
        "handoff_complete",
        "error",
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
        self.output_alphabet = output_alphabet or self.DEFAULT_OUTPUT_ALPHABET
        
        # Build transition table from events
        self._transitions: Dict[Tuple[int, str], Tuple[int, str]] = {}
        self._current_state = 0
        self._num_states = 1
        
        self._build_transitions()
    
    def _build_transitions(self) -> None:
        """Build transition table from event sequences."""
        # Track state transitions across all sequences
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
                
                # Create state key
                state_key = f"s{current_state}_{inp_norm}"
                
                if state_key not in state_map:
                    state_map[state_key] = next_state_id
                    next_state_id += 1
                
                next_state = state_map[state_key]
                
                # Record transition
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
) -> AgentBehaviorSUL:
    """Create SUL from a flat list of events.
    
    Args:
        events: List of event dictionaries from event store.
        agent_id: Optional agent ID to filter events.
        
    Returns:
        AgentBehaviorSUL instance.
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
    
    return AgentBehaviorSUL(event_sequences)


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
    
    return list(inputs), list(outputs)

