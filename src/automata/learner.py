"""L* Automata learner for agent behavior patterns."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from src.automata.sul import AgentBehaviorSUL, EventSequence, create_sul_from_events

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """Result of automata learning."""
    
    success: bool
    num_states: int
    num_transitions: int
    learning_time_ms: float
    queries_made: int = 0
    equivalence_queries: int = 0
    model_type: str = "mealy"
    automaton: Optional[Any] = None  # The learned automaton
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without automaton object)."""
        return {
            "success": self.success,
            "num_states": self.num_states,
            "num_transitions": self.num_transitions,
            "learning_time_ms": self.learning_time_ms,
            "queries_made": self.queries_made,
            "equivalence_queries": self.equivalence_queries,
            "model_type": self.model_type,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class AutomataLearner:
    """L* automata learner for agent behavior.
    
    Uses AALpy to learn Mealy machines from agent event logs.
    The learned automaton can predict agent behavior based on input sequences.
    """
    
    def __init__(
        self,
        model_type: str = "mealy",
        eq_oracle_type: str = "random_walk",
        max_learning_rounds: int = 100,
        random_walk_steps: int = 5000,
    ):
        """Initialize the learner.
        
        Args:
            model_type: Type of automaton to learn ("mealy", "moore", "dfa").
            eq_oracle_type: Equivalence oracle type ("random_walk", "w_method").
            max_learning_rounds: Maximum learning rounds.
            random_walk_steps: Number of steps for random walk oracle.
        """
        self.model_type = model_type
        self.eq_oracle_type = eq_oracle_type
        self.max_learning_rounds = max_learning_rounds
        self.random_walk_steps = random_walk_steps
        
        self._learned_model = None
        self._sul: Optional[AgentBehaviorSUL] = None
    
    def learn(
        self,
        events: List[Dict[str, Any]],
        agent_id: Optional[str] = None,
    ) -> LearningResult:
        """Learn an automaton from event logs.
        
        Args:
            events: List of event dictionaries from event store.
            agent_id: Optional agent ID to filter events.
            
        Returns:
            LearningResult with the learned automaton.
        """
        start_time = time.perf_counter()
        
        if len(events) < 10:
            return LearningResult(
                success=False,
                num_states=0,
                num_transitions=0,
                learning_time_ms=0,
                error="Insufficient events for learning (need at least 10)",
            )
        
        try:
            # Create SUL from events
            self._sul = create_sul_from_events(events, agent_id)
            
            # Try to use AALpy for learning
            try:
                result = self._learn_with_aalpy()
            except (ImportError, TypeError, AttributeError) as aalpy_error:
                # AALpy not available or has API compatibility issues, use fallback
                logger.warning(f"AALpy learning failed ({aalpy_error}), using simple automaton")
                result = self._learn_simple()
            
        except ImportError:
            # AALpy not available, use fallback
            logger.warning("AALpy not available, using simple automaton")
            result = self._learn_simple()
            
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            result = LearningResult(
                success=False,
                num_states=0,
                num_transitions=0,
                learning_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )
        
        result.learning_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    def _learn_with_aalpy(self) -> LearningResult:
        """Learn using AALpy L* algorithm."""
        from aalpy.SULs import MealySUL
        from aalpy.oracles import RandomWalkEqOracle
        from aalpy.learning_algs import run_Lstar
        
        # Create AALpy-compatible SUL wrapper
        class AALpySULWrapper(MealySUL):
            def __init__(wrapper_self, sul: AgentBehaviorSUL):
                super().__init__()
                wrapper_self.sul = sul
                wrapper_self.input_alphabet = sul.input_alphabet
            
            def pre(wrapper_self):
                wrapper_self.sul.pre()
            
            def post(wrapper_self):
                wrapper_self.sul.post()
            
            def step(wrapper_self, letter):
                return wrapper_self.sul.step(letter)
        
        # Wrap our SUL
        aalpy_sul = AALpySULWrapper(self._sul)
        
        # Create equivalence oracle - try different API signatures for compatibility
        try:
            # Try newer AALpy API (sul, alphabet, num_steps)
            eq_oracle = RandomWalkEqOracle(
                aalpy_sul,
                aalpy_sul.input_alphabet,
                num_steps=self.random_walk_steps,
            )
        except TypeError:
            # Fall back to older API (alphabet, sul, num_steps)
            eq_oracle = RandomWalkEqOracle(
                aalpy_sul.input_alphabet,
                aalpy_sul,
                num_steps=self.random_walk_steps,
            )
        
        # Run L* learning
        learned_automaton = run_Lstar(
            aalpy_sul.input_alphabet,
            aalpy_sul,
            eq_oracle,
            automaton_type=self.model_type,
            print_level=0,
        )
        
        self._learned_model = learned_automaton
        
        # Count states and transitions
        num_states = len(learned_automaton.states) if hasattr(learned_automaton, 'states') else 0
        num_transitions = sum(
            len(s.transitions) for s in learned_automaton.states
        ) if hasattr(learned_automaton, 'states') else 0
        
        return LearningResult(
            success=True,
            num_states=num_states,
            num_transitions=num_transitions,
            learning_time_ms=0,  # Will be set by caller
            model_type=self.model_type,
            automaton=learned_automaton,
        )
    
    def _learn_simple(self) -> LearningResult:
        """Simple fallback learning without AALpy."""
        # Use our SUL's built-in transition table as a simple automaton
        self._learned_model = SimpleAutomaton(self._sul)
        
        return LearningResult(
            success=True,
            num_states=self._sul.num_states,
            num_transitions=self._sul.num_transitions,
            learning_time_ms=0,
            model_type="simple",
            automaton=self._learned_model,
        )
    
    def predict_next_output(self, input_sequence: List[str]) -> Optional[str]:
        """Predict the next output given an input sequence.
        
        Args:
            input_sequence: Sequence of inputs seen so far.
            
        Returns:
            Predicted output for the next input, or None if not learned.
        """
        if self._learned_model is None:
            return None
        
        # Check for SimpleAutomaton first (since it also has execute_sequence)
        if isinstance(self._learned_model, SimpleAutomaton):
            # Simple model uses predict() which resets and executes
            return self._learned_model.predict(input_sequence)
        elif hasattr(self._learned_model, 'execute_sequence'):
            # AALpy model
            self._learned_model.reset_to_initial()
            outputs = self._learned_model.execute_sequence(
                self._learned_model.initial_state,
                input_sequence,
            )
            return outputs[-1] if outputs else None
        
        return None
    
    def get_model(self) -> Optional[Any]:
        """Get the learned automaton model."""
        return self._learned_model
    
    def save_model(self, filepath: str) -> bool:
        """Save the learned model to a file.
        
        Args:
            filepath: Path to save the model.
            
        Returns:
            True if saved successfully.
        """
        if self._learned_model is None:
            return False
        
        try:
            if hasattr(self._learned_model, 'save'):
                self._learned_model.save(filepath)
            else:
                # Save as DOT format for visualization
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self._learned_model, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False


class SimpleAutomaton:
    """Simple automaton implementation as fallback when AALpy unavailable."""
    
    def __init__(self, sul: AgentBehaviorSUL):
        """Initialize from SUL."""
        self.sul = sul
        self._current_state = 0
        self.initial_state = 0  # For compatibility with AALpy models
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._current_state = 0
    
    def reset_to_initial(self) -> None:
        """Reset to initial state (AALpy compatibility)."""
        self._current_state = 0
    
    def step(self, input_symbol: str) -> str:
        """Execute one step."""
        key = (self._current_state, input_symbol)
        if key in self.sul._transitions:
            next_state, output = self.sul._transitions[key]
            self._current_state = next_state
            return output
        return "unknown"
    
    def predict(self, input_sequence: List[str]) -> Optional[str]:
        """Predict output after executing input sequence."""
        self.reset()
        last_output = None
        for inp in input_sequence:
            last_output = self.step(inp)
        return last_output
    
    def execute_sequence(self, input_sequence: List[str]) -> List[str]:
        """Execute sequence and return all outputs."""
        self.reset()
        return [self.step(inp) for inp in input_sequence]


def learn_agent_behavior(
    events: List[Dict[str, Any]],
    agent_id: Optional[str] = None,
    model_type: str = "mealy",
) -> LearningResult:
    """Convenience function to learn agent behavior.
    
    Args:
        events: Event log data.
        agent_id: Optional agent ID filter.
        model_type: Type of automaton.
        
    Returns:
        LearningResult with learned automaton.
    """
    learner = AutomataLearner(model_type=model_type)
    return learner.learn(events, agent_id)

