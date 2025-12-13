"""Behavior predictor using learned automata for state reconstruction."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.automata.learner import AutomataLearner, LearningResult, SimpleAutomaton
from src.automata.sul import abstract_action, abstract_sequence, AbstractionSUL, ABSTRACT_ALPHABET

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Confidence level for predictions."""
    
    HIGH = "high"      # Exact match in training data
    MEDIUM = "medium"  # Pattern matched, some uncertainty
    LOW = "low"        # Extrapolation from limited data
    UNKNOWN = "unknown"  # Cannot predict


@dataclass
class Prediction:
    """A prediction from the behavior predictor."""
    
    predicted_action: str
    predicted_status: str
    confidence: PredictionConfidence
    confidence_score: float  # 0.0 to 1.0
    input_sequence: List[str] = field(default_factory=list)
    alternative_actions: List[Tuple[str, float]] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_action": self.predicted_action,
            "predicted_status": self.predicted_status,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "input_sequence": self.input_sequence,
            "alternative_actions": [
                {"action": a, "score": s}
                for a, s in self.alternative_actions
            ],
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


class BehaviorPredictor:
    """Predictor for agent behavior using learned automata.
    
    Uses the learned automaton to predict the next action an agent
    should take based on its event history.
    """
    
    # Action to status mapping
    ACTION_STATUS_MAP = {
        "validate_product_data": "validated",
        "generate_listing": "generated",
        "confirm_upload": "completed",
        "generate_marketing_copy": "generated",
        "handoff": "handoff_complete",
        "TASK_ASSIGN": "in_progress",
        "TASK_COMPLETE": "completed",
        "failure": "failed",
    }
    
    def __init__(
        self,
        learner: Optional[AutomataLearner] = None,
        min_confidence_threshold: float = 0.5,
    ):
        """Initialize the predictor.
        
        Args:
            learner: AutomataLearner with learned model.
            min_confidence_threshold: Minimum confidence for predictions.
        """
        self.learner = learner or AutomataLearner()
        self.min_confidence_threshold = min_confidence_threshold
        self._learning_result: Optional[LearningResult] = None
        self._event_history: List[Dict[str, Any]] = []
    
    def train(
        self,
        events: List[Dict[str, Any]],
        agent_id: Optional[str] = None,
    ) -> LearningResult:
        """Train the predictor on event data.
        
        Args:
            events: Event log data.
            agent_id: Optional agent ID filter.
            
        Returns:
            LearningResult from training.
        """
        self._learning_result = self.learner.learn(events, agent_id)
        self._event_history = events
        
        logger.info(
            f"Trained predictor: {self._learning_result.num_states} states, "
            f"{self._learning_result.num_transitions} transitions"
        )
        
        return self._learning_result
    
    def predict_next_action(
        self,
        recent_actions: List[str],
        current_status: str = "in_progress",
    ) -> Prediction:
        """Predict the next action given recent action history.
        
        Args:
            recent_actions: List of recent action types.
            current_status: Current workflow status.
            
        Returns:
            Prediction with predicted action and confidence.
        """
        if not self._learning_result or not self._learning_result.success:
            return Prediction(
                predicted_action="unknown",
                predicted_status="unknown",
                confidence=PredictionConfidence.UNKNOWN,
                confidence_score=0.0,
                input_sequence=recent_actions,
                reasoning="No trained model available",
            )
        
        # For sequence learning model: output of input[i] is input[i+1]
        # So we need to execute the sequence and get the output of the last action
        # which represents the next action after the sequence
        
        model = self.learner.get_model()
        if model is None:
            return self._fallback_prediction(recent_actions, current_status)
        
        # Check if using abstraction mode
        use_abstraction = getattr(self.learner, 'use_abstraction', False)
        
        # Abstract input sequence if in abstraction mode
        if use_abstraction:
            query_sequence = abstract_sequence(recent_actions)
        else:
            query_sequence = recent_actions
        
        # Execute the sequence to get to current state, then get next output
        predicted_action = None
        
        if isinstance(model, SimpleAutomaton):
            # SimpleAutomaton: execute sequence and get last output
            model.reset()
            for inp in query_sequence:
                output = model.step(inp)
            # The output of the last action is the next action
            predicted_action = output
        elif hasattr(model, 'execute_sequence'):
            # AALpy model: execute sequence and get last output
            try:
                model.reset_to_initial()
                # AALpy execute_sequence takes (initial_state, input_sequence)
                outputs = model.execute_sequence(model.initial_state, query_sequence)
                predicted_action = outputs[-1] if outputs else None
            except KeyError as e:
                # Action not in learned automaton's alphabet - OOD input
                logger.debug(f"AALpy model doesn't know action: {e}")
                predicted_action = None
        else:
            # Fallback: use learner's method
            predicted_action = self.learner.predict_next_output(query_sequence)
        
        if predicted_action is None or predicted_action == "unknown" or predicted_action == "TERMINAL":
            return self._fallback_prediction(recent_actions, current_status)
        
        # Infer status from predicted action
        predicted_status = self.ACTION_STATUS_MAP.get(predicted_action, "in_progress")
        
        # Calculate confidence
        confidence, confidence_score = self._calculate_confidence(
            recent_actions, predicted_action
        )
        
        # Get alternatives (from transition table)
        alternatives = self._get_alternative_predictions(recent_actions)
        
        return Prediction(
            predicted_action=predicted_action,
            predicted_status=predicted_status,
            confidence=confidence,
            confidence_score=confidence_score,
            input_sequence=recent_actions,
            alternative_actions=alternatives,
            reasoning=f"Predicted sequence based on {len(recent_actions)} recent actions",
        )
    
    def _output_to_action(self, output: str, recent_actions: List[str]) -> str:
        """Legacy method, no longer needed as output IS the action."""
        return output
    
    def _calculate_confidence(
        self,
        input_sequence: List[str],
        predicted_action: str,
    ) -> Tuple[PredictionConfidence, float]:
        """Calculate calibrated confidence that distinguishes memorization from generalization.
        
        Confidence levels:
        - HIGH: Exact sequence seen in training AND automaton predicts correctly (memorization)
        - MEDIUM: Prefix seen, suffix extrapolated via learned automaton (partial generalization)
        - LOW: Sequence not in training, pure automaton generalization (full generalization)
        - UNKNOWN: No valid transition exists in automaton
        
        Args:
            input_sequence: Sequence of actions leading to prediction.
            predicted_action: The predicted next action.
            
        Returns:
            Tuple of (confidence_level, confidence_score).
        """
        if not self._event_history:
            return PredictionConfidence.UNKNOWN, 0.0
        
        # Check if exact sequence + prediction was seen in training (memorization)
        exact_match_count = 0
        prefix_match_count = 0
        total_sequences = 0
        
        for event_seq in self._group_events_by_thread():
            actions = [e.get("action_type", "") for e in event_seq]
            
            # Check for exact sequence match (memorization)
            for i in range(len(actions) - len(input_sequence)):
                if actions[i:i+len(input_sequence)] == input_sequence:
                    total_sequences += 1
                    # Check if next action matches prediction
                    if i + len(input_sequence) < len(actions):
                        next_action = actions[i+len(input_sequence)]
                        if next_action == predicted_action:
                            exact_match_count += 1
                    
                    # Check if prefix matches (for partial generalization)
                    if len(input_sequence) > 1:
                        prefix = input_sequence[:-1]
                        if i + len(prefix) < len(actions) and actions[i:i+len(prefix)] == prefix:
                            prefix_match_count += 1
        
        # Check if automaton has a valid transition (even if not in training)
        has_valid_transition = False
        model = self.learner.get_model() if self.learner else None
        if model:
            try:
                # Try to execute the sequence and see if we get a valid prediction
                if isinstance(model, SimpleAutomaton):
                    model.reset()
                    for inp in input_sequence:
                        output = model.step(inp)
                    # If we got here without error, transition exists
                    has_valid_transition = (output != "unknown" and output != "TERMINAL")
                elif hasattr(model, 'execute_sequence'):
                    model.reset_to_initial()
                    outputs = model.execute_sequence(model.initial_state, input_sequence)
                    has_valid_transition = (outputs and outputs[-1] not in ("unknown", "TERMINAL"))
            except Exception:
                has_valid_transition = False
        
        # Determine confidence level
        if total_sequences == 0:
            # No training data match - check if automaton can generalize
            if has_valid_transition:
                # Pure generalization - automaton learned pattern but sequence unseen
                return PredictionConfidence.LOW, 0.4
            else:
                # No valid transition - cannot predict
                return PredictionConfidence.UNKNOWN, 0.0
        
        # Calculate exact match ratio (memorization)
        exact_match_ratio = exact_match_count / total_sequences if total_sequences > 0 else 0.0
        
        if exact_match_ratio >= 0.8:
            # High memorization - exact sequence seen frequently
            return PredictionConfidence.HIGH, exact_match_ratio
        elif exact_match_ratio >= 0.3 or prefix_match_count > 0:
            # Medium - partial match or prefix seen, suffix extrapolated
            # Blend exact match with prefix match for score
            prefix_ratio = prefix_match_count / max(total_sequences, 1)
            blended_score = 0.6 * exact_match_ratio + 0.4 * prefix_ratio
            return PredictionConfidence.MEDIUM, min(0.7, blended_score)
        elif has_valid_transition:
            # Low - sequence not in training but automaton can generalize
            return PredictionConfidence.LOW, 0.3
        else:
            # Unknown - no valid transition
            return PredictionConfidence.UNKNOWN, 0.0
            
    def _group_events_by_thread(self) -> List[List[Dict[str, Any]]]:
        """Helper to group events by thread."""
        threads = {}
        for event in self._event_history:
            tid = event.get("thread_id", "default")
            if tid not in threads:
                threads[tid] = []
            threads[tid].append(event)
        
        # Sort each thread by timestamp
        for tid in threads:
            threads[tid].sort(key=lambda x: x.get("timestamp", ""))
            
        return list(threads.values())
    
    def _get_alternative_predictions(
        self,
        input_sequence: List[str],
    ) -> List[Tuple[str, float]]:
        """Get alternative predictions by querying the automaton."""
        if not self.learner or not self.learner.get_model():
            return []
            
        # Get learned model (SimpleAutomaton or AALpy model)
        model = self.learner.get_model()
        
        # Try to find current state
        current_state = 0
        if hasattr(model, 'initial_state'):
            current_state = model.initial_state
            
        # Advance state
        valid_path = True
        
        # Handle AALpy model vs SimpleAutomaton
        is_aalpy = not hasattr(model, 'sul') 
        
        if is_aalpy:
            # AALpy logic would go here, simplified fallback
            return []
        
        # SimpleAutomaton logic
        for inp in input_sequence:
            key = (current_state, inp)
            if hasattr(model, 'sul') and key in model.sul._transitions:
                current_state, _ = model.sul._transitions[key]
            else:
                valid_path = False
                break
        
        if not valid_path:
            return []
            
        # Check outgoing transitions from current state
        alternatives = []
        if hasattr(model, 'sul'):
            total_trans = 0
            counts = {}
            
            for (state, inp), (next_s, out) in model.sul._transitions.items():
                if state == current_state:
                    # In our sequence model, the input IS the next valid action
                    counts[inp] = counts.get(inp, 0) + 1
                    total_trans += 1
            
            for act, count in counts.items():
                alternatives.append((act, count/total_trans))
                
        return alternatives
    
    def _fallback_prediction(
        self,
        recent_actions: List[str],
        current_status: str,
    ) -> Prediction:
        """Fallback prediction when model can't predict."""
        # Use workflow heuristics
        if not recent_actions:
            return Prediction(
                predicted_action="validate_product_data",
                predicted_status="pending",
                confidence=PredictionConfidence.LOW,
                confidence_score=0.4,
                input_sequence=recent_actions,
                reasoning="Fallback: starting workflow from beginning",
            )
        
        last_action = recent_actions[-1]
        
        # Workflow progression
        next_actions = {
            "validate_product_data": "generate_listing",
            "generate_listing": "confirm_upload",
            "confirm_upload": "handoff",
            "TASK_ASSIGN": "validate_product_data",
        }
        
        predicted = next_actions.get(last_action, "continue")
        
        return Prediction(
            predicted_action=predicted,
            predicted_status="in_progress",
            confidence=PredictionConfidence.LOW,
            confidence_score=0.4,
            input_sequence=recent_actions,
            reasoning=f"Fallback: workflow heuristic after {last_action}",
        )
    
    def predict_state_for_reconstruction(
        self,
        agent_id: str,
        last_checkpoint_state: Dict[str, Any],
        events_since_checkpoint: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Predict reconstructed state using automata.
        
        This is the main entry point for state reconstruction.
        
        Args:
            agent_id: ID of the agent to reconstruct.
            last_checkpoint_state: State from last checkpoint.
            events_since_checkpoint: Events that occurred after checkpoint.
            
        Returns:
            Predicted state dictionary.
        """
        # Extract action sequence from events
        action_sequence = [
            e.get("action_type", "unknown")
            for e in events_since_checkpoint
        ]
        
        # Get prediction
        prediction = self.predict_next_action(
            action_sequence,
            current_status=last_checkpoint_state.get("status", "in_progress"),
        )
        
        # Build reconstructed state
        reconstructed_state = dict(last_checkpoint_state)
        
        # Update state based on prediction
        if prediction.confidence_score >= self.min_confidence_threshold:
            # High confidence: use prediction
            reconstructed_state["current_step"] = len(action_sequence)
            reconstructed_state["status"] = prediction.predicted_status
            reconstructed_state["next_action"] = prediction.predicted_action
            reconstructed_state["reconstruction_method"] = "automata"
            reconstructed_state["reconstruction_confidence"] = prediction.confidence_score
        else:
            # Low confidence: mark for LLM reconstruction
            reconstructed_state["reconstruction_method"] = "needs_llm"
            reconstructed_state["reconstruction_confidence"] = prediction.confidence_score
        
        # Add prediction metadata
        reconstructed_state["prediction"] = prediction.to_dict()
        
        return reconstructed_state
    
    @property
    def is_trained(self) -> bool:
        """Check if predictor has been trained."""
        return self._learning_result is not None and self._learning_result.success
    
    @property
    def model_accuracy(self) -> float:
        """Estimate model accuracy (based on training)."""
        if not self._learning_result:
            return 0.0
        
        # Simple heuristic based on model complexity
        if self._learning_result.num_states == 0:
            return 0.0
        
        # More states usually means better fit to data
        return min(0.95, 0.5 + (self._learning_result.num_states / 20) * 0.45)

