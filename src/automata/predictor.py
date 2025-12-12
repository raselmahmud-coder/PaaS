"""Behavior predictor using learned automata for state reconstruction."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.automata.learner import AutomataLearner, LearningResult

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
        
        # Get prediction from learned model
        predicted_output = self.learner.predict_next_output(recent_actions)
        
        if predicted_output is None or predicted_output == "unknown":
            return self._fallback_prediction(recent_actions, current_status)
        
        # Determine action from output
        predicted_action = self._output_to_action(predicted_output, recent_actions)
        
        # Calculate confidence
        confidence, confidence_score = self._calculate_confidence(
            recent_actions, predicted_output
        )
        
        # Get alternatives
        alternatives = self._get_alternative_predictions(recent_actions)
        
        return Prediction(
            predicted_action=predicted_action,
            predicted_status=predicted_output,
            confidence=confidence,
            confidence_score=confidence_score,
            input_sequence=recent_actions,
            alternative_actions=alternatives,
            reasoning=f"Predicted based on {len(recent_actions)} recent actions",
        )
    
    def _output_to_action(
        self,
        output: str,
        recent_actions: List[str],
    ) -> str:
        """Map output status to predicted action."""
        # Reverse lookup from ACTION_STATUS_MAP
        for action, status in self.ACTION_STATUS_MAP.items():
            if status == output:
                return action
        
        # Infer from recent actions
        if recent_actions:
            last_action = recent_actions[-1]
            
            # Predict next step in workflow
            workflow_sequence = [
                "validate_product_data",
                "generate_listing",
                "confirm_upload",
            ]
            
            if last_action in workflow_sequence:
                idx = workflow_sequence.index(last_action)
                if idx < len(workflow_sequence) - 1:
                    return workflow_sequence[idx + 1]
        
        return "continue"
    
    def _calculate_confidence(
        self,
        input_sequence: List[str],
        predicted_output: str,
    ) -> Tuple[PredictionConfidence, float]:
        """Calculate prediction confidence."""
        if not self._event_history:
            return PredictionConfidence.LOW, 0.3
        
        # Count how many times this pattern appears in training data
        pattern_count = 0
        total_sequences = 0
        
        for event in self._event_history:
            action = event.get("action_type", "")
            if action in input_sequence:
                total_sequences += 1
                output = event.get("output_data", {})
                if isinstance(output, dict) and output.get("status") == predicted_output:
                    pattern_count += 1
        
        if total_sequences == 0:
            return PredictionConfidence.LOW, 0.3
        
        confidence_score = pattern_count / total_sequences
        
        if confidence_score >= 0.8:
            return PredictionConfidence.HIGH, confidence_score
        elif confidence_score >= 0.5:
            return PredictionConfidence.MEDIUM, confidence_score
        else:
            return PredictionConfidence.LOW, confidence_score
    
    def _get_alternative_predictions(
        self,
        input_sequence: List[str],
    ) -> List[Tuple[str, float]]:
        """Get alternative predictions with scores."""
        # Simple heuristic: return common next actions
        alternatives = []
        
        common_next = {
            "validate_product_data": ("generate_listing", 0.8),
            "generate_listing": ("confirm_upload", 0.8),
            "confirm_upload": ("handoff", 0.7),
            "TASK_ASSIGN": ("validate_product_data", 0.7),
        }
        
        if input_sequence:
            last_action = input_sequence[-1]
            if last_action in common_next:
                alternatives.append(common_next[last_action])
        
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

