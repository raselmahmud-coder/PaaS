"""Automata-based state reconstruction using L* learned patterns."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.automata.learner import AutomataLearner, LearningResult
from src.automata.predictor import BehaviorPredictor, Prediction

logger = logging.getLogger(__name__)


@dataclass
class AutomataReconstructionResult:
    """Result of automata-based reconstruction."""
    
    success: bool
    reconstructed_state: Dict[str, Any]
    predicted_action: str
    predicted_status: str
    confidence: float
    model_accuracy: float
    agent_id: str
    thread_id: str
    events_used: int = 0
    reconstruction_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "reconstructed_state": self.reconstructed_state,
            "predicted_action": self.predicted_action,
            "predicted_status": self.predicted_status,
            "confidence": self.confidence,
            "model_accuracy": self.model_accuracy,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "events_used": self.events_used,
            "reconstruction_time_ms": self.reconstruction_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


class AutomataReconstructor:
    """State reconstructor using L* learned automata.
    
    Uses the learned behavior patterns to predict the next state
    an agent should be in based on its event history.
    """
    
    MIN_EVENTS_FOR_TRAINING = 20
    MIN_CONFIDENCE = 0.5
    
    def __init__(
        self,
        min_events: int = MIN_EVENTS_FOR_TRAINING,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        """Initialize the automata reconstructor.
        
        Args:
            min_events: Minimum events required for training.
            min_confidence: Minimum confidence for successful reconstruction.
        """
        self.min_events = min_events
        self.min_confidence = min_confidence
        
        self._predictor: Optional[BehaviorPredictor] = None
        self._learning_result: Optional[LearningResult] = None
        self._training_events: List[Dict[str, Any]] = []
    
    def train(self, events: List[Dict[str, Any]]) -> LearningResult:
        """Train the automata on event data.
        
        Args:
            events: Event log data.
            
        Returns:
            LearningResult from training.
        """
        self._training_events = events
        self._predictor = BehaviorPredictor()
        self._learning_result = self._predictor.train(events)
        
        logger.info(
            f"Trained automata reconstructor: "
            f"{self._learning_result.num_states} states, "
            f"{self._learning_result.num_transitions} transitions"
        )
        
        return self._learning_result
    
    def reconstruct(
        self,
        agent_id: str,
        thread_id: str,
        last_checkpoint: Optional[Dict[str, Any]] = None,
        events_since_checkpoint: Optional[List[Dict[str, Any]]] = None,
    ) -> AutomataReconstructionResult:
        """Reconstruct agent state using learned automata.
        
        Args:
            agent_id: ID of the agent.
            thread_id: Thread/workflow ID.
            last_checkpoint: State from last checkpoint.
            events_since_checkpoint: Events after checkpoint.
            
        Returns:
            AutomataReconstructionResult.
        """
        start_time = time.perf_counter()
        events = events_since_checkpoint or []
        checkpoint = last_checkpoint or {}
        
        # Check if we need to train
        if self._predictor is None or not self._predictor.is_trained:
            all_events = self._training_events + events
            if len(all_events) < self.min_events:
                return AutomataReconstructionResult(
                    success=False,
                    reconstructed_state=checkpoint,
                    predicted_action="unknown",
                    predicted_status="unknown",
                    confidence=0.0,
                    model_accuracy=0.0,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    error=f"Insufficient events for training (need {self.min_events})",
                    reconstruction_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            
            self.train(all_events)
        
        try:
            # Get prediction
            predicted_state = self._predictor.predict_state_for_reconstruction(
                agent_id=agent_id,
                last_checkpoint_state=checkpoint,
                events_since_checkpoint=events,
            )
            
            # Extract prediction details
            prediction = predicted_state.get("prediction", {})
            confidence = predicted_state.get("reconstruction_confidence", 0.0)
            
            success = confidence >= self.min_confidence
            
            return AutomataReconstructionResult(
                success=success,
                reconstructed_state=predicted_state,
                predicted_action=prediction.get("predicted_action", "unknown"),
                predicted_status=prediction.get("predicted_status", "unknown"),
                confidence=confidence,
                model_accuracy=self._predictor.model_accuracy,
                agent_id=agent_id,
                thread_id=thread_id,
                events_used=len(events),
                reconstruction_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.error(f"Automata reconstruction failed: {e}")
            return AutomataReconstructionResult(
                success=False,
                reconstructed_state=checkpoint,
                predicted_action="unknown",
                predicted_status="unknown",
                confidence=0.0,
                model_accuracy=0.0,
                agent_id=agent_id,
                thread_id=thread_id,
                error=str(e),
                reconstruction_time_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    @property
    def is_trained(self) -> bool:
        """Check if the reconstructor is trained."""
        return self._predictor is not None and self._predictor.is_trained
    
    @property
    def model_accuracy(self) -> float:
        """Get the model accuracy estimate."""
        if self._predictor:
            return self._predictor.model_accuracy
        return 0.0


def reconstruct_with_automata(
    agent_id: str,
    thread_id: str,
    training_events: List[Dict[str, Any]],
    last_checkpoint: Optional[Dict[str, Any]] = None,
    events_since_checkpoint: Optional[List[Dict[str, Any]]] = None,
) -> AutomataReconstructionResult:
    """Convenience function for automata reconstruction.
    
    Args:
        agent_id: Agent ID.
        thread_id: Thread ID.
        training_events: Events for training the automata.
        last_checkpoint: Last checkpoint state.
        events_since_checkpoint: Events since checkpoint.
        
    Returns:
        AutomataReconstructionResult.
    """
    reconstructor = AutomataReconstructor()
    reconstructor.train(training_events)
    return reconstructor.reconstruct(
        agent_id, thread_id, last_checkpoint, events_since_checkpoint
    )
