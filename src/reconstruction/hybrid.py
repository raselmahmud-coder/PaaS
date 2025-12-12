"""Hybrid reconstruction combining L* automata and LLM strategies."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.automata.learner import AutomataLearner, LearningResult
from src.automata.predictor import BehaviorPredictor, Prediction, PredictionConfidence

logger = logging.getLogger(__name__)


class ReconstructionStrategy(Enum):
    """Strategy used for reconstruction."""
    
    CHECKPOINT = "checkpoint"        # Fresh checkpoint available
    AUTOMATA = "automata"            # L* automata-based prediction
    LLM = "llm"                      # LLM-based inference
    HYBRID = "hybrid"                # Combined automata + LLM
    FALLBACK = "fallback"            # Simple heuristic fallback


@dataclass
class HybridReconstructionResult:
    """Result of hybrid reconstruction."""
    
    success: bool
    strategy: ReconstructionStrategy
    reconstructed_state: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    agent_id: str
    thread_id: str
    
    # Strategy-specific details
    automata_prediction: Optional[Prediction] = None
    llm_result: Optional[Dict[str, Any]] = None
    peer_context_used: bool = False
    peer_agents_queried: int = 0
    
    # Timing
    reconstruction_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Error info
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "strategy": self.strategy.value,
            "reconstructed_state": self.reconstructed_state,
            "confidence": self.confidence,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "automata_prediction": self.automata_prediction.to_dict() if self.automata_prediction else None,
            "llm_result": self.llm_result,
            "peer_context_used": self.peer_context_used,
            "peer_agents_queried": self.peer_agents_queried,
            "reconstruction_time_ms": self.reconstruction_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


class HybridReconstructor:
    """Hybrid reconstructor combining multiple strategies.
    
    Uses a decision tree to select the best reconstruction strategy:
    1. If fresh checkpoint exists (< threshold), use checkpoint
    2. If enough structured events, use L* automata learning
    3. Otherwise, use LLM with peer context
    """
    
    # Thresholds
    CHECKPOINT_FRESHNESS_THRESHOLD = 30  # seconds
    MIN_EVENTS_FOR_AUTOMATA = 50  # minimum events for L* learning
    AUTOMATA_CONFIDENCE_THRESHOLD = 0.7  # minimum confidence for automata-only
    
    def __init__(
        self,
        enable_automata: bool = True,
        enable_llm: bool = True,
        enable_peer_context: bool = True,
        checkpoint_freshness: int = 30,
        min_events_for_automata: int = 50,
        automata_confidence_threshold: float = 0.7,
    ):
        """Initialize the hybrid reconstructor.
        
        Args:
            enable_automata: Whether to use automata-based reconstruction.
            enable_llm: Whether to use LLM-based reconstruction.
            enable_peer_context: Whether to query peer agents for context.
            checkpoint_freshness: Max age (seconds) for checkpoint to be "fresh".
            min_events_for_automata: Minimum events needed for L* learning.
            automata_confidence_threshold: Min confidence to use automata alone.
        """
        self.enable_automata = enable_automata
        self.enable_llm = enable_llm
        self.enable_peer_context = enable_peer_context
        self.checkpoint_freshness = checkpoint_freshness
        self.min_events_for_automata = min_events_for_automata
        self.automata_confidence_threshold = automata_confidence_threshold
        
        # Components
        self._behavior_predictor: Optional[BehaviorPredictor] = None
        self._llm_reconstructor = None  # Lazy loaded
        self._training_events: List[Dict[str, Any]] = []
    
    def train_automata(self, events: List[Dict[str, Any]]) -> LearningResult:
        """Train the automata predictor on event data.
        
        Args:
            events: Event log data for training.
            
        Returns:
            LearningResult from training.
        """
        self._training_events = events
        self._behavior_predictor = BehaviorPredictor()
        return self._behavior_predictor.train(events)
    
    async def reconstruct(
        self,
        agent_id: str,
        thread_id: str,
        checkpoint: Optional[Dict[str, Any]] = None,
        events_since_checkpoint: Optional[List[Dict[str, Any]]] = None,
        all_events: Optional[List[Dict[str, Any]]] = None,
    ) -> HybridReconstructionResult:
        """Reconstruct agent state using hybrid strategy.
        
        Args:
            agent_id: ID of the agent to reconstruct.
            thread_id: Thread/workflow ID.
            checkpoint: Last checkpoint state (if available).
            events_since_checkpoint: Events after last checkpoint.
            all_events: All available events (for training if needed).
            
        Returns:
            HybridReconstructionResult with reconstructed state.
        """
        start_time = time.perf_counter()
        
        events = events_since_checkpoint or []
        
        # Step 1: Check checkpoint freshness
        if checkpoint and self._is_checkpoint_fresh(checkpoint):
            return HybridReconstructionResult(
                success=True,
                strategy=ReconstructionStrategy.CHECKPOINT,
                reconstructed_state=checkpoint,
                confidence=1.0,
                agent_id=agent_id,
                thread_id=thread_id,
                reconstruction_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Step 2: Try automata-based reconstruction
        automata_result = None
        if self.enable_automata and len(events) >= 10:
            automata_result = await self._try_automata_reconstruction(
                agent_id, thread_id, checkpoint, events, all_events
            )
            
            if automata_result and automata_result.confidence >= self.automata_confidence_threshold:
                automata_result.reconstruction_time_ms = (time.perf_counter() - start_time) * 1000
                return automata_result
        
        # Step 3: Use LLM reconstruction
        llm_result = None
        if self.enable_llm:
            llm_result = await self._try_llm_reconstruction(
                agent_id, thread_id, checkpoint, events
            )
            
            if llm_result and llm_result.success:
                llm_result.reconstruction_time_ms = (time.perf_counter() - start_time) * 1000
                return llm_result
        
        # Step 4: Combine strategies if both have partial results
        if automata_result and llm_result:
            combined = self._combine_strategies(automata_result, llm_result)
            combined.reconstruction_time_ms = (time.perf_counter() - start_time) * 1000
            return combined
        
        # Step 5: Fallback
        return self._fallback_reconstruction(
            agent_id, thread_id, checkpoint, events,
            reconstruction_time=(time.perf_counter() - start_time) * 1000
        )
    
    def _is_checkpoint_fresh(self, checkpoint: Dict[str, Any]) -> bool:
        """Check if checkpoint is fresh enough to use directly."""
        if not checkpoint:
            return False
        
        checkpoint_time = checkpoint.get("timestamp")
        if not checkpoint_time:
            return False
        
        if isinstance(checkpoint_time, str):
            try:
                checkpoint_time = datetime.fromisoformat(checkpoint_time.replace("Z", "+00:00"))
            except ValueError:
                return False
        
        age = (datetime.utcnow() - checkpoint_time.replace(tzinfo=None)).total_seconds()
        return age < self.checkpoint_freshness
    
    async def _try_automata_reconstruction(
        self,
        agent_id: str,
        thread_id: str,
        checkpoint: Optional[Dict[str, Any]],
        events: List[Dict[str, Any]],
        all_events: Optional[List[Dict[str, Any]]],
    ) -> Optional[HybridReconstructionResult]:
        """Try automata-based reconstruction."""
        try:
            # Train if needed
            if self._behavior_predictor is None or not self._behavior_predictor.is_trained:
                training_data = all_events or events
                if len(training_data) < self.min_events_for_automata:
                    return None
                self.train_automata(training_data)
            
            if not self._behavior_predictor.is_trained:
                return None
            
            # Get prediction
            last_checkpoint = checkpoint or {}
            predicted_state = self._behavior_predictor.predict_state_for_reconstruction(
                agent_id=agent_id,
                last_checkpoint_state=last_checkpoint,
                events_since_checkpoint=events,
            )
            
            prediction = predicted_state.get("prediction", {})
            confidence = predicted_state.get("reconstruction_confidence", 0.0)
            
            # Build prediction object
            automata_prediction = None
            if prediction:
                automata_prediction = Prediction(
                    predicted_action=prediction.get("predicted_action", "unknown"),
                    predicted_status=prediction.get("predicted_status", "unknown"),
                    confidence=PredictionConfidence(prediction.get("confidence", "unknown")),
                    confidence_score=prediction.get("confidence_score", 0.0),
                )
            
            return HybridReconstructionResult(
                success=confidence >= 0.5,
                strategy=ReconstructionStrategy.AUTOMATA,
                reconstructed_state=predicted_state,
                confidence=confidence,
                agent_id=agent_id,
                thread_id=thread_id,
                automata_prediction=automata_prediction,
            )
            
        except Exception as e:
            logger.warning(f"Automata reconstruction failed: {e}")
            return None
    
    async def _try_llm_reconstruction(
        self,
        agent_id: str,
        thread_id: str,
        checkpoint: Optional[Dict[str, Any]],
        events: List[Dict[str, Any]],
    ) -> Optional[HybridReconstructionResult]:
        """Try LLM-based reconstruction."""
        try:
            # Lazy load LLM reconstructor
            if self._llm_reconstructor is None:
                from src.reconstruction.reconstructor import AgentReconstructor
                self._llm_reconstructor = AgentReconstructor(
                    enable_peer_context=self.enable_peer_context
                )
            
            # Query peer context if enabled
            peer_context = []
            if self.enable_peer_context:
                peer_context = await self._query_peer_context(agent_id)
            
            # Reconstruct using LLM
            result = await self._llm_reconstructor.reconstruct_async(
                agent_id=agent_id,
                thread_id=thread_id,
                peer_context=peer_context,
            )
            
            return HybridReconstructionResult(
                success=result.get("success", False),
                strategy=ReconstructionStrategy.LLM,
                reconstructed_state=result.get("reconstructed_state", {}),
                confidence=result.get("confidence", 0.5),
                agent_id=agent_id,
                thread_id=thread_id,
                llm_result=result,
                peer_context_used=len(peer_context) > 0,
                peer_agents_queried=len(peer_context),
            )
            
        except Exception as e:
            logger.warning(f"LLM reconstruction failed: {e}")
            return None
    
    async def _query_peer_context(self, agent_id: str) -> List[Dict[str, Any]]:
        """Query peer agents for context."""
        try:
            from src.messaging.producer import KafkaProducer
            from src.messaging.consumer import KafkaConsumer
            from src.protocol.messages import RequestContextMessage
            
            # This is a simplified version - real implementation would use Kafka
            # For now, return empty list (no peers available in test mode)
            return []
            
        except ImportError:
            return []
    
    def _combine_strategies(
        self,
        automata_result: HybridReconstructionResult,
        llm_result: HybridReconstructionResult,
    ) -> HybridReconstructionResult:
        """Combine automata and LLM results."""
        # Weight results by confidence
        automata_conf = automata_result.confidence
        llm_conf = llm_result.confidence
        
        # Choose primary based on confidence
        if automata_conf >= llm_conf:
            primary = automata_result
            secondary = llm_result
        else:
            primary = llm_result
            secondary = automata_result
        
        # Merge states, preferring primary
        merged_state = dict(secondary.reconstructed_state)
        merged_state.update(primary.reconstructed_state)
        merged_state["hybrid_sources"] = {
            "automata_confidence": automata_conf,
            "llm_confidence": llm_conf,
        }
        
        # Combined confidence is weighted average
        combined_confidence = (automata_conf * 0.4 + llm_conf * 0.6)
        
        return HybridReconstructionResult(
            success=True,
            strategy=ReconstructionStrategy.HYBRID,
            reconstructed_state=merged_state,
            confidence=combined_confidence,
            agent_id=primary.agent_id,
            thread_id=primary.thread_id,
            automata_prediction=automata_result.automata_prediction,
            llm_result=llm_result.llm_result,
            peer_context_used=llm_result.peer_context_used,
            peer_agents_queried=llm_result.peer_agents_queried,
        )
    
    def _fallback_reconstruction(
        self,
        agent_id: str,
        thread_id: str,
        checkpoint: Optional[Dict[str, Any]],
        events: List[Dict[str, Any]],
        reconstruction_time: float,
    ) -> HybridReconstructionResult:
        """Fallback reconstruction using simple heuristics."""
        # Build state from available information
        state = dict(checkpoint) if checkpoint else {}
        
        state["agent_id"] = agent_id
        state["thread_id"] = thread_id
        state["status"] = "in_progress"
        state["reconstruction_method"] = "fallback"
        
        # Infer current step from events
        if events:
            last_event = events[-1]
            state["current_step"] = len(events)
            state["last_action"] = last_event.get("action_type", "unknown")
            
            # Try to infer next action
            action_sequence = {
                "validate_product_data": "generate_listing",
                "generate_listing": "confirm_upload",
                "confirm_upload": "handoff",
                "TASK_ASSIGN": "validate_product_data",
            }
            last_action = last_event.get("action_type", "")
            state["next_action"] = action_sequence.get(last_action, "continue")
        
        return HybridReconstructionResult(
            success=True,
            strategy=ReconstructionStrategy.FALLBACK,
            reconstructed_state=state,
            confidence=0.3,  # Low confidence for fallback
            agent_id=agent_id,
            thread_id=thread_id,
            reconstruction_time_ms=reconstruction_time,
        )
    
    def reconstruct_sync(
        self,
        agent_id: str,
        thread_id: str,
        checkpoint: Optional[Dict[str, Any]] = None,
        events_since_checkpoint: Optional[List[Dict[str, Any]]] = None,
        all_events: Optional[List[Dict[str, Any]]] = None,
    ) -> HybridReconstructionResult:
        """Synchronous version of reconstruct."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.reconstruct(
                agent_id, thread_id, checkpoint,
                events_since_checkpoint, all_events
            )
        )


# Convenience functions
async def hybrid_reconstruct(
    agent_id: str,
    thread_id: str,
    checkpoint: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    training_events: Optional[List[Dict[str, Any]]] = None,
) -> HybridReconstructionResult:
    """Convenience function for hybrid reconstruction.
    
    Args:
        agent_id: Agent ID.
        thread_id: Thread ID.
        checkpoint: Last checkpoint.
        events: Events since checkpoint.
        training_events: Events for automata training.
        
    Returns:
        HybridReconstructionResult.
    """
    reconstructor = HybridReconstructor()
    
    # Pre-train if we have training events
    if training_events and len(training_events) >= 50:
        reconstructor.train_automata(training_events)
    
    return await reconstructor.reconstruct(
        agent_id, thread_id, checkpoint, events, training_events
    )
