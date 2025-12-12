"""L* Automata Learning module for agent behavior prediction."""

from src.automata.sul import (
    AgentBehaviorSUL,
    EventSequence,
    create_sul_from_events,
)
from src.automata.learner import (
    AutomataLearner,
    LearningResult,
    learn_agent_behavior,
)
from src.automata.predictor import (
    BehaviorPredictor,
    Prediction,
    PredictionConfidence,
)
from src.automata.event_generator import (
    SyntheticEventGenerator,
    GeneratedEvent,
    generate_training_events,
)

__all__ = [
    # SUL
    "AgentBehaviorSUL",
    "EventSequence",
    "create_sul_from_events",
    # Learner
    "AutomataLearner",
    "LearningResult",
    "learn_agent_behavior",
    # Predictor
    "BehaviorPredictor",
    "Prediction",
    "PredictionConfidence",
    # Event Generator
    "SyntheticEventGenerator",
    "GeneratedEvent",
    "generate_training_events",
]

