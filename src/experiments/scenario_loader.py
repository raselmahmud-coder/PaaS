"""YAML scenario loader for experiments."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default scenarios directory
SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"


@dataclass
class TermConflict:
    """Definition of a potential term conflict in a step (Gap 4)."""
    
    terms: List[str]  # Terms that may conflict between agents
    probability: float = 0.3  # Probability of conflict occurring
    severity: str = "medium"  # low, medium, high - affects resolution difficulty
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TermConflict":
        """Create from dictionary."""
        return cls(
            terms=data.get("terms", []),
            probability=data.get("probability", 0.3),
            severity=data.get("severity", "medium"),
        )


@dataclass
class ScenarioStep:
    """A single step in a scenario."""
    
    name: str
    agent: str
    action: str
    expected_status: str
    timeout_seconds: int = 30
    target_agent: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    # Semantic term conflicts that may occur at this step (Gap 4)
    term_conflicts: Optional[TermConflict] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioStep":
        """Create from dictionary."""
        # Parse term_conflicts if present
        term_conflicts = None
        if "term_conflicts" in data:
            term_conflicts = TermConflict.from_dict(data["term_conflicts"])
        
        return cls(
            name=data["name"],
            agent=data["agent"],
            action=data.get("action", "execute"),
            expected_status=data["expected_status"],
            timeout_seconds=data.get("timeout_seconds", 30),
            target_agent=data.get("target_agent"),
            config=data.get("config", {}),
            term_conflicts=term_conflicts,
        )
    
    def has_term_conflicts(self) -> bool:
        """Check if this step has potential term conflicts."""
        return self.term_conflicts is not None and len(self.term_conflicts.terms) > 0


@dataclass
class CascadeConfig:
    """Configuration for cascade failure behavior.
    
    When enabled, a failure at the trigger step can propagate to downstream
    steps with a specified probability, simulating real-world cascade failures.
    """
    
    enabled: bool = False
    trigger_step: int = 0  # Step index that triggers cascade
    downstream_probability: float = 0.7  # Probability cascade propagates to next step
    max_depth: int = 2  # Maximum number of steps the cascade can propagate
    delay_between_failures_ms: int = 100  # Simulated delay between cascading failures
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CascadeConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            trigger_step=data.get("trigger_step", 0),
            downstream_probability=data.get("downstream_probability", 0.7),
            max_depth=data.get("max_depth", 2),
            delay_between_failures_ms=data.get("delay_between_failures_ms", 100),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "trigger_step": self.trigger_step,
            "downstream_probability": self.downstream_probability,
            "max_depth": self.max_depth,
            "delay_between_failures_ms": self.delay_between_failures_ms,
        }


@dataclass
class FailureInjectionConfig:
    """Configuration for failure injection."""
    
    enabled: bool = True
    probability: float = 0.3
    target_steps: List[int] = field(default_factory=list)
    failure_types: List[str] = field(default_factory=lambda: ["crash", "timeout"])
    cascade: Optional[CascadeConfig] = None  # Cascade failure configuration
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureInjectionConfig":
        """Create from dictionary."""
        # Parse cascade config if present
        cascade = None
        if "cascade" in data:
            cascade = CascadeConfig.from_dict(data["cascade"])
        
        return cls(
            enabled=data.get("enabled", True),
            probability=data.get("probability", 0.3),
            target_steps=data.get("target_steps", []),
            failure_types=data.get("failure_types", ["crash", "timeout"]),
            cascade=cascade,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "enabled": self.enabled,
            "probability": self.probability,
            "target_steps": self.target_steps,
            "failure_types": self.failure_types,
        }
        if self.cascade:
            result["cascade"] = self.cascade.to_dict()
        return result
    
    def has_cascade(self) -> bool:
        """Check if cascade failure is configured."""
        return self.cascade is not None and self.cascade.enabled


@dataclass
class Scenario:
    """A complete experiment scenario."""
    
    name: str
    description: str
    complexity: str
    version: str
    agents: List[Dict[str, str]]
    steps: List[ScenarioStep]
    failure_injection: FailureInjectionConfig
    initial_state: Dict[str, Any]
    success_criteria: Dict[str, Any]
    file_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], file_path: Optional[str] = None) -> "Scenario":
        """Create from dictionary."""
        steps = [ScenarioStep.from_dict(s) for s in data.get("steps", [])]
        failure_config = FailureInjectionConfig.from_dict(
            data.get("failure_injection", {})
        )
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            complexity=data.get("complexity", "medium"),
            version=data.get("version", "1.0"),
            agents=data.get("agents", []),
            steps=steps,
            failure_injection=failure_config,
            initial_state=data.get("initial_state", {}),
            success_criteria=data.get("success_criteria", {}),
            file_path=file_path,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        steps_data = []
        for s in self.steps:
            step_dict = {
                "name": s.name,
                "agent": s.agent,
                "action": s.action,
                "expected_status": s.expected_status,
            }
            if s.term_conflicts:
                step_dict["term_conflicts"] = {
                    "terms": s.term_conflicts.terms,
                    "probability": s.term_conflicts.probability,
                    "severity": s.term_conflicts.severity,
                }
            steps_data.append(step_dict)
        
        return {
            "name": self.name,
            "description": self.description,
            "complexity": self.complexity,
            "version": self.version,
            "agents": self.agents,
            "steps": steps_data,
            "failure_injection": self.failure_injection.to_dict(),
            "initial_state": self.initial_state,
            "success_criteria": self.success_criteria,
        }
    
    def get_initial_state(self, run_id: str) -> Dict[str, Any]:
        """Get initial state with run_id substituted."""
        state = dict(self.initial_state)
        
        # Substitute {{run_id}} placeholders
        if "task_id" in state:
            state["task_id"] = state["task_id"].replace("{{run_id}}", run_id)
        
        return state
    
    @property
    def num_steps(self) -> int:
        """Get number of steps."""
        return len(self.steps)
    
    @property
    def agent_ids(self) -> List[str]:
        """Get list of agent IDs."""
        return [a["id"] for a in self.agents]


class ScenarioLoader:
    """Loader for YAML scenario files."""
    
    def __init__(self, scenarios_dir: Optional[Path] = None):
        """Initialize the loader.
        
        Args:
            scenarios_dir: Directory containing scenario YAML files.
        """
        self.scenarios_dir = scenarios_dir or SCENARIOS_DIR
        self._cache: Dict[str, Scenario] = {}
    
    def load(self, scenario_name: str) -> Scenario:
        """Load a scenario by name.
        
        Args:
            scenario_name: Name of the scenario (without .yaml extension).
            
        Returns:
            Loaded Scenario object.
            
        Raises:
            FileNotFoundError: If scenario file not found.
        """
        # Check cache
        if scenario_name in self._cache:
            return self._cache[scenario_name]
        
        # Find file
        file_path = self.scenarios_dir / f"{scenario_name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario not found: {file_path}")
        
        # Load YAML
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Create scenario
        scenario = Scenario.from_dict(data, str(file_path))
        
        # Cache
        self._cache[scenario_name] = scenario
        
        logger.info(f"Loaded scenario: {scenario.name} ({scenario.num_steps} steps)")
        return scenario
    
    def load_all(self) -> List[Scenario]:
        """Load all scenarios from the directory.
        
        Returns:
            List of all loaded scenarios.
        """
        scenarios = []
        
        for file_path in self.scenarios_dir.glob("*.yaml"):
            scenario_name = file_path.stem
            try:
                scenario = self.load(scenario_name)
                scenarios.append(scenario)
            except Exception as e:
                logger.error(f"Failed to load {scenario_name}: {e}")
        
        return scenarios
    
    def list_scenarios(self) -> List[str]:
        """List available scenario names."""
        return [f.stem for f in self.scenarios_dir.glob("*.yaml")]
    
    def clear_cache(self) -> None:
        """Clear the scenario cache."""
        self._cache.clear()


# Convenience functions
def load_scenario(name: str) -> Scenario:
    """Load a scenario by name."""
    loader = ScenarioLoader()
    return loader.load(name)


def load_all_scenarios() -> List[Scenario]:
    """Load all available scenarios."""
    loader = ScenarioLoader()
    return loader.load_all()


def list_available_scenarios() -> List[str]:
    """List available scenario names."""
    loader = ScenarioLoader()
    return loader.list_scenarios()

