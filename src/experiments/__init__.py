"""Experiment framework for thesis evaluation and benchmarking."""

from src.experiments.scenario_loader import (
    ScenarioLoader,
    Scenario,
    ScenarioStep,
    load_scenario,
    load_all_scenarios,
)
from src.experiments.conditions import (
    ExperimentCondition,
    BaselineCondition,
    ReconstructionCondition,
    FullSystemCondition,
    get_condition,
    list_conditions,
)
from src.experiments.runner import (
    ExperimentRunner,
    ExperimentResult,
    run_experiment,
    run_all_experiments,
)
from src.experiments.collector import (
    MetricsCollector,
    ExperimentMetrics,
    aggregate_metrics,
)

__all__ = [
    # Scenario Loader
    "ScenarioLoader",
    "Scenario",
    "ScenarioStep",
    "load_scenario",
    "load_all_scenarios",
    # Conditions
    "ExperimentCondition",
    "BaselineCondition",
    "ReconstructionCondition",
    "FullSystemCondition",
    "get_condition",
    "list_conditions",
    # Runner
    "ExperimentRunner",
    "ExperimentResult",
    "run_experiment",
    "run_all_experiments",
    # Collector
    "MetricsCollector",
    "ExperimentMetrics",
    "aggregate_metrics",
]

