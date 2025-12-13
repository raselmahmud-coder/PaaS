"""Event generator for L* automata training using chaos scenarios."""

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GeneratedEvent:
    """A generated event for training."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    thread_id: str = ""
    action_type: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "action_type": self.action_type,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "timestamp": self.timestamp.isoformat(),
        }


class SyntheticEventGenerator:
    """Generate synthetic events for L* automata training.
    
    Creates realistic event sequences that mimic agent behavior
    during workflow execution, including failures and recoveries.
    """
    
    # Workflow steps
    PRODUCT_WORKFLOW_STEPS = [
        ("validate_product_data", "validated"),
        ("generate_listing", "generated"),
        ("confirm_upload", "completed"),
    ]
    
    MARKETING_WORKFLOW_STEPS = [
        ("generate_marketing_copy", "generated"),
        ("review_copy", "reviewed"),
        ("publish_campaign", "published"),
    ]
    
    # Protocol message types
    PROTOCOL_MESSAGES = [
        "TASK_ASSIGN",
        "TASK_COMPLETE",
        "REQUEST_CONTEXT",
        "PROVIDE_CONTEXT",
    ]
    
    def __init__(
        self,
        failure_probability: float = 0.1,
        recovery_probability: float = 0.9,
        random_seed: Optional[int] = None,
    ):
        """Initialize the generator.
        
        Args:
            failure_probability: Probability of failure at each step.
            recovery_probability: Probability of successful recovery.
            random_seed: Seed for reproducibility.
        """
        self.failure_probability = failure_probability
        self.recovery_probability = recovery_probability
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def generate_workflow_events(
        self,
        num_workflows: int = 50,
        agent_id: str = "product-agent-1",
        include_failures: bool = True,
        workflow_type: str = "product",
    ) -> List[Dict[str, Any]]:
        """Generate events from complete workflow executions.
        
        Args:
            num_workflows: Number of workflows to simulate.
            agent_id: Agent ID for events.
            include_failures: Whether to include failure scenarios.
            workflow_type: Type of workflow - "product" or "marketing" (for OOD testing).
            
        Returns:
            List of event dictionaries.
        """
        all_events = []
        base_time = datetime.utcnow()
        
        # Select workflow steps based on type
        if workflow_type == "marketing":
            workflow_steps = self.MARKETING_WORKFLOW_STEPS
            task_name = "marketing_campaign"
        else:
            workflow_steps = self.PRODUCT_WORKFLOW_STEPS
            task_name = "product_upload"
        
        for i in range(num_workflows):
            thread_id = f"{workflow_type}-workflow-{i:04d}"
            events = self._generate_single_workflow(
                agent_id=agent_id,
                thread_id=thread_id,
                base_time=base_time + timedelta(minutes=i * 5),
                include_failure=include_failures and random.random() < self.failure_probability,
                workflow_steps=workflow_steps,
                task_name=task_name,
            )
            all_events.extend(events)
        
        logger.info(f"Generated {len(all_events)} events from {num_workflows} {workflow_type} workflows")
        return all_events
    
    def _generate_single_workflow(
        self,
        agent_id: str,
        thread_id: str,
        base_time: datetime,
        include_failure: bool = False,
        workflow_steps: Optional[List[Tuple[str, str]]] = None,
        task_name: str = "product_upload",
    ) -> List[Dict[str, Any]]:
        """Generate events for a single workflow execution.
        
        Args:
            agent_id: Agent ID for events.
            thread_id: Thread ID for the workflow.
            base_time: Base timestamp for events.
            include_failure: Whether to include a failure scenario.
            workflow_steps: List of (action, status) tuples. Uses PRODUCT_WORKFLOW_STEPS if None.
            task_name: Name of the task (for TASK_ASSIGN/TASK_COMPLETE events).
        """
        events = []
        current_time = base_time
        
        # Start with TASK_ASSIGN
        events.append(GeneratedEvent(
            agent_id=agent_id,
            thread_id=thread_id,
            action_type="TASK_ASSIGN",
            input_data={"task": task_name},
            output_data={"status": "in_progress"},
            timestamp=current_time,
        ).to_dict())
        
        current_time += timedelta(seconds=1)
        
        # Choose workflow steps
        steps = workflow_steps or self.PRODUCT_WORKFLOW_STEPS
        
        # Determine failure point if including failure
        failure_step = -1
        if include_failure:
            failure_step = random.randint(0, len(steps) - 1)
        
        for step_idx, (action, success_status) in enumerate(steps):
            # Check for failure
            if step_idx == failure_step:
                # Generate failure event
                events.append(GeneratedEvent(
                    agent_id=agent_id,
                    thread_id=thread_id,
                    action_type=action,
                    input_data={"step": step_idx},
                    output_data={"status": "failed", "error": "Simulated failure"},
                    timestamp=current_time,
                ).to_dict())
                
                current_time += timedelta(seconds=2)
                
                # Attempt recovery
                if random.random() < self.recovery_probability:
                    # Recovery successful
                    events.append(GeneratedEvent(
                        agent_id=agent_id,
                        thread_id=thread_id,
                        action_type="recovery",
                        input_data={"failed_step": action},
                        output_data={"status": "recovered"},
                        timestamp=current_time,
                    ).to_dict())
                    
                    current_time += timedelta(seconds=1)
                    
                    # Retry the step
                    events.append(GeneratedEvent(
                        agent_id=agent_id,
                        thread_id=thread_id,
                        action_type=action,
                        input_data={"step": step_idx, "retry": True},
                        output_data={"status": success_status},
                        timestamp=current_time,
                    ).to_dict())
                else:
                    # Recovery failed, end workflow
                    events.append(GeneratedEvent(
                        agent_id=agent_id,
                        thread_id=thread_id,
                        action_type="TASK_COMPLETE",
                        input_data={"final_step": action},
                        output_data={"status": "failed"},
                        timestamp=current_time,
                    ).to_dict())
                    return events
            else:
                # Normal step execution
                events.append(GeneratedEvent(
                    agent_id=agent_id,
                    thread_id=thread_id,
                    action_type=action,
                    input_data={"step": step_idx},
                    output_data={"status": success_status},
                    timestamp=current_time,
                ).to_dict())
            
            current_time += timedelta(seconds=random.uniform(1, 3))
        
        # Workflow completed successfully
        events.append(GeneratedEvent(
            agent_id=agent_id,
            thread_id=thread_id,
            action_type="TASK_COMPLETE",
            input_data={"workflow": task_name},
            output_data={"status": "completed"},
            timestamp=current_time,
        ).to_dict())
        
        return events
    
    def generate_handoff_events(
        self,
        num_handoffs: int = 20,
        source_agent: str = "product-agent-1",
        target_agent: str = "marketing-agent-1",
    ) -> List[Dict[str, Any]]:
        """Generate events for agent handoffs.
        
        Args:
            num_handoffs: Number of handoffs to simulate.
            source_agent: Source agent ID.
            target_agent: Target agent ID.
            
        Returns:
            List of event dictionaries.
        """
        events = []
        base_time = datetime.utcnow()
        
        for i in range(num_handoffs):
            thread_id = f"handoff-{i:04d}"
            current_time = base_time + timedelta(minutes=i * 2)
            
            # Source completes and initiates handoff
            events.append(GeneratedEvent(
                agent_id=source_agent,
                thread_id=thread_id,
                action_type="TASK_COMPLETE",
                input_data={"product_id": f"prod-{i}"},
                output_data={"status": "completed"},
                timestamp=current_time,
            ).to_dict())
            
            current_time += timedelta(milliseconds=500)
            
            # Handoff message
            events.append(GeneratedEvent(
                agent_id=source_agent,
                thread_id=thread_id,
                action_type="handoff",
                input_data={"target": target_agent},
                output_data={"status": "handoff_complete"},
                timestamp=current_time,
            ).to_dict())
            
            current_time += timedelta(milliseconds=500)
            
            # Target receives task
            events.append(GeneratedEvent(
                agent_id=target_agent,
                thread_id=thread_id,
                action_type="TASK_ASSIGN",
                input_data={"source": source_agent, "product_id": f"prod-{i}"},
                output_data={"status": "in_progress"},
                timestamp=current_time,
            ).to_dict())
        
        logger.info(f"Generated {len(events)} handoff events")
        return events
    
    def generate_context_request_events(
        self,
        num_requests: int = 15,
        requesting_agent: str = "reconstruction-module",
        responding_agents: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate events for context request/response.
        
        Args:
            num_requests: Number of context requests.
            requesting_agent: Agent requesting context.
            responding_agents: Agents that respond.
            
        Returns:
            List of event dictionaries.
        """
        if responding_agents is None:
            responding_agents = ["product-agent-1", "marketing-agent-1"]
        
        events = []
        base_time = datetime.utcnow()
        
        for i in range(num_requests):
            thread_id = f"context-{i:04d}"
            current_time = base_time + timedelta(minutes=i)
            
            # Context request
            events.append(GeneratedEvent(
                agent_id=requesting_agent,
                thread_id=thread_id,
                action_type="REQUEST_CONTEXT",
                input_data={"failed_agent": random.choice(responding_agents)},
                output_data={"status": "pending"},
                timestamp=current_time,
            ).to_dict())
            
            # Responses from agents
            for agent in responding_agents:
                current_time += timedelta(milliseconds=random.randint(100, 500))
                events.append(GeneratedEvent(
                    agent_id=agent,
                    thread_id=thread_id,
                    action_type="PROVIDE_CONTEXT",
                    input_data={"requester": requesting_agent},
                    output_data={"status": "provided", "context_size": random.randint(1, 10)},
                    timestamp=current_time,
                ).to_dict())
        
        logger.info(f"Generated {len(events)} context request events")
        return events
    
    def generate_novel_sequences(
        self,
        num_sequences: int = 20,
        agent_id: str = "test-agent-1",
        sequence_type: str = "valid_partial",
    ) -> List[Dict[str, Any]]:
        """Generate novel sequences for out-of-distribution testing.
        
        Creates sequences that test generalization:
        - valid_partial: Valid structure but incomplete workflows
        - valid_reordered: Valid actions but in different order
        - invalid_mixed: Mix of valid and invalid action combinations
        
        Args:
            num_sequences: Number of novel sequences to generate.
            agent_id: Agent ID for events.
            sequence_type: Type of novel sequence - "valid_partial", "valid_reordered", or "invalid_mixed".
            
        Returns:
            List of event dictionaries.
        """
        all_events = []
        base_time = datetime.utcnow()
        
        # All possible actions from both workflow types
        all_actions = [
            action for action, _ in self.PRODUCT_WORKFLOW_STEPS + self.MARKETING_WORKFLOW_STEPS
        ]
        
        for i in range(num_sequences):
            thread_id = f"novel-{sequence_type}-{i:04d}"
            current_time = base_time + timedelta(minutes=i * 2)
            
            # Start with TASK_ASSIGN
            events = [GeneratedEvent(
                agent_id=agent_id,
                thread_id=thread_id,
                action_type="TASK_ASSIGN",
                input_data={"task": "novel_test"},
                output_data={"status": "in_progress"},
                timestamp=current_time,
            ).to_dict()]
            
            current_time += timedelta(seconds=1)
            
            if sequence_type == "valid_partial":
                # Valid structure but incomplete (e.g., only first 2 steps)
                steps = random.choice([
                    self.PRODUCT_WORKFLOW_STEPS[:2],
                    self.MARKETING_WORKFLOW_STEPS[:2],
                ])
                for action, status in steps:
                    events.append(GeneratedEvent(
                        agent_id=agent_id,
                        thread_id=thread_id,
                        action_type=action,
                        input_data={"step": "partial"},
                        output_data={"status": status},
                        timestamp=current_time,
                    ).to_dict())
                    current_time += timedelta(seconds=random.uniform(1, 2))
                # No TASK_COMPLETE - incomplete workflow
                
            elif sequence_type == "valid_reordered":
                # Valid actions but in wrong order (should have lower confidence)
                steps = list(self.PRODUCT_WORKFLOW_STEPS)
                random.shuffle(steps)  # Reorder
                for action, status in steps:
                    events.append(GeneratedEvent(
                        agent_id=agent_id,
                        thread_id=thread_id,
                        action_type=action,
                        input_data={"step": "reordered"},
                        output_data={"status": status},
                        timestamp=current_time,
                    ).to_dict())
                    current_time += timedelta(seconds=random.uniform(1, 2))
                events.append(GeneratedEvent(
                    agent_id=agent_id,
                    thread_id=thread_id,
                    action_type="TASK_COMPLETE",
                    input_data={"workflow": "reordered"},
                    output_data={"status": "completed"},
                    timestamp=current_time,
                ).to_dict())
                
            elif sequence_type == "invalid_mixed":
                # Mix actions from different workflow types (invalid combination)
                mixed_actions = random.sample(all_actions, k=min(3, len(all_actions)))
                for action in mixed_actions:
                    events.append(GeneratedEvent(
                        agent_id=agent_id,
                        thread_id=thread_id,
                        action_type=action,
                        input_data={"step": "mixed"},
                        output_data={"status": "unknown"},
                        timestamp=current_time,
                    ).to_dict())
                    current_time += timedelta(seconds=random.uniform(1, 2))
                events.append(GeneratedEvent(
                    agent_id=agent_id,
                    thread_id=thread_id,
                    action_type="TASK_COMPLETE",
                    input_data={"workflow": "mixed"},
                    output_data={"status": "completed"},
                    timestamp=current_time,
                ).to_dict())
            
            all_events.extend(events)
        
        logger.info(f"Generated {len(all_events)} events from {num_sequences} novel {sequence_type} sequences")
        return all_events
    
    def generate_training_dataset(
        self,
        num_workflows: int = 50,
        num_handoffs: int = 20,
        num_context_requests: int = 15,
        include_failures: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate a complete training dataset.
        
        Args:
            num_workflows: Number of workflow executions.
            num_handoffs: Number of agent handoffs.
            num_context_requests: Number of context requests.
            include_failures: Whether to include failure scenarios.
            
        Returns:
            Combined list of all events.
        """
        all_events = []
        
        # Generate workflow events
        all_events.extend(
            self.generate_workflow_events(
                num_workflows=num_workflows,
                include_failures=include_failures,
            )
        )
        
        # Generate handoff events
        all_events.extend(
            self.generate_handoff_events(num_handoffs=num_handoffs)
        )
        
        # Generate context request events
        all_events.extend(
            self.generate_context_request_events(num_requests=num_context_requests)
        )
        
        # Sort by timestamp
        all_events.sort(key=lambda e: e.get("timestamp", ""))
        
        logger.info(f"Generated training dataset with {len(all_events)} total events")
        return all_events


def generate_training_events(
    num_events: int = 100,
    include_failures: bool = True,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to generate training events.
    
    Args:
        num_events: Approximate number of events to generate.
        include_failures: Whether to include failure scenarios.
        random_seed: Seed for reproducibility.
        
    Returns:
        List of event dictionaries.
    """
    # Calculate proportions
    num_workflows = max(10, num_events // 5)  # ~5 events per workflow
    num_handoffs = max(5, num_events // 15)   # ~3 events per handoff
    num_context = max(3, num_events // 20)    # ~4 events per context req
    
    generator = SyntheticEventGenerator(
        failure_probability=0.2 if include_failures else 0.0,
        random_seed=random_seed,
    )
    
    return generator.generate_training_dataset(
        num_workflows=num_workflows,
        num_handoffs=num_handoffs,
        num_context_requests=num_context,
        include_failures=include_failures,
    )

