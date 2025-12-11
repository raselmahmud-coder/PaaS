"""End-to-end reconstruction demo script."""

import logging
import uuid
import time
from src.config import settings
from src.workflows.vendor_workflow import create_vendor_workflow
from src.agents.base import AgentState
from src.reconstruction.detector import FailureDetector
from src.reconstruction.reconstructor import AgentReconstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def simulate_failure_demo():
    """Demonstrate failure detection and reconstruction."""
    
    logger.info("=" * 60)
    logger.info("Phase 1 MVP: Failure Detection and Reconstruction Demo")
    logger.info("=" * 60)
    
    # Initialize database
    from src.persistence.models import init_db
    init_db()
    logger.info("✓ Database initialized")
    
    # Create workflow
    workflow = create_vendor_workflow()
    logger.info("✓ Vendor workflow created")
    
    # Create initial state
    thread_id = str(uuid.uuid4())
    agent_id = "product-agent-1"
    
    initial_state: AgentState = {
        "task_id": f"task-{uuid.uuid4()}",
        "agent_id": agent_id,
        "thread_id": thread_id,
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": {
            "name": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation",
            "price": 79.99,
            "category": "Electronics",
            "sku": "WBH-001"
        },
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info("\n--- Step 1: Starting workflow ---")
    logger.info(f"Thread ID: {thread_id}")
    logger.info(f"Agent ID: {agent_id}")
    
    try:
        # Run workflow up to step 2 (simulate crash before step 3)
        logger.info("Executing workflow steps 1-2...")
        
        # Invoke workflow - it will checkpoint after each step
        result = workflow.invoke(initial_state, config)
        
        logger.info(f"✓ Workflow completed successfully!")
        logger.info(f"  Final status: {result.get('status')}")
        logger.info(f"  Final step: {result.get('current_step')}")
        
        # Check events
        from src.persistence.event_store import event_store
        events = event_store.get_events(thread_id=thread_id)
        logger.info(f"  Total events logged: {len(events)}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        logger.info("\n--- Step 2: Simulating failure ---")
        logger.info("Simulating agent crash after step 2...")
        
        # Simulate failure by checking events
        from src.persistence.event_store import event_store
        events = event_store.get_events(thread_id=thread_id)
        logger.info(f"Events before failure: {len(events)}")
        
        # Wait a bit to simulate time passing
        time.sleep(1)
        
        logger.info("\n--- Step 3: Failure Detection ---")
        detector = FailureDetector(timeout_seconds=30)
        
        # Check if agent appears failed (it should since we're not running it)
        # In real scenario, this would detect timeout
        failed_threads = detector.get_failed_threads()
        logger.info(f"Failed threads detected: {len(failed_threads)}")
        
        logger.info("\n--- Step 4: State Reconstruction ---")
        reconstructor = AgentReconstructor()
        
        try:
            reconstruction_result = reconstructor.reconstruct(
                agent_id=agent_id,
                thread_id=thread_id
            )
            
            logger.info("✓ Reconstruction successful!")
            logger.info(f"  Checkpoint timestamp: {reconstruction_result['checkpoint'].ts}")
            logger.info(f"  Events since checkpoint: {len(reconstruction_result['events_since'])}")
            logger.info(f"  Inferred next action: {reconstruction_result['inferred_next_action'].get('next_action', 'N/A')}")
            logger.info(f"  Reconstructed status: {reconstruction_result['reconstructed_state'].get('status', 'N/A')}")
            
            logger.info("\n--- Step 5: Resume Workflow ---")
            logger.info("Resuming workflow from reconstructed state...")
            
            # Get the reconstructed state
            reconstructed_state = reconstruction_result['reconstructed_state']
            
            # Update initial state with reconstructed data
            resume_state: AgentState = {
                **initial_state,
                **reconstructed_state,
                "status": "in_progress",  # Reset to in_progress for resume
            }
            
            # Resume workflow
            final_result = workflow.invoke(resume_state, config)
            
            logger.info("✓ Workflow resumed and completed!")
            logger.info(f"  Final status: {final_result.get('status')}")
            logger.info(f"  Final step: {final_result.get('current_step')}")
            
        except Exception as recon_error:
            logger.error(f"Reconstruction failed: {recon_error}", exc_info=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    simulate_failure_demo()

