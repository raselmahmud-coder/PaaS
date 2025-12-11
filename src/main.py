"""Main entry point for the PaaS system."""

import logging
import uuid
from src.config import settings
from src.workflows.product_workflow import create_product_upload_workflow
from src.agents.base import AgentState

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    logger.info("Starting PaaS system...")
    logger.info(f"Database URL: {settings.database_url}")
    
    # Initialize database
    from src.persistence.models import init_db
    init_db()
    logger.info("Database initialized")
    
    # Create workflow
    workflow = create_product_upload_workflow()
    logger.info("Product Upload workflow created")
    
    # Create initial state
    thread_id = str(uuid.uuid4())
    initial_state: AgentState = {
        "task_id": f"task-{uuid.uuid4()}",
        "agent_id": "product-agent-1",
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
    
    # Run workflow
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info("Starting workflow execution...")
    try:
        result = workflow.invoke(initial_state, config)
        logger.info(f"Workflow completed successfully!")
        logger.info(f"Final status: {result.get('status')}")
        logger.info(f"Final step: {result.get('current_step')}")
        if result.get("generated_listing"):
            logger.info(f"Generated listing preview: {result['generated_listing'][:100]}...")
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
