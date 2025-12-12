#!/usr/bin/env python
"""Script to run real API experiments against Shopify.

This script executes controlled experiments using the Shopify Admin API
to validate PaaS resilience with real e-commerce operations.

Prerequisites:
1. Configure Shopify credentials in .env file:
   SHOPIFY_STORE_URL=your-store.myshopify.com
   SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxx

2. Ensure your Shopify app has Products read/write permissions

Usage:
    poetry run python scripts/run_real_api_experiments.py
    poetry run python scripts/run_real_api_experiments.py --runs 50
    poetry run python scripts/run_real_api_experiments.py --dry-run
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.experiments.runner import ExperimentRunner
from src.experiments.conditions import get_condition
from src.integrations.shopify import get_shopify_client, cleanup_test_products


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def check_shopify_connection():
    """Verify Shopify API connection before running experiments."""
    client = get_shopify_client()
    
    if not settings.shopify_store_url or not settings.shopify_access_token:
        logger.error(
            "Shopify credentials not configured!\n"
            "Please set in .env file:\n"
            "  SHOPIFY_STORE_URL=your-store.myshopify.com\n"
            "  SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxx"
        )
        return False
    
    logger.info(f"Testing connection to: {settings.shopify_store_url}")
    
    try:
        healthy = await client.health_check()
        if healthy:
            logger.info("✓ Shopify API connection successful")
            return True
        else:
            logger.error("✗ Shopify API health check failed")
            return False
    except Exception as e:
        logger.error(f"✗ Shopify connection error: {e}")
        return False


async def cleanup_before_run(dry_run: bool = True):
    """Clean up any existing test products before running experiments."""
    client = get_shopify_client()
    
    logger.info("Checking for existing test products...")
    
    deleted = await cleanup_test_products(client, dry_run=dry_run)
    
    if deleted > 0:
        if dry_run:
            logger.info(f"Would delete {deleted} existing test products")
        else:
            logger.info(f"Cleaned up {deleted} existing test products")
    else:
        logger.info("No existing test products found")
    
    return deleted


async def run_experiments(num_runs: int, condition_name: str = "real_api"):
    """Run real API experiments.
    
    Args:
        num_runs: Number of experiment runs
        condition_name: Condition to use
        
    Returns:
        List of experiment results
    """
    runner = ExperimentRunner()
    condition = get_condition(condition_name)
    
    logger.info(f"Starting {num_runs} real API experiments with {condition_name}...")
    logger.info(f"Estimated duration: {num_runs * 5} - {num_runs * 15} seconds")
    
    def progress_callback(current, total):
        if current % 10 == 0 or current == total:
            logger.info(f"Progress: {current}/{total} runs completed")
    
    results = await runner.run_real_api_batch(
        condition=condition,
        num_runs=num_runs,
        progress_callback=progress_callback,
    )
    
    return results


def analyze_results(results):
    """Analyze and print experiment results."""
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    
    recovery_attempted = [r for r in results if r.recovery_attempted]
    recovery_successful = [r for r in recovery_attempted if r.recovery_success]
    
    mttr_values = [r.mttr_seconds for r in results if r.mttr_seconds is not None]
    
    print("\n" + "=" * 60)
    print("REAL API EXPERIMENT RESULTS")
    print("=" * 60)
    
    print(f"\nTotal runs: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    
    if recovery_attempted:
        print(f"\nRecovery attempted: {len(recovery_attempted)}")
        print(f"Recovery successful: {len(recovery_successful)} ({len(recovery_successful)/len(recovery_attempted)*100:.1f}%)")
    
    if mttr_values:
        import statistics
        print(f"\nMTTR Statistics:")
        print(f"  Mean: {statistics.mean(mttr_values):.3f}s")
        print(f"  Median: {statistics.median(mttr_values):.3f}s")
        if len(mttr_values) > 1:
            print(f"  Std Dev: {statistics.stdev(mttr_values):.3f}s")
        print(f"  Min: {min(mttr_values):.3f}s")
        print(f"  Max: {max(mttr_values):.3f}s")
    
    # Failure breakdown
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\nFailure breakdown:")
        error_types = {}
        for f in failures:
            error = f.failure_step or "unknown"
            error_types[error] = error_types.get(error, 0) + 1
        for error, count in sorted(error_types.items(), key=lambda x: -x[1])[:5]:
            print(f"  {error}: {count}")
    
    print("\n" + "=" * 60)


def save_results(results, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "successful": sum(1 for r in results if r.success),
        "success_rate": sum(1 for r in results if r.success) / len(results),
        "recovery_attempted": sum(1 for r in results if r.recovery_attempted),
        "recovery_successful": sum(1 for r in results if r.recovery_success),
    }
    
    recovery_results = [r for r in results if r.recovery_attempted]
    if recovery_results:
        summary["recovery_rate"] = (
            sum(1 for r in recovery_results if r.recovery_success) / len(recovery_results)
        )
    
    mttr_values = [r.mttr_seconds for r in results if r.mttr_seconds is not None]
    if mttr_values:
        import statistics
        summary["mttr_mean"] = statistics.mean(mttr_values)
        summary["mttr_median"] = statistics.median(mttr_values)
    
    with open(output_dir / "real_api_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save raw results
    raw_results = [r.to_dict() for r in results]
    with open(output_dir / "real_api_results.json", "w") as f:
        json.dump(raw_results, f, indent=2)
    
    # Save as CSV
    import csv
    with open(output_dir / "real_api_results.csv", "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
    
    logger.info(f"Results saved to {output_dir}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run real API experiments against Shopify"
    )
    parser.add_argument(
        "--runs", type=int, default=100,
        help="Number of experiment runs (default: 100)"
    )
    parser.add_argument(
        "--condition", type=str, default="real_api",
        help="Experimental condition (default: real_api)"
    )
    parser.add_argument(
        "--output", type=str, default="data/experiments/real_api",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check connection and show what would be done"
    )
    parser.add_argument(
        "--cleanup-only", action="store_true",
        help="Only clean up existing test products"
    )
    parser.add_argument(
        "--skip-cleanup", action="store_true",
        help="Skip cleanup of existing test products"
    )
    
    args = parser.parse_args()
    
    # Check connection
    if not await check_shopify_connection():
        sys.exit(1)
    
    # Handle cleanup-only mode
    if args.cleanup_only:
        await cleanup_before_run(dry_run=False)
        return
    
    # Dry run mode
    if args.dry_run:
        logger.info("\n=== DRY RUN MODE ===")
        logger.info(f"Would run {args.runs} experiments with condition: {args.condition}")
        await cleanup_before_run(dry_run=True)
        return
    
    # Clean up before running
    if not args.skip_cleanup:
        await cleanup_before_run(dry_run=False)
    
    # Run experiments
    try:
        results = await run_experiments(args.runs, args.condition)
        
        # Analyze and save
        analyze_results(results)
        save_results(results, Path(args.output))
        
    finally:
        # Final cleanup
        logger.info("Running final cleanup...")
        client = get_shopify_client()
        await cleanup_test_products(client, dry_run=False)
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

