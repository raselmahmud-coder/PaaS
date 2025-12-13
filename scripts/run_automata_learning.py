#!/usr/bin/env python3
"""Run actual L* automata learning and report results.

This script demonstrates the L* automata learning component of PaaS by:
1. Generating synthetic agent events
2. Running actual AALpy L* learning
3. Testing prediction accuracy on held-out data
4. Exporting DOT visualization

Usage:
    poetry run python scripts/run_automata_learning.py [--events N] [--seed S]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.automata.event_generator import SyntheticEventGenerator, generate_training_events
from src.automata.learner import AutomataLearner, LearningResult, SimpleAutomaton
from src.automata.predictor import BehaviorPredictor, PredictionConfidence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_actual_lstar(
    num_events: int = 500,
    random_seed: int = 42,
    output_dir: str = "data/automata",
) -> dict:
    """Run actual L* learning and report comprehensive results.
    
    Args:
        num_events: Number of events to generate for training.
        random_seed: Random seed for reproducibility.
        output_dir: Directory to save outputs.
        
    Returns:
        Dictionary with learning results and metrics.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_events": num_events,
            "random_seed": random_seed,
        }
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("L* AUTOMATA LEARNING - PaaS Agent Behavior Analysis")
    print("=" * 70)
    
    # Step 1: Generate synthetic events with separate train/test workflows
    print("\n1. GENERATING EVENTS (Out-of-Distribution Testing)")
    print("-" * 40)
    
    generator = SyntheticEventGenerator(
        failure_probability=0.15,
        recovery_probability=0.85,
        random_seed=random_seed,
    )
    
    # TRAINING: Product workflows only (for learning)
    print("   Generating TRAINING events (product workflows)...")
    train_events = generator.generate_workflow_events(
        num_workflows=num_events // 5,  # ~5 events per workflow
        agent_id="product-agent-1",
        include_failures=True,
        workflow_type="product",  # Train on product workflows
    )
    
    # Add handoffs and context requests to training
    train_events.extend(generator.generate_handoff_events(
        num_handoffs=num_events // 15,
    ))
    train_events.extend(generator.generate_context_request_events(
        num_requests=num_events // 20,
    ))
    
    # TESTING: Marketing workflows (different workflow type - OOD)
    print("   Generating TEST events (marketing workflows - OOD)...")
    test_events_ood = generator.generate_workflow_events(
        num_workflows=max(10, num_events // 20),  # Smaller test set
        agent_id="marketing-agent-1",
        include_failures=True,
        workflow_type="marketing",  # Test on marketing workflows (unseen)
    )
    
    # In-distribution test: Product workflows (same as training)
    print("   Generating IN-DISTRIBUTION test events (product workflows)...")
    test_events_indist = generator.generate_workflow_events(
        num_workflows=max(10, num_events // 20),
        agent_id="product-agent-2",  # Different agent ID
        include_failures=True,
        workflow_type="product",  # Same as training
    )
    
    # Novel sequences for generalization testing
    print("   Generating NOVEL sequences (generalization test)...")
    test_events_novel = []
    for seq_type in ["valid_partial", "valid_reordered", "invalid_mixed"]:
        test_events_novel.extend(generator.generate_novel_sequences(
            num_sequences=5,
            agent_id="test-agent-1",
            sequence_type=seq_type,
        ))
    
    total_train = len(train_events)
    total_test_ood = len(test_events_ood)
    total_test_indist = len(test_events_indist)
    total_test_novel = len(test_events_novel)
    
    print(f"\n   Training set: {total_train} events (product workflows)")
    print(f"   Test (OOD): {total_test_ood} events (marketing workflows)")
    print(f"   Test (In-Dist): {total_test_indist} events (product workflows)")
    print(f"   Test (Novel): {total_test_novel} events (novel sequences)")
    
    results["events"] = {
        "train": total_train,
        "test_ood": total_test_ood,
        "test_indist": total_test_indist,
        "test_novel": total_test_novel,
    }
    
    # Extract action types for analysis (from all event sets)
    all_test_events = test_events_ood + test_events_indist + test_events_novel
    action_types = set(e.get("action_type", "unknown") for e in train_events + all_test_events)
    print(f"   Unique action types: {len(action_types)}")
    print(f"   Actions: {sorted(action_types)}")
    
    results["action_types"] = sorted(action_types)
    
    # Step 2: Run L* Learning (Both Concrete and Abstraction modes)
    print("\n2. RUNNING L* LEARNING")
    print("-" * 40)
    
    # 2a. Concrete mode (memorization baseline)
    print("\n   [2a] CONCRETE MODE (Memorization Baseline)")
    learner = AutomataLearner(
        model_type="mealy",
        eq_oracle_type="random_walk",
        max_learning_rounds=100,
        random_walk_steps=5000,
        use_abstraction=False,  # Concrete mode
    )
    
    print(f"   Model type: {learner.model_type}")
    print(f"   Use abstraction: {learner.use_abstraction}")
    
    learning_result = learner.learn(train_events)
    
    print(f"\n   Concrete Learning Result:")
    print(f"   - Success: {learning_result.success}")
    print(f"   - States: {learning_result.num_states}")
    print(f"   - Transitions: {learning_result.num_transitions}")
    print(f"   - Learning time: {learning_result.learning_time_ms:.1f}ms")
    print(f"   - Model type: {learning_result.model_type}")
    
    if learning_result.error:
        print(f"   Warning: {learning_result.error}")
    
    results["learning_concrete"] = learning_result.to_dict()
    
    # 2b. Abstraction mode (generalization)
    print("\n   [2b] ABSTRACTION MODE (Generalization)")
    learner_abstract = AutomataLearner(
        model_type="mealy",
        eq_oracle_type="random_walk",
        max_learning_rounds=100,
        random_walk_steps=5000,
        use_abstraction=True,  # Abstraction mode
    )
    
    print(f"   Model type: {learner_abstract.model_type}")
    print(f"   Use abstraction: {learner_abstract.use_abstraction}")
    
    learning_result_abstract = learner_abstract.learn(train_events)
    
    print(f"\n   Abstraction Learning Result:")
    print(f"   - Success: {learning_result_abstract.success}")
    print(f"   - States: {learning_result_abstract.num_states}")
    print(f"   - Transitions: {learning_result_abstract.num_transitions}")
    print(f"   - Learning time: {learning_result_abstract.learning_time_ms:.1f}ms")
    print(f"   - Model type: {learning_result_abstract.model_type}")
    
    if learning_result_abstract.error:
        print(f"   Warning: {learning_result_abstract.error}")
    
    results["learning_abstract"] = learning_result_abstract.to_dict()
    results["learning"] = learning_result.to_dict()  # Keep for compatibility
    
    # Step 3: Test Prediction Accuracy (In-Distribution, OOD, and Novel)
    print("\n3. TESTING PREDICTION ACCURACY")
    print("-" * 40)
    
    # Import abstraction functions for evaluation
    from src.automata.sul import abstract_action
    
    def evaluate_predictions(predictor, test_events, test_name, use_abstraction=False):
        """Evaluate predictions on a test set.
        
        Args:
            predictor: BehaviorPredictor instance.
            test_events: List of test events.
            test_name: Name for logging.
            use_abstraction: If True, compare abstract predictions to abstract actuals.
        """
        correct = 0
        total_predictions = 0
        confidence_counts = {conf.value: 0 for conf in PredictionConfidence}
        confidence_correct = {conf.value: 0 for conf in PredictionConfidence}
        
        # Evaluate per thread_id to avoid mixing independent workflows/streams.
        threads: dict[str, list[dict]] = {}
        for e in test_events:
            tid = e.get("thread_id", "default")
            threads.setdefault(tid, []).append(e)
        for tid in threads:
            threads[tid].sort(key=lambda x: x.get("timestamp", ""))
 
        for tid, events_in_thread in threads.items():
            current_sequence: list[str] = []
            # predict next action within thread
            for i in range(len(events_in_thread) - 1):
                action = events_in_thread[i].get("action_type", "unknown")
                current_sequence.append(action)

                prediction = predictor.predict_next_action(current_sequence.copy())
                actual_next = events_in_thread[i + 1].get("action_type", "unknown")

                # For abstraction mode, compare at abstract level
                if use_abstraction:
                    predicted = prediction.predicted_action
                    actual = abstract_action(actual_next)
                    is_correct = (predicted == actual)
                else:
                    is_correct = prediction.predicted_action == actual_next
                
                if is_correct:
                    correct += 1
                    confidence_correct[prediction.confidence.value] += 1

                confidence_counts[prediction.confidence.value] += 1
                total_predictions += 1

                # Reset at explicit terminal-like actions within a thread
                if action in ["TASK_COMPLETE", "failure"]:
                    current_sequence = []
        
        accuracy = correct / total_predictions if total_predictions > 0 else 0
        
        # Calculate accuracy per confidence level
        accuracy_by_confidence = {}
        for conf in confidence_counts:
            if confidence_counts[conf] > 0:
                accuracy_by_confidence[conf] = confidence_correct[conf] / confidence_counts[conf]
            else:
                accuracy_by_confidence[conf] = 0.0
        
        return {
            "total_predictions": total_predictions,
            "correct": correct,
            "accuracy": accuracy,
            "confidence_distribution": confidence_counts,
            "accuracy_by_confidence": accuracy_by_confidence,
        }
    
    if learning_result.success:
        # ===============================================
        # 3a. CONCRETE MODE EVALUATION (Memorization)
        # ===============================================
        print("\n   [3a] CONCRETE MODE (Memorization)")
        predictor = BehaviorPredictor(learner=learner)
        predictor._learning_result = learning_result  # noqa: SLF001
        predictor._event_history = train_events  # noqa: SLF001
        
        # Test in-distribution
        print("\n   In-Distribution Test (product workflows):")
        indist_results = evaluate_predictions(predictor, test_events_indist, "in-dist", use_abstraction=False)
        print(f"   - Accuracy: {indist_results['accuracy']:.1%}")
        print(f"   - Predictions: {indist_results['total_predictions']}")
        
        # Test out-of-distribution
        print("\n   Out-of-Distribution Test (marketing workflows):")
        ood_results = evaluate_predictions(predictor, test_events_ood, "ood", use_abstraction=False)
        print(f"   - Accuracy: {ood_results['accuracy']:.1%}")
        print(f"   - Predictions: {ood_results['total_predictions']}")
        
        # Test novel sequences
        print("\n   Novel Sequences Test:")
        novel_results = evaluate_predictions(predictor, test_events_novel, "novel", use_abstraction=False)
        print(f"   - Accuracy: {novel_results['accuracy']:.1%}")
        print(f"   - Predictions: {novel_results['total_predictions']}")
        
        # Calculate generalization gap
        generalization_gap = indist_results['accuracy'] - ood_results['accuracy']
        print(f"\n   Concrete Generalization Gap: {generalization_gap:.1%}")
        
        # ===============================================
        # 3b. ABSTRACTION MODE EVALUATION (Generalization)
        # ===============================================
        print("\n   [3b] ABSTRACTION MODE (Generalization)")
        predictor_abstract = BehaviorPredictor(learner=learner_abstract)
        predictor_abstract._learning_result = learning_result_abstract  # noqa: SLF001
        predictor_abstract._event_history = train_events  # noqa: SLF001
        
        # Test in-distribution (abstract)
        print("\n   In-Distribution Test (abstract):")
        indist_results_abs = evaluate_predictions(predictor_abstract, test_events_indist, "in-dist-abs", use_abstraction=True)
        print(f"   - Accuracy: {indist_results_abs['accuracy']:.1%}")
        print(f"   - Predictions: {indist_results_abs['total_predictions']}")
        
        # Test out-of-distribution (abstract) - THIS IS THE KEY TEST
        print("\n   Out-of-Distribution Test (abstract):")
        ood_results_abs = evaluate_predictions(predictor_abstract, test_events_ood, "ood-abs", use_abstraction=True)
        print(f"   - Accuracy: {ood_results_abs['accuracy']:.1%}")
        print(f"   - Predictions: {ood_results_abs['total_predictions']}")
        
        # Test novel sequences (abstract)
        print("\n   Novel Sequences Test (abstract):")
        novel_results_abs = evaluate_predictions(predictor_abstract, test_events_novel, "novel-abs", use_abstraction=True)
        print(f"   - Accuracy: {novel_results_abs['accuracy']:.1%}")
        print(f"   - Predictions: {novel_results_abs['total_predictions']}")
        
        # Calculate abstraction generalization gap
        generalization_gap_abs = indist_results_abs['accuracy'] - ood_results_abs['accuracy']
        print(f"\n   Abstraction Generalization Gap: {generalization_gap_abs:.1%}")
        
        # ===============================================
        # COMPARISON SUMMARY
        # ===============================================
        print("\n   " + "=" * 50)
        print("   GENERALIZATION COMPARISON")
        print("   " + "=" * 50)
        print(f"   {'Mode':<20} {'In-Dist':<12} {'OOD':<12} {'Gap':<12}")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        concrete_in = f"{indist_results['accuracy']:.1%}"
        concrete_ood = f"{ood_results['accuracy']:.1%}"
        concrete_gap = f"{generalization_gap:.1%}"
        abstract_in = f"{indist_results_abs['accuracy']:.1%}"
        abstract_ood = f"{ood_results_abs['accuracy']:.1%}"
        abstract_gap = f"{generalization_gap_abs:.1%}"
        print(f"   {'Concrete (Memo)':<20} {concrete_in:<12} {concrete_ood:<12} {concrete_gap:<12}")
        print(f"   {'Abstract (Gen)':<20} {abstract_in:<12} {abstract_ood:<12} {abstract_gap:<12}")
        improvement = generalization_gap - generalization_gap_abs
        print(f"\n   Improvement: {improvement:.1%} smaller gap with abstraction")
        
        # Store all results
        results["prediction"] = {
            "concrete": {
                "in_distribution": indist_results,
                "out_of_distribution": ood_results,
                "novel_sequences": novel_results,
                "generalization_gap": generalization_gap,
            },
            "abstract": {
                "in_distribution": indist_results_abs,
                "out_of_distribution": ood_results_abs,
                "novel_sequences": novel_results_abs,
                "generalization_gap": generalization_gap_abs,
            },
            "improvement": generalization_gap - generalization_gap_abs,
        }
        # Keep for backwards compatibility
        results["prediction"]["in_distribution"] = indist_results
        results["prediction"]["out_of_distribution"] = ood_results
        results["prediction"]["generalization_gap"] = generalization_gap
    else:
        print("   Skipped: No learned model")
        results["prediction"] = {"skipped": True}
    
    # Step 4: Compare SimpleAutomaton vs True L* (if AALpy worked)
    print("\n4. COMPARING SIMPLEAUTOMATON vs TRUE L*")
    print("-" * 40)
    
    if learning_result.model_type == "simple":
        print("   Warning: AALpy L* failed, using SimpleAutomaton (memorization)")
        print("   This indicates memorization, not generalization")
    else:
        print(f"   Successfully used AALpy L* ({learning_result.model_type})")
        print(f"   Queries made: {learning_result.queries_made}")
        print(f"   Equivalence queries: {learning_result.equivalence_queries}")
        
        # Also test SimpleAutomaton for comparison
        print("\n   Testing SimpleAutomaton (memorization baseline)...")
        simple_learner = AutomataLearner()
        simple_result = simple_learner.learn(train_events)
        if simple_result.success and simple_result.model_type == "simple":
            simple_predictor = BehaviorPredictor(learner=simple_learner)
            simple_predictor._learning_result = simple_result  # noqa: SLF001
            simple_predictor._event_history = train_events  # noqa: SLF001
            
            simple_indist = evaluate_predictions(simple_predictor, test_events_indist, "simple-indist")
            simple_ood = evaluate_predictions(simple_predictor, test_events_ood, "simple-ood")
            simple_gap = simple_indist['accuracy'] - simple_ood['accuracy']
            
            print(f"   SimpleAutomaton In-Dist: {simple_indist['accuracy']:.1%}")
            print(f"   SimpleAutomaton OOD: {simple_ood['accuracy']:.1%}")
            print(f"   SimpleAutomaton Gap: {simple_gap:.1%}")
            
            if "prediction" in results and not results["prediction"].get("skipped"):
                lstar_gap = results["prediction"]["generalization_gap"]
                print(f"\n   Comparison:")
                print(f"   - L* Generalization Gap: {lstar_gap:.1%}")
                print(f"   - SimpleAutomaton Gap: {simple_gap:.1%}")
                print(f"   - Improvement: {simple_gap - lstar_gap:.1%} (lower gap = better generalization)")
            
            results["comparison"] = {
                "simple_automaton": {
                    "in_distribution": simple_indist['accuracy'],
                    "out_of_distribution": simple_ood['accuracy'],
                    "generalization_gap": simple_gap,
                },
                "lstar": {
                    "in_distribution": results.get("prediction", {}).get("in_distribution", {}).get("accuracy", 0),
                    "out_of_distribution": results.get("prediction", {}).get("out_of_distribution", {}).get("accuracy", 0),
                    "generalization_gap": results.get("prediction", {}).get("generalization_gap", 0),
                } if "prediction" in results else {},
            }
    
    # Step 5: Export Visualization
    print("\n5. EXPORTING VISUALIZATION")
    print("-" * 40)
    
    dot_path = output_path / "learned_automaton.dot"
    json_path = output_path / "learning_results.json"
    
    # Save DOT file
    if learner.save_model(str(dot_path)):
        print(f"   Automaton saved to: {dot_path}")
        results["outputs"] = {"dot": str(dot_path)}
    else:
        # Create a simple DOT representation manually
        dot_content = generate_dot_from_learner(learner, learning_result)
        with open(dot_path, 'w') as f:
            f.write(dot_content)
        print(f"   Automaton DOT saved to: {dot_path}")
        results["outputs"] = {"dot": str(dot_path)}
    
    # Save JSON results
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {json_path}")
    
    # Step 6: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Training events: {total_train} (product workflows)")
    print(f"   Test events (OOD): {total_test_ood} (marketing workflows)")
    print(f"   Test events (In-Dist): {total_test_indist} (product workflows)")
    print(f"   Test events (Novel): {total_test_novel} (novel sequences)")
    print(f"\n   Automaton:")
    print(f"   - Model type: {learning_result.model_type}")
    print(f"   - States: {learning_result.num_states}")
    print(f"   - Transitions: {learning_result.num_transitions}")
    print(f"   - Learning time: {learning_result.learning_time_ms:.1f}ms")
    if learning_result.queries_made > 0:
        print(f"   - Queries made: {learning_result.queries_made}")
        print(f"   - Equivalence queries: {learning_result.equivalence_queries}")
    
    if "prediction" in results and not results["prediction"].get("skipped"):
        print(f"\n   Prediction Results (Concrete Mode):")
        print(f"   - In-Distribution Accuracy: {results['prediction']['concrete']['in_distribution']['accuracy']:.1%}")
        print(f"   - Out-of-Distribution Accuracy: {results['prediction']['concrete']['out_of_distribution']['accuracy']:.1%}")
        print(f"   - Generalization Gap: {results['prediction']['concrete']['generalization_gap']:.1%}")
        print(f"\n   Prediction Results (Abstract Mode):")
        print(f"   - In-Distribution Accuracy: {results['prediction']['abstract']['in_distribution']['accuracy']:.1%}")
        print(f"   - Out-of-Distribution Accuracy: {results['prediction']['abstract']['out_of_distribution']['accuracy']:.1%}")
        print(f"   - Generalization Gap: {results['prediction']['abstract']['generalization_gap']:.1%}")
        print(f"\n   IMPROVEMENT: {results['prediction']['improvement']:.1%} smaller gap with abstraction")
    
    print(f"\n   Output files:")
    print(f"   - {dot_path}")
    print(f"   - {json_path}")
    print("=" * 70)
    
    return results


def generate_dot_from_learner(learner: AutomataLearner, result: LearningResult) -> str:
    """Generate DOT format visualization from learner state.
    
    Args:
        learner: The AutomataLearner instance.
        result: The LearningResult from learning.
        
    Returns:
        DOT format string for visualization.
    """
    lines = [
        'digraph AgentBehavior {',
        '    rankdir=LR;',
        '    node [shape=circle];',
        f'    // States: {result.num_states}',
        f'    // Transitions: {result.num_transitions}',
        '',
    ]
    
    # Try to extract transitions from SUL
    if learner._sul is not None:
        transitions = getattr(learner._sul, '_transitions', {})
        for (state, input_sym), (next_state, output) in transitions.items():
            label = f"{input_sym}/{output}"
            lines.append(f'    s{state} -> s{next_state} [label="{label}"];')
    else:
        # Fallback: create placeholder visualization
        lines.append('    s0 [label="Initial"];')
        lines.append('    s1 [label="Processing"];')
        lines.append('    s2 [label="Complete", shape=doublecircle];')
        lines.append('    s0 -> s1 [label="TASK_ASSIGN"];')
        lines.append('    s1 -> s1 [label="step/status"];')
        lines.append('    s1 -> s2 [label="TASK_COMPLETE"];')
    
    lines.append('}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run L* automata learning on synthetic agent events"
    )
    parser.add_argument(
        "--events", "-n",
        type=int,
        default=500,
        help="Number of events to generate (default: 500)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/automata",
        help="Output directory (default: data/automata)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_actual_lstar(
            num_events=args.events,
            random_seed=args.seed,
            output_dir=args.output,
        )
        
        # Return appropriate exit code
        if results.get("learning", {}).get("success", False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running automata learning: {e}")
        raise


if __name__ == "__main__":
    main()

