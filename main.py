"""
Main script for DIVERSE CoT project
Orchestrates the complete pipeline: data loading, verifier training, and evaluation
"""

import argparse
import os
import time
import json
from datetime import datetime
import random
import numpy as np
import torch

from config import config
from data_loader import load_aqua_data, ReasoningPathGenerator
from llm_client import DeepSeekClient, test_deepseek_connection
from verifier import VerifierTrainer, download_and_prepare_verifier
from diverse_cot import DiverseCoT, BaselineCoT, compare_methods, save_results

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_experiment():
    """Setup experiment environment"""
    print("="*60)
    print("DIVERSE Chain-of-Thought Reasoning System")
    print("="*60)
    
    # Set seeds
    set_seed(config.experiment.seed)
    
    # Create directories
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    os.makedirs(config.experiment.model_save_dir, exist_ok=True)
    os.makedirs(config.experiment.results_dir, exist_ok=True)
    
    print(f"Experiment setup complete. Seed: {config.experiment.seed}")
    print(f"Output directory: {config.experiment.output_dir}")

def test_api_connection():
    """Test DeepSeek API connection"""
    print("\n" + "="*40)
    print("Testing API Connection")
    print("="*40)
    
    success = test_deepseek_connection()
    if not success:
        print("❌ API connection failed. Please check your API key and network connection.")
        return False
    
    print("✅ API connection successful!")
    return True

def prepare_data():
    """Load and prepare AQuA dataset"""
    print("\n" + "="*40)
    print("Loading AQuA Dataset")
    print("="*40)
    
    data_loader = load_aqua_data("all")
    
    # Get data subsets
    train_subset = data_loader.get_train_subset(config.data.max_train_samples)
    eval_subset = data_loader.get_eval_subset("dev", config.data.max_eval_samples)
    
    print(f"Training subset: {len(train_subset)} examples")
    print(f"Evaluation subset: {len(eval_subset)} examples")
    
    return data_loader, train_subset, eval_subset

def generate_verifier_training_data(train_subset, llm_client):
    """Generate training data for the verifier"""
    print("\n" + "="*40)
    print("Generating Verifier Training Data")
    print("="*40)
    print(f"total training examples: {len(train_subset)}")
    
    path_generator = ReasoningPathGenerator(llm_client)
    
    # Generate reasoning paths for training
    print("Generating diverse reasoning paths...")
    training_data = path_generator.create_verifier_training_data(train_subset)
    
    print(f"Generated {len(training_data)} training instances")
    
    # Save training data
    training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
    with open(training_data_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training data saved to {training_data_path}")
    return training_data

def train_verifier(training_data):
    """Train the verifier model"""
    print("\n" + "="*40)
    print("Training Verifier Model")
    print("="*40)
    
    # Download base model if needed
    try:
        verifier_trainer = VerifierTrainer()
        print("Base model loaded successfully")
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Downloading base model...")
        download_and_prepare_verifier()
        verifier_trainer = VerifierTrainer()
    
    # Split training data for train/validation
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Train the model
    training_history = verifier_trainer.train(train_data, val_data)
    
    print("✅ Verifier training completed!")
    return verifier_trainer

def run_evaluation(eval_subset, llm_client, verifier_trainer=None):
    """Run evaluation comparing DIVERSE and baseline methods"""
    print("\n" + "="*40)
    print("Running Evaluation")
    print("="*40)
    
    # Initialize methods
    diverse_cot = DiverseCoT(llm_client, verifier_trainer)
    baseline_cot = BaselineCoT(llm_client)
    
    # Run comparison
    print("Starting method comparison...")
    start_time = time.time()
    
    comparison_results = compare_methods(eval_subset, diverse_cot, baseline_cot)
    
    elapsed_time = time.time() - start_time
    comparison_results['evaluation_time_seconds'] = elapsed_time
    comparison_results['evaluation_time_formatted'] = f"{elapsed_time/60:.2f} minutes"
    
    # Print results
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total examples: {comparison_results['total_examples']}")
    print(f"DIVERSE accuracy: {comparison_results['diverse_accuracy']:.3f} ({comparison_results['diverse_correct']}/{comparison_results['total_examples']})")
    print(f"Baseline accuracy: {comparison_results['baseline_accuracy']:.3f} ({comparison_results['baseline_correct']}/{comparison_results['total_examples']})")
    print(f"Improvement: {comparison_results['improvement']:.3f} ({comparison_results['relative_improvement_percent']:.1f}%)")
    print(f"Evaluation time: {comparison_results['evaluation_time_formatted']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{timestamp}.json"
    save_results(comparison_results, results_filename)
    
    return comparison_results

def run_ablation_study(eval_subset, llm_client, verifier_trainer):
    """Run ablation study to analyze different components"""
    print("\n" + "="*40)
    print("Running Ablation Study")
    print("="*40)
    
    diverse_cot = DiverseCoT(llm_client, verifier_trainer)
    baseline_cot = BaselineCoT(llm_client)
    
    # Test different voting methods
    methods = ["majority", "verifier", "step_aware_verifier"]
    results = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        
        if method != "majority" and verifier_trainer is None:
            print(f"Skipping {method} - no verifier available")
            continue
        
        try:
            method_results = []
            for example in eval_subset:
                if method == "baseline":
                    result = baseline_cot.solve(example)
                    method_results.append({
                        'predicted_answer': result['predicted_answer'],
                        'confidence': result['confidence']
                    })
                else:
                    result = diverse_cot.solve(example, method=method)
                    method_results.append({
                        'predicted_answer': result.predicted_answer,
                        'confidence': result.confidence
                    })
            
            # Calculate accuracy
            correct = sum(1 for i, result in enumerate(method_results) 
                         if result['predicted_answer'] == eval_subset[i].correct)
            accuracy = correct / len(eval_subset)
            
            results[method] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(eval_subset)
            }
            
            print(f"{method} accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Error testing {method}: {e}")
    
    # Save ablation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_filename = f"ablation_study_{timestamp}.json"
    save_results(results, ablation_filename)
    
    return results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="DIVERSE CoT Reasoning System")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip verifier training and use existing model")
    parser.add_argument("--skip-data-generation", action="store_true",
                       help="Skip training data generation and use existing data")
    parser.add_argument("--evaluation-only", action="store_true",
                       help="Run evaluation only (requires trained verifier)")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    parser.add_argument("--num-eval", type=int, default=None,
                       help="Number of evaluation examples to use")
    
    args = parser.parse_args()
    
    # Setup
    setup_experiment()
    
    # Test API connection
    # if not test_api_connection():
    #     return
    
    # Initialize LLM client
    llm_client = DeepSeekClient()
    
    # Load data
    data_loader, train_subset, eval_subset = prepare_data()
    
    # Limit evaluation samples if specified
    if args.num_eval is not None:
        eval_subset = eval_subset[:args.num_eval]
        print(f"Limited evaluation to {len(eval_subset)} examples")
    
    verifier_trainer = None
    
    if not args.evaluation_only:
        # Generate training data for verifier
        if not args.skip_data_generation:
            training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
            
            if os.path.exists(training_data_path):
                print(f"Loading existing training data from {training_data_path}")
                with open(training_data_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
            else:
                training_data = generate_verifier_training_data(train_subset, llm_client)
        else:
            training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
        
        # Train verifier
        if not args.skip_training:
            verifier_trainer = train_verifier(training_data)
        else:
            # Load existing verifier
            verifier_trainer = VerifierTrainer()
            try:
                verifier_trainer.load_model()
                print("✅ Existing verifier model loaded")
            except Exception as e:
                print(f"❌ Error loading existing verifier: {e}")
                print("Training new verifier...")
                verifier_trainer = train_verifier(training_data)
    else:
        # Load existing verifier for evaluation only
        verifier_trainer = VerifierTrainer()
        try:
            verifier_trainer.load_model()
            print("✅ Existing verifier model loaded for evaluation")
        except Exception as e:
            print(f"❌ Error loading verifier for evaluation: {e}")
            print("Running without verifier (majority voting only)")
    
    # Run evaluation
    evaluation_results = run_evaluation(eval_subset, llm_client, verifier_trainer)
    
    # Run ablation study if requested
    if args.ablation and verifier_trainer is not None:
        ablation_results = run_ablation_study(eval_subset, llm_client, verifier_trainer)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved in: {config.experiment.results_dir}")
    
    # Print final summary
    if evaluation_results:
        print(f"\nFinal Results:")
        print(f"DIVERSE Method: {evaluation_results['diverse_accuracy']:.3f} accuracy")
        print(f"Baseline Method: {evaluation_results['baseline_accuracy']:.3f} accuracy") 
        print(f"Improvement: {evaluation_results['relative_improvement_percent']:.1f}%")

if __name__ == "__main__":
    main() 