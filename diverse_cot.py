"""
DIVERSE: Diverse Verifier on Reasoning Step
Main implementation of the DIVERSE method for enhanced chain-of-thought reasoning
"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import random
from dataclasses import dataclass

from config import config
from data_loader import AQuAExample, AQuADataLoader
from llm_client import DeepSeekClient, ResponseAnalyzer
from verifier import VerifierTrainer

@dataclass
class DiverseResult:
    """Result of DIVERSE inference"""
    predicted_answer: str
    confidence: float
    voting_scores: Dict[str, float]
    reasoning_paths: List[Dict]
    final_reasoning: str

class DiverseCoT:
    """DIVERSE Chain-of-Thought implementation"""
    
    def __init__(self, llm_client: DeepSeekClient, verifier: VerifierTrainer = None):
        self.llm_client = llm_client
        self.verifier = verifier
        self.data_loader = AQuADataLoader()
        
    def generate_diverse_reasoning_paths(self, example: AQuAExample) -> List[Dict]:
        """Generate diverse reasoning paths for a single example"""
        print(f"Generating diverse paths for: {example.question[:50]}...")
        
        # Generate responses using different prompts and sampling
        responses = self.llm_client.generate_diverse_responses(
            example.question,
            example.options,
            num_prompts=config.diverse.num_prompts,
            num_samples_per_prompt=config.diverse.num_samples_per_prompt
        )
        
        # Process responses
        reasoning_paths = []
        for response_data in responses:
            response = response_data['response']
            predicted_answer = self.data_loader.extract_answer_from_response(response)
            steps = self.data_loader.parse_reasoning_steps(response)
            
            path_info = {
                'response': response,
                'predicted_answer': predicted_answer,
                'steps': steps,
                'prompt_idx': response_data['prompt_idx'],
                'sample_idx': response_data['sample_idx'],
                'temperature': response_data.get('temperature', config.model.temperature)
            }
            reasoning_paths.append(path_info)
        
        return reasoning_paths
    
    def majority_voting(self, reasoning_paths: List[Dict]) -> DiverseResult:
        """Simple majority voting among reasoning paths"""
        # Count votes for each answer
        answer_counts = Counter()
        answer_paths = defaultdict(list)
        
        for path in reasoning_paths:
            answer = path['predicted_answer']
            answer_counts[answer] += 1
            answer_paths[answer].append(path)
        
        # Get most common answer
        most_common_answer, vote_count = answer_counts.most_common(1)[0]
        confidence = vote_count / len(reasoning_paths)
        
        # Create voting scores
        voting_scores = {}
        for answer, count in answer_counts.items():
            voting_scores[answer] = count / len(reasoning_paths)
        
        # Select best reasoning path for the winning answer
        winning_paths = answer_paths[most_common_answer]
        final_reasoning = winning_paths[0]['response']  # Take first one as representative
        
        return DiverseResult(
            predicted_answer=most_common_answer,
            confidence=confidence,
            voting_scores=voting_scores,
            reasoning_paths=reasoning_paths,
            final_reasoning=final_reasoning
        )
    
    def verifier_voting(self, example: AQuAExample, reasoning_paths: List[Dict]) -> DiverseResult:
        """Voting with verifier scores"""
        if self.verifier is None:
            raise ValueError("Verifier not available. Use majority voting instead.")
        
        # Get verifier scores for each path
        verifier_inputs = []
        for path in reasoning_paths:
            verifier_inputs.append({
                'question': example.question,
                'options': example.options,
                'reasoning_path': path['response']
            })
        
        # Batch prediction from verifier
        verifier_scores = self.verifier.batch_predict(verifier_inputs)
        
        # Add scores to paths
        for path, score in zip(reasoning_paths, verifier_scores):
            path['verifier_score'] = score
        
        # Weighted voting based on verifier scores
        answer_scores = defaultdict(float)
        answer_paths = defaultdict(list)
        
        for path in reasoning_paths:
            answer = path['predicted_answer']
            score = path['verifier_score']
            answer_scores[answer] += score
            answer_paths[answer].append(path)
        
        # Get answer with highest total score
        best_answer = max(answer_scores.keys(), key=lambda x: answer_scores[x])
        
        # Calculate confidence and voting scores
        total_score = sum(answer_scores.values())
        confidence = answer_scores[best_answer] / total_score if total_score > 0 else 0
        
        voting_scores = {}
        for answer, score in answer_scores.items():
            voting_scores[answer] = score / total_score if total_score > 0 else 0
        
        # Select best reasoning path for the winning answer
        winning_paths = answer_paths[best_answer]
        # Sort by verifier score and take the best one
        winning_paths.sort(key=lambda x: x['verifier_score'], reverse=True)
        final_reasoning = winning_paths[0]['response']
        
        return DiverseResult(
            predicted_answer=best_answer,
            confidence=confidence,
            voting_scores=voting_scores,
            reasoning_paths=reasoning_paths,
            final_reasoning=final_reasoning
        )
    
    def step_aware_verifier_voting(self, example: AQuAExample, reasoning_paths: List[Dict]) -> DiverseResult:
        """Advanced voting with step-aware verifier"""
        if self.verifier is None:
            raise ValueError("Verifier not available. Use majority voting instead.")
        
        # Get path-level scores
        path_inputs = []
        step_inputs = []
        path_indices = []
        step_to_path_mapping = []
        
        for path_idx, path in enumerate(reasoning_paths):
            # Path-level input
            path_inputs.append({
                'question': example.question,
                'options': example.options,
                'reasoning_path': path['response']
            })
            path_indices.append(path_idx)
            
            # Step-level inputs
            for step in path['steps']:
                step_inputs.append({
                    'question': example.question,
                    'options': example.options,
                    'reasoning_step': step
                })
                step_to_path_mapping.append(path_idx)
        
        # Get predictions
        path_scores = self.verifier.batch_predict(path_inputs)
        step_scores = self.verifier.batch_predict(step_inputs)
        
        # Assign scores back to paths
        for path_idx, score in zip(path_indices, path_scores):
            reasoning_paths[path_idx]['path_score'] = score
        
        # Assign step scores
        step_idx = 0
        for path_idx, path in enumerate(reasoning_paths):
            path['step_scores'] = []
            for _ in path['steps']:
                path['step_scores'].append(step_scores[step_idx])
                step_idx += 1
            
            # Calculate average step score
            if path['step_scores']:
                path['avg_step_score'] = np.mean(path['step_scores'])
            else:
                path['avg_step_score'] = 0.0
        
        # Combine path and step scores
        alpha = config.verifier.step_aware_alpha
        for path in reasoning_paths:
            combined_score = (1 - alpha) * path['path_score'] + alpha * path['avg_step_score']
            path['combined_score'] = combined_score
        
        # Weighted voting based on combined scores
        answer_scores = defaultdict(float)
        answer_paths = defaultdict(list)
        
        for path in reasoning_paths:
            answer = path['predicted_answer']
            score = path['combined_score']
            answer_scores[answer] += score
            answer_paths[answer].append(path)
        
        # Get answer with highest total score
        best_answer = max(answer_scores.keys(), key=lambda x: answer_scores[x])
        
        # Calculate confidence and voting scores
        total_score = sum(answer_scores.values())
        confidence = answer_scores[best_answer] / total_score if total_score > 0 else 0
        
        voting_scores = {}
        for answer, score in answer_scores.items():
            voting_scores[answer] = score / total_score if total_score > 0 else 0
        
        # Select best reasoning path for the winning answer
        winning_paths = answer_paths[best_answer]
        # Sort by combined score and take the best one
        winning_paths.sort(key=lambda x: x['combined_score'], reverse=True)
        final_reasoning = winning_paths[0]['response']
        
        return DiverseResult(
            predicted_answer=best_answer,
            confidence=confidence,
            voting_scores=voting_scores,
            reasoning_paths=reasoning_paths,
            final_reasoning=final_reasoning
        )
    
    def solve(self, example: AQuAExample, method: str = "auto") -> DiverseResult:
        """Solve a single problem using DIVERSE method"""
        # Generate diverse reasoning paths
        reasoning_paths = self.generate_diverse_reasoning_paths(example)
        
        # Choose voting method
        if method == "auto":
            if self.verifier is not None:
                if config.diverse.use_step_aware:
                    method = "step_aware_verifier"
                else:
                    method = "verifier"
            else:
                method = "majority"
        
        # Apply voting method
        if method == "majority":
            result = self.majority_voting(reasoning_paths)
        elif method == "verifier":
            result = self.verifier_voting(example, reasoning_paths)
        elif method == "step_aware_verifier":
            result = self.step_aware_verifier_voting(example, reasoning_paths)
        else:
            raise ValueError(f"Unknown voting method: {method}")
        
        return result
    
    def solve_batch(self, examples: List[AQuAExample], method: str = "auto") -> List[DiverseResult]:
        """Solve a batch of problems"""
        results = []
        
        for i, example in enumerate(examples):
            print(f"Solving problem {i+1}/{len(examples)}")
            try:
                result = self.solve(example, method=method)
                results.append(result)
            except Exception as e:
                print(f"Error solving problem {i+1}: {e}")
                # Create a dummy result with error
                error_result = DiverseResult(
                    predicted_answer="A",  # Default
                    confidence=0.0,
                    voting_scores={"A": 1.0},
                    reasoning_paths=[],
                    final_reasoning=f"Error: {str(e)}"
                )
                results.append(error_result)
        
        return results

class BaselineCoT:
    """Baseline Chain-of-Thought for comparison"""
    
    def __init__(self, llm_client: DeepSeekClient):
        self.llm_client = llm_client
        self.data_loader = AQuADataLoader()
    
    def solve(self, example: AQuAExample) -> Dict:
        """Solve using standard chain-of-thought"""
        response = self.llm_client.generate_cot_response(
            example.question,
            example.options,
            prompt_template_idx=0
        )
        
        predicted_answer = self.data_loader.extract_answer_from_response(response)
        
        return {
            'predicted_answer': predicted_answer,
            'reasoning': response,
            'confidence': 1.0  # No confidence estimation for baseline
        }
    
    def solve_batch(self, examples: List[AQuAExample]) -> List[Dict]:
        """Solve a batch of problems with baseline method"""
        results = []
        
        for i, example in enumerate(examples):
            print(f"Solving problem {i+1}/{len(examples)} (Baseline)")
            try:
                result = self.solve(example)
                results.append(result)
            except Exception as e:
                print(f"Error solving problem {i+1}: {e}")
                error_result = {
                    'predicted_answer': "A",
                    'reasoning': f"Error: {str(e)}",
                    'confidence': 0.0
                }
                results.append(error_result)
        
        return results

def compare_methods(examples: List[AQuAExample], diverse_cot: DiverseCoT, 
                   baseline_cot: BaselineCoT) -> Dict[str, Any]:
    """Compare DIVERSE and baseline methods"""
    print("Running DIVERSE method...")
    diverse_results = diverse_cot.solve_batch(examples)
    
    print("Running baseline method...")
    baseline_results = baseline_cot.solve_batch(examples)
    
    # Calculate metrics
    diverse_correct = 0
    baseline_correct = 0
    total = len(examples)
    
    detailed_results = []
    
    for i, (example, diverse_result, baseline_result) in enumerate(zip(examples, diverse_results, baseline_results)):
        diverse_pred = diverse_result.predicted_answer
        baseline_pred = baseline_result['predicted_answer']
        ground_truth = example.correct
        
        diverse_is_correct = diverse_pred == ground_truth
        baseline_is_correct = baseline_pred == ground_truth
        
        if diverse_is_correct:
            diverse_correct += 1
        if baseline_is_correct:
            baseline_correct += 1
        
        detailed_results.append({
            'example_id': example.id,
            'question': example.question,
            'ground_truth': ground_truth,
            'diverse_prediction': diverse_pred,
            'baseline_prediction': baseline_pred,
            'diverse_correct': diverse_is_correct,
            'baseline_correct': baseline_is_correct,
            'diverse_confidence': diverse_result.confidence,
            'diverse_reasoning': diverse_result.final_reasoning,
            'baseline_reasoning': baseline_result['reasoning']
        })
    
    # Calculate accuracies
    diverse_accuracy = diverse_correct / total
    baseline_accuracy = baseline_correct / total
    
    # Calculate improvement
    improvement = diverse_accuracy - baseline_accuracy
    relative_improvement = (improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
    
    comparison_results = {
        'total_examples': total,
        'diverse_accuracy': diverse_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement': improvement,
        'relative_improvement_percent': relative_improvement,
        'diverse_correct': diverse_correct,
        'baseline_correct': baseline_correct,
        'detailed_results': detailed_results
    }
    
    return comparison_results

def save_results(results: Dict[str, Any], filename: str):
    """Save results to JSON file"""
    filepath = os.path.join(config.experiment.results_dir, filename)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def clean_for_json(data):
        if isinstance(data, dict):
            return {k: clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_for_json(item) for item in data]
        else:
            return convert_numpy(data)
    
    cleaned_results = clean_for_json(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {filepath}")

if __name__ == "__main__":
    # This module is meant to be imported and used by the main script
    print("DIVERSE CoT module loaded successfully")
    print("Use this module through main.py or evaluation scripts") 