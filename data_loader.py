"""
Data loading and processing module for AQuA dataset
Handles loading, preprocessing, and formatting of AQuA data
"""

import json
import random
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from config import config

@dataclass
class AQuAExample:
    """Data structure for a single AQuA example"""
    question: str
    options: List[str]
    rationale: str
    correct: str
    id: Optional[str] = None

class AQuADataLoader:
    """Data loader for AQuA dataset"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or config.data.data_dir
        self.train_data = None
        self.dev_data = None
        self.test_data = None
    
    def load_data(self, split: str = "all") -> None:
        """Load data from JSON files"""
        if split in ["train", "all"]:
            self.train_data = self._load_json_file(config.data.train_file)
            print(f"Loaded {len(self.train_data)} training examples")
        
        if split in ["dev", "all"]:
            self.dev_data = self._load_json_file(config.data.dev_file)
            print(f"Loaded {len(self.dev_data)} development examples")
        
        if split in ["test", "all"]:
            self.test_data = self._load_json_file(config.data.test_file)
            print(f"Loaded {len(self.test_data)} test examples")
    
    def _load_json_file(self, filename: str) -> List[AQuAExample]:
        """Load data from a single JSON file"""
        filepath = f"{self.data_dir}/{filename}"
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    example = AQuAExample(
                        question=data['question'],
                        options=data['options'],
                        rationale=data['rationale'],
                        correct=data['correct'],
                        id=f"{filename}_{line_idx}"
                    )
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_idx} in {filename}: {e}")
                except KeyError as e:
                    print(f"Missing key {e} in line {line_idx} of {filename}")
        
        return examples
    
    def get_train_subset(self, max_samples: int = None) -> List[AQuAExample]:
        """Get a subset of training data for verifier training"""
        if self.train_data is None:
            self.load_data("train")
        
        max_samples = max_samples or config.data.max_train_samples
        if len(self.train_data) <= max_samples:
            return self.train_data
        
        # Randomly sample while maintaining balance
        random.shuffle(self.train_data)
        return self.train_data[:max_samples]
    
    def get_eval_subset(self, split: str = "dev", max_samples: int = None) -> List[AQuAExample]:
        """Get evaluation data subset"""
        if split == "dev":
            if self.dev_data is None:
                self.load_data("dev")
            data = self.dev_data
        else:
            if self.test_data is None:
                self.load_data("test")
            data = self.test_data
        
        max_samples = max_samples or config.data.max_eval_samples
        if len(data) <= max_samples:
            return data
        
        # Take first N samples for consistent evaluation
        return data[:max_samples]
    
    def format_options(self, options: List[str]) -> str:
        """Format options list into a readable string"""
        return " ".join(options)
    
    def extract_answer_from_response(self, response: str) -> str:
        """Extract the final answer from a model response"""
        # Look for patterns like "The answer is A", "Answer: B", etc.
        patterns = [
            r"[Tt]he answer is ([A-E])",
            r"[Aa]nswer:?\s*([A-E])",
            r"[Ff]inal answer:?\s*([A-E])",
            r"\b([A-E])\)?\s*$",  # Answer at the end
            r"[Cc]hoice\s*([A-E])",
            r"[Oo]ption\s*([A-E])"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, look for any A-E at the end
        words = response.strip().split()
        for word in reversed(words):
            if word.upper() in ['A', 'B', 'C', 'D', 'E']:
                return word.upper()
        
        return "A"  # Default fallback
    
    def parse_reasoning_steps(self, reasoning: str) -> List[str]:
        """Parse reasoning text into individual steps"""
        # Split by common step indicators
        step_patterns = [
            r'\n\d+\.',  # Numbered steps
            r'\nStep \d+:',  # Step N:
            r'\nFirst,',  # First, Second, etc.
            r'\nNext,',
            r'\nThen,',
            r'\nFinally,',
            r'\n-',  # Bullet points
            r'\n\*',  # Asterisk bullets
        ]
        
        steps = [reasoning]
        for pattern in step_patterns:
            new_steps = []
            for step in steps:
                new_steps.extend(re.split(pattern, step))
            steps = new_steps
        
        # Clean and filter steps
        cleaned_steps = []
        for step in steps:
            step = step.strip()
            if step and len(step) > 10:  # Filter very short steps
                cleaned_steps.append(step)
        
        return cleaned_steps if cleaned_steps else [reasoning]

class ReasoningPathGenerator:
    """Generate reasoning paths for training the verifier"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_reasoning_paths(self, example: AQuAExample, num_paths: int = 10) -> List[Dict]:
        """Generate multiple reasoning paths for a single example"""
        paths = []
        
        for i in range(num_paths):
            try:
                # Use different prompts for diversity
                prompt_idx = i % len(config.COT_PROMPTS)
                response = self.llm_client.generate_cot_response(
                    example.question, 
                    example.options,
                    prompt_template_idx=prompt_idx
                )
                
                predicted_answer = AQuADataLoader().extract_answer_from_response(response)
                is_correct = predicted_answer == example.correct
                
                # Parse reasoning steps
                steps = AQuADataLoader().parse_reasoning_steps(response)
                
                path_data = {
                    'response': response,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'steps': steps,
                    'prompt_idx': prompt_idx
                }
                paths.append(path_data)
                
            except Exception as e:
                print(f"Error generating reasoning path {i}: {e}")
                continue
        
        return paths
    
    def create_verifier_training_data(self, examples: List[AQuAExample]) -> List[Dict]:
        """Create training data for the verifier"""
        training_data = []
        
        for example in examples:
            print(f"Processing example: {example.id}")
            
            # Generate multiple reasoning paths
            paths = self.generate_reasoning_paths(example, num_paths=5)
            
            for path in paths:
                # Create training instance for path-level verification
                training_instance = {
                    'question': example.question,
                    'options': example.options,
                    'reasoning_path': path['response'],
                    'label': 1 if path['is_correct'] else 0,
                    'ground_truth': example.correct,
                    'predicted_answer': path['predicted_answer']
                }
                training_data.append(training_instance)
                
                # Create training instances for step-level verification
                if config.diverse.use_step_aware:
                    step_labels = self._get_step_labels(
                        path['steps'], 
                        path['is_correct'], 
                        example
                    )
                    
                    for step, step_label in zip(path['steps'], step_labels):
                        step_instance = {
                            'question': example.question,
                            'options': example.options,
                            'reasoning_step': step,
                            'step_label': step_label,
                            'is_step_level': True
                        }
                        training_data.append(step_instance)
        
        return training_data
    
    def _get_step_labels(self, steps: List[str], is_correct: bool, example: AQuAExample) -> List[int]:
        """Get labels for individual reasoning steps"""
        if is_correct:
            # If the final answer is correct, assume all steps are correct
            return [1] * len(steps)
        else:
            # For incorrect paths, we need to identify where the error occurs
            # This is a simplified heuristic - in practice, you might want more sophisticated logic
            step_labels = []
            for i, step in enumerate(steps):
                # Check if step contains obvious errors or contradictions
                if self._step_contains_error(step):
                    # Mark this step and all subsequent steps as incorrect
                    step_labels.extend([0] * (len(steps) - i))
                    break
                else:
                    step_labels.append(1)
            
            # If no obvious errors found, mark the last step as incorrect
            if len(step_labels) == len(steps):
                step_labels[-1] = 0
            
            return step_labels
    
    def _step_contains_error(self, step: str) -> bool:
        """Simple heuristic to detect erroneous reasoning steps"""
        error_indicators = [
            "mistake",
            "error",
            "wrong",
            "incorrect",
            "contradiction",
            "impossible"
        ]
        
        step_lower = step.lower()
        return any(indicator in step_lower for indicator in error_indicators)

# Utility functions
def load_aqua_data(split: str = "all") -> AQuADataLoader:
    """Convenience function to load AQuA data"""
    loader = AQuADataLoader()
    loader.load_data(split)
    return loader

def create_few_shot_prompt(examples: List[AQuAExample], target_example: AQuAExample, 
                          template_idx: int = 0) -> str:
    """Create a few-shot prompt with examples"""
    from config import COT_PROMPTS, FEW_SHOT_EXAMPLES
    
    prompt_parts = []
    
    # Add few-shot examples
    for ex in FEW_SHOT_EXAMPLES:
        example_text = f"""Problem: {ex['question']}
Options: {' '.join(ex['options'])}

Solution: {ex['reasoning']}
The answer is {ex['answer']}.

"""
        prompt_parts.append(example_text)
    
    # Add target problem
    template = COT_PROMPTS[template_idx]
    target_prompt = template.format(
        question=target_example.question,
        options=' '.join(target_example.options)
    )
    prompt_parts.append(target_prompt)
    
    return '\n'.join(prompt_parts) 