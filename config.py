"""
Configuration file for DIVERSE CoT project
Contains all hyperparameters and settings
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
# Prompt templates for AQuA dataset

COT_PROMPTS = [
    """Solve the following algebra word problem step by step:

Problem: {question}
Options: {options}

Let me work through this step by step:""",

    """I need to solve this algebra problem carefully:

Question: {question}
Choices: {options}

Step-by-step solution:""",

    """Let me analyze this mathematical word problem:

Problem: {question}
Available answers: {options}

My reasoning process:""",

    """Here's an algebra word problem to solve:

{question}
Options: {options}

I'll solve this systematically:""",

    """Time to tackle this math problem:

Question: {question}
Multiple choice options: {options}

Breaking it down step by step:"""
]

# Few-shot examples for chain-of-thought prompting
FEW_SHOT_EXAMPLES = [
    {
        "question": "A grocery sells a bag of ice for $1.25, and makes 20% profit. If it sells 500 bags of ice, how much total profit does it make?",
        "options": ["A)125", "B)150", "C)225", "D)250", "E)275"],
        "reasoning": "First, I need to find the profit per bag. The selling price is $1.25 and the profit margin is 20%. So profit per bag = $1.25 × 0.20 = $0.25. Next, I calculate the total profit from 500 bags: Total profit = 500 × $0.25 = $125.",
        "answer": "A"
    },
    {
        "question": "If 3x + 7 = 22, what is the value of x?",
        "options": ["A)3", "B)5", "C)7", "D)9", "E)11"],
        "reasoning": "I need to solve for x in the equation 3x + 7 = 22. First, I subtract 7 from both sides: 3x = 22 - 7 = 15. Then I divide both sides by 3: x = 15 ÷ 3 = 5.",
        "answer": "B"
    },
    {
        "question": "A rectangular garden has length 12 meters and width 8 meters. What is its perimeter?",
        "options": ["A)20", "B)32", "C)40", "D)96", "E)160"],
        "reasoning": "The perimeter of a rectangle is calculated as 2 × (length + width). Given length = 12 meters and width = 8 meters, the perimeter = 2 × (12 + 8) = 2 × 20 = 40 meters.",
        "answer": "C"
    }
] 

@dataclass
class ModelConfig:
    """Configuration for language models"""
    deepseek_api_key: str = "sk-225ac3f431704f6c8632adceb9fb47b1"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"
    temperature: float = 0.5
    max_tokens: int = 2048
    timeout: int = 60

@dataclass
class VerifierConfig:
    """Configuration for the verifier model"""
    model_name: str = "microsoft/deberta-v3-large"
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 5
    max_length: int = 512
    step_aware_alpha: float = 0.2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

@dataclass
class DiverseConfig:
    """Configuration for DIVERSE method"""
    num_prompts: int = 5  # M1 in paper
    num_samples_per_prompt: int = 20  # M2 in paper
    total_samples: int = 100  # M1 * M2
    use_step_aware: bool = True
    voting_method: str = "verifier"  # "majority" or "verifier"

@dataclass
class DataConfig:
    """Configuration for data processing"""
    data_dir: str = "./AQuA-master"
    train_file: str = "train.json"
    dev_file: str = "dev.json"
    test_file: str = "test.json"
    max_train_samples: int = 1000  # For verifier training
    max_eval_samples: int = 500    # For evaluation

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    model_save_dir: str = "./models"
    results_dir: str = "./results"
    wandb_project: str = "diverse-cot"
    seed: int = 42

# Global configuration instance
class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.verifier = VerifierConfig()
        self.diverse = DiverseConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        
        # Add prompt templates to config
        self.COT_PROMPTS = COT_PROMPTS
        self.FEW_SHOT_EXAMPLES = FEW_SHOT_EXAMPLES
        
        # Create necessary directories
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        os.makedirs(self.experiment.log_dir, exist_ok=True)
        os.makedirs(self.experiment.model_save_dir, exist_ok=True)
        os.makedirs(self.experiment.results_dir, exist_ok=True)

# Create global config instance
config = Config()

