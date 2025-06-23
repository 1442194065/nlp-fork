"""
Verifier model for evaluating reasoning paths
Based on DeBERTa-v3-large as described in DIVERSE paper
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
    Trainer, TrainingArguments
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from config import config

@dataclass
class VerifierTrainingInstance:
    """Training instance for verifier"""
    question: str
    options: List[str]
    reasoning_path: str
    label: int  # 1 for correct, 0 for incorrect
    reasoning_step: str = None  # For step-level training
    step_label: int = None
    is_step_level: bool = False

class VerifierDataset(Dataset):
    """Dataset for verifier training"""
    
    def __init__(self, instances: List[VerifierTrainingInstance], tokenizer, max_length: int = 512):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        # Create input text
        if instance.is_step_level:
            # For step-level verification
            input_text = f"Question: {instance.question}\nOptions: {' '.join(instance.options)}\nReasoning Step: {instance.reasoning_step}"
            label = instance.step_label
        else:
            # For path-level verification
            input_text = f"Question: {instance.question}\nOptions: {' '.join(instance.options)}\nReasoning: {instance.reasoning_path}"
            label = instance.label
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'is_step_level': torch.tensor(instance.is_step_level, dtype=torch.bool)
        }

class VerifierModel(nn.Module):
    """Verifier model based on DeBERTa"""
    
    def __init__(self, model_name: str = None, use_step_aware: bool = True):
        super().__init__()
        self.model_name = model_name or config.verifier.model_name
        self.use_step_aware = use_step_aware
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name, config=self.config)
        
        # Classification heads
        self.path_classifier = nn.Linear(self.config.hidden_size, 2)
        
        if self.use_step_aware:
            self.step_classifier = nn.Linear(self.config.hidden_size, 2)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, labels=None, is_step_level=None):
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Determine which classifier to use
        if is_step_level is not None and self.use_step_aware:
            # Mixed batch - use appropriate classifier for each sample
            path_mask = ~is_step_level
            step_mask = is_step_level
            
            logits = torch.zeros(pooled_output.size(0), 2, device=pooled_output.device)
            
            if path_mask.any():
                logits[path_mask] = self.path_classifier(pooled_output[path_mask])
            
            if step_mask.any():
                logits[step_mask] = self.step_classifier(pooled_output[step_mask])
        else:
            # Single type batch - use path classifier by default
            logits = self.path_classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            outputs['loss'] = loss
        
        return outputs
    
    def predict_proba(self, input_ids, attention_mask, is_step_level=None):
        """Get prediction probabilities"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, is_step_level=is_step_level)
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            return probabilities[:, 1]  # Return probability of positive class

class VerifierTrainer:
    """Trainer for verifier model"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.verifier.model_name
        # Use AutoTokenizer which handles different tokenizer types automatically
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        except Exception as e:
            print(f"Failed to load fast tokenizer, trying slow tokenizer: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def prepare_training_data(self, training_instances: List[Dict]) -> List[VerifierTrainingInstance]:
        """Convert raw training data to VerifierTrainingInstance objects"""
        instances = []
        
        for item in training_instances:
            if item.get('is_step_level', False):
                # Step-level instance
                instance = VerifierTrainingInstance(
                    question=item['question'],
                    options=item['options'],
                    reasoning_path="",
                    label=0,  # Not used for step-level
                    reasoning_step=item['reasoning_step'],
                    step_label=item['step_label'],
                    is_step_level=True
                )
            else:
                # Path-level instance
                instance = VerifierTrainingInstance(
                    question=item['question'],
                    options=item['options'],
                    reasoning_path=item['reasoning_path'],
                    label=item['label'],
                    is_step_level=False
                )
            instances.append(instance)
        
        return instances
    
    def train(self, training_instances: List[Dict], validation_instances: List[Dict] = None):
        """Train the verifier model"""
        print("Preparing training data...")
        train_instances = self.prepare_training_data(training_instances)
        
        # Create datasets
        train_dataset = VerifierDataset(
            train_instances, 
            self.tokenizer, 
            max_length=config.verifier.max_length
        )
        
        val_dataset = None
        if validation_instances:
            val_instances = self.prepare_training_data(validation_instances)
            val_dataset = VerifierDataset(
                val_instances, 
                self.tokenizer, 
                max_length=config.verifier.max_length
            )
        
        # Initialize model
        self.model = VerifierModel(
            model_name=self.model_name,
            use_step_aware=config.diverse.use_step_aware
        )
        self.model.to(self.device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.experiment.model_save_dir,
            num_train_epochs=config.verifier.num_epochs,
            per_device_train_batch_size=config.verifier.batch_size,
            per_device_eval_batch_size=config.verifier.batch_size,
            warmup_ratio=config.verifier.warmup_ratio,
            weight_decay=config.verifier.weight_decay,
            logging_dir=config.experiment.log_dir,
            logging_steps=100,
            eval_steps=500,
            save_steps=500,
            eval_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_accuracy" if val_dataset else None,
            greater_is_better=True,
            report_to=None,  # 禁用wandb
            run_name=f"verifier_training_{config.diverse.use_step_aware}"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        self.save_model()
        
        return trainer.state.log_history
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, path: str = None):
        """Save the trained model"""
        if path is None:
            path = os.path.join(config.experiment.model_save_dir, "verifier_model")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save custom model components
        torch.save({
            'path_classifier': self.model.path_classifier.state_dict(),
            'step_classifier': self.model.step_classifier.state_dict() if self.model.use_step_aware else None,
            'config': {
                'model_name': self.model_name,
                'use_step_aware': self.model.use_step_aware
            }
        }, os.path.join(path, "verifier_components.pt"))
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load a trained model"""
        if path is None:
            path = os.path.join(config.experiment.model_save_dir, "verifier_model")
        
        # Load components
        components = torch.load(os.path.join(path, "verifier_components.pt"))
        
        # Initialize model
        self.model = VerifierModel(
            model_name=components['config']['model_name'],
            use_step_aware=components['config']['use_step_aware']
        )
        
        # Load encoder
        self.model.encoder = AutoModel.from_pretrained(path)
        
        # Load classifiers
        self.model.path_classifier.load_state_dict(components['path_classifier'])
        if components['step_classifier'] is not None:
            self.model.step_classifier.load_state_dict(components['step_classifier'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {path}")
    
    def predict(self, question: str, options: List[str], reasoning_path: str = None, 
                reasoning_step: str = None) -> float:
        """Predict the probability that a reasoning path/step is correct"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare input
        if reasoning_step is not None:
            input_text = f"Question: {question}\nOptions: {' '.join(options)}\nReasoning Step: {reasoning_step}"
            is_step_level = True
        else:
            input_text = f"Question: {question}\nOptions: {' '.join(options)}\nReasoning: {reasoning_path}"
            is_step_level = False
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=config.verifier.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        is_step_level_tensor = torch.tensor([is_step_level], dtype=torch.bool, device=self.device)
        
        # Predict
        probability = self.model.predict_proba(
            input_ids, 
            attention_mask, 
            is_step_level=is_step_level_tensor
        )
        
        return probability.item()
    
    def batch_predict(self, inputs: List[Dict]) -> List[float]:
        """Batch prediction for multiple inputs"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        probabilities = []
        
        # Process in batches
        batch_size = config.verifier.batch_size
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Prepare batch
            input_texts = []
            is_step_levels = []
            
            for item in batch:
                if 'reasoning_step' in item and item['reasoning_step'] is not None:
                    input_text = f"Question: {item['question']}\nOptions: {' '.join(item['options'])}\nReasoning Step: {item['reasoning_step']}"
                    is_step_levels.append(True)
                else:
                    input_text = f"Question: {item['question']}\nOptions: {' '.join(item['options'])}\nReasoning: {item['reasoning_path']}"
                    is_step_levels.append(False)
                
                input_texts.append(input_text)
            
            # Tokenize batch
            encodings = self.tokenizer(
                input_texts,
                truncation=True,
                padding='max_length',
                max_length=config.verifier.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            is_step_level_tensor = torch.tensor(is_step_levels, dtype=torch.bool, device=self.device)
            
            # Predict
            batch_probs = self.model.predict_proba(
                input_ids, 
                attention_mask, 
                is_step_level=is_step_level_tensor
            )
            
            probabilities.extend(batch_probs.cpu().tolist())
        
        return probabilities

def download_and_prepare_verifier():
    """Download and prepare the base verifier model"""
    print("Downloading base model...")
    # Use specific tokenizer to avoid conversion issues
    if "deberta" in config.verifier.model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(config.verifier.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.verifier.model_name, use_fast=False)
    model = AutoModel.from_pretrained(config.verifier.model_name)
    
    # Save to local directory
    model_dir = os.path.join(config.experiment.model_save_dir, "base_model")
    os.makedirs(model_dir, exist_ok=True)
    
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    
    print(f"Base model saved to {model_dir}")
    return model_dir

if __name__ == "__main__":
    # Test model loading
    download_and_prepare_verifier() 