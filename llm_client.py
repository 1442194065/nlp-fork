"""
DeepSeek API client for language model inference
Handles API calls to DeepSeek for chain-of-thought reasoning
"""

import requests
import time
import json
from typing import List, Dict, Optional, Any
import random
from config import config, COT_PROMPTS
from data_loader import AQuAExample, create_few_shot_prompt

class DeepSeekClient:
    """Client for DeepSeek API interactions"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or config.model.deepseek_api_key
        self.base_url = base_url or config.model.deepseek_base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _make_request(self, messages: List[Dict], temperature: float = None, 
                     max_tokens: int = None, timeout: int = None) -> str:
        """Make a request to DeepSeek API"""
        payload = {
            "model": config.model.model_name,
            "messages": messages,
            "temperature": temperature or config.model.temperature,
            "max_tokens": max_tokens or config.model.max_tokens,
        }
        
        timeout = timeout or config.model.timeout
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    
    def generate_cot_response(self, question: str, options: List[str], 
                             prompt_template_idx: int = 0, 
                             temperature: float = None) -> str:
        """Generate chain-of-thought response for a single question"""
        
        # Create the prompt using the specified template
        template = COT_PROMPTS[prompt_template_idx]
        options_str = " ".join(options)
        
        # Create few-shot prompt
        from config import FEW_SHOT_EXAMPLES
        
        messages = []
        
        # Add system message
        system_message = {
            "role": "system",
            "content": "You are an expert at solving algebra word problems. Always show your step-by-step reasoning and end with 'The answer is [letter]'."
        }
        messages.append(system_message)
        
        # Add few-shot examples
        for example in FEW_SHOT_EXAMPLES:
            user_msg = {
                "role": "user",
                "content": f"Problem: {example['question']}\nOptions: {' '.join(example['options'])}\n\nSolve this step by step:"
            }
            assistant_msg = {
                "role": "assistant", 
                "content": f"{example['reasoning']}\nThe answer is {example['answer']}."
            }
            messages.extend([user_msg, assistant_msg])
        
        # Add the actual question
        user_message = {
            "role": "user",
            "content": template.format(question=question, options=options_str)
        }
        messages.append(user_message)
        
        return self._make_request(messages, temperature=temperature)
    
    def generate_diverse_responses(self, question: str, options: List[str],
                                 num_prompts: int = None, 
                                 num_samples_per_prompt: int = None) -> List[Dict]:
        """Generate diverse reasoning paths using different prompts and sampling"""
        
        num_prompts = num_prompts or config.diverse.num_prompts
        num_samples_per_prompt = num_samples_per_prompt or config.diverse.num_samples_per_prompt
        
        all_responses = []
        
        for prompt_idx in range(num_prompts):
            for sample_idx in range(num_samples_per_prompt):
                try:
                    # Use slightly different temperature for diversity
                    temperature = config.model.temperature + random.uniform(-0.1, 0.1)
                    temperature = max(0.1, min(1.0, temperature))  # Clamp to valid range
                    
                    response = self.generate_cot_response(
                        question, 
                        options, 
                        prompt_template_idx=prompt_idx % len(COT_PROMPTS),
                        temperature=temperature
                    )
                    
                    response_data = {
                        'response': response,
                        'prompt_idx': prompt_idx,
                        'sample_idx': sample_idx,
                        'temperature': temperature
                    }
                    all_responses.append(response_data)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error generating response {prompt_idx}-{sample_idx}: {e}")
                    continue
        
        return all_responses
    
    def batch_generate(self, examples: List[AQuAExample], method: str = "diverse") -> List[Dict]:
        """Generate responses for a batch of examples"""
        results = []
        
        for i, example in enumerate(examples):
            print(f"Processing example {i+1}/{len(examples)}: {example.id}")
            
            try:
                if method == "diverse":
                    responses = self.generate_diverse_responses(
                        example.question, 
                        example.options
                    )
                elif method == "single":
                    response = self.generate_cot_response(
                        example.question, 
                        example.options
                    )
                    responses = [{'response': response, 'prompt_idx': 0, 'sample_idx': 0}]
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                result = {
                    'example': example,
                    'responses': responses
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing example {example.id}: {e}")
                continue
        
        return results

class ResponseAnalyzer:
    """Analyze and process model responses"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def analyze_responses(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze the quality and diversity of responses"""
        analysis = {
            'total_examples': len(results),
            'total_responses': 0,
            'correct_responses': 0,
            'response_diversity': {},
            'prompt_performance': {},
            'temperature_effects': []
        }
        
        for result in results:
            example = result['example']
            responses = result['responses']
            analysis['total_responses'] += len(responses)
            
            for response_data in responses:
                response = response_data['response']
                predicted = self.data_loader.extract_answer_from_response(response)
                is_correct = predicted == example.correct
                
                if is_correct:
                    analysis['correct_responses'] += 1
                
                # Track prompt performance
                prompt_idx = response_data['prompt_idx']
                if prompt_idx not in analysis['prompt_performance']:
                    analysis['prompt_performance'][prompt_idx] = {'total': 0, 'correct': 0}
                analysis['prompt_performance'][prompt_idx]['total'] += 1
                if is_correct:
                    analysis['prompt_performance'][prompt_idx]['correct'] += 1
                
                # Track temperature effects
                if 'temperature' in response_data:
                    analysis['temperature_effects'].append({
                        'temperature': response_data['temperature'],
                        'correct': is_correct
                    })
        
        # Calculate accuracies
        if analysis['total_responses'] > 0:
            analysis['overall_accuracy'] = analysis['correct_responses'] / analysis['total_responses']
        
        for prompt_idx in analysis['prompt_performance']:
            perf = analysis['prompt_performance'][prompt_idx]
            if perf['total'] > 0:
                perf['accuracy'] = perf['correct'] / perf['total']
        
        return analysis
    
    def extract_reasoning_paths(self, results: List[Dict]) -> List[Dict]:
        """Extract reasoning paths for verifier training"""
        training_data = []
        
        for result in results:
            example = result['example']
            responses = result['responses']
            
            for response_data in responses:
                response = response_data['response']
                predicted = self.data_loader.extract_answer_from_response(response)
                is_correct = predicted == example.correct
                
                # Parse reasoning steps
                steps = self.data_loader.parse_reasoning_steps(response)
                
                # Create training instance
                training_instance = {
                    'question': example.question,
                    'options': example.options,
                    'reasoning_path': response,
                    'steps': steps,
                    'predicted_answer': predicted,
                    'ground_truth': example.correct,
                    'is_correct': is_correct,
                    'prompt_idx': response_data['prompt_idx']
                }
                training_data.append(training_instance)
        
        return training_data

def test_deepseek_connection():
    """Test DeepSeek API connection"""
    client = DeepSeekClient()
    
    try:
        test_question = "What is 2 + 2?"
        test_options = ["A)3", "B)4", "C)5", "D)6", "E)7"]
        
        response = client.generate_cot_response(test_question, test_options)
        print("DeepSeek API connection successful!")
        print(f"Test response: {response}")
        return True
        
    except Exception as e:
        print(f"DeepSeek API connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test the API connection
    test_deepseek_connection() 