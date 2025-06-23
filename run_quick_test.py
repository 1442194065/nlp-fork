#!/usr/bin/env python3
"""
Quick test script for DIVERSE CoT project
This script runs a minimal test to verify the setup is working correctly
"""

import sys
import os
from config import config
from llm_client import test_deepseek_connection
from data_loader import load_aqua_data

def test_data_loading():
    """Test if AQuA dataset can be loaded"""
    print("Testing data loading...")
    try:
        data_loader = load_aqua_data("dev")  # Load only dev set for testing
        dev_subset = data_loader.get_eval_subset("dev", 5)  # Just 5 examples
        
        if len(dev_subset) > 0:
            print(f"✅ Successfully loaded {len(dev_subset)} examples")
            
            # Show first example
            example = dev_subset[0]
            print(f"Example question: {example.question[:100]}...")
            print(f"Options: {example.options}")
            print(f"Correct answer: {example.correct}")
            return True
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('requests', 'Requests'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn')
    ]
    
    missing_packages = []
    
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            print(f"✅ {display_name} is available")
        except ImportError:
            print(f"❌ {display_name} is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + ' '.join(missing_packages))
        return False
    
    return True

def test_directories():
    """Test if required directories exist or can be created"""
    print("Testing directory structure...")
    
    directories = [
        config.experiment.output_dir,
        config.experiment.log_dir,
        config.experiment.model_save_dir,
        config.experiment.results_dir,
        config.data.data_dir
    ]
    
    all_good = True
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory} exists")
        else:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created {directory}")
            except Exception as e:
                print(f"❌ Cannot create {directory}: {e}")
                all_good = False
    
    return all_good

def test_config():
    """Test configuration settings"""
    print("Testing configuration...")
    
    # Check API key
    if config.model.deepseek_api_key and len(config.model.deepseek_api_key) > 10:
        print("✅ DeepSeek API key is configured")
    else:
        print("❌ DeepSeek API key is missing or invalid")
        return False
    
    # Check other important settings
    print(f"✅ Model: {config.model.model_name}")
    print(f"✅ Max train samples: {config.data.max_train_samples}")
    print(f"✅ Max eval samples: {config.data.max_eval_samples}")
    print(f"✅ DIVERSE settings: {config.diverse.num_prompts} prompts, {config.diverse.num_samples_per_prompt} samples each")
    
    return True

def run_mini_evaluation():
    """Run a very small evaluation to test the complete pipeline"""
    print("Running mini evaluation with 2 examples...")
    
    try:
        from llm_client import DeepSeekClient
        from diverse_cot import BaselineCoT
        
        # Load client and data
        llm_client = DeepSeekClient()
        data_loader = load_aqua_data("dev")
        examples = data_loader.get_eval_subset("dev", 2)  # Just 2 examples
        
        # Test baseline method
        baseline = BaselineCoT(llm_client)
        
        print("Testing baseline CoT...")
        for i, example in enumerate(examples):
            print(f"Example {i+1}: {example.question[:50]}...")
            result = baseline.solve(example)
            print(f"Predicted: {result['predicted_answer']}, Correct: {example.correct}")
            
            is_correct = result['predicted_answer'] == example.correct
            print(f"Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
        
        print("✅ Mini evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Mini evaluation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("DIVERSE CoT Quick Test")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Data Loading", test_data_loading),
        ("API Connection", test_deepseek_connection),
        ("Mini Evaluation", run_mini_evaluation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running {test_name} Test")
        print(f"{'-'*40}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your setup is ready. You can now run:")
        print("  python main.py --num-eval 10")
        print("to start a small-scale experiment.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before running the main script.")
        sys.exit(1)
    
    print("="*60)

if __name__ == "__main__":
    main() 