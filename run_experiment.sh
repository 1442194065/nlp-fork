#!/bin/bash

# DIVERSE CoT Experiment Runner
# This script provides easy commands to run different types of experiments

set -e  # Exit on any error

echo "======================================"
echo "DIVERSE Chain-of-Thought Experiment Runner"
echo "======================================"

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  test         - Run quick setup test"
    echo "  quick        - Quick experiment (50 samples)"
    echo "  small        - Small experiment (100 samples)"
    echo "  medium       - Medium experiment (250 samples)"
    echo "  full         - Full experiment (500 samples)"
    echo "  ablation     - Full experiment with ablation study"
    echo "  eval-only    - Evaluation only (requires trained model)"
    echo "  help         - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test      # Test setup"
    echo "  $0 quick     # Quick test with 50 samples"
    echo "  $0 full      # Full experiment"
}

# Function to run test
run_test() {
    echo "Running setup test..."
    python run_quick_test.py
}

# Function to run quick experiment
run_quick() {
    echo "Running quick experiment (50 samples)..."
    python main.py --num-eval 50
}

# Function to run small experiment
run_small() {
    echo "Running small experiment (100 samples)..."
    python main.py --num-eval 100
}

# Function to run medium experiment
run_medium() {
    echo "Running medium experiment (250 samples)..."
    python main.py --num-eval 250
}

# Function to run full experiment
run_full() {
    echo "Running full experiment (500 samples)..."
    python main.py
}

# Function to run ablation study
run_ablation() {
    echo "Running full experiment with ablation study..."
    python main.py --ablation
}

# Function to run evaluation only
run_eval_only() {
    echo "Running evaluation only (requires trained model)..."
    python main.py --evaluation-only
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Process command
case "$1" in
    "test")
        run_test
        ;;
    "quick")
        run_quick
        ;;
    "small")
        run_small
        ;;
    "medium")
        run_medium
        ;;
    "full")
        run_full
        ;;
    "ablation")
        run_ablation
        ;;
    "eval-only")
        run_eval_only
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "Error: Unknown command '$1'"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Experiment completed!"
echo "Check the results/ directory for output files."
echo "======================================" 