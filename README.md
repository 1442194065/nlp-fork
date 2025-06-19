# DIVERSE Chain-of-Thought Reasoning

This project implements the DIVERSE (Diverse Verifier on Reasoning Step) method for enhanced chain-of-thought reasoning, as described in the research paper. The implementation uses the DeepSeek API for language model inference and trains a DeBERTa-based verifier to improve reasoning accuracy on the AQuA dataset.

## 🎯 Overview

DIVERSE enhances chain-of-thought reasoning through three key components:
1. **Diverse Prompts**: Generate multiple reasoning paths using different prompt templates
2. **Voting Verifier**: Train a verifier model to score reasoning path quality
3. **Step-aware Verification**: Evaluate individual reasoning steps for fine-grained quality assessment

## 📁 Project Structure

```
diverse-cot/
├── config.py                 # Configuration settings and hyperparameters
├── data_loader.py            # AQuA dataset loading and preprocessing
├── llm_client.py             # DeepSeek API client for LLM inference
├── verifier.py               # DeBERTa-based verifier model training
├── diverse_cot.py            # Main DIVERSE implementation
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── AQuA-master/             # AQuA dataset directory
├── outputs/                 # Generated training data
├── models/                  # Trained verifier models
├── results/                 # Evaluation results
└── logs/                    # Training logs
```

## 🚀 Quick Start

### Prerequisites

1. Python 3.8 or higher
2. CUDA GPU (recommended for verifier training)
3. DeepSeek API access

### Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the AQuA dataset is in the `AQuA-master/` directory

### Running the Complete Pipeline

Run the full DIVERSE pipeline with default settings:

```bash
python main.py
```

This will:
1. Test DeepSeek API connection
2. Load AQuA dataset
3. Generate verifier training data using diverse reasoning paths
4. Train the verifier model
5. Evaluate DIVERSE vs baseline CoT methods
6. Save results to the `results/` directory

## 🛠 Configuration

Key settings can be modified in `config.py`:

### Model Configuration
- `deepseek_api_key`: Your DeepSeek API key (default: "sk-225ac3f431704f6c8632adceb9fb47b1")
- `model_name`: DeepSeek model to use (default: "deepseek-chat")
- `temperature`: Sampling temperature (default: 0.5)

### DIVERSE Configuration
- `num_prompts`: Number of different prompts (M1 = 5)
- `num_samples_per_prompt`: Samples per prompt (M2 = 20)
- `use_step_aware`: Enable step-aware verification (default: True)

### Training Configuration
- `max_train_samples`: Training examples for verifier (default: 1000)
- `max_eval_samples`: Evaluation examples (default: 500)
- `batch_size`: Training batch size (default: 16)
- `num_epochs`: Training epochs (default: 5)

## 📋 Usage Options

### Command Line Arguments

```bash
# Skip verifier training (use existing model)
python main.py --skip-training

# Skip training data generation (use existing data)
python main.py --skip-data-generation

# Run evaluation only
python main.py --evaluation-only

# Run ablation study
python main.py --ablation

# Limit evaluation examples
python main.py --num-eval 100
```

### Example Workflows

**Quick Test (Small Scale):**
```bash
python main.py --num-eval 50
```

**Full Experiment:**
```bash
python main.py --ablation
```

**Resume from Existing Model:**
```bash
python main.py --skip-training --evaluation-only
```

## 🔬 Method Details

### 1. Diverse Prompts
The system uses 5 different prompt templates to generate varied reasoning approaches:
- Systematic step-by-step analysis
- Careful problem breakdown
- Mathematical word problem analysis
- Systematic solution approach
- Step-by-step problem tackling

### 2. Verifier Training
- Base model: DeBERTa-v3-large
- Training data: Generated reasoning paths with correctness labels
- Architecture: Dual classification heads for path-level and step-level verification
- Loss function: Combined path and step losses with weighting parameter α

### 3. Voting Mechanisms
- **Majority Voting**: Simple vote counting across reasoning paths
- **Verifier Voting**: Weighted voting using verifier confidence scores
- **Step-aware Voting**: Combined path and step-level verification scores

## 📊 Expected Results

Based on the original DIVERSE paper, you should expect:
- Baseline CoT accuracy: ~40-50% on AQuA
- DIVERSE improvement: +5-15% absolute accuracy improvement
- Step-aware verification providing additional gains over simple verifier voting

## 🗂 Output Files

### Results Directory (`results/`)
- `evaluation_results_YYYYMMDD_HHMMSS.json`: Main comparison results
- `ablation_study_YYYYMMDD_HHMMSS.json`: Ablation study results

### Model Directory (`models/`)
- `verifier_model/`: Trained verifier model files
- `base_model/`: Downloaded base DeBERTa model

### Outputs Directory (`outputs/`)
- `verifier_training_data.json`: Generated training data for verifier

## 🐛 Troubleshooting

### Common Issues

**API Connection Failed:**
- Check your DeepSeek API key in `config.py`
- Verify internet connection
- Ensure API key has sufficient credits

**CUDA Out of Memory:**
- Reduce `batch_size` in `config.py`
- Use CPU training by setting CUDA_VISIBLE_DEVICES=""
- Reduce `max_train_samples`

**Missing Dependencies:**
```bash
pip install torch transformers datasets accelerate
```

**Dataset Not Found:**
- Ensure `AQuA-master/` directory contains the dataset files
- Check that `train.json`, `dev.json`, and `test.json` exist

## 📈 Performance Optimization

### For Faster Training:
- Use GPU with sufficient VRAM (8GB+ recommended)
- Reduce `max_train_samples` for quicker experiments
- Use `--skip-data-generation` after first run

### For Better Accuracy:
- Increase `num_prompts` and `num_samples_per_prompt`
- Increase `max_train_samples` for verifier training
- Tune `step_aware_alpha` parameter

## 🔍 Understanding Results

The evaluation produces several metrics:

- **Accuracy**: Percentage of correct predictions
- **Improvement**: Absolute improvement over baseline
- **Relative Improvement**: Percentage improvement over baseline
- **Confidence**: Model confidence scores (DIVERSE only)

Example output:
```
Total examples: 500
DIVERSE accuracy: 0.642 (321/500)
Baseline accuracy: 0.584 (292/500)
Improvement: 0.058 (9.9%)
Evaluation time: 45.2 minutes
```

## 🤝 Contributing

To extend this project:
1. Add new prompt templates in `config.py`
2. Implement additional voting mechanisms in `diverse_cot.py`
3. Add new evaluation metrics in the comparison functions
4. Experiment with different verifier architectures in `verifier.py`

## 📚 References

- Original DIVERSE paper: "Diverse Verifier on Reasoning Step"
- AQuA dataset: "Program Induction by Rationale Generation"
- DeBERTa: "Decoding-enhanced BERT with Disentangled Attention"
- DeepSeek: Large language model API service

## ⚠ Important Notes

- API calls may incur costs - monitor your DeepSeek API usage
- Training the verifier requires significant computational resources
- Results may vary based on API response variations
- Save your trained models to avoid retraining

---

For questions or issues, please check the troubleshooting section or refer to the detailed comments in the source code. 