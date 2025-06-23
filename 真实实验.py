#!/usr/bin/env python3
"""
真实DIVERSE实验 - 使用真正的模型
包含真实的DeBERTa verifier训练和DeepSeek API调用
"""

import os
import json
import time
import random
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any

# 导入项目模块
from config import config
from data_loader import load_aqua_data, ReasoningPathGenerator
from llm_client import DeepSeekClient, test_deepseek_connection
from verifier import VerifierTrainer, download_and_prepare_verifier
from diverse_cot import DiverseCoT, BaselineCoT, compare_methods, save_results

def check_environment():
    """检查实验环境"""
    print("="*60)
    print("检查实验环境")
    print("="*60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")
    
    # 检查API连接
    print("\n测试DeepSeek API连接...")
    if test_deepseek_connection():
        print("✅ DeepSeek API连接成功")
        return True
    else:
        print("❌ DeepSeek API连接失败，请检查网络和API密钥")
        return False

def setup_experiment():
    """设置实验环境"""
    print("="*60)
    print("设置实验环境")
    print("="*60)
    
    # 设置随机种子
    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.experiment.seed)
    
    print(f"✅ 随机种子设置为: {config.experiment.seed}")
    
    # 创建目录
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    os.makedirs(config.experiment.model_save_dir, exist_ok=True)
    os.makedirs(config.experiment.results_dir, exist_ok=True)
    
    print(f"✅ 实验目录创建完成")

def prepare_data():
    """准备AQuA数据"""
    print("\n" + "="*60)
    print("准备AQuA数据集")
    print("="*60)
    
    # 加载数据
    data_loader = load_aqua_data("all")
    
    # 获取训练和评估子集
    train_subset = data_loader.get_train_subset(config.data.max_train_samples)
    eval_subset = data_loader.get_eval_subset("dev", config.data.max_eval_samples)
    
    print(f"✅ 训练样本: {len(train_subset)} 个")
    print(f"✅ 评估样本: {len(eval_subset)} 个")
    
    return data_loader, train_subset, eval_subset

def generate_verifier_training_data(train_subset, llm_client):
    """生成验证器训练数据 - 真实API调用"""
    print("\n" + "="*60)
    print("生成验证器训练数据")
    print("="*60)
    
    path_generator = ReasoningPathGenerator(llm_client)
    
    print(f"开始为 {len(train_subset)} 个样本生成推理路径...")
    print("这可能需要几分钟时间...")
    
    # 生成训练数据
    training_data = []
    
    for i, example in enumerate(train_subset):
        print(f"处理样本 {i+1}/{len(train_subset)}: {example.question[:50]}...")
        
        try:
            # 为每个样本生成多条推理路径
            paths = path_generator.generate_reasoning_paths(
                example, 
                num_paths=5  # 每个样本生成5条路径
            )
            
            # 创建路径级训练实例
            for path in paths:
                training_instance = {
                    'question': example.question,
                    'options': example.options,
                    'reasoning_path': path['response'],
                    'label': 1 if path['is_correct'] else 0,
                    'is_step_level': False
                }
                training_data.append(training_instance)
                
                # 创建步骤级训练实例
                for j, step in enumerate(path['steps']):
                    if len(step.strip()) > 10:  # 过滤短步骤
                        step_label = path_generator._get_step_label(step, path['is_correct'])
                        step_instance = {
                            'question': example.question,
                            'options': example.options,
                            'reasoning_step': step,
                            'step_label': step_label,
                            'is_step_level': True
                        }
                        training_data.append(step_instance)
            
            # 添加延迟避免API限流
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ 处理样本 {i+1} 时出错: {e}")
            continue
    
    print(f"✅ 成功生成 {len(training_data)} 个训练实例")
    
    # 保存训练数据
    training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
    with open(training_data_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练数据已保存到: {training_data_path}")
    return training_data

def train_verifier(training_data):
    """训练真实的DeBERTa验证器"""
    print("\n" + "="*60)
    print("训练DeBERTa验证器模型")
    print("="*60)
    
    try:
        # 尝试下载预训练模型
        print("检查DeBERTa-v3-large模型...")
        download_and_prepare_verifier()
        
        # 初始化trainer
        verifier_trainer = VerifierTrainer()
        print("✅ 验证器初始化成功")
        
        # 划分训练/验证集
        split_idx = int(0.8 * len(training_data))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"训练样本: {len(train_data)}")
        print(f"验证样本: {len(val_data)}")
        
        # 开始训练
        print("开始训练验证器...")
        print("模型配置:")
        print(f"  - 基础模型: {config.verifier.model_name}")
        print(f"  - 学习率: {config.verifier.learning_rate}")
        print(f"  - 批大小: {config.verifier.batch_size}")
        print(f"  - 训练轮数: {config.verifier.num_epochs}")
        print(f"  - 最大长度: {config.verifier.max_length}")
        
        training_history = verifier_trainer.train(train_data, val_data)
        
        print("✅ 验证器训练完成!")
        return verifier_trainer
        
    except Exception as e:
        print(f"❌ 验证器训练失败: {e}")
        print("这可能是由于:")
        print("1. GPU内存不足")
        print("2. 网络连接问题")
        print("3. 依赖包版本问题")
        return None

def run_diverse_evaluation(eval_subset, llm_client, verifier_trainer=None):
    """运行DIVERSE方法评估"""
    print("\n" + "="*60)
    print("运行DIVERSE方法评估")
    print("="*60)
    
    # 初始化方法
    diverse_cot = DiverseCoT(llm_client, verifier_trainer)
    baseline_cot = BaselineCoT(llm_client)
    
    print("配置:")
    print(f"  - 每题推理路径数: {config.diverse.total_samples}")
    print(f"  - 提示模板数: {config.diverse.num_prompts}")
    print(f"  - 每模板采样数: {config.diverse.num_samples_per_prompt}")
    print(f"  - 使用步骤感知: {config.diverse.use_step_aware}")
    
    # 运行对比实验
    print("\n开始方法对比...")
    start_time = time.time()
    
    try:
        comparison_results = compare_methods(eval_subset, diverse_cot, baseline_cot)
        
        elapsed_time = time.time() - start_time
        comparison_results['evaluation_time_seconds'] = elapsed_time
        comparison_results['evaluation_time_formatted'] = f"{elapsed_time/60:.2f} minutes"
        
        return comparison_results
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        return None

def run_ablation_study(eval_subset, llm_client, verifier_trainer):
    """运行消融实验"""
    print("\n" + "="*60)
    print("运行消融实验")
    print("="*60)
    
    if verifier_trainer is None:
        print("⚠️ 没有可用的验证器，跳过消融实验")
        return None
    
    diverse_cot = DiverseCoT(llm_client, verifier_trainer)
    
    # 测试不同的投票方法
    methods = ["majority", "verifier", "step_aware_verifier"]
    results = {}
    
    for method in methods:
        print(f"\n测试方法: {method}")
        
        try:
            method_results = []
            for i, example in enumerate(eval_subset):
                print(f"  处理样本 {i+1}/{len(eval_subset)}")
                result = diverse_cot.solve(example, method=method)
                method_results.append({
                    'predicted_answer': result.predicted_answer,
                    'confidence': result.confidence,
                    'correct': result.predicted_answer == example.correct
                })
            
            # 计算准确率
            correct = sum(1 for result in method_results if result['correct'])
            accuracy = correct / len(eval_subset)
            
            results[method] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(eval_subset),
                'avg_confidence': np.mean([r['confidence'] for r in method_results])
            }
            
            print(f"  {method} 准确率: {accuracy:.3f}")
            
        except Exception as e:
            print(f"❌ 测试 {method} 时出错: {e}")
    
    return results

def save_final_results(comparison_results, ablation_results=None):
    """保存最终结果"""
    print("\n" + "="*60)
    print("保存实验结果")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存主要结果
    if comparison_results:
        results_filename = f"diverse_experiment_results_{timestamp}.json"
        save_results(comparison_results, results_filename)
        print(f"✅ 主要结果已保存到: results/{results_filename}")
    
    # 保存消融结果
    if ablation_results:
        ablation_filename = f"ablation_study_{timestamp}.json"
        ablation_path = os.path.join(config.experiment.results_dir, ablation_filename)
        with open(ablation_path, 'w', encoding='utf-8') as f:
            json.dump(ablation_results, f, indent=2, ensure_ascii=False)
        print(f"✅ 消融结果已保存到: results/{ablation_filename}")

def print_final_summary(comparison_results, ablation_results=None):
    """打印最终总结"""
    print("\n" + "="*60)
    print("🎉 实验完成！最终结果总结")
    print("="*60)
    
    if comparison_results:
        print("\n📊 主要结果:")
        print(f"总样本数: {comparison_results['total_examples']}")
        print(f"DIVERSE方法准确率: {comparison_results['diverse_accuracy']:.3f} ({comparison_results['diverse_correct']}/{comparison_results['total_examples']})")
        print(f"基线方法准确率: {comparison_results['baseline_accuracy']:.3f} ({comparison_results['baseline_correct']}/{comparison_results['total_examples']})")
        print(f"准确率提升: {comparison_results['improvement']:.3f}")
        print(f"相对提升: {comparison_results['relative_improvement_percent']:.1f}%")
        print(f"评估时间: {comparison_results['evaluation_time_formatted']}")
        
        if comparison_results['improvement'] > 0:
            print("✅ DIVERSE方法表现更好!")
        else:
            print("⚠️ 需要进一步优化")
    
    if ablation_results:
        print("\n🔬 消融实验结果:")
        for method, result in ablation_results.items():
            print(f"{method:20s}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")

def main():
    """主实验流程"""
    print("="*60)
    print("🚀 DIVERSE Chain-of-Thought 真实实验系统")
    print("="*60)
    
    # 1. 环境检查
    if not check_environment():
        print("❌ 环境检查失败，实验终止")
        return
    
    # 2. 设置实验
    setup_experiment()
    
    # 3. 准备数据
    data_loader, train_subset, eval_subset = prepare_data()
    
    # 4. 初始化LLM客户端
    llm_client = DeepSeekClient()
    
    # 5. 生成验证器训练数据
    print("\n是否生成新的训练数据？这将调用DeepSeek API")
    training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
    
    if os.path.exists(training_data_path):
        print(f"发现已存在的训练数据: {training_data_path}")
        print("加载已有数据...")
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        print(f"✅ 加载了 {len(training_data)} 个训练实例")
    else:
        training_data = generate_verifier_training_data(train_subset, llm_client)
    
    # 6. 训练验证器
    verifier_trainer = train_verifier(training_data)
    
    # 7. 运行评估
    comparison_results = run_diverse_evaluation(eval_subset, llm_client, verifier_trainer)
    
    # 8. 运行消融实验
    ablation_results = run_ablation_study(eval_subset, llm_client, verifier_trainer)
    
    # 9. 保存结果
    save_final_results(comparison_results, ablation_results)
    
    # 10. 打印总结
    print_final_summary(comparison_results, ablation_results)
    
    print("\n🎉 真实实验完成!")
    print(f"📁 结果文件位于: {config.experiment.results_dir}")

if __name__ == "__main__":
    main() 