#!/usr/bin/env python3
"""
改进版Verifier效果测试
对比改进前后的verifier性能
"""

import os
import json
import random
import numpy as np
from datetime import datetime

from config import config
from data_loader import AQuADataLoader
from improved_verifier import ImprovedVerifierTrainer
from verifier import VerifierModel

def load_test_data(num_samples=50):
    """加载AQuA测试数据"""
    print(f"📚 加载AQuA测试数据 (样本数: {num_samples})")
    
    data_loader = AQuADataLoader()
    aqua_data = data_loader.load_data()
    
    if aqua_data is None or len(aqua_data) == 0:
        print("❌ AQuA数据加载失败，使用模拟数据")
        # 创建一些模拟测试数据
        test_samples = []
        for i in range(num_samples):
            test_samples.append({
                'question': f'模拟数学题 {i+1}: 计算 2+3 等于多少？',
                'options': ['A)4', 'B)5', 'C)6', 'D)7', 'E)8'],
                'correct': 'B'
            })
    else:
        # 随机选择测试样本
        test_samples = random.sample(aqua_data, min(num_samples, len(aqua_data)))
    
    print(f"✅ 成功加载 {len(test_samples)} 个测试样本")
    return test_samples

def generate_realistic_reasoning_paths(problem, num_paths=5):
    """为每个问题生成真实的推理路径（模拟不同质量）"""
    
    question = problem['question']
    correct_answer = problem['correct']
    options = problem['options']
    
    paths = []
    
    # 1. 高质量推理路径（30%概率正确）
    if random.random() < 0.3:
        reasoning = f"根据题目分析，我需要仔细计算各个选项。通过逐步分析，答案是 {correct_answer}。"
        paths.append({
            'reasoning': reasoning,
            'predicted_answer': correct_answer,
            'is_correct': True,
            'quality': 'high'
        })
    
    # 2. 中等质量推理路径（20%概率正确）
    for _ in range(2):
        if random.random() < 0.2:
            reasoning = f"分析题目条件，进行计算，我认为答案是 {correct_answer}。"
            predicted = correct_answer
            is_correct = True
        else:
            wrong_answers = [opt for opt in options if opt != correct_answer]
            predicted = random.choice(wrong_answers)
            reasoning = f"根据我的理解，经过计算，答案应该是 {predicted}。"
            is_correct = False
        
        paths.append({
            'reasoning': reasoning,
            'predicted_answer': predicted,
            'is_correct': is_correct,
            'quality': 'medium'
        })
    
    # 3. 低质量推理路径（10%概率正确）
    for _ in range(2):
        if random.random() < 0.1:
            reasoning = f"我觉得这道题应该选择 {correct_answer}。"
            predicted = correct_answer
            is_correct = True
        else:
            wrong_answers = [opt for opt in options if opt != correct_answer]
            predicted = random.choice(wrong_answers)
            reasoning = f"看起来答案是 {predicted}。"
            is_correct = False
        
        paths.append({
            'reasoning': reasoning,
            'predicted_answer': predicted,
            'is_correct': is_correct,
            'quality': 'low'
        })
    
    return paths

def majority_voting(reasoning_paths):
    """多数投票法"""
    from collections import Counter
    
    predictions = [path['predicted_answer'] for path in reasoning_paths]
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    
    return most_common

def verifier_weighted_voting(reasoning_paths, verifier, question, options):
    """Verifier加权投票"""
    answer_scores = {}
    
    for path in reasoning_paths:
        # 获取verifier分数
        score = verifier.predict(question, options, path['reasoning'])
        answer = path['predicted_answer']
        
        if answer not in answer_scores:
            answer_scores[answer] = 0
        answer_scores[answer] += score
    
    # 选择得分最高的答案
    if answer_scores:
        best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
        return best_answer
    else:
        # 如果没有预测，使用多数投票
        return majority_voting(reasoning_paths)

def perfect_voting(reasoning_paths):
    """理想情况：知道真实质量的投票"""
    correct_paths = [path for path in reasoning_paths if path['is_correct']]
    
    if correct_paths:
        # 如果有正确路径，从中随机选择一个
        selected = random.choice(correct_paths)
        return selected['predicted_answer']
    else:
        # 如果没有正确路径，使用多数投票
        return majority_voting(reasoning_paths)

def run_comparative_experiment(test_data, old_verifier=None, new_verifier=None):
    """运行对比实验"""
    print(f"\n🧪 开始对比实验")
    print(f"   测试样本数: {len(test_data)}")
    
    results = {
        'majority_vote': {'correct': 0, 'total': 0},
        'old_verifier': {'correct': 0, 'total': 0} if old_verifier else None,
        'new_verifier': {'correct': 0, 'total': 0} if new_verifier else None,
        'perfect': {'correct': 0, 'total': 0},
        'details': []
    }
    
    for i, problem in enumerate(test_data):
        print(f"\r   处理进度: {i+1}/{len(test_data)}", end='', flush=True)
        
        question = problem['question']
        correct_answer = problem['correct']
        options = problem['options']
        
        # 生成推理路径
        reasoning_paths = generate_realistic_reasoning_paths(problem)
        
        # 1. 多数投票
        majority_pred = majority_voting(reasoning_paths)
        majority_correct = (majority_pred == correct_answer)
        results['majority_vote']['correct'] += majority_correct
        results['majority_vote']['total'] += 1
        
        # 2. 旧verifier (如果有)
        old_verifier_pred = None
        old_verifier_correct = False
        if old_verifier:
            old_verifier_pred = verifier_weighted_voting(reasoning_paths, old_verifier, question, options)
            old_verifier_correct = (old_verifier_pred == correct_answer)
            results['old_verifier']['correct'] += old_verifier_correct
            results['old_verifier']['total'] += 1
        
        # 3. 新verifier (如果有)
        new_verifier_pred = None
        new_verifier_correct = False
        if new_verifier:
            new_verifier_pred = verifier_weighted_voting(reasoning_paths, new_verifier, question, options)
            new_verifier_correct = (new_verifier_pred == correct_answer)
            results['new_verifier']['correct'] += new_verifier_correct
            results['new_verifier']['total'] += 1
        
        # 4. 理想情况
        perfect_pred = perfect_voting(reasoning_paths)
        perfect_correct = (perfect_pred == correct_answer)
        results['perfect']['correct'] += perfect_correct
        results['perfect']['total'] += 1
        
        # 记录详细结果
        detail = {
            'question': question[:100] + "..." if len(question) > 100 else question,
            'correct_answer': correct_answer,
            'majority_vote': {'prediction': majority_pred, 'correct': majority_correct},
            'old_verifier': {'prediction': old_verifier_pred, 'correct': old_verifier_correct} if old_verifier else None,
            'new_verifier': {'prediction': new_verifier_pred, 'correct': new_verifier_correct} if new_verifier else None,
            'perfect': {'prediction': perfect_pred, 'correct': perfect_correct},
            'reasoning_quality': {
                'total_paths': len(reasoning_paths),
                'correct_paths': len([p for p in reasoning_paths if p['is_correct']])
            }
        }
        results['details'].append(detail)
    
    print(f"\n✅ 实验完成!")
    return results

def analyze_results(results):
    """分析实验结果"""
    print(f"\n📊 实验结果分析")
    print("="*70)
    
    # 计算准确率
    majority_acc = results['majority_vote']['correct'] / results['majority_vote']['total'] * 100
    perfect_acc = results['perfect']['correct'] / results['perfect']['total'] * 100
    
    print(f"🎯 基线方法 (多数投票):     {majority_acc:.2f}% ({results['majority_vote']['correct']}/{results['majority_vote']['total']})")
    
    if results['old_verifier']:
        old_acc = results['old_verifier']['correct'] / results['old_verifier']['total'] * 100
        improvement_old = old_acc - majority_acc
        print(f"🤖 旧Verifier (BERT):       {old_acc:.2f}% ({results['old_verifier']['correct']}/{results['old_verifier']['total']}) " +
              f"[{improvement_old:+.2f}%]")
    
    if results['new_verifier']:
        new_acc = results['new_verifier']['correct'] / results['new_verifier']['total'] * 100
        improvement_new = new_acc - majority_acc
        print(f"🚀 新Verifier (RoBERTa):    {new_acc:.2f}% ({results['new_verifier']['correct']}/{results['new_verifier']['total']}) " +
              f"[{improvement_new:+.2f}%]")
        
        if results['old_verifier']:
            relative_improvement = new_acc - (results['old_verifier']['correct'] / results['old_verifier']['total'] * 100)
            print(f"   相对于旧verifier提升:    {relative_improvement:+.2f}%")
    
    print(f"⭐ 理想情况 (质量感知):     {perfect_acc:.2f}% ({results['perfect']['correct']}/{results['perfect']['total']})")
    
    # 效果分析
    if results['new_verifier']:
        potential = perfect_acc
        actual = results['new_verifier']['correct'] / results['new_verifier']['total'] * 100
        efficiency = (actual / potential * 100) if potential > 0 else 0
        print(f"\n💡 Verifier效果分析:")
        print(f"   理论潜力: {potential:.2f}%")
        print(f"   实际效果: {actual:.2f}%")
        print(f"   效果发挥: {efficiency:.1f}%")

def main():
    """主函数"""
    print("="*70)
    print("🔬 改进版Verifier对比实验")
    print("="*70)
    
    # 1. 加载测试数据
    test_data = load_test_data(num_samples=40)
    
    # 2. 加载模型
    print(f"\n🔄 加载Verifier模型...")
    
    # 尝试加载旧verifier
    old_verifier = None
    old_model_path = os.path.join(config.experiment.model_save_dir, "verifier_model")
    if os.path.exists(old_model_path):
        try:
            from verifier import VerifierModel
            old_verifier = VerifierModel()
            old_verifier.load_model(old_model_path)
            print(f"✅ 旧Verifier加载成功 (BERT)")
        except Exception as e:
            print(f"⚠️  旧Verifier加载失败: {e}")
            old_verifier = None
    
    # 尝试加载新verifier
    new_verifier = None
    new_model_path = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
    if os.path.exists(new_model_path):
        try:
            new_verifier = ImprovedVerifierTrainer(model_name="roberta-base")
            new_verifier.load_model(new_model_path)
            print(f"✅ 新Verifier加载成功 (RoBERTa)")
        except Exception as e:
            print(f"⚠️  新Verifier加载失败: {e}")
            new_verifier = None
    
    if not old_verifier and not new_verifier:
        print(f"❌ 没有找到可用的Verifier模型")
        return
    
    # 3. 运行对比实验
    results = run_comparative_experiment(test_data, old_verifier, new_verifier)
    
    # 4. 分析结果
    analyze_results(results)
    
    # 5. 保存结果
    output_file = os.path.join(config.experiment.output_dir, f"verifier_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 保存时去掉不能序列化的详细信息
    save_results = {
        'experiment_info': {
            'test_samples': len(test_data),
            'timestamp': datetime.now().isoformat(),
            'models_tested': {
                'old_verifier': old_verifier is not None,
                'new_verifier': new_verifier is not None
            }
        },
        'accuracy_results': {
            'majority_vote': results['majority_vote']['correct'] / results['majority_vote']['total'],
            'old_verifier': results['old_verifier']['correct'] / results['old_verifier']['total'] if results['old_verifier'] else None,
            'new_verifier': results['new_verifier']['correct'] / results['new_verifier']['total'] if results['new_verifier'] else None,
            'perfect': results['perfect']['correct'] / results['perfect']['total']
        },
        'sample_details': results['details'][:10]  # 只保存前10个样本的详细信息
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 