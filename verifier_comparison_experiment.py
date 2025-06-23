#!/usr/bin/env python3
"""
Verifier vs 非Verifier准确率对比实验
比较使用验证器和不使用验证器的效果差异
"""

import os
import json
import random
import torch
from collections import Counter
from datetime import datetime

from config import config
from test_improved_verifier import ImprovedVerifierPredictor

def generate_reasoning_paths_for_problem(problem, num_paths=5):
    """为数学问题生成多条推理路径"""
    question = problem['question']
    correct_answer = problem['correct']
    options = problem['options']
    
    # 所有选项
    all_options = [opt.split(')')[0] for opt in options]  # 提取A, B, C, D, E
    wrong_options = [opt for opt in all_options if opt != correct_answer]
    
    paths = []
    
    # 1. 生成正确推理路径（30%概率）
    if random.random() < 0.3:
        reasoning = f"根据题目条件，我需要仔细分析和计算。经过逐步推理，答案是{correct_answer}。"
        paths.append({
            'reasoning': reasoning,
            'predicted_answer': correct_answer,
            'is_correct': True,
            'quality_score': 0.9
        })
    
    # 2. 生成中等质量推理（部分正确，20%概率正确答案）
    for _ in range(2):
        if random.random() < 0.2:
            reasoning = f"分析题目，进行计算，我认为答案是{correct_answer}。"
            predicted = correct_answer
            is_correct = True
            quality = 0.7
        else:
            predicted = random.choice(wrong_options)
            reasoning = f"根据我的理解，经过计算，答案应该是{predicted}。"
            is_correct = False
            quality = 0.3
        
        paths.append({
            'reasoning': reasoning,
            'predicted_answer': predicted,
            'is_correct': is_correct,
            'quality_score': quality
        })
    
    # 3. 生成低质量推理（10%概率正确答案）
    for _ in range(2):
        if random.random() < 0.1:
            reasoning = f"我觉得这道题答案是{correct_answer}。"
            predicted = correct_answer
            is_correct = True
            quality = 0.2
        else:
            predicted = random.choice(wrong_options)
            reasoning = f"看起来答案是{predicted}。"
            is_correct = False
            quality = 0.1
        
        paths.append({
            'reasoning': reasoning,
            'predicted_answer': predicted,
            'is_correct': is_correct,
            'quality_score': quality
        })
    
    return paths

def majority_voting(reasoning_paths):
    """多数投票法（不使用verifier）"""
    predictions = [path['predicted_answer'] for path in reasoning_paths]
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common

def verifier_weighted_voting(reasoning_paths, verifier, question, options):
    """Verifier加权投票法"""
    answer_scores = {}
    
    for path in reasoning_paths:
        # 获取verifier评分
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
        # 如果没有预测，回退到多数投票
        return majority_voting(reasoning_paths)

def quality_aware_voting(reasoning_paths):
    """理想情况：基于真实质量的投票（作为上限参考）"""
    answer_scores = {}
    
    for path in reasoning_paths:
        answer = path['predicted_answer']
        quality = path['quality_score']
        
        if answer not in answer_scores:
            answer_scores[answer] = 0
        answer_scores[answer] += quality
    
    # 选择得分最高的答案
    if answer_scores:
        best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
        return best_answer
    else:
        return majority_voting(reasoning_paths)

def create_test_problems():
    """创建测试问题集"""
    problems = [
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'correct': 'A'
        },
        {
            'question': '一个班级有24个学生，其中25%是男生，有多少个女生？',
            'options': ['A)6', 'B)18', 'C)12', 'D)8', 'E)16'],
            'correct': 'B'
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'correct': 'B'
        },
        {
            'question': '一个长方形的长是8米，宽是5米，面积是多少平方米？',
            'options': ['A)13', 'B)26', 'C)40', 'D)35', 'E)30'],
            'correct': 'C'
        },
        {
            'question': '从1到10的整数中，有多少个偶数？',
            'options': ['A)4', 'B)5', 'C)6', 'D)3', 'E)7'],
            'correct': 'B'
        },
        {
            'question': '如果一个数的3倍是15，这个数是多少？',
            'options': ['A)3', 'B)4', 'C)5', 'D)6', 'E)12'],
            'correct': 'C'
        },
        {
            'question': '一个圆的半径是3厘米，直径是多少厘米？',
            'options': ['A)9', 'B)6', 'C)12', 'D)3', 'E)15'],
            'correct': 'B'
        },
        {
            'question': '100除以4等于多少？',
            'options': ['A)20', 'B)25', 'C)30', 'D)15', 'E)40'],
            'correct': 'B'
        },
        {
            'question': '一个三角形的三条边长分别是3、4、5，这是什么三角形？',
            'options': ['A)等边三角形', 'B)等腰三角形', 'C)直角三角形', 'D)钝角三角形', 'E)不是三角形'],
            'correct': 'C'
        },
        {
            'question': '2的3次方等于多少？',
            'options': ['A)6', 'B)8', 'C)9', 'D)4', 'E)12'],
            'correct': 'B'
        }
    ]
    return problems

def run_comparison_experiment(problems, verifier):
    """运行对比实验"""
    print(f"\n🧪 开始准确率对比实验")
    print(f"   测试问题数: {len(problems)}")
    print("="*80)
    
    results = {
        'majority_vote': {'correct': 0, 'total': 0},
        'verifier_vote': {'correct': 0, 'total': 0},
        'quality_aware': {'correct': 0, 'total': 0},
        'details': []
    }
    
    for i, problem in enumerate(problems):
        print(f"\n📋 问题 {i+1}: {problem['question']}")
        print(f"   正确答案: {problem['correct']}")
        
        question = problem['question']
        correct_answer = problem['correct']
        options = problem['options']
        
        # 生成推理路径
        reasoning_paths = generate_reasoning_paths_for_problem(problem)
        
        print(f"   生成了 {len(reasoning_paths)} 条推理路径")
        
        # 1. 多数投票法（无verifier）
        majority_pred = majority_voting(reasoning_paths)
        majority_correct = (majority_pred == correct_answer)
        results['majority_vote']['correct'] += majority_correct
        results['majority_vote']['total'] += 1
        
        # 2. Verifier加权投票
        verifier_pred = verifier_weighted_voting(reasoning_paths, verifier, question, options)
        verifier_correct = (verifier_pred == correct_answer)
        results['verifier_vote']['correct'] += verifier_correct
        results['verifier_vote']['total'] += 1
        
        # 3. 理想情况（质量感知）
        quality_pred = quality_aware_voting(reasoning_paths)
        quality_correct = (quality_pred == correct_answer)
        results['quality_aware']['correct'] += quality_correct
        results['quality_aware']['total'] += 1
        
        # 显示结果
        status_majority = "✅" if majority_correct else "❌"
        status_verifier = "✅" if verifier_correct else "❌"
        status_quality = "✅" if quality_correct else "❌"
        
        print(f"   {status_majority} 多数投票:    {majority_pred}")
        print(f"   {status_verifier} Verifier投票: {verifier_pred}")
        print(f"   {status_quality} 理想投票:    {quality_pred}")
        
        # 记录详细结果
        detail = {
            'question': question,
            'correct_answer': correct_answer,
            'majority_vote': {'prediction': majority_pred, 'correct': majority_correct},
            'verifier_vote': {'prediction': verifier_pred, 'correct': verifier_correct},
            'quality_aware': {'prediction': quality_pred, 'correct': quality_correct},
            'reasoning_paths_count': len(reasoning_paths)
        }
        results['details'].append(detail)
    
    return results

def analyze_results(results):
    """分析实验结果"""
    print(f"\n📊 实验结果分析")
    print("="*80)
    
    # 计算准确率
    majority_acc = results['majority_vote']['correct'] / results['majority_vote']['total'] * 100
    verifier_acc = results['verifier_vote']['correct'] / results['verifier_vote']['total'] * 100
    quality_acc = results['quality_aware']['correct'] / results['quality_aware']['total'] * 100
    
    improvement = verifier_acc - majority_acc
    
    print(f"🎯 基线方法 (多数投票):     {majority_acc:.2f}% ({results['majority_vote']['correct']}/{results['majority_vote']['total']})")
    print(f"🤖 Verifier加权投票:       {verifier_acc:.2f}% ({results['verifier_vote']['correct']}/{results['verifier_vote']['total']}) [{improvement:+.2f}%]")
    print(f"⭐ 理想情况 (质量感知):     {quality_acc:.2f}% ({results['quality_aware']['correct']}/{results['quality_aware']['total']})")
    
    # 效果分析
    if improvement > 0:
        print(f"\n✅ Verifier方法显著优于基线！提升了 {improvement:.2f} 个百分点")
    elif improvement == 0:
        print(f"\n➡️ Verifier方法与基线持平")
    else:
        print(f"\n❌ Verifier方法表现不如基线，下降了 {abs(improvement):.2f} 个百分点")
    
    # 潜力分析
    potential_improvement = quality_acc - majority_acc
    verifier_efficiency = (improvement / potential_improvement * 100) if potential_improvement > 0 else 0
    
    print(f"\n💡 潜力分析:")
    print(f"   理论最大提升: {potential_improvement:.2f}%")
    print(f"   Verifier实际提升: {improvement:.2f}%")
    print(f"   效果发挥程度: {verifier_efficiency:.1f}%")
    
    return {
        'majority_accuracy': majority_acc,
        'verifier_accuracy': verifier_acc,
        'improvement': improvement,
        'quality_accuracy': quality_acc,
        'verifier_efficiency': verifier_efficiency
    }

def main():
    """主函数"""
    print("="*80)
    print("🔬 Verifier vs 非Verifier准确率对比实验")
    print("="*80)
    
    # 1. 加载改进版verifier
    model_path = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
    
    if not os.path.exists(model_path):
        print(f"❌ Verifier模型不存在: {model_path}")
        return
    
    try:
        print(f"🔄 加载改进版Verifier...")
        verifier = ImprovedVerifierPredictor(model_path)
        
    except Exception as e:
        print(f"❌ Verifier加载失败: {e}")
        return
    
    # 2. 创建测试问题
    problems = create_test_problems()
    print(f"✅ 创建了 {len(problems)} 个测试问题")
    
    # 3. 运行对比实验
    results = run_comparison_experiment(problems, verifier)
    
    # 4. 分析结果
    analysis = analyze_results(results)
    
    # 5. 保存结果
    experiment_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'verifier_vs_baseline_comparison',
        'test_problems_count': len(problems),
        'results': results,
        'analysis': analysis
    }
    
    output_file = os.path.join(config.experiment.output_dir, f"verifier_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 保存时去掉详细推理路径以减小文件大小
    save_results = experiment_results.copy()
    save_results['results'] = {
        'majority_vote': results['majority_vote'],
        'verifier_vote': results['verifier_vote'],
        'quality_aware': results['quality_aware'],
        'sample_details': results['details'][:5]  # 只保存前5个样本的详细信息
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 实验结果已保存: {output_file}")
    
    # 6. 总结
    print(f"\n🎉 实验总结:")
    if analysis['improvement'] > 0:
        print(f"   ✅ 使用Verifier比不使用Verifier准确率提升了 {analysis['improvement']:.2f}%")
        print(f"   📈 从 {analysis['majority_accuracy']:.2f}% 提升到 {analysis['verifier_accuracy']:.2f}%")
    else:
        print(f"   ❌ 使用Verifier的效果不如预期，需要进一步优化")
    
    print(f"   🎯 Verifier效果发挥了理论潜力的 {analysis['verifier_efficiency']:.1f}%")

if __name__ == "__main__":
    main() 