#!/usr/bin/env python3
"""
Verifier诊断分析
分析verifier的评分行为，找出问题所在
"""

import os
import json
import numpy as np
from datetime import datetime

from config import config
from test_improved_verifier import ImprovedVerifierPredictor

def analyze_verifier_scores():
    """分析verifier的评分分布"""
    print("🔍 分析Verifier评分行为")
    print("="*60)
    
    # 加载verifier
    model_path = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
    
    if not os.path.exists(model_path):
        print(f"❌ Verifier模型不存在: {model_path}")
        return
    
    try:
        verifier = ImprovedVerifierPredictor(model_path)
    except Exception as e:
        print(f"❌ Verifier加载失败: {e}")
        return
    
    # 创建测试案例：明显的好/差推理
    test_cases = [
        # 明显正确的推理
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning': '设x + 5 = 12，两边同时减去5，得到x = 12 - 5 = 7。答案是A。',
            'expected_quality': 'high',
            'correct_answer': 'A'
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning': '总价格是20元，数量是4本，每本价格 = 20 ÷ 4 = 5元。答案是B。',
            'expected_quality': 'high',
            'correct_answer': 'B'
        },
        {
            'question': '100除以4等于多少？',
            'options': ['A)20', 'B)25', 'C)30', 'D)15', 'E)40'],
            'reasoning': '100 ÷ 4 = 25。答案是B。',
            'expected_quality': 'high',
            'correct_answer': 'B'
        },
        
        # 明显错误的推理
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning': '我觉得答案是C，因为6看起来不错。',
            'expected_quality': 'low',
            'correct_answer': 'A'
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning': '随便猜一个，选择A)4元。',
            'expected_quality': 'low',
            'correct_answer': 'B'
        },
        {
            'question': '100除以4等于多少？',
            'options': ['A)20', 'B)25', 'C)30', 'D)15', 'E)40'],
            'reasoning': '100 + 4 = 104，但这不对。我选择D)15。',
            'expected_quality': 'low',
            'correct_answer': 'B'
        },
        
        # 中等质量推理
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning': '这道题需要计算，x应该是7，选择A。',
            'expected_quality': 'medium',
            'correct_answer': 'A'
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning': '20除以4等于5，答案是B。',
            'expected_quality': 'medium',
            'correct_answer': 'B'
        }
    ]
    
    print(f"📋 测试 {len(test_cases)} 个案例...")
    
    # 收集评分
    high_scores = []
    medium_scores = []
    low_scores = []
    
    detailed_results = []
    
    for i, case in enumerate(test_cases):
        score = verifier.predict(case['question'], case['options'], case['reasoning'])
        
        if case['expected_quality'] == 'high':
            high_scores.append(score)
        elif case['expected_quality'] == 'medium':
            medium_scores.append(score)
        else:
            low_scores.append(score)
        
        # 判断预测是否与答案一致
        predicted_answer = case['reasoning'].split('答案是')[-1].split('。')[0].strip() if '答案是' in case['reasoning'] else 'Unknown'
        is_answer_correct = case['correct_answer'] in predicted_answer
        
        detailed_results.append({
            'case_id': i+1,
            'expected_quality': case['expected_quality'],
            'verifier_score': score,
            'reasoning': case['reasoning'][:50] + '...',
            'is_answer_correct': is_answer_correct
        })
        
        print(f"   案例 {i+1}: 期望质量={case['expected_quality']}, Verifier评分={score:.3f}, 推理={case['reasoning'][:40]}...")
    
    # 统计分析
    print(f"\n📊 评分统计分析:")
    print(f"   高质量推理 (期望>0.7): 平均={np.mean(high_scores):.3f}, 范围=[{np.min(high_scores):.3f}, {np.max(high_scores):.3f}]")
    print(f"   中等质量推理 (期望0.4-0.7): 平均={np.mean(medium_scores):.3f}, 范围=[{np.min(medium_scores):.3f}, {np.max(medium_scores):.3f}]")
    print(f"   低质量推理 (期望<0.4): 平均={np.mean(low_scores):.3f}, 范围=[{np.min(low_scores):.3f}, {np.max(low_scores):.3f}]")
    
    # 问题诊断
    print(f"\n🔍 问题诊断:")
    
    # 1. 评分区分度
    high_avg = np.mean(high_scores)
    low_avg = np.mean(low_scores)
    discrimination = high_avg - low_avg
    
    print(f"   评分区分度: {discrimination:.3f} (高质量 - 低质量)")
    if discrimination < 0.1:
        print(f"   ❌ 区分度过低！Verifier无法有效区分推理质量")
    elif discrimination < 0.2:
        print(f"   ⚠️  区分度较低，需要改进")
    else:
        print(f"   ✅ 区分度良好")
    
    # 2. 评分范围
    all_scores = high_scores + medium_scores + low_scores
    score_range = np.max(all_scores) - np.min(all_scores)
    print(f"   评分范围: {score_range:.3f}")
    if score_range < 0.1:
        print(f"   ❌ 评分范围过窄！模型输出过于集中")
    elif score_range < 0.3:
        print(f"   ⚠️  评分范围较窄")
    else:
        print(f"   ✅ 评分范围正常")
    
    # 3. 评分偏向
    overall_avg = np.mean(all_scores)
    print(f"   整体平均评分: {overall_avg:.3f}")
    if overall_avg > 0.7:
        print(f"   ⚠️  评分偏高，可能过于乐观")
    elif overall_avg < 0.3:
        print(f"   ⚠️  评分偏低，可能过于保守")
    else:
        print(f"   ✅ 评分分布合理")
    
    # 4. 推荐改进措施
    print(f"\n💡 改进建议:")
    
    if discrimination < 0.1:
        print(f"   🔧 提高区分度:")
        print(f"      - 增加对比学习训练")
        print(f"      - 改进训练数据标注质量")
        print(f"      - 调整损失函数，增大正负样本差异")
    
    if score_range < 0.2:
        print(f"   🔧 扩大评分范围:")
        print(f"      - 检查模型输出层设计")
        print(f"      - 调整训练超参数")
        print(f"      - 增加数据多样性")
    
    if overall_avg > 0.6 or overall_avg < 0.4:
        print(f"   🔧 调整评分分布:")
        print(f"      - 平衡训练数据中正负样本比例")
        print(f"      - 调整分类阈值")
    
    # 保存诊断结果
    diagnosis_results = {
        'timestamp': datetime.now().isoformat(),
        'test_cases_count': len(test_cases),
        'scores_by_quality': {
            'high': high_scores,
            'medium': medium_scores,
            'low': low_scores
        },
        'statistics': {
            'discrimination': discrimination,
            'score_range': score_range,
            'overall_average': overall_avg
        },
        'detailed_results': detailed_results
    }
    
    output_file = os.path.join(config.experiment.output_dir, f"verifier_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diagnosis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 诊断结果已保存: {output_file}")
    
    return diagnosis_results

def main():
    """主函数"""
    print("="*60)
    print("🩺 Verifier诊断分析")
    print("="*60)
    
    analyze_verifier_scores()

if __name__ == "__main__":
    main() 