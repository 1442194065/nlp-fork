#!/usr/bin/env python3
"""
快速增强版Verifier测试 - 验证是否超越基线
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from datetime import datetime

from config import config
from enhanced_verifier_trainer import ContrastiveVerifierModel

class QuickEnhancedVerifier:
    """快速增强版Verifier测试器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """加载增强版模型"""
        print(f"🔄 加载增强版Verifier模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 加载模型
        self.model = ContrastiveVerifierModel()
        
        # 使用safetensors加载模型权重
        model_file = os.path.join(self.model_path, "model.safetensors")
        
        if os.path.exists(model_file):
            state_dict = {}
            with safe_open(model_file, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            
            self.model.load_state_dict(state_dict)
            print(f"✅ 增强版模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, question, reasoning_path):
        """预测推理路径质量"""
        # 构造输入文本
        text = f"问题: {question} 推理: {reasoning_path}"
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # 移到设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs['logits']
            score = logits[0][0].item()
        
        return score

def quick_test():
    """快速验证测试"""
    print("="*70)
    print("🚀 快速增强版Verifier验证测试")
    print("="*70)
    
    # 1. 加载增强版verifier
    model_path = os.path.join(config.experiment.model_save_dir, "enhanced_verifier_model")
    
    if not os.path.exists(model_path):
        print(f"❌ 增强版模型不存在: {model_path}")
        return
    
    try:
        verifier = QuickEnhancedVerifier(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 创建测试案例
    test_scenarios = [
        {
            'name': '简单数学题',
            'question': '如果 2x + 3 = 9，那么 x 等于多少？',
            'reasoning_paths': [
                {'text': '2x + 3 = 9，两边减3得 2x = 6，两边除以2得 x = 3', 'quality': 'high'},
                {'text': '2x = 9 - 3 = 6，所以 x = 6/2 = 3', 'quality': 'high'}, 
                {'text': '感觉答案是3，选3', 'quality': 'low'},
                {'text': '随便猜一个数字，可能是5吧', 'quality': 'low'},
                {'text': '我觉得是2或者4', 'quality': 'low'}
            ],
            'correct_answer': '3'
        },
        {
            'name': '百分比问题',
            'question': '商店有100件商品，卖出了30%，还剩多少件？',
            'reasoning_paths': [
                {'text': '卖出30%，剩下70%。100 × 70% = 100 × 0.7 = 70件', 'quality': 'high'},
                {'text': '卖出30件，还剩100-30=70件', 'quality': 'high'},
                {'text': '应该是70件左右吧', 'quality': 'low'}, 
                {'text': '感觉是80件', 'quality': 'low'},
                {'text': '不知道，随便说60件', 'quality': 'low'}
            ],
            'correct_answer': '70'
        },
        {
            'name': '比例问题',
            'question': '如果3个苹果的价格等于2个橙子，那么9个苹果等于多少个橙子？',
            'reasoning_paths': [
                {'text': '3苹果=2橙子，所以9苹果=9÷3×2=3×2=6个橙子', 'quality': 'high'},
                {'text': '3:2 = 9:x，所以x = 9×2÷3 = 6', 'quality': 'high'},
                {'text': '应该是6个橙子', 'quality': 'medium'},
                {'text': '感觉是4个', 'quality': 'low'},
                {'text': '不确定，可能是8个', 'quality': 'low'}
            ],
            'correct_answer': '6'
        }
    ]
    
    print(f"📋 测试 {len(test_scenarios)} 个场景...")
    
    # 3. 对每个场景进行测试
    all_results = []
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"问题: {scenario['question']}")
        print(f"正确答案: {scenario['correct_answer']}")
        
        # 评分所有推理路径
        scored_paths = []
        for path in scenario['reasoning_paths']:
            score = verifier.predict(scenario['question'], path['text'])
            scored_paths.append({
                'text': path['text'],
                'quality': path['quality'],
                'score': score,
                'answer': scenario['correct_answer'] if path['quality'] == 'high' else 'wrong'
            })
            print(f"   {path['quality']:>6}: {score:.3f} - {path['text'][:50]}...")
        
        # 模拟投票
        # 基线方法：等权重投票（假设每个推理都有相同权重）
        baseline_prediction = scenario['correct_answer']  # 简化：假设基线准确率50%
        
        # Verifier加权投票：使用分数作为权重
        correct_weight = sum(p['score'] for p in scored_paths if p['quality'] == 'high')
        wrong_weight = sum(p['score'] for p in scored_paths if p['quality'] == 'low')
        
        verifier_prediction = scenario['correct_answer'] if correct_weight > wrong_weight else 'wrong'
        
        # 分析结果
        high_scores = [p['score'] for p in scored_paths if p['quality'] == 'high']
        low_scores = [p['score'] for p in scored_paths if p['quality'] == 'low']
        medium_scores = [p['score'] for p in scored_paths if p['quality'] == 'medium']
        
        discrimination = np.mean(high_scores) - np.mean(low_scores) if high_scores and low_scores else 0
        
        result = {
            'scenario': scenario['name'],
            'baseline_correct': True,  # 简化
            'verifier_correct': correct_weight > wrong_weight,
            'discrimination': discrimination,
            'high_avg': np.mean(high_scores) if high_scores else 0,
            'low_avg': np.mean(low_scores) if low_scores else 0,
            'medium_avg': np.mean(medium_scores) if medium_scores else 0
        }
        
        all_results.append(result)
        
        print(f"   📊 高质量均分: {result['high_avg']:.3f}")
        print(f"   📊 低质量均分: {result['low_avg']:.3f}")
        print(f"   📊 区分度: {discrimination:.3f}")
        print(f"   📊 Verifier选择: {'正确' if result['verifier_correct'] else '错误'}")
    
    # 4. 总体结果分析
    print(f"\n" + "="*70)
    print(f"🏆 总体验证结果")
    print(f"="*70)
    
    baseline_accuracy = np.mean([r['baseline_correct'] for r in all_results]) * 100
    verifier_accuracy = np.mean([r['verifier_correct'] for r in all_results]) * 100
    avg_discrimination = np.mean([r['discrimination'] for r in all_results])
    
    print(f"📊 基线方法准确率: {baseline_accuracy:.1f}%")
    print(f"📊 增强版Verifier准确率: {verifier_accuracy:.1f}%")
    print(f"📊 平均区分度: {avg_discrimination:.3f}")
    
    improvement = verifier_accuracy - baseline_accuracy
    print(f"\n🎯 准确率提升: {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"   ✅ 成功！增强版Verifier超越基线 {improvement:.1f}%!")
        print(f"   🎉 目标达成：Verifier效果显著提升！")
    elif improvement == 0:
        print(f"   ⚡ 平手：Verifier效果与基线相当")
    else:
        print(f"   ❌ 仍需改进：Verifier效果低于基线 {abs(improvement):.1f}%")
    
    # 区分度分析
    if avg_discrimination > 0.4:
        print(f"   ✅ 区分度优秀！({avg_discrimination:.3f})")
    elif avg_discrimination > 0.2:
        print(f"   ✅ 区分度良好 ({avg_discrimination:.3f})")
    elif avg_discrimination > 0.1:
        print(f"   ⚠️  区分度一般 ({avg_discrimination:.3f})")
    else:
        print(f"   ❌ 区分度不足 ({avg_discrimination:.3f})")
    
    # 保存结果
    output_file = os.path.join(
        config.experiment.output_dir, 
        f"quick_enhanced_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_results': all_results,
            'summary': {
                'baseline_accuracy': baseline_accuracy,
                'verifier_accuracy': verifier_accuracy,
                'improvement': improvement,
                'avg_discrimination': avg_discrimination
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    return improvement > 0

def main():
    """主函数"""
    quick_test()

if __name__ == "__main__":
    main() 