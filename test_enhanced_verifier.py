#!/usr/bin/env python3
"""
测试增强版Verifier效果
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from datetime import datetime

from config import config
from enhanced_verifier_trainer import ContrastiveVerifierModel

class EnhancedVerifierPredictor:
    """增强版Verifier预测器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """加载增强版模型"""
        print(f"🔄 加载增强版Verifier模型...")
        print(f"   模型路径: {self.model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 加载模型
        self.model = ContrastiveVerifierModel()
        
        # 使用safetensors加载模型权重
        from safetensors import safe_open
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
    
    def predict(self, question, options, reasoning_path):
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
            # 直接返回回归分数（已经是0-1范围）
            score = logits[0][0].item()
        
        return score

def test_enhanced_verifier():
    """测试增强版verifier效果"""
    print("🧪 测试增强版Verifier效果")
    print("="*60)
    
    # 加载增强版verifier
    model_path = os.path.join(config.experiment.model_save_dir, "enhanced_verifier_model")
    
    if not os.path.exists(model_path):
        print(f"❌ 增强版模型不存在: {model_path}")
        return
    
    try:
        verifier = EnhancedVerifierPredictor(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 创建更严格的测试案例
    test_cases = [
        # 明显高质量推理
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning': '根据方程 x + 5 = 12，两边同时减去5，得到 x = 12 - 5 = 7。因此答案是A)7。',
            'expected_quality': 'high',
            'expected_score_range': (0.7, 1.0)
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning': '总价格是20元，书的数量是4本。每本书的价格 = 总价格 ÷ 数量 = 20 ÷ 4 = 5元。答案是B)5元。',
            'expected_quality': 'high',
            'expected_score_range': (0.7, 1.0)
        },
        {
            'question': '一个班级有30个学生，其中60%是女生，有多少个男生？',
            'options': ['A)12', 'B)18', 'C)15', 'D)10', 'E)20'],
            'reasoning': '女生人数 = 30 × 60% = 30 × 0.6 = 18人。男生人数 = 总人数 - 女生人数 = 30 - 18 = 12人。答案是A)12。',
            'expected_quality': 'high',
            'expected_score_range': (0.7, 1.0)
        },
        
        # 明显低质量推理
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning': '我觉得答案是C，因为6这个数字看起来比较合适。',
            'expected_quality': 'low',
            'expected_score_range': (0.0, 0.3)
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning': '随便猜一个，我选择A)4元。',
            'expected_quality': 'low',
            'expected_score_range': (0.0, 0.3)
        },
        {
            'question': '一个班级有30个学生，其中60%是女生，有多少个男生？',
            'options': ['A)12', 'B)18', 'C)15', 'D)10', 'E)20'],
            'reasoning': '感觉应该是15个，答案选C。',
            'expected_quality': 'low',
            'expected_score_range': (0.0, 0.3)
        },
        
        # 中等质量推理
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning': '这道题需要计算，x应该是7，选择A。',
            'expected_quality': 'medium',
            'expected_score_range': (0.3, 0.7)
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning': '20除以4等于5，答案是B。',
            'expected_quality': 'medium',
            'expected_score_range': (0.3, 0.7)
        }
    ]
    
    print(f"📋 测试 {len(test_cases)} 个案例...")
    
    # 收集评分
    high_scores = []
    medium_scores = []
    low_scores = []
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, case in enumerate(test_cases):
        score = verifier.predict(case['question'], case['options'], case['reasoning'])
        
        # 根据质量分类
        if case['expected_quality'] == 'high':
            high_scores.append(score)
        elif case['expected_quality'] == 'medium':
            medium_scores.append(score)
        else:
            low_scores.append(score)
        
        # 检查预测是否在期望范围内
        min_score, max_score = case['expected_score_range']
        is_correct = min_score <= score <= max_score
        correct_predictions += is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"   {status} 案例 {i+1}: 质量={case['expected_quality']}, 评分={score:.3f}, 期望=[{min_score:.1f}, {max_score:.1f}]")
        print(f"       推理: {case['reasoning'][:50]}...")
    
    # 统计分析
    print(f"\n📊 增强版Verifier评分分析:")
    if high_scores:
        print(f"   高质量推理: 平均={np.mean(high_scores):.3f}, 范围=[{np.min(high_scores):.3f}, {np.max(high_scores):.3f}]")
    if medium_scores:
        print(f"   中等质量推理: 平均={np.mean(medium_scores):.3f}, 范围=[{np.min(medium_scores):.3f}, {np.max(medium_scores):.3f}]")
    if low_scores:
        print(f"   低质量推理: 平均={np.mean(low_scores):.3f}, 范围=[{np.min(low_scores):.3f}, {np.max(low_scores):.3f}]")
    
    # 计算区分度
    if high_scores and low_scores:
        discrimination = np.mean(high_scores) - np.mean(low_scores)
        print(f"\n🎯 评分区分度: {discrimination:.3f}")
        
        if discrimination > 0.4:
            print(f"   ✅ 区分度优秀！")
        elif discrimination > 0.2:
            print(f"   ✅ 区分度良好")
        elif discrimination > 0.1:
            print(f"   ⚠️  区分度一般")
        else:
            print(f"   ❌ 区分度不足")
    
    # 预测准确性
    accuracy = correct_predictions / total_predictions * 100
    print(f"\n🎯 预测准确性: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    if accuracy >= 80:
        print(f"   ✅ 增强版Verifier表现优秀！")
    elif accuracy >= 60:
        print(f"   ✅ 增强版Verifier表现良好")
    else:
        print(f"   ⚠️  增强版Verifier仍需优化")
    
    return {
        'accuracy': accuracy,
        'discrimination': discrimination if high_scores and low_scores else 0,
        'high_avg': np.mean(high_scores) if high_scores else 0,
        'low_avg': np.mean(low_scores) if low_scores else 0
    }

def main():
    """主函数"""
    print("="*60)
    print("🔬 增强版Verifier效果测试")
    print("="*60)
    
    test_enhanced_verifier()

if __name__ == "__main__":
    main() 