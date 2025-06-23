#!/usr/bin/env python3
"""
Verifier标签诊断调试
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer
from safetensors import safe_open

from config import config
from enhanced_verifier_trainer import ContrastiveVerifierModel

class VerifierDebugger:
    """Verifier调试器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
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
            print(f"✅ 模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict_score(self, question, reasoning):
        """预测单个推理的分数"""
        text = f"问题: {question} 推理: {reasoning}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            score = outputs['logits'][0][0].item()
        
        return score

def test_verifier_understanding():
    """测试verifier对高低质量推理的理解"""
    print("="*70)
    print("🔬 Verifier标签理解诊断")
    print("="*70)
    
    # 加载verifier
    model_path = os.path.join(config.experiment.model_save_dir, "enhanced_verifier_model")
    debugger = VerifierDebugger(model_path)
    
    # 创建极端对比的测试案例
    test_cases = [
        {
            'category': '极高质量推理',
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'reasoning': '根据方程 x + 5 = 12，两边同时减去5，得到 x = 12 - 5 = 7。这是标准的代数运算，结果准确无误。',
            'expected_high': True
        },
        {
            'category': '极高质量推理',
            'question': '买4本书花了20元，每本书多少钱？',
            'reasoning': '总价格是20元，书的数量是4本。每本书的价格 = 总价格 ÷ 数量 = 20 ÷ 4 = 5元。这是基本的除法运算。',
            'expected_high': True
        },
        {
            'category': '极低质量推理',
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'reasoning': '我不知道怎么算，随便猜一个数字，可能是100吧。',
            'expected_high': False
        },
        {
            'category': '极低质量推理',
            'question': '买4本书花了20元，每本书多少钱？',
            'reasoning': '这个问题太难了，我觉得答案是1000元。',
            'expected_high': False
        },
        {
            'category': '训练时的高质量样本',
            'question': '一个班级有30个学生，其中60%是女生，有多少个男生？',
            'reasoning': '女生人数 = 30 × 60% = 30 × 0.6 = 18人。男生人数 = 总人数 - 女生人数 = 30 - 18 = 12人。答案是A)12。',
            'expected_high': True
        },
        {
            'category': '训练时的低质量样本',
            'question': '一个班级有30个学生，其中60%是女生，有多少个男生？',
            'reasoning': '感觉应该是15个，答案选C。',
            'expected_high': False
        }
    ]
    
    print(f"📋 测试 {len(test_cases)} 个极端对比案例...\n")
    
    # 收集结果
    high_scores = []
    low_scores = []
    
    for i, case in enumerate(test_cases):
        score = debugger.predict_score(case['question'], case['reasoning'])
        
        if case['expected_high']:
            high_scores.append(score)
        else:
            low_scores.append(score)
        
        expected_level = "高质量" if case['expected_high'] else "低质量"
        print(f"案例 {i+1}: [{case['category']}]")
        print(f"   问题: {case['question'][:50]}...")
        print(f"   推理: {case['reasoning'][:60]}...")
        print(f"   期望: {expected_level}")
        print(f"   评分: {score:.3f}")
        print()
    
    # 分析结果
    print("="*70)
    print("📊 评分分析结果")
    print("="*70)
    
    print(f"高质量推理评分:")
    for i, score in enumerate(high_scores):
        print(f"   案例{i+1}: {score:.3f}")
    print(f"   平均分: {np.mean(high_scores):.3f}")
    print(f"   范围: [{np.min(high_scores):.3f}, {np.max(high_scores):.3f}]")
    
    print(f"\n低质量推理评分:")
    for i, score in enumerate(low_scores):
        print(f"   案例{i+1}: {score:.3f}")
    print(f"   平均分: {np.mean(low_scores):.3f}")
    print(f"   范围: [{np.min(low_scores):.3f}, {np.max(low_scores):.3f}]")
    
    discrimination = np.mean(high_scores) - np.mean(low_scores)
    print(f"\n🎯 区分度: {discrimination:.3f}")
    
    if discrimination > 0.1:
        print(f"   ✅ 区分度正常：高质量 > 低质量")
    elif discrimination > -0.1:
        print(f"   ⚠️  区分度微弱：几乎无差别")
    else:
        print(f"   ❌ 区分度反向：高质量 < 低质量 (标签可能搞反了！)")
    
    # 检查是否标签反向
    if discrimination < -0.05:
        print(f"\n🚨 诊断结果：标签理解可能反向！")
        print(f"   模型可能认为：")
        print(f"   - 评分越低 = 质量越高")
        print(f"   - 评分越高 = 质量越低")
        print(f"   建议：检查训练数据标签或重新训练")
    else:
        print(f"\n✅ 诊断结果：标签理解基本正常")
    
    return discrimination

def main():
    """主函数"""
    test_verifier_understanding()

if __name__ == "__main__":
    main() 