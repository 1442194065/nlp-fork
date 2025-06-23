#!/usr/bin/env python3
"""
测试改进版Verifier效果
"""

import os
import json
import random
import torch
from transformers import AutoTokenizer
from datetime import datetime

from config import config
from train_improved_verifier_simple import SimpleImprovedVerifier

class ImprovedVerifierPredictor:
    """改进版Verifier预测器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        print(f"🔄 加载改进版Verifier模型...")
        print(f"   模型路径: {self.model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 加载模型
        self.model = SimpleImprovedVerifier()
        
        # 使用safetensors加载模型权重
        from safetensors import safe_open
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
    
    def predict(self, question, options, reasoning_path):
        """预测推理路径质量"""
        # 构造输入文本
        options_str = " ".join(options) if isinstance(options, list) else str(options)
        text = f"{question} [SEP] {reasoning_path}"
        
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
            probabilities = torch.softmax(logits, dim=-1)
            # 返回"正确"类别的概率
            score = probabilities[0][1].item()
        
        return score

def generate_test_cases():
    """生成测试案例"""
    test_cases = [
        {
            'question': '如果一个数字加上15等于37，这个数字是多少？',
            'options': ['A)22', 'B)52', 'C)12', 'D)42', 'E)32'],
            'correct': 'A',
            'reasoning_paths': [
                ('设这个数字为x，则x + 15 = 37，所以x = 37 - 15 = 22。答案是A。', True),
                ('37 + 15 = 52，所以答案是B。', False),
                ('我觉得应该是12，答案是C。', False),
                ('根据题意，37 - 15 = 22，所以答案是A。', True),
                ('随便猜一个，选择D。', False)
            ]
        },
        {
            'question': '一个班级有30个学生，其中60%是女生，有多少个男生？',
            'options': ['A)12', 'B)18', 'C)15', 'D)10', 'E)20'],
            'correct': 'A',
            'reasoning_paths': [
                ('女生有30 × 60% = 18人，所以男生有30 - 18 = 12人。答案是A。', True),
                ('60%的学生是女生，那就是18人，答案是B。', False),
                ('30 ÷ 2 = 15，所以答案是C。', False),
                ('总共30人，女生占60%即18人，男生就是30-18=12人，选A。', True),
                ('我觉得答案是D。', False)
            ]
        },
        {
            'question': '买3支笔需要9元，买7支同样的笔需要多少元？',
            'options': ['A)21', 'B)28', 'C)15', 'D)18', 'E)24'],
            'correct': 'A',
            'reasoning_paths': [
                ('每支笔9÷3=3元，7支笔需要7×3=21元。答案是A。', True),
                ('9 + 7 = 16，不对。答案应该是21，选A。', True),
                ('7 × 4 = 28，答案是B。', False),
                ('根据比例，3支9元，7支应该是21元，选择A。', True),
                ('随机选择E。', False)
            ]
        }
    ]
    return test_cases

def test_verifier(verifier, test_cases):
    """测试verifier效果"""
    print(f"\n🧪 测试改进版Verifier效果")
    print("="*60)
    
    total_correct = 0
    total_predictions = 0
    
    for i, case in enumerate(test_cases):
        print(f"\n📋 测试案例 {i+1}: {case['question'][:50]}...")
        
        question = case['question']
        options = case['options']
        correct_answer = case['correct']
        
        # 测试每个推理路径
        for reasoning, is_correct in case['reasoning_paths']:
            score = verifier.predict(question, options, reasoning)
            
            # 判断verifier是否正确识别了推理质量
            predicted_good = score > 0.5
            actual_good = is_correct
            
            is_prediction_correct = (predicted_good == actual_good)
            total_correct += is_prediction_correct
            total_predictions += 1
            
            status = "✅" if is_prediction_correct else "❌"
            print(f"   {status} 推理: {reasoning[:40]}...")
            print(f"       真实质量: {'好' if actual_good else '差'}, Verifier评分: {score:.3f}, 预测: {'好' if predicted_good else '差'}")
    
    accuracy = total_correct / total_predictions * 100
    print(f"\n📊 Verifier准确率: {accuracy:.2f}% ({total_correct}/{total_predictions})")
    
    return accuracy

def compare_with_baseline():
    """与基线方法对比"""
    print(f"\n📈 与随机基线对比")
    print("="*60)
    
    # 随机基线：50%准确率（随机猜测）
    baseline_accuracy = 50.0
    
    # 加载改进版verifier
    model_path = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    try:
        verifier = ImprovedVerifierPredictor(model_path)
        
        # 生成测试案例
        test_cases = generate_test_cases()
        
        # 测试verifier
        verifier_accuracy = test_verifier(verifier, test_cases)
        
        # 结果对比
        improvement = verifier_accuracy - baseline_accuracy
        
        print(f"\n🎯 结果对比:")
        print(f"   随机基线准确率: {baseline_accuracy:.2f}%")
        print(f"   改进版Verifier:  {verifier_accuracy:.2f}%")
        print(f"   提升幅度:       {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"✅ 改进版Verifier表现优于基线！")
        else:
            print(f"❌ 改进版Verifier需要进一步优化")
        
        # 保存结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_accuracy': baseline_accuracy,
            'verifier_accuracy': verifier_accuracy,
            'improvement': improvement,
            'test_cases': len(test_cases)
        }
        
        output_file = os.path.join(config.experiment.output_dir, f"improved_verifier_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 测试结果已保存: {output_file}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    """主函数"""
    print("="*60)
    print("🔬 改进版Verifier效果测试")
    print("="*60)
    
    compare_with_baseline()

if __name__ == "__main__":
    main() 