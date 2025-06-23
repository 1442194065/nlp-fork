#!/usr/bin/env python3
"""
最终Verifier对比实验 - 增强版 vs 原版 vs 基线
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer
import torch
from safetensors import safe_open

from config import config
from data_loader import AQuADataLoader
from diverse_cot import DiverseCoT
from llm_client import DeepSeekClient

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

def final_comparison_experiment():
    """最终对比实验"""
    print("="*80)
    print("🏆 最终Verifier对比实验 - 验证改进效果")
    print("="*80)
    
    # 1. 初始化组件
    data_loader = AQuADataLoader()
    llm_client = DeepSeekClient()
    diverse_cot = DiverseCoT(llm_client)
    
    # 加载测试数据
    test_data = data_loader.load_data('test')[:15]  # 使用15个问题进行测试
    print(f"📋 使用 {len(test_data)} 个测试题目")
    
    # 2. 加载不同版本的verifier
    verifiers = {}
    
    # 加载增强版verifier
    enhanced_model_path = os.path.join(config.experiment.model_save_dir, "enhanced_verifier_model")
    if os.path.exists(enhanced_model_path):
        try:
            verifiers['enhanced'] = EnhancedVerifierPredictor(enhanced_model_path)
            print(f"✅ 增强版Verifier加载成功")
        except Exception as e:
            print(f"❌ 增强版Verifier加载失败: {e}")
    
         # 暂时只测试增强版verifier
     # improved_model_path = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
     # 可以在这里添加其他verifier版本的测试
    
    if not verifiers:
        print(f"❌ 没有可用的verifier模型")
        return
    
    # 3. 实验结果收集
    results = {
        'baseline': {'correct': 0, 'total': 0, 'details': []},
    }
    
    for verifier_name in verifiers.keys():
        results[verifier_name] = {'correct': 0, 'total': 0, 'details': []}
    
    print(f"\n🚀 开始对比实验...")
    
    for i, question_data in enumerate(test_data):
        print(f"\n--- 问题 {i+1}/{len(test_data)} ---")
        question = question_data['question']
        options = question_data['options']
        correct_answer = question_data['correct']
        
        print(f"问题: {question[:80]}...")
        print(f"正确答案: {correct_answer}")
        
        # 生成多个推理路径
        reasoning_paths = diverse_cot.generate_diverse_reasoning(
            question, options, num_paths=5
        )
        
        if not reasoning_paths:
            print(f"   ⚠️  无法生成推理路径，跳过")
            continue
        
        print(f"   生成 {len(reasoning_paths)} 个推理路径")
        
        # 手动评估推理质量（用于参考）
        correct_count = sum(1 for path in reasoning_paths if path['prediction'] == correct_answer)
        print(f"   正确推理: {correct_count}/{len(reasoning_paths)}")
        
        # 基线方法：多数投票
        predictions = [path['prediction'] for path in reasoning_paths]
        majority_vote = max(set(predictions), key=predictions.count)
        baseline_correct = (majority_vote == correct_answer)
        
        results['baseline']['correct'] += baseline_correct
        results['baseline']['total'] += 1
        results['baseline']['details'].append({
            'question_idx': i,
            'prediction': majority_vote,
            'correct': baseline_correct,
            'reasoning_count': len(reasoning_paths)
        })
        
        print(f"   基线预测: {majority_vote} {'✅' if baseline_correct else '❌'}")
        
        # 各种verifier方法
        for verifier_name, verifier in verifiers.items():
            # 计算每个推理路径的质量分数
            scores = []
            for path in reasoning_paths:
                try:
                    score = verifier.predict(question, options, path['reasoning'])
                    scores.append(score)
                except Exception as e:
                    print(f"     ❌ {verifier_name} 评分失败: {e}")
                    scores.append(0.5)  # 默认分数
            
            # 使用verifier分数进行加权投票
            weighted_predictions = {}
            for path, score in zip(reasoning_paths, scores):
                pred = path['prediction']
                if pred not in weighted_predictions:
                    weighted_predictions[pred] = 0
                weighted_predictions[pred] += score
            
            # 选择得分最高的预测
            if weighted_predictions:
                verifier_prediction = max(weighted_predictions, key=weighted_predictions.get)
                verifier_correct = (verifier_prediction == correct_answer)
            else:
                verifier_prediction = majority_vote
                verifier_correct = baseline_correct
            
            results[verifier_name]['correct'] += verifier_correct
            results[verifier_name]['total'] += 1
            results[verifier_name]['details'].append({
                'question_idx': i,
                'prediction': verifier_prediction,
                'correct': verifier_correct,
                'scores': scores,
                'score_avg': np.mean(scores),
                'score_std': np.std(scores)
            })
            
            print(f"   {verifier_name}预测: {verifier_prediction} {'✅' if verifier_correct else '❌'} (分数均值:{np.mean(scores):.3f})")
    
    # 4. 结果分析
    print(f"\n" + "="*80)
    print(f"🏆 最终实验结果分析")
    print(f"="*80)
    
    baseline_accuracy = results['baseline']['correct'] / results['baseline']['total'] * 100
    print(f"📊 基线方法（多数投票）: {baseline_accuracy:.2f}% ({results['baseline']['correct']}/{results['baseline']['total']})")
    
    for verifier_name in verifiers.keys():
        verifier_accuracy = results[verifier_name]['correct'] / results[verifier_name]['total'] * 100
        improvement = verifier_accuracy - baseline_accuracy
        
        print(f"📊 {verifier_name.capitalize()}版Verifier: {verifier_accuracy:.2f}% ({results[verifier_name]['correct']}/{results[verifier_name]['total']})")
        print(f"     相对基线提升: {improvement:+.2f}% {'✅' if improvement > 0 else '❌'}")
    
    # 寻找最佳方法
    best_method = 'baseline'
    best_accuracy = baseline_accuracy
    
    for verifier_name in verifiers.keys():
        verifier_accuracy = results[verifier_name]['correct'] / results[verifier_name]['total'] * 100
        if verifier_accuracy > best_accuracy:
            best_method = verifier_name
            best_accuracy = verifier_accuracy
    
    print(f"\n🏆 最佳方法: {best_method.capitalize()} ({best_accuracy:.2f}%)")
    
    if best_method != 'baseline':
        improvement = best_accuracy - baseline_accuracy
        print(f"🎯 Verifier成功提升准确率 {improvement:.2f}%!")
        print(f"   ✅ 目标达成：Verifier效果超越基线！")
    else:
        print(f"⚠️  所有Verifier方法仍未超越基线")
        
        # 分析问题
        print(f"\n🔍 问题分析:")
        for verifier_name in verifiers.keys():
            scores_all = []
            for detail in results[verifier_name]['details']:
                scores_all.extend(detail['scores'])
            
            if scores_all:
                print(f"   {verifier_name}: 平均分={np.mean(scores_all):.3f}, 标准差={np.std(scores_all):.3f}, 范围=[{np.min(scores_all):.3f}, {np.max(scores_all):.3f}]")
    
    # 5. 保存详细结果
    output_file = os.path.join(
        config.experiment.output_dir, 
        f"final_verifier_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    return results

def main():
    """主函数"""
    final_comparison_experiment()

if __name__ == "__main__":
    main() 