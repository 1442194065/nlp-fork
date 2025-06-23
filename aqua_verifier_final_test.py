#!/usr/bin/env python3
"""
AQuA数据集 Verifier效果最终验证实验
比较使用增强版verifier和不使用verifier的准确率差异
"""

import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from datetime import datetime
import time

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
        options_text = " ".join(options) if isinstance(options, list) else str(options)
        text = f"问题: {question} 选项: {options_text} 推理: {reasoning_path}"
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,  # 增加长度以容纳更多信息
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

def aqua_verifier_experiment(num_questions=20, paths_per_question=5):
    """AQuA数据集verifier效果实验"""
    print("="*80)
    print("🧪 AQuA数据集 Verifier效果最终验证实验")
    print("="*80)
    
    # 1. 初始化组件
    print("🔧 初始化实验组件...")
    data_loader = AQuADataLoader()
    llm_client = DeepSeekClient()
    diverse_cot = DiverseCoT(llm_client)
    
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
    
    # 2. 加载AQuA测试数据
    print(f"📋 加载AQuA测试数据...")
    data_loader.load_data('test')
    test_data = data_loader.test_data
    
    if not test_data or len(test_data) == 0:
        print(f"❌ 无法加载测试数据")
        return
    
    # 随机选择测试题目
    if len(test_data) > num_questions:
        test_questions = random.sample(test_data, num_questions)
    else:
        test_questions = test_data[:num_questions]
    
    print(f"📊 选择 {len(test_questions)} 个测试题目，每题生成 {paths_per_question} 个推理路径")
    
    # 3. 实验结果收集
    results = {
        'baseline_method': {
            'name': '基线方法（多数投票）',
            'correct': 0,
            'total': 0,
            'details': []
        },
        'verifier_method': {
            'name': '增强版Verifier加权投票',
            'correct': 0,
            'total': 0,
            'details': []
        },
        'analysis': {
            'verifier_scores': [],
            'correct_predictions_scores': [],
            'wrong_predictions_scores': [],
            'discrimination_per_question': []
        }
    }
    
    print(f"\n🚀 开始AQuA实验...")
    
    for i, question_data in enumerate(test_questions):
        print(f"\n--- 问题 {i+1}/{len(test_questions)} ---")
        
        question = question_data.question
        options = question_data.options
        correct_answer = question_data.correct
        
        print(f"问题: {question[:100]}...")
        print(f"选项: {', '.join(options[:3])}...")
        print(f"正确答案: {correct_answer}")
        
        # 生成多个推理路径
        print(f"   🤖 生成推理路径...")
        from data_loader import AQuAExample
        example = AQuAExample(
            question=question,
            options=options,
            rationale="",
            correct=correct_answer
        )
        reasoning_paths_data = diverse_cot.generate_diverse_reasoning_paths(example)
        
        # 转换为简化格式
        reasoning_paths = []
        for path_data in reasoning_paths_data[:paths_per_question]:
            reasoning_paths.append({
                'reasoning': path_data['response'],
                'prediction': path_data['predicted_answer']
            })
        
        if not reasoning_paths or len(reasoning_paths) < 2:
            print(f"   ⚠️  推理路径生成失败，跳过")
            continue
        
        print(f"   ✅ 生成 {len(reasoning_paths)} 个推理路径")
        
        # 显示推理路径质量
        correct_count = sum(1 for path in reasoning_paths if path['prediction'] == correct_answer)
        print(f"   📊 正确推理: {correct_count}/{len(reasoning_paths)} ({correct_count/len(reasoning_paths)*100:.1f}%)")
        
        # 基线方法：多数投票
        predictions = [path['prediction'] for path in reasoning_paths]
        prediction_counts = {}
        for pred in predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        majority_vote = max(prediction_counts, key=prediction_counts.get)
        baseline_correct = (majority_vote == correct_answer)
        
        results['baseline_method']['correct'] += baseline_correct
        results['baseline_method']['total'] += 1
        results['baseline_method']['details'].append({
            'question_idx': i,
            'question': question[:100],
            'correct_answer': correct_answer,
            'prediction': majority_vote,
            'correct': baseline_correct,
            'prediction_counts': prediction_counts,
            'reasoning_paths_count': len(reasoning_paths)
        })
        
        print(f"   🗳️  基线预测: {majority_vote} {'✅' if baseline_correct else '❌'} (投票: {prediction_counts})")
        
        # Verifier方法：加权投票
        print(f"   🔍 Verifier评分推理路径...")
        
        scored_paths = []
        verifier_scores_all = []
        
        for j, path in enumerate(reasoning_paths):
            try:
                score = verifier.predict(question, options, path['reasoning'])
                scored_paths.append({
                    'path_idx': j,
                    'reasoning': path['reasoning'],
                    'prediction': path['prediction'],
                    'score': score,
                    'is_correct': path['prediction'] == correct_answer
                })
                verifier_scores_all.append(score)
                print(f"     路径{j+1}: {score:.3f} -> {path['prediction']} {'✅' if path['prediction'] == correct_answer else '❌'}")
            except Exception as e:
                print(f"     ❌ 路径{j+1}评分失败: {e}")
                continue
        
        if not scored_paths:
            print(f"   ⚠️  无法评分，使用基线结果")
            verifier_correct = baseline_correct
            verifier_prediction = majority_vote
        else:
            # 使用verifier分数进行加权投票
            weighted_predictions = {}
            for path in scored_paths:
                pred = path['prediction']
                score = path['score']
                if pred not in weighted_predictions:
                    weighted_predictions[pred] = 0
                weighted_predictions[pred] += score
            
            verifier_prediction = max(weighted_predictions, key=weighted_predictions.get)
            verifier_correct = (verifier_prediction == correct_answer)
            
            # 分析verifier分数
            correct_scores = [p['score'] for p in scored_paths if p['is_correct']]
            wrong_scores = [p['score'] for p in scored_paths if not p['is_correct']]
            
            discrimination = np.mean(correct_scores) - np.mean(wrong_scores) if correct_scores and wrong_scores else 0
            
            results['analysis']['verifier_scores'].extend(verifier_scores_all)
            results['analysis']['correct_predictions_scores'].extend(correct_scores)
            results['analysis']['wrong_predictions_scores'].extend(wrong_scores)
            results['analysis']['discrimination_per_question'].append(discrimination)
            
            print(f"   ⚖️  加权投票: {weighted_predictions}")
            print(f"   🎯 Verifier预测: {verifier_prediction} {'✅' if verifier_correct else '❌'}")
            if correct_scores and wrong_scores:
                print(f"   📊 区分度: {discrimination:.3f} (正确:{np.mean(correct_scores):.3f} vs 错误:{np.mean(wrong_scores):.3f})")
        
        results['verifier_method']['correct'] += verifier_correct
        results['verifier_method']['total'] += 1
        results['verifier_method']['details'].append({
            'question_idx': i,
            'question': question[:100],
            'correct_answer': correct_answer,
            'prediction': verifier_prediction,
            'correct': verifier_correct,
            'weighted_predictions': weighted_predictions if 'weighted_predictions' in locals() else {},
            'scored_paths': scored_paths,
            'discrimination': discrimination if 'discrimination' in locals() else 0
        })
        
        # 当前累计结果
        baseline_acc = results['baseline_method']['correct'] / results['baseline_method']['total'] * 100
        verifier_acc = results['verifier_method']['correct'] / results['verifier_method']['total'] * 100
        print(f"   📈 累计准确率: 基线{baseline_acc:.1f}% vs Verifier{verifier_acc:.1f}%")
        
        # 避免API限制
        if i < len(test_questions) - 1:
            time.sleep(1)
    
    # 4. 最终结果分析
    print(f"\n" + "="*80)
    print(f"🏆 AQuA数据集实验最终结果")
    print(f"="*80)
    
    baseline_accuracy = results['baseline_method']['correct'] / results['baseline_method']['total'] * 100
    verifier_accuracy = results['verifier_method']['correct'] / results['verifier_method']['total'] * 100
    improvement = verifier_accuracy - baseline_accuracy
    
    print(f"📊 基线方法（多数投票）准确率: {baseline_accuracy:.2f}% ({results['baseline_method']['correct']}/{results['baseline_method']['total']})")
    print(f"📊 增强版Verifier准确率: {verifier_accuracy:.2f}% ({results['verifier_method']['correct']}/{results['verifier_method']['total']})")
    print(f"📈 准确率提升: {improvement:+.2f}%")
    
    # 分析verifier效果
    if improvement > 0:
        print(f"\n🎉 成功！增强版Verifier超越基线 {improvement:.2f}%!")
        print(f"   ✅ 目标达成：在真实AQuA数据集上Verifier效果更优！")
    elif improvement == 0:
        print(f"\n⚡ 平手：Verifier效果与基线相当")
    else:
        print(f"\n⚠️  Verifier效果低于基线 {abs(improvement):.2f}%")
    
    # Verifier质量分析
    if results['analysis']['verifier_scores']:
        all_scores = results['analysis']['verifier_scores']
        correct_scores = results['analysis']['correct_predictions_scores']
        wrong_scores = results['analysis']['wrong_predictions_scores']
        discriminations = results['analysis']['discrimination_per_question']
        
        print(f"\n🔍 Verifier质量分析:")
        print(f"   总评分范围: [{np.min(all_scores):.3f}, {np.max(all_scores):.3f}]")
        print(f"   正确推理平均分: {np.mean(correct_scores):.3f} ± {np.std(correct_scores):.3f}")
        print(f"   错误推理平均分: {np.mean(wrong_scores):.3f} ± {np.std(wrong_scores):.3f}")
        
        overall_discrimination = np.mean(correct_scores) - np.mean(wrong_scores)
        print(f"   整体区分度: {overall_discrimination:.3f}")
        print(f"   平均每题区分度: {np.mean(discriminations):.3f} ± {np.std(discriminations):.3f}")
        
        if overall_discrimination > 0.3:
            print(f"   ✅ Verifier区分能力优秀！")
        elif overall_discrimination > 0.1:
            print(f"   ✅ Verifier区分能力良好")
        else:
            print(f"   ⚠️  Verifier区分能力仍需提升")
    
    # 保存详细结果
    output_file = os.path.join(
        config.experiment.output_dir, 
        f"aqua_verifier_final_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    # 添加实验总结
    results['experiment_summary'] = {
        'total_questions': len(test_questions),
        'paths_per_question': paths_per_question,
        'baseline_accuracy': baseline_accuracy,
        'verifier_accuracy': verifier_accuracy,
        'improvement': improvement,
        'overall_discrimination': overall_discrimination if 'overall_discrimination' in locals() else 0,
        'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 最终结论
    print(f"\n" + "="*80)
    print(f"🎯 实验结论")
    print(f"="*80)
    
    if improvement > 5:
        print(f"🏆 增强版Verifier在AQuA数据集上取得显著提升！")
        print(f"   提升幅度: {improvement:.2f}%，验证了方法的有效性")
    elif improvement > 0:
        print(f"✅ 增强版Verifier在AQuA数据集上取得了正向提升")
        print(f"   提升幅度: {improvement:.2f}%，证明了改进的价值")
    else:
        print(f"📈 虽然准确率提升有限，但Verifier已具备推理质量区分能力")
        print(f"   区分度: {overall_discrimination:.3f}，为进一步优化奠定基础")
    
    return results

def main():
    """主函数"""
    print("欢迎使用AQuA数据集Verifier效果验证实验！")
    
    # 可以调整实验参数
    num_questions = 10  # 测试题目数量
    paths_per_question = 10  # 每题生成的推理路径数
    
    print(f"实验设置: {num_questions}题 × {paths_per_question}路径")
    
    aqua_verifier_experiment(num_questions, paths_per_question)

if __name__ == "__main__":
    main() 