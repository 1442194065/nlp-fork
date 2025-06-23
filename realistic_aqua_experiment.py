#!/usr/bin/env python3
"""
更现实的AQuA Verifier实验
模拟真实的推理错误和verifier效果
"""

import os
import json
import random
import numpy as np
from collections import Counter
from datetime import datetime

# 导入必要模块
from config import config
from data_loader import AQuADataLoader
from verifier import VerifierTrainer

class RealisticAQuAExperiment:
    """更现实的AQuA实验"""
    
    def __init__(self):
        self.data_loader = AQuADataLoader()
        self.verifier_trainer = None
        
        print("🧪 现实AQuA Verifier实验初始化完成")
    
    def load_test_data(self):
        """加载测试数据"""
        print("\n" + "="*60)
        print("📊 加载AQuA测试数据")
        print("="*60)
        
        self.data_loader.load_data("dev")  
        test_data = self.data_loader.get_eval_subset("dev", max_samples=30)
        print(f"✅ 加载了 {len(test_data)} 个测试样本")
        
        # 转换为字典格式
        test_data_dict = []
        for example in test_data:
            test_data_dict.append({
                'question': example.question,
                'options': example.options,
                'correct': example.correct,
                'rationale': example.rationale
            })
        
        return test_data_dict
    
    def load_verifier(self):
        """加载verifier"""
        print("\n" + "="*60)
        print("🏋️ 加载Verifier模型")
        print("="*60)
        
        model_path = os.path.join(config.experiment.model_save_dir, "verifier_model")
        
        if os.path.exists(model_path):
            print("✅ 发现已训练的verifier模型，加载中...")
            self.verifier_trainer = VerifierTrainer()
            try:
                self.verifier_trainer.load_model(model_path)
                print("✅ Verifier模型加载成功")
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return False
        else:
            print("❌ 未找到verifier模型")
            return False
    
    def generate_realistic_reasoning_paths(self, question, options, correct_answer):
        """生成更现实的推理路径，包含常见错误"""
        paths = []
        
        # 1. 正确推理路径 (30%概率)
        if random.random() < 0.3:
            correct_reasoning = self._generate_correct_reasoning(question, correct_answer)
            paths.append({'reasoning': correct_reasoning, 'answer': correct_answer, 'quality': 0.9})
        
        # 2. 部分正确推理 (40%概率)
        if random.random() < 0.4:
            partial_reasoning = self._generate_partial_reasoning(question, correct_answer)
            paths.append({'reasoning': partial_reasoning, 'answer': correct_answer, 'quality': 0.7})
        
        # 3. 错误推理路径 (填满剩余空间)
        wrong_answers = [opt for opt in ['A', 'B', 'C', 'D', 'E'] if opt != correct_answer]
        
        while len(paths) < 5:
            wrong_answer = random.choice(wrong_answers)
            wrong_reasoning = self._generate_wrong_reasoning(question, wrong_answer)
            quality = random.uniform(0.1, 0.4)  # 错误推理质量较低
            paths.append({'reasoning': wrong_reasoning, 'answer': wrong_answer, 'quality': quality})
        
        return paths
    
    def _generate_correct_reasoning(self, question, answer):
        """生成正确的推理路径"""
        reasoning_templates = [
            f"仔细分析题目: {question[:40]}... 通过正确的数学计算，我得出答案是 {answer}。这个解法考虑了所有关键因素。",
            f"解题思路: {question[:30]}... 运用相关公式和逻辑推理，最终确定答案是 {answer}。验算结果无误。",
            f"分步骤求解: {question[:35]}... 经过仔细计算和验证，确信答案是 {answer}。"
        ]
        return random.choice(reasoning_templates)
    
    def _generate_partial_reasoning(self, question, answer):
        """生成部分正确的推理路径"""
        reasoning_templates = [
            f"初步分析: {question[:30]}... 虽然推理过程有些粗糙，但我认为答案是 {answer}。",
            f"简单计算: {question[:25]}... 没有完全验证，但倾向于选择 {answer}。",
            f"直觉判断: {question[:35]}... 基于经验和简单分析，答案应该是 {answer}。"
        ]
        return random.choice(reasoning_templates)
    
    def _generate_wrong_reasoning(self, question, wrong_answer):
        """生成错误的推理路径"""
        error_types = [
            f"计算错误: {question[:25]}... 在运算过程中犯了错误，得出答案是 {wrong_answer}。",
            f"理解偏差: {question[:30]}... 对题目理解有误，错误地认为答案是 {wrong_answer}。",
            f"方法错误: {question[:20]}... 使用了不合适的解法，导致答案是 {wrong_answer}。",
            f"遗漏条件: {question[:35]}... 忽略了重要条件，计算结果是 {wrong_answer}。"
        ]
        return random.choice(error_types)
    
    def evaluate_baseline_majority_vote(self, test_data):
        """评估基线方法：多数投票"""
        print("\n" + "="*60)
        print("📈 评估基线方法（多数投票）")
        print("="*60)
        
        correct_count = 0
        total_count = len(test_data)
        
        for i, item in enumerate(test_data):
            print(f"测试题目 {i+1}/{total_count}: {item['question'][:30]}...")
            
            # 生成现实的推理路径
            paths = self.generate_realistic_reasoning_paths(
                item['question'], 
                item['options'], 
                item['correct']
            )
            
            # 多数投票
            answers = [path['answer'] for path in paths]
            answer_counts = Counter(answers)
            predicted_answer = answer_counts.most_common(1)[0][0]
            
            if predicted_answer == item['correct']:
                correct_count += 1
                result = "✅ 正确"
            else:
                result = "❌ 错误"
            
            print(f"  {result} - 预测: {predicted_answer}, 实际: {item['correct']}")
            print(f"    路径答案: {answers}")
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 多数投票结果:")
        print(f"   正确数量: {correct_count}/{total_count}")
        print(f"   准确率: {accuracy:.2%}")
        
        return accuracy
    
    def evaluate_verifier_weighted_vote(self, test_data):
        """评估verifier加权投票方法"""
        print("\n" + "="*60)
        print("🧠 评估Verifier加权投票方法")
        print("="*60)
        
        if self.verifier_trainer is None:
            print("❌ Verifier未加载")
            return 0
        
        correct_count = 0
        total_count = len(test_data)
        
        for i, item in enumerate(test_data):
            print(f"测试题目 {i+1}/{total_count}: {item['question'][:30]}...")
            
            # 生成现实的推理路径
            paths = self.generate_realistic_reasoning_paths(
                item['question'], 
                item['options'], 
                item['correct']
            )
            
            # 使用verifier评分进行加权投票
            answer_scores = {}
            
            for path in paths:
                try:
                    # Verifier评分
                    score = self.verifier_trainer.predict(
                        question=item['question'],
                        options=item['options'],
                        reasoning_path=path['reasoning']
                    )
                    
                    answer = path['answer']
                    if answer not in answer_scores:
                        answer_scores[answer] = 0
                    
                    # 加权累加（verifier评分作为权重）
                    answer_scores[answer] += score
                    
                except Exception as e:
                    print(f"    ⚠️ 路径评分失败: {e}")
                    continue
            
            if answer_scores:
                # 选择加权得分最高的答案
                predicted_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
                best_score = answer_scores[predicted_answer]
                
                if predicted_answer == item['correct']:
                    correct_count += 1
                    result = "✅ 正确"
                else:
                    result = "❌ 错误"
                
                print(f"  {result} - 预测: {predicted_answer}, 实际: {item['correct']}")
                print(f"    加权得分: {[(ans, f'{score:.3f}') for ans, score in answer_scores.items()]}")
            else:
                print(f"  ⚠️ 无法获得有效评分")
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 Verifier加权投票结果:")
        print(f"   正确数量: {correct_count}/{total_count}")
        print(f"   准确率: {accuracy:.2%}")
        
        return accuracy
    
    def evaluate_quality_aware_vote(self, test_data):
        """评估质量感知投票（基于已知质量）"""
        print("\n" + "="*60)
        print("🎯 评估质量感知投票（理想情况）")
        print("="*60)
        
        correct_count = 0
        total_count = len(test_data)
        
        for i, item in enumerate(test_data):
            print(f"测试题目 {i+1}/{total_count}: {item['question'][:30]}...")
            
            # 生成现实的推理路径
            paths = self.generate_realistic_reasoning_paths(
                item['question'], 
                item['options'], 
                item['correct']
            )
            
            # 基于质量进行加权投票
            answer_scores = {}
            
            for path in paths:
                answer = path['answer']
                quality = path['quality']
                
                if answer not in answer_scores:
                    answer_scores[answer] = 0
                
                answer_scores[answer] += quality
            
            # 选择质量加权得分最高的答案
            predicted_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
            
            if predicted_answer == item['correct']:
                correct_count += 1
                result = "✅ 正确"
            else:
                result = "❌ 错误"
            
            print(f"  {result} - 预测: {predicted_answer}, 实际: {item['correct']}")
            print(f"    质量得分: {[(ans, f'{score:.3f}') for ans, score in answer_scores.items()]}")
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 质量感知投票结果:")
        print(f"   正确数量: {correct_count}/{total_count}")
        print(f"   准确率: {accuracy:.2%}")
        
        return accuracy
    
    def run_experiment(self):
        """运行完整实验"""
        print("="*70)
        print("🎯 现实AQuA Verifier效果对比实验")
        print("="*70)
        
        # 设置随机种子保证可重现性
        random.seed(42)
        np.random.seed(42)
        
        # 1. 加载测试数据
        test_data = self.load_test_data()
        
        # 2. 加载verifier
        verifier_loaded = self.load_verifier()
        
        # 3. 评估基线方法
        baseline_accuracy = self.evaluate_baseline_majority_vote(test_data)
        
        # 4. 评估verifier方法
        if verifier_loaded:
            verifier_accuracy = self.evaluate_verifier_weighted_vote(test_data)
        else:
            print("⚠️ Verifier加载失败，跳过verifier评估")
            verifier_accuracy = 0
        
        # 5. 评估理想情况（质量感知）
        ideal_accuracy = self.evaluate_quality_aware_vote(test_data)
        
        # 6. 汇总结果
        print("\n" + "="*70)
        print("📈 实验结果汇总")
        print("="*70)
        
        print(f"📊 基线方法（多数投票）准确率: {baseline_accuracy:.2%}")
        
        if verifier_loaded:
            print(f"🧠 Verifier加权投票准确率: {verifier_accuracy:.2%}")
            if verifier_accuracy > baseline_accuracy:
                improvement = verifier_accuracy - baseline_accuracy
                print(f"🎉 Verifier提升了 {improvement:.2%} 的准确率！")
            elif verifier_accuracy == baseline_accuracy:
                print(f"➡️ Verifier与基线方法效果相同")
            else:
                decline = baseline_accuracy - verifier_accuracy
                print(f"⚠️ Verifier准确率下降了 {decline:.2%}")
        
        print(f"🎯 理想情况（质量感知）准确率: {ideal_accuracy:.2%}")
        
        if verifier_loaded:
            verifier_vs_ideal = verifier_accuracy / ideal_accuracy if ideal_accuracy > 0 else 0
            print(f"📏 Verifier效果是理想情况的 {verifier_vs_ideal:.1%}")
        
        # 保存结果
        results = {
            "experiment_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "test_sample_size": len(test_data),
            "baseline_accuracy": baseline_accuracy,
            "verifier_accuracy": verifier_accuracy if verifier_loaded else None,
            "ideal_accuracy": ideal_accuracy,
            "verifier_loaded": verifier_loaded
        }
        
        results_path = os.path.join(config.experiment.output_dir, "realistic_experiment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 实验结果已保存到: {results_path}")
        
        return results

def main():
    """主函数"""
    experiment = RealisticAQuAExperiment()
    results = experiment.run_experiment()
    
    print("\n🎊 现实实验完成！")
    print("\n📝 实验总结:")
    print("   - 基线方法使用简单多数投票")
    print("   - Verifier方法使用质量评分加权投票")
    print("   - 理想情况假设我们知道每个推理路径的真实质量")
    print("   - 这个实验更接近真实场景中的推理错误分布")
    
    return results

if __name__ == "__main__":
    main() 