#!/usr/bin/env python3
"""
AQuA数据集Verifier实验
比较采用verifier和不采用verifier的正确率
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
from verifier import VerifierTrainer, download_and_prepare_verifier
from llm_client import DeepSeekClient

class AQuAVerifierExperiment:
    """AQuA数据集上的Verifier实验"""
    
    def __init__(self):
        self.data_loader = AQuADataLoader()
        self.llm_client = DeepSeekClient()
        self.verifier_trainer = None
        self.results = {}
        
        # 实验参数
        self.sample_size = 50  # 测试样本数量
        self.num_reasoning_paths = 5  # 每题生成的推理路径数量
        
        print("🧪 AQuA Verifier实验初始化完成")
        
    def load_data(self):
        """加载AQuA数据"""
        print("\n" + "="*60)
        print("📊 加载AQuA数据集")
        print("="*60)
        
        # 加载训练和测试数据
        train_data = self.data_loader.load_data("train", limit=200)  # 训练用200题
        test_data = self.data_loader.load_data("dev", limit=self.sample_size)  # 测试用50题
        
        print(f"✅ 训练数据: {len(train_data)} 题")
        print(f"✅ 测试数据: {len(test_data)} 题") 
        
        return train_data, test_data
    
    def generate_training_data(self, train_data):
        """生成verifier训练数据"""
        print("\n" + "="*60)
        print("🚀 生成Verifier训练数据")
        print("="*60)
        
        training_instances = []
        
        for i, item in enumerate(train_data):
            print(f"处理题目 {i+1}/{len(train_data)}: {item['question'][:50]}...")
            
            try:
                # 生成多条推理路径
                reasoning_paths = self.llm_client.generate_diverse_reasoning(
                    question=item['question'],
                    options=item['options'],
                    num_paths=self.num_reasoning_paths
                )
                
                # 为每条推理路径创建训练样本
                for j, path in enumerate(reasoning_paths):
                    # 根据推理路径是否得出正确答案来设置标签
                    predicted_answer = self._extract_answer_from_reasoning(path)
                    is_correct = predicted_answer == item['correct']
                    
                    # 路径级训练样本
                    training_instances.append({
                        'question': item['question'],
                        'options': item['options'],
                        'reasoning_path': path,
                        'label': 1 if is_correct else 0,
                        'is_step_level': False
                    })
                    
                    # 步骤级训练样本（将推理路径分解为步骤）
                    steps = self._split_reasoning_into_steps(path)
                    for step_idx, step in enumerate(steps):
                        # 假设前面的步骤都是正确的，最后一步决定整体正确性
                        step_is_correct = is_correct if step_idx == len(steps)-1 else True
                        
                        training_instances.append({
                            'question': item['question'],
                            'options': item['options'],
                            'reasoning_path': path,
                            'reasoning_step': step,
                            'label': 1 if step_is_correct else 0,
                            'is_step_level': True
                        })
                
            except Exception as e:
                print(f"❌ 处理题目 {i+1} 失败: {e}")
                continue
        
        print(f"\n✅ 生成 {len(training_instances)} 个训练样本")
        
        # 保存训练数据
        output_path = os.path.join(config.experiment.output_dir, "aqua_verifier_training_data.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_instances, f, indent=2, ensure_ascii=False)
        
        print(f"💾 训练数据已保存到: {output_path}")
        return training_instances
    
    def train_verifier(self, training_data):
        """训练verifier模型"""
        print("\n" + "="*60)
        print("🏋️ 训练Verifier模型")
        print("="*60)
        
        try:
            # 下载基础模型
            download_and_prepare_verifier()
            
            # 初始化verifier训练器
            self.verifier_trainer = VerifierTrainer()
            
            # 训练模型
            self.verifier_trainer.train(training_data)
            
            print("✅ Verifier训练完成")
            return True
            
        except Exception as e:
            print(f"❌ Verifier训练失败: {e}")
            return False
    
    def evaluate_baseline(self, test_data):
        """评估基线方法（多数投票，不使用verifier）"""
        print("\n" + "="*60)
        print("📈 评估基线方法（多数投票）")
        print("="*60)
        
        correct_count = 0
        total_count = len(test_data)
        
        for i, item in enumerate(test_data):
            print(f"测试题目 {i+1}/{total_count}: {item['question'][:30]}...")
            
            try:
                # 生成多条推理路径
                reasoning_paths = self.llm_client.generate_diverse_reasoning(
                    question=item['question'],
                    options=item['options'],
                    num_paths=self.num_reasoning_paths
                )
                
                # 提取每条路径的答案
                predicted_answers = []
                for path in reasoning_paths:
                    answer = self._extract_answer_from_reasoning(path)
                    if answer:
                        predicted_answers.append(answer)
                
                # 多数投票
                if predicted_answers:
                    answer_counts = Counter(predicted_answers)
                    final_answer = answer_counts.most_common(1)[0][0]
                    
                    if final_answer == item['correct']:
                        correct_count += 1
                        print(f"  ✅ 正确 - 预测: {final_answer}, 实际: {item['correct']}")
                    else:
                        print(f"  ❌ 错误 - 预测: {final_answer}, 实际: {item['correct']}")
                else:
                    print(f"  ⚠️ 无法提取答案")
                
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        
        baseline_accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 基线方法结果:")
        print(f"   正确数量: {correct_count}/{total_count}")
        print(f"   准确率: {baseline_accuracy:.2%}")
        
        return baseline_accuracy
    
    def evaluate_with_verifier(self, test_data):
        """评估使用verifier的方法"""
        print("\n" + "="*60)
        print("🧠 评估使用Verifier的方法")
        print("="*60)
        
        if self.verifier_trainer is None:
            print("❌ Verifier未训练，加载已保存的模型...")
            self.verifier_trainer = VerifierTrainer()
            try:
                self.verifier_trainer.load_model()
            except:
                print("❌ 无法加载Verifier模型")
                return 0
        
        correct_count = 0
        total_count = len(test_data)
        
        for i, item in enumerate(test_data):
            print(f"测试题目 {i+1}/{total_count}: {item['question'][:30]}...")
            
            try:
                # 生成多条推理路径
                reasoning_paths = self.llm_client.generate_diverse_reasoning(
                    question=item['question'],
                    options=item['options'],
                    num_paths=self.num_reasoning_paths
                )
                
                # 使用verifier为每条路径打分
                path_scores = []
                path_answers = []
                
                for path in reasoning_paths:
                    try:
                        # Verifier评分
                        score = self.verifier_trainer.predict(
                            question=item['question'],
                            options=item['options'],
                            reasoning_path=path
                        )
                        
                        # 提取答案
                        answer = self._extract_answer_from_reasoning(path)
                        
                        if answer:
                            path_scores.append(score)
                            path_answers.append(answer)
                    
                    except Exception as e:
                        print(f"    ⚠️ 路径评分失败: {e}")
                        continue
                
                # 根据verifier评分选择最佳答案
                if path_scores and path_answers:
                    # 选择评分最高的路径对应的答案
                    best_idx = np.argmax(path_scores)
                    final_answer = path_answers[best_idx]
                    best_score = path_scores[best_idx]
                    
                    if final_answer == item['correct']:
                        correct_count += 1
                        print(f"  ✅ 正确 - 预测: {final_answer}, 实际: {item['correct']}, 评分: {best_score:.3f}")
                    else:
                        print(f"  ❌ 错误 - 预测: {final_answer}, 实际: {item['correct']}, 评分: {best_score:.3f}")
                else:
                    print(f"  ⚠️ 无法获得有效评分")
                
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        
        verifier_accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 Verifier方法结果:")
        print(f"   正确数量: {correct_count}/{total_count}")
        print(f"   准确率: {verifier_accuracy:.2%}")
        
        return verifier_accuracy
    
    def _extract_answer_from_reasoning(self, reasoning_text):
        """从推理文本中提取答案"""
        # 简单的答案提取逻辑
        for option in ['A', 'B', 'C', 'D', 'E']:
            if f"答案是{option}" in reasoning_text or f"Answer: {option}" in reasoning_text or f"answer is {option}" in reasoning_text:
                return option
        
        # 如果没有明确答案标识，寻找选项标记
        import re
        matches = re.findall(r'\b[A-E]\)', reasoning_text)
        if matches:
            return matches[-1][0]  # 返回最后一个找到的选项
        
        return None
    
    def _split_reasoning_into_steps(self, reasoning_text):
        """将推理文本分解为步骤"""
        # 简单的步骤分割逻辑
        sentences = reasoning_text.split('.')
        steps = [s.strip() for s in sentences if len(s.strip()) > 10]
        return steps[:5]  # 最多5个步骤
    
    def run_experiment(self):
        """运行完整实验"""
        print("="*70)
        print("🎯 AQuA数据集Verifier效果对比实验")
        print("="*70)
        
        # 1. 加载数据
        train_data, test_data = self.load_data()
        
        # 2. 生成训练数据
        training_data = self.generate_training_data(train_data)
        
        # 3. 训练verifier
        verifier_success = self.train_verifier(training_data)
        
        # 4. 评估基线方法
        baseline_accuracy = self.evaluate_baseline(test_data)
        
        # 5. 评估verifier方法（如果训练成功）
        if verifier_success:
            verifier_accuracy = self.evaluate_with_verifier(test_data)
        else:
            print("⚠️ Verifier训练失败，跳过verifier评估")
            verifier_accuracy = 0
        
        # 6. 汇总结果
        print("\n" + "="*70)
        print("📈 实验结果汇总")
        print("="*70)
        
        print(f"📊 基线方法（多数投票）准确率: {baseline_accuracy:.2%}")
        if verifier_success:
            print(f"🧠 Verifier方法准确率: {verifier_accuracy:.2%}")
            if verifier_accuracy > baseline_accuracy:
                improvement = verifier_accuracy - baseline_accuracy
                print(f"🎉 Verifier提升了 {improvement:.2%} 的准确率！")
            else:
                decline = baseline_accuracy - verifier_accuracy
                print(f"⚠️ Verifier准确率下降了 {decline:.2%}")
        
        # 保存结果
        results = {
            "experiment_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "sample_size": self.sample_size,
            "num_reasoning_paths": self.num_reasoning_paths,
            "baseline_accuracy": baseline_accuracy,
            "verifier_accuracy": verifier_accuracy if verifier_success else None,
            "verifier_trained": verifier_success
        }
        
        results_path = os.path.join(config.experiment.output_dir, "aqua_verifier_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 实验结果已保存到: {results_path}")
        
        return results

def main():
    """主函数"""
    experiment = AQuAVerifierExperiment()
    results = experiment.run_experiment()
    
    print("\n🎊 实验完成！")
    return results

if __name__ == "__main__":
    main() 