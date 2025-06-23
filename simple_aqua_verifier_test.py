#!/usr/bin/env python3
"""
简化的AQuA Verifier测试
使用已有的训练数据测试verifier效果
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

class SimpleAQuATest:
    """简化的AQuA测试"""
    
    def __init__(self):
        self.data_loader = AQuADataLoader()
        self.verifier_trainer = None
        
        print("🧪 简化AQuA Verifier测试初始化完成")
    
    def load_existing_training_data(self):
        """加载已存在的训练数据"""
        print("\n" + "="*60)
        print("📊 加载已有的训练数据")
        print("="*60)
        
        training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
        
        if not os.path.exists(training_data_path):
            print(f"❌ 未找到训练数据: {training_data_path}")
            return None
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"✅ 加载了 {len(training_data)} 个训练样本")
        return training_data
    
    def load_test_data(self):
        """加载测试数据"""
        print("\n" + "="*60)
        print("📊 加载AQuA测试数据")
        print("="*60)
        
        # 加载AQuA dev数据作为测试
        self.data_loader.load_data("dev")  
        test_data = self.data_loader.get_eval_subset("dev", max_samples=20)  # 只测试20题
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
    
    def train_verifier_if_needed(self, training_data):
        """如果需要，训练verifier"""
        print("\n" + "="*60)
        print("🏋️ 检查Verifier模型")
        print("="*60)
        
        # 检查是否已有训练好的模型
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
                # 如果加载失败，重新训练
        
        # 如果没有模型或加载失败，重新训练
        print("🚀 开始训练新的verifier模型...")
        try:
            self.verifier_trainer = VerifierTrainer()
            self.verifier_trainer.train(training_data)
            print("✅ Verifier训练完成")
            return True
        except Exception as e:
            print(f"❌ Verifier训练失败: {e}")
            return False
    
    def simulate_reasoning_paths(self, question, options, correct_answer):
        """模拟生成多条推理路径（用于演示）"""
        # 这里我们模拟生成5条推理路径
        paths = []
        
        # 模拟正确推理路径
        correct_path = f"分析问题: {question[:50]}... 通过计算得出答案是 {correct_answer}。"
        paths.append({'reasoning': correct_path, 'answer': correct_answer})
        
        # 模拟错误推理路径
        wrong_answers = [opt for opt in ['A', 'B', 'C', 'D', 'E'] if opt != correct_answer]
        for i, wrong_ans in enumerate(random.sample(wrong_answers, min(3, len(wrong_answers)))):
            wrong_path = f"计算过程{i+1}: {question[:30]}... 得出答案是 {wrong_ans}。"
            paths.append({'reasoning': wrong_path, 'answer': wrong_ans})
        
        # 再添加一个正确路径（不同推理）
        if len(paths) < 5:
            alt_correct_path = f"换个思路: {question[:40]}... 最终答案是 {correct_answer}。"
            paths.append({'reasoning': alt_correct_path, 'answer': correct_answer})
        
        return paths
    
    def evaluate_baseline(self, test_data):
        """评估基线方法（多数投票）"""
        print("\n" + "="*60)
        print("📈 评估基线方法（多数投票）")
        print("="*60)
        
        correct_count = 0
        total_count = len(test_data)
        
        for i, item in enumerate(test_data):
            print(f"测试题目 {i+1}/{total_count}: {item['question'][:30]}...")
            
            # 模拟生成多条推理路径
            paths = self.simulate_reasoning_paths(
                item['question'], 
                item['options'], 
                item['correct']
            )
            
            # 提取答案并进行多数投票
            answers = [path['answer'] for path in paths]
            answer_counts = Counter(answers)
            predicted_answer = answer_counts.most_common(1)[0][0]
            
            if predicted_answer == item['correct']:
                correct_count += 1
                print(f"  ✅ 正确 - 预测: {predicted_answer}, 实际: {item['correct']}")
            else:
                print(f"  ❌ 错误 - 预测: {predicted_answer}, 实际: {item['correct']}")
            
            print(f"    路径答案: {answers}")
        
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
            print("❌ Verifier未加载")
            return 0
        
        correct_count = 0
        total_count = len(test_data)
        
        for i, item in enumerate(test_data):
            print(f"测试题目 {i+1}/{total_count}: {item['question'][:30]}...")
            
            # 模拟生成多条推理路径
            paths = self.simulate_reasoning_paths(
                item['question'], 
                item['options'], 
                item['correct']
            )
            
            # 使用verifier为每条路径打分
            path_scores = []
            path_answers = []
            
            for path in paths:
                try:
                    score = self.verifier_trainer.predict(
                        question=item['question'],
                        options=item['options'],
                        reasoning_path=path['reasoning']
                    )
                    path_scores.append(score)
                    path_answers.append(path['answer'])
                except Exception as e:
                    print(f"    ⚠️ 路径评分失败: {e}")
                    continue
            
            if path_scores and path_answers:
                # 选择评分最高的路径
                best_idx = np.argmax(path_scores)
                predicted_answer = path_answers[best_idx]
                best_score = path_scores[best_idx]
                
                if predicted_answer == item['correct']:
                    correct_count += 1
                    print(f"  ✅ 正确 - 预测: {predicted_answer}, 实际: {item['correct']}, 评分: {best_score:.3f}")
                else:
                    print(f"  ❌ 错误 - 预测: {predicted_answer}, 实际: {item['correct']}, 评分: {best_score:.3f}")
                
                print(f"    路径评分: {[f'{ans}:{score:.3f}' for ans, score in zip(path_answers, path_scores)]}")
            else:
                print(f"  ⚠️ 无法获得有效评分")
        
        verifier_accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 Verifier方法结果:")
        print(f"   正确数量: {correct_count}/{total_count}")
        print(f"   准确率: {verifier_accuracy:.2%}")
        
        return verifier_accuracy
    
    def run_test(self):
        """运行测试"""
        print("="*70)
        print("🎯 简化AQuA Verifier效果测试")
        print("="*70)
        
        # 1. 加载已有训练数据
        training_data = self.load_existing_training_data()
        if training_data is None:
            print("❌ 无法进行测试，缺少训练数据")
            return
        
        # 2. 加载测试数据
        test_data = self.load_test_data()
        
        # 3. 训练或加载verifier
        verifier_success = self.train_verifier_if_needed(training_data)
        
        # 4. 评估基线方法
        baseline_accuracy = self.evaluate_baseline(test_data)
        
        # 5. 评估verifier方法
        if verifier_success:
            verifier_accuracy = self.evaluate_with_verifier(test_data)
        else:
            print("⚠️ Verifier加载失败，跳过verifier评估")
            verifier_accuracy = 0
        
        # 6. 汇总结果
        print("\n" + "="*70)
        print("📈 测试结果汇总")
        print("="*70)
        
        print(f"📊 基线方法（多数投票）准确率: {baseline_accuracy:.2%}")
        if verifier_success:
            print(f"🧠 Verifier方法准确率: {verifier_accuracy:.2%}")
            if verifier_accuracy > baseline_accuracy:
                improvement = verifier_accuracy - baseline_accuracy
                print(f"🎉 Verifier提升了 {improvement:.2%} 的准确率！")
            elif verifier_accuracy == baseline_accuracy:
                print(f"➡️ Verifier与基线方法效果相同")
            else:
                decline = baseline_accuracy - verifier_accuracy
                print(f"⚠️ Verifier准确率下降了 {decline:.2%}")
        
        # 保存结果
        results = {
            "test_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "baseline_accuracy": baseline_accuracy,
            "verifier_accuracy": verifier_accuracy if verifier_success else None,
            "verifier_loaded": verifier_success,
            "test_sample_size": len(test_data)
        }
        
        results_path = os.path.join(config.experiment.output_dir, "simple_test_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 测试结果已保存到: {results_path}")
        
        return results

def main():
    """主函数"""
    test = SimpleAQuATest()
    results = test.run_test()
    
    print("\n🎊 测试完成！")
    return results

if __name__ == "__main__":
    main() 