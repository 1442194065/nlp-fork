#!/usr/bin/env python3
"""
测试训练好的Verifier模型
"""

import os
from verifier import VerifierTrainer

def test_verifier():
    """测试verifier模型的功能"""
    print("="*60)
    print("🧪 测试Verifier模型")
    print("="*60)
    
    # 初始化verifier
    verifier_trainer = VerifierTrainer()
    
    # 加载训练好的模型
    model_path = os.path.join("models", "verifier_model")
    if os.path.exists(model_path):
        print(f"📁 加载模型: {model_path}")
        verifier_trainer.load_model(model_path)
    else:
        print("❌ 未找到训练好的模型")
        return
    
    print(f"✅ 模型加载成功，设备: {verifier_trainer.device}")
    
    # 测试案例
    test_cases = [
        {
            "name": "数学题 - 正确推理",
            "question": "If 2x + 5 = 13, what is the value of x?",
            "options": ["A) 2", "B) 4", "C) 6", "D) 8"],
            "reasoning": "To solve for x: 2x + 5 = 13. Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4.",
            "expected": "高分 (正确推理)"
        },
        {
            "name": "数学题 - 错误推理", 
            "question": "If 2x + 5 = 13, what is the value of x?",
            "options": ["A) 2", "B) 4", "C) 6", "D) 8"],
            "reasoning": "To solve for x: 2x + 5 = 13. Add 5 to both sides: 2x = 18. Divide by 2: x = 9.",
            "expected": "低分 (错误推理)"
        },
        {
            "name": "应用题 - 正确推理",
            "question": "A store sells apples for $2 each and oranges for $3 each. If John buys 4 apples and 2 oranges, how much does he spend?",
            "options": ["A) $10", "B) $12", "C) $14", "D) $16"],
            "reasoning": "Cost of apples: 4 × $2 = $8. Cost of oranges: 2 × $3 = $6. Total cost: $8 + $6 = $14.",
            "expected": "高分 (正确推理)"
        },
        {
            "name": "应用题 - 错误推理",
            "question": "A store sells apples for $2 each and oranges for $3 each. If John buys 4 apples and 2 oranges, how much does he spend?", 
            "options": ["A) $10", "B) $12", "C) $14", "D) $16"],
            "reasoning": "Cost of apples: 4 × $2 = $8. Cost of oranges: 2 × $3 = $6. Total cost: $8 × $6 = $48.",
            "expected": "低分 (错误推理)"
        }
    ]
    
    print("\n" + "="*60)
    print("🔍 开始测试")
    print("="*60)
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}: {case['name']}")
        print(f"问题: {case['question']}")
        print(f"选项: {' '.join(case['options'])}")
        print(f"推理: {case['reasoning']}")
        print(f"期望: {case['expected']}")
        
        try:
            score = verifier_trainer.predict(
                question=case['question'],
                options=case['options'],
                reasoning_path=case['reasoning']
            )
            
            print(f"🎯 Verifier评分: {score:.3f}")
            
            # 判断结果
            if ("正确" in case['expected'] and score > 0.5) or ("错误" in case['expected'] and score <= 0.5):
                result = "✅ 正确"
            else:
                result = "❌ 错误"
            
            print(f"📊 判断结果: {result}")
            results.append((case['name'], score, result))
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results.append((case['name'], 0.0, "❌ 失败"))
        
        print("-" * 50)
    
    # 汇总结果
    print("\n" + "="*60)
    print("📈 测试结果汇总")
    print("="*60)
    
    correct_count = sum(1 for _, _, result in results if "✅" in result)
    total_count = len(results)
    
    for name, score, result in results:
        print(f"{result} {name}: {score:.3f}")
    
    print(f"\n🎊 总体准确率: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    
    if correct_count == total_count:
        print("🌟 所有测试都通过了！Verifier工作正常！")
    elif correct_count >= total_count * 0.75:
        print("👍 大部分测试通过，Verifier表现良好！")
    else:
        print("⚠️ 部分测试失败，可能需要更多训练数据或调整参数")

if __name__ == "__main__":
    test_verifier() 