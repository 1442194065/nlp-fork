#!/usr/bin/env python3
"""
专门训练Verifier模型的脚本
只专注于verifier训练，不涉及完整DIVERSE实验
"""

import os
import json
import torch
from datetime import datetime

# 导入必要模块
from config import config
from verifier import VerifierTrainer, download_and_prepare_verifier

def check_gpu():
    """检查GPU状态"""
    print("="*50)
    print("🔥 GPU环境检查")
    print("="*50)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU可用: {gpu_name}")
        print(f"✅ GPU内存: {gpu_memory:.1f}GB")
        print(f"✅ CUDA版本: {torch.version.cuda}")
        return True
    else:
        print("⚠️ GPU不可用，将使用CPU训练")
        return False

def load_training_data():
    """加载训练数据"""
    print("\n" + "="*50)
    print("📊 加载训练数据")
    print("="*50)
    
    training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
    
    if not os.path.exists(training_data_path):
        print(f"❌ 未找到训练数据: {training_data_path}")
        print("请先运行完整实验生成训练数据")
        return None
    
    with open(training_data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"✅ 成功加载 {len(training_data)} 个训练实例")
    
    # 统计数据类型
    path_level = sum(1 for item in training_data if not item.get('is_step_level', False))
    step_level = sum(1 for item in training_data if item.get('is_step_level', False))
    
    print(f"📈 路径级实例: {path_level}")
    print(f"🔍 步骤级实例: {step_level}")
    
    return training_data

def train_verifier_model(training_data):
    """训练verifier模型"""
    print("\n" + "="*50)
    print("🚀 开始训练Verifier模型")
    print("="*50)
    
    try:
        # 下载基础模型（如果需要）
        print("检查基础模型...")
        download_and_prepare_verifier()
        
        # 初始化trainer
        print("初始化训练器...")
        verifier_trainer = VerifierTrainer()
        
        print(f"✅ 使用模型: {config.verifier.model_name}")
        print(f"✅ 设备: {verifier_trainer.device}")
        
        # 划分训练/验证集
        split_idx = int(0.8 * len(training_data))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"\n📊 数据划分:")
        print(f"训练样本: {len(train_data)}")
        print(f"验证样本: {len(val_data)}")
        
        # 显示训练配置
        print(f"\n⚙️ 训练配置:")
        print(f"  学习率: {config.verifier.learning_rate}")
        print(f"  批大小: {config.verifier.batch_size}")
        print(f"  训练轮数: {config.verifier.num_epochs}")
        print(f"  最大长度: {config.verifier.max_length}")
        print(f"  步骤感知: {config.diverse.use_step_aware}")
        
        # 开始训练
        print(f"\n🏃‍♂️ 开始训练...")
        print("="*50)
        
        training_history = verifier_trainer.train(train_data, val_data)
        
        print("\n" + "="*50)
        print("🎉 训练完成!")
        print("="*50)
        
        return verifier_trainer, training_history
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_verifier(verifier_trainer):
    """测试训练好的verifier"""
    print("\n" + "="*50)
    print("🧪 测试Verifier模型")
    print("="*50)
    
    if verifier_trainer is None:
        print("❌ 没有可用的verifier模型")
        return
    
    # 简单测试
    test_question = "If 3x + 7 = 22, what is the value of x?"
    test_options = ["A)3", "B)5", "C)7", "D)9", "E)11"]
    
    # 测试正确推理路径
    correct_reasoning = "I need to solve for x. First, subtract 7 from both sides: 3x = 15. Then divide by 3: x = 5."
    
    # 测试错误推理路径
    wrong_reasoning = "I need to solve for x. First, add 7 to both sides: 3x = 29. Then divide by 3: x = 9.67."
    
    try:
        correct_score = verifier_trainer.predict(test_question, test_options, reasoning_path=correct_reasoning)
        wrong_score = verifier_trainer.predict(test_question, test_options, reasoning_path=wrong_reasoning)
        
        print(f"✅ 正确推理路径评分: {correct_score:.3f}")
        print(f"❌ 错误推理路径评分: {wrong_score:.3f}")
        
        if correct_score > wrong_score:
            print("🎯 Verifier能够区分正确和错误的推理！")
        else:
            print("⚠️ Verifier可能需要更多训练")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def save_model_info(verifier_trainer, training_history):
    """保存模型信息"""
    if verifier_trainer is None:
        return
        
    print("\n" + "="*50)
    print("💾 保存模型信息")
    print("="*50)
    
    # 保存模型信息
    model_info = {
        "model_type": config.verifier.model_name,
        "training_config": {
            "learning_rate": config.verifier.learning_rate,
            "batch_size": config.verifier.batch_size,
            "num_epochs": config.verifier.num_epochs,
            "max_length": config.verifier.max_length,
        },
        "device": str(verifier_trainer.device),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "training_completed": True
    }
    
    info_path = os.path.join(config.experiment.model_save_dir, "verifier_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 模型信息已保存到: {info_path}")

def main():
    """主函数"""
    print("="*60)
    print("🧠 VERIFIER 模型训练专用脚本")
    print("="*60)
    
    # 1. 检查GPU
    gpu_available = check_gpu()
    
    # 2. 加载训练数据
    training_data = load_training_data()
    if training_data is None:
        return
    
    # 3. 训练verifier
    verifier_trainer, training_history = train_verifier_model(training_data)
    
    # 4. 测试verifier
    test_verifier(verifier_trainer)
    
    # 5. 保存模型信息
    save_model_info(verifier_trainer, training_history)
    
    print("\n" + "="*60)
    print("🎊 Verifier训练流程完成!")
    print("="*60)
    
    if verifier_trainer is not None:
        print("✅ 训练成功！现在可以在DIVERSE实验中使用这个verifier了。")
    else:
        print("❌ 训练失败，请检查错误信息。")

if __name__ == "__main__":
    main() 