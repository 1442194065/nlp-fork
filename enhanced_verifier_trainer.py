#!/usr/bin/env python3
"""
增强版Verifier训练器
专门解决区分度低的问题，提升verifier效果
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score
from datetime import datetime

from config import config

class ContrastiveVerifierModel(nn.Module):
    """对比学习增强的Verifier模型"""
    
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # 增强的分类头，提高区分度
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(self.config.hidden_size, 256)
        self.activation1 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(256, 64)
        self.activation2 = nn.ReLU()
        
        self.classifier = nn.Linear(64, 1)  # 回归输出，增加表达范围
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.dense1, self.dense2, self.classifier]:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # 多层前向传播
        x = self.dropout1(pooled_output)
        x = self.activation1(self.dense1(x))
        
        x = self.dropout2(x)
        x = self.activation2(self.dense2(x))
        
        # 回归输出，使用sigmoid压缩到[0,1]
        logits = torch.sigmoid(self.classifier(x))
        
        loss = None
        if labels is not None:
            # 使用MSE损失增强区分度
            labels = labels.float().unsqueeze(-1)
            loss = F.mse_loss(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class ContrastiveDataset(Dataset):
    """对比学习数据集"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 创建高质量标签映射
        self.quality_labels = {
            0: 0.0,  # 低质量
            1: 1.0   # 高质量
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构造输入文本
        question = item['question']
        reasoning = item.get('reasoning_path', item.get('reasoning_step', ''))
        text = f"问题: {question} 推理: {reasoning}"
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 获取标签并转换为回归目标
        original_label = item.get('label', item.get('step_label', 0))
        quality_score = self.quality_labels.get(original_label, 0.0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(quality_score, dtype=torch.float)
        }

def create_enhanced_training_data():
    """创建增强的训练数据，专注于提高区分度"""
    
    # 加载原始数据
    training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
    
    if not os.path.exists(training_data_path):
        print(f"❌ 未找到训练数据: {training_data_path}")
        return []
    
    with open(training_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"✅ 加载原始训练数据: {len(original_data)} 样本")
    
    # 创建对比性数据
    enhanced_data = []
    
    # 1. 保留原始数据
    enhanced_data.extend(original_data)
    
    # 2. 添加明显对比的高质量推理
    high_quality_examples = [
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning_path': '根据方程 x + 5 = 12，两边同时减去5，得到 x = 12 - 5 = 7。因此答案是A)7。',
            'label': 1,
            'is_step_level': False
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning_path': '总价格是20元，书的数量是4本。每本书的价格 = 总价格 ÷ 数量 = 20 ÷ 4 = 5元。答案是B)5元。',
            'label': 1,
            'is_step_level': False
        },
        {
            'question': '一个班级有30个学生，其中60%是女生，有多少个男生？',
            'options': ['A)12', 'B)18', 'C)15', 'D)10', 'E)20'],
            'reasoning_path': '女生人数 = 30 × 60% = 30 × 0.6 = 18人。男生人数 = 总人数 - 女生人数 = 30 - 18 = 12人。答案是A)12。',
            'label': 1,
            'is_step_level': False
        }
    ]
    
    # 3. 添加明显对比的低质量推理
    low_quality_examples = [
        {
            'question': '如果 x + 5 = 12，那么 x 等于多少？',
            'options': ['A)7', 'B)17', 'C)6', 'D)8', 'E)9'],
            'reasoning_path': '我觉得答案是C，因为6这个数字看起来比较合适。',
            'label': 0,
            'is_step_level': False
        },
        {
            'question': '买4本书花了20元，每本书多少钱？',
            'options': ['A)4元', 'B)5元', 'C)6元', 'D)3元', 'E)8元'],
            'reasoning_path': '随便猜一个，我选择A)4元。',
            'label': 0,
            'is_step_level': False
        },
        {
            'question': '一个班级有30个学生，其中60%是女生，有多少个男生？',
            'options': ['A)12', 'B)18', 'C)15', 'D)10', 'E)20'],
            'reasoning_path': '感觉应该是15个，答案选C。',
            'label': 0,
            'is_step_level': False
        }
    ]
    
    # 4. 大量复制对比样本以增强学习
    for _ in range(50):  # 增加50倍的对比样本
        enhanced_data.extend(high_quality_examples)
        enhanced_data.extend(low_quality_examples)
    
    # 5. 平衡数据集
    positive_samples = [item for item in enhanced_data if item.get('label', item.get('step_label', 0)) == 1]
    negative_samples = [item for item in enhanced_data if item.get('label', item.get('step_label', 0)) == 0]
    
    print(f"   正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")
    
    # 确保样本数量平衡
    min_samples = min(len(positive_samples), len(negative_samples))
    balanced_data = random.sample(positive_samples, min_samples) + random.sample(negative_samples, min_samples)
    random.shuffle(balanced_data)
    
    print(f"✅ 增强后数据量: {len(balanced_data)} (平衡后)")
    return balanced_data

def compute_enhanced_metrics(eval_pred):
    """计算增强的评估指标"""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # 计算回归指标
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    # 计算分类指标（使用0.5阈值）
    pred_binary = (predictions > 0.5).astype(int)
    label_binary = (labels > 0.5).astype(int)
    accuracy = accuracy_score(label_binary, pred_binary)
    
    # 计算区分度
    high_quality_preds = predictions[labels > 0.5]
    low_quality_preds = predictions[labels <= 0.5]
    
    if len(high_quality_preds) > 0 and len(low_quality_preds) > 0:
        discrimination = np.mean(high_quality_preds) - np.mean(low_quality_preds)
    else:
        discrimination = 0.0
    
    return {
        'accuracy': accuracy,
        'mse': mse,
        'mae': mae,
        'discrimination': discrimination
    }

def main():
    """主函数：训练增强版verifier"""
    print("="*70)
    print("🚀 增强版Verifier训练 - 专注提升区分度")
    print("="*70)
    
    # 1. 创建增强训练数据
    enhanced_data = create_enhanced_training_data()
    
    if len(enhanced_data) == 0:
        print("❌ 无法创建训练数据")
        return
    
    # 2. 训练验证集分割
    split_idx = int(len(enhanced_data) * 0.8)
    train_data = enhanced_data[:split_idx]
    val_data = enhanced_data[split_idx:]
    
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(val_data)} 样本")
    
    # 3. 初始化模型和分词器
    print(f"\n🔧 初始化增强模型...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ContrastiveVerifierModel(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")
    
    # 4. 创建数据集
    train_dataset = ContrastiveDataset(train_data, tokenizer)
    val_dataset = ContrastiveDataset(val_data, tokenizer)
    
    # 5. 训练参数（针对对比学习优化）
    output_dir = os.path.join(config.experiment.model_save_dir, "enhanced_verifier_model")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # 增加训练轮数
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,  # 降低学习率，精细调优
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=50,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_discrimination",
        greater_is_better=True,
        report_to=None,
        save_total_limit=2,
        dataloader_pin_memory=False,
    )
    
    # 6. 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_enhanced_metrics,
    )
    
    # 7. 开始训练
    print(f"\n🚀 开始增强训练...")
    train_result = trainer.train()
    
    # 8. 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 9. 评估
    eval_result = trainer.evaluate()
    
    print(f"\n✅ 增强训练完成!")
    print(f"   最终训练损失: {train_result.training_loss:.4f}")
    print(f"   最终验证准确率: {eval_result['eval_accuracy']:.4f}")
    print(f"   最终区分度: {eval_result['eval_discrimination']:.4f}")
    print(f"   MSE: {eval_result['eval_mse']:.4f}")
    
    # 保存训练信息
    training_info = {
        "model_name": model_name,
        "model_type": "ContrastiveVerifierModel",
        "training_loss": train_result.training_loss,
        "eval_accuracy": eval_result['eval_accuracy'],
        "eval_discrimination": eval_result['eval_discrimination'],
        "eval_mse": eval_result['eval_mse'],
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "training_time": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 增强版Verifier训练完成!")
    print(f"   模型保存位置: {output_dir}")
    
    # 检查区分度改进
    if eval_result['eval_discrimination'] > 0.2:
        print(f"   ✅ 区分度显著提升！({eval_result['eval_discrimination']:.3f})")
    else:
        print(f"   ⚠️  区分度仍需改进 ({eval_result['eval_discrimination']:.3f})")
    
    return training_info

if __name__ == "__main__":
    main() 