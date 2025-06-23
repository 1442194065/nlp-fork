#!/usr/bin/env python3
"""
简化版改进Verifier训练器
专注于核心改进，减少复杂度
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score
from datetime import datetime

from config import config

class SimpleImprovedVerifier(nn.Module):
    """简化版改进Verifier模型"""
    
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # 简化的分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class SimpleVerifierDataset(Dataset):
    """简化版数据集"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构造输入文本
        question = item['question']
        reasoning = item.get('reasoning_path', item.get('reasoning_step', ''))
        text = f"{question} [SEP] {reasoning}"
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 获取标签
        label = item.get('label', item.get('step_label', 0))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def main():
    """主函数：简化版训练"""
    print("="*60)
    print("🎯 简化版改进Verifier训练")
    print("="*60)
    
    # 1. 加载训练数据
    training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
    
    if not os.path.exists(training_data_path):
        print(f"❌ 未找到训练数据: {training_data_path}")
        return
    
    with open(training_data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"✅ 加载训练数据: {len(training_data)} 样本")
    
    # 2. 简单数据增强：复制2倍数据
    enhanced_data = training_data * 2
    random.shuffle(enhanced_data)
    
    # 3. 训练验证集分割
    split_idx = int(len(enhanced_data) * 0.8)
    train_data = enhanced_data[:split_idx]
    val_data = enhanced_data[split_idx:]
    
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(val_data)} 样本")
    
    # 4. 初始化模型和分词器
    print(f"\n🔧 初始化模型...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SimpleImprovedVerifier(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")
    
    # 5. 创建数据集
    train_dataset = SimpleVerifierDataset(train_data, tokenizer)
    val_dataset = SimpleVerifierDataset(val_data, tokenizer)
    
    # 6. 训练参数（简化）
    output_dir = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=None,
        save_total_limit=2,
        dataloader_pin_memory=False,
    )
    
    # 7. 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 8. 开始训练
    print(f"\n🚀 开始训练...")
    train_result = trainer.train()
    
    # 9. 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 10. 评估
    eval_result = trainer.evaluate()
    
    print(f"\n✅ 训练完成!")
    print(f"   最终训练损失: {train_result.training_loss:.4f}")
    print(f"   最终验证准确率: {eval_result['eval_accuracy']:.4f}")
    
    # 保存训练信息
    training_info = {
        "model_name": model_name,
        "training_loss": train_result.training_loss,
        "eval_accuracy": eval_result['eval_accuracy'],
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "training_time": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 改进版Verifier训练完成!")
    print(f"   模型保存位置: {output_dir}")
    
    return training_info

if __name__ == "__main__":
    main() 