#!/usr/bin/env python3
"""
改进版Verifier训练器
目标：提高verifier的推理路径判断准确率
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
    TrainingArguments, Trainer, 
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import logging

# 导入配置
from config import config
from data_loader import AQuADataLoader

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedVerifierModel(nn.Module):
    """改进的Verifier模型架构"""
    
    def __init__(self, model_name="roberta-base", dropout_rate=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 多层分类头
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(self.config.hidden_size, 512)
        self.activation1 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(dropout_rate/2)
        self.dense2 = nn.Linear(512, 128)
        self.activation2 = nn.ReLU()
        
        self.dropout3 = nn.Dropout(dropout_rate/3)
        self.classifier = nn.Linear(128, 2)  # 二分类
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类头权重"""
        for module in [self.dense1, self.dense2, self.classifier]:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用CLS token的表示
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # 多层前向传播
        x = self.dropout1(pooled_output)
        x = self.activation1(self.dense1(x))
        
        x = self.dropout2(x)
        x = self.activation2(self.dense2(x))
        
        x = self.dropout3(x)
        logits = self.classifier(x)
        
        return logits

class VerifierDataset(Dataset):
    """改进的Verifier数据集"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构造输入文本：[问题] [SEP] [推理路径]
        question = item['question']
        options = " ".join(item['options']) if isinstance(item['options'], list) else str(item['options'])
        reasoning = item.get('reasoning_path', item.get('reasoning_step', ''))
        
        # 组合文本
        text = f"问题: {question} 选项: {options} [SEP] 推理: {reasoning}"
        
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
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ImprovedVerifierTrainer:
    """改进版Verifier训练器"""
    
    def __init__(self, model_name="roberta-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练参数
        self.training_config = {
            'learning_rate': 2e-5,
            'num_epochs': 5,
            'batch_size': 8,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'early_stopping_patience': 3,
            'gradient_accumulation_steps': 2
        }
        
        print(f"🚀 初始化改进版Verifier训练器")
        print(f"   模型: {model_name}")
        print(f"   设备: {self.device}")
    
    def _get_label(self, item):
        """获取数据项的标签"""
        if 'label' in item:
            return item['label']
        elif 'step_label' in item:
            return item['step_label']
        else:
            raise ValueError(f"数据项没有标签字段: {item}")
    
    def prepare_enhanced_training_data(self, original_data, augment_factor=3):
        """准备增强的训练数据"""
        print(f"\n📊 准备增强训练数据 (增强倍数: {augment_factor})")
        
        enhanced_data = []
        
        # 1. 原始数据
        enhanced_data.extend(original_data)
        
        # 2. 数据增强
        for _ in range(augment_factor - 1):
            for item in original_data:
                # 创建变种
                augmented_item = self._augment_training_instance(item)
                enhanced_data.append(augmented_item)
        
        # 3. 平衡数据集
        enhanced_data = self._balance_dataset(enhanced_data)
        
        print(f"✅ 增强后数据量: {len(enhanced_data)}")
        return enhanced_data
    
    def _augment_training_instance(self, item):
        """数据增强：创建训练实例的变种"""
        augmented = item.copy()
        
        # 对推理路径进行轻微变换
        reasoning = item.get('reasoning_path', item.get('reasoning_step', ''))
        
        # 简单的文本变换
        variations = [
            reasoning,
            reasoning.replace('计算', '运算'),
            reasoning.replace('得出', '求得'),
            reasoning.replace('答案是', '结果是'),
            reasoning.replace('因此', '所以')
        ]
        
        augmented_reasoning = random.choice(variations)
        
        if 'reasoning_path' in augmented:
            augmented['reasoning_path'] = augmented_reasoning
        else:
            augmented['reasoning_step'] = augmented_reasoning
        
        # 确保标签字段一致
        if 'label' not in augmented and 'step_label' in augmented:
            augmented['label'] = augmented['step_label']
        elif 'step_label' not in augmented and 'label' in augmented:
            augmented['step_label'] = augmented['label']
        
        return augmented
    
    def _balance_dataset(self, data):
        """平衡数据集：确保正负样本比例合理"""
        positive_samples = [item for item in data if self._get_label(item) == 1]
        negative_samples = [item for item in data if self._get_label(item) == 0]
        
        print(f"   原始正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")
        
        # 如果负样本过多，进行下采样
        if len(negative_samples) > len(positive_samples) * 2:
            negative_samples = random.sample(negative_samples, len(positive_samples) * 2)
        
        # 如果正样本过少，进行上采样
        elif len(positive_samples) < len(negative_samples) * 0.5:
            additional_positive = random.choices(positive_samples, k=int(len(negative_samples) * 0.5) - len(positive_samples))
            positive_samples.extend(additional_positive)
        
        balanced_data = positive_samples + negative_samples
        random.shuffle(balanced_data)
        
        final_positive = len([item for item in balanced_data if self._get_label(item) == 1])
        final_negative = len([item for item in balanced_data if self._get_label(item) == 0])
        print(f"   平衡后正样本: {final_positive}, 负样本: {final_negative}")
        
        return balanced_data
    
    def train(self, training_data, validation_split=0.2):
        """改进的训练流程"""
        print(f"\n🏋️ 开始改进版Verifier训练")
        
        # 1. 数据增强
        enhanced_data = self.prepare_enhanced_training_data(training_data, augment_factor=3)
        
        # 2. 训练验证集分割
        random.shuffle(enhanced_data)
        split_idx = int(len(enhanced_data) * (1 - validation_split))
        train_data = enhanced_data[:split_idx]
        val_data = enhanced_data[split_idx:]
        
        print(f"   训练集: {len(train_data)} 样本")
        print(f"   验证集: {len(val_data)} 样本")
        
        # 3. 初始化模型和分词器
        print(f"\n🔧 初始化模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ImprovedVerifierModel(self.model_name)
        self.model.to(self.device)
        
        # 4. 创建数据集
        train_dataset = VerifierDataset(train_data, self.tokenizer)
        val_dataset = VerifierDataset(val_data, self.tokenizer)
        
        # 5. 训练参数
        output_dir = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config['num_epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            warmup_ratio=self.training_config['warmup_ratio'],
            weight_decay=self.training_config['weight_decay'],
            logging_dir=config.experiment.log_dir,
            logging_steps=50,
            eval_steps=100,
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,  # 禁用wandb
            save_total_limit=3,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # 使用混合精度训练
        )
        
        # 6. 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.training_config['early_stopping_patience'])]
        )
        
        # 7. 开始训练
        print(f"\n🚀 开始训练...")
        train_result = self.trainer.train()
        
        # 8. 保存模型
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # 9. 评估
        eval_result = self.trainer.evaluate()
        
        print(f"\n✅ 训练完成!")
        print(f"   最终训练损失: {train_result.training_loss:.4f}")
        print(f"   最终验证准确率: {eval_result['eval_accuracy']:.4f}")
        print(f"   最终验证F1: {eval_result['eval_f1']:.4f}")
        
        # 保存训练信息
        training_info = {
            "model_name": self.model_name,
            "training_config": self.training_config,
            "training_loss": train_result.training_loss,
            "eval_accuracy": eval_result['eval_accuracy'],
            "eval_f1": eval_result['eval_f1'],
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "training_time": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        info_path = os.path.join(output_dir, "training_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        return training_info
    
    def _compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def load_model(self, model_path=None):
        """加载训练好的模型"""
        if model_path is None:
            model_path = os.path.join(config.experiment.model_save_dir, "improved_verifier_model")
        
        print(f"🔄 加载改进版模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ImprovedVerifierModel(self.model_name)
        
        # 加载模型权重
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载成功，设备: {self.device}")
    
    def predict(self, question, options, reasoning_path):
        """预测推理路径质量"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()或train()")
        
        # 构造输入文本
        options_str = " ".join(options) if isinstance(options, list) else str(options)
        text = f"问题: {question} 选项: {options_str} [SEP] 推理: {reasoning_path}"
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # 移到设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # 预测
        with torch.no_grad():
            logits = self.model(**encoding)
            probabilities = torch.softmax(logits, dim=-1)
            # 返回"正确"类别的概率
            score = probabilities[0][1].item()
        
        return score

def main():
    """主函数：训练改进版verifier"""
    print("="*70)
    print("🎯 改进版Verifier训练")
    print("="*70)
    
    # 1. 加载训练数据
    training_data_path = os.path.join(config.experiment.output_dir, "verifier_training_data.json")
    
    if not os.path.exists(training_data_path):
        print(f"❌ 未找到训练数据: {training_data_path}")
        return
    
    with open(training_data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"✅ 加载训练数据: {len(training_data)} 样本")
    
    # 2. 初始化改进版训练器
    trainer = ImprovedVerifierTrainer(model_name="roberta-base")
    
    # 3. 训练模型
    training_info = trainer.train(training_data)
    
    print(f"\n🎉 改进版Verifier训练完成!")
    print(f"   模型保存位置: {config.experiment.model_save_dir}/improved_verifier_model")
    
    return training_info

if __name__ == "__main__":
    main() 