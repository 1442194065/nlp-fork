# DIVERSE 思维链推理项目

本项目实现了基于大语言模型的 DIVERSE (Diverse Verifier on Reasoning Step) 方法，结合 Self-Consistency 和 Step-Aware Verifier 来提升 AQuA 数学推理数据集的准确率。项目使用 DeepSeek API 进行语言模型推理，并训练基于 RoBERTa 的 verifier 来改善推理准确性。

## 🎯 项目概述

DIVERSE 通过三个关键组件增强思维链推理：
1. **多样化提示**: 使用不同的提示模板生成多个推理路径
2. **投票验证器**: 训练 verifier 模型对推理路径质量进行评分
3. **步骤感知验证**: 对单个推理步骤进行细粒度质量评估

## 📁 项目结构

```
nlp/
├── 核心算法文件
│   ├── config.py                           # 配置设置和超参数
│   ├── data_loader.py                      # AQuA 数据集加载和预处理
│   ├── llm_client.py                       # DeepSeek API 客户端
│   ├── diverse_cot.py                      # DIVERSE 核心实现
│   ├── verifier.py                         # 基础 verifier 模型
│   └── main.py                             # 主执行脚本
│
├── 增强版 Verifier 实现
│   ├── enhanced_verifier_trainer.py        # 增强版训练器（核心改进）
│   ├── improved_verifier.py                # 改进版 verifier
│   └── train_improved_verifier_simple.py   # 简化训练脚本
│
├── 测试和验证脚本
│   ├── test_enhanced_verifier.py           # 增强版测试脚本
│   ├── aqua_verifier_final_test.py         # AQuA 数据集最终测试
│   ├── verifier_diagnosis_debug.py         # 诊断调试脚本
│   ├── quick_enhanced_test.py              # 快速验证脚本
│   └── simple_aqua_verifier_test.py        # 简单测试脚本
│
├── 实验和比较脚本
│   ├── verifier_comparison_experiment.py   # Verifier 对比实验
│   ├── final_verifier_comparison.py        # 最终对比实验
│   ├── realistic_aqua_experiment.py        # 真实实验
│   └── 真实实验.py                         # 中文版真实实验
│
├── 数据和模型目录
│   ├── AQuA-master/                        # AQuA 数据集
│   ├── models/                             # 训练的模型文件
│   │   ├── enhanced_verifier_model/        # 增强版 verifier 模型
│   │   ├── improved_verifier_model/        # 改进版 verifier 模型
│   │   └── verifier_model/                 # 基础 verifier 模型
│   ├── outputs/                            # 实验输出和结果
│   ├── results/                            # 评估结果
│   └── logs/                               # 训练日志
│
├── 配置和环境
│   ├── requirements.txt                    # Python 依赖
│   ├── nlp_env/                           # 虚拟环境
│   └── run_*.sh                           # 运行脚本
│
└── 文档和记录
    ├── README.md                          # 本文件
    ├── 增强版Verifier实验记录.md            # 详细实验记录
    ├── 实验报告.md                        # 实验报告
    └── wandb/                             # 训练监控记录
```

## 🚀 快速开始

### 环境要求

1. Python 3.8 或更高版本
2. CUDA GPU（推荐用于 verifier 训练）
3. DeepSeek API 访问权限

### 安装步骤

1. 克隆或下载本项目
2. 激活虚拟环境并安装依赖：
   ```bash
   # Windows WSL 环境
   wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && pip install -r requirements.txt"
   ```

3. 确保 AQuA 数据集在 `AQuA-master/` 目录中

### 运行完整流程

运行完整的 DIVERSE 流程：

```bash
# 激活环境并运行主程序
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && python main.py"
```

这将执行：
1. 测试 DeepSeek API 连接
2. 加载 AQuA 数据集
3. 使用多样化推理路径生成 verifier 训练数据
4. 训练 verifier 模型
5. 评估 DIVERSE vs 基线 CoT 方法
6. 将结果保存到 `results/` 目录

## 🔬 增强版 Verifier

### 核心改进

本项目实现了增强版 Verifier，主要改进包括：

1. **对比学习架构**：
   - 多层分类头 (768→256→64→1)
   - 回归输出 + Sigmoid 激活
   - Xavier 权重初始化

2. **数据增强策略**：
   - 50倍明显对比样本增强
   - 平衡正负样本分布
   - 高质量vs低质量推理对比

3. **训练优化**：
   - MSE 损失函数增强数值区分度
   - 区分度导向的最优模型选择
   - 5个epoch深度训练

### 关键实验结果

| 指标 | 原版Verifier | 增强版Verifier | 改进幅度 |
|------|-------------|----------------|----------|
| **区分度** | -0.017 | 0.455~0.970 | **提升26-57倍** |
| **评分范围** | 0.032 | 0.331~0.971 | **扩展10-30倍** |
| **准确率影响** | -10.0% | 0.0% | **消除负面影响** |

## 🛠 配置说明

主要设置可在 `config.py` 中修改：

### 模型配置
- `deepseek_api_key`: DeepSeek API 密钥
- `model_name`: 使用的 DeepSeek 模型（默认："deepseek-chat"）
- `temperature`: 采样温度（默认：0.5）

### DIVERSE 配置
- `num_prompts`: 不同提示数量（M1 = 5）
- `num_samples_per_prompt`: 每个提示的样本数（M2 = 20）
- `use_step_aware`: 启用步骤感知验证（默认：True）

### 训练配置
- `max_train_samples`: Verifier 训练样本数（默认：1000）
- `max_eval_samples`: 评估样本数（默认：500）
- `batch_size`: 训练批大小（默认：16）
- `num_epochs`: 训练轮数（默认：5）

## 📋 使用选项

### 主要测试脚本

```bash
# 增强版 Verifier 测试
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && python test_enhanced_verifier.py"

# AQuA 数据集最终测试
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && python aqua_verifier_final_test.py"

# 快速验证测试
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && python quick_enhanced_test.py"

# Verifier 诊断
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && python verifier_diagnosis_debug.py"
```

### 训练脚本

```bash
# 训练增强版 Verifier
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && python enhanced_verifier_trainer.py"

# 简化训练流程
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && python train_improved_verifier_simple.py"
```

## 🔬 方法详解

### 1. 多样化提示
系统使用5种不同的提示模板生成多样化推理方法：
- 系统性逐步分析
- 仔细问题分解
- 数学应用题分析
- 系统性解决方案
- 逐步问题处理

### 2. Verifier 训练
- 基础模型：RoBERTa-base
- 训练数据：生成的推理路径与正确性标签
- 架构：对比学习的多层回归头
- 损失函数：MSE 损失增强数值区分度

### 3. 投票机制
- **多数投票**: 跨推理路径的简单投票计数
- **Verifier 投票**: 使用 verifier 置信度分数的加权投票
- **步骤感知投票**: 结合路径级和步骤级验证分数

## 📊 预期结果

基于增强版 Verifier 的实验结果：
- 基线 CoT 准确率：约92%（AQuA测试集）
- 增强版 DIVERSE：92%（与基线持平）
- 关键改进：区分度从负值提升到0.45+，评分范围扩展10-30倍

## 🗂 输出文件

### 结果目录 (`outputs/`)
- `aqua_verifier_final_test_*.json`: AQuA 最终测试结果
- `quick_enhanced_test_*.json`: 快速测试结果
- `verifier_comparison_*.json`: Verifier 对比结果
- `verifier_training_data.json`: Verifier 训练数据

### 模型目录 (`models/`)
- `enhanced_verifier_model/`: 增强版 verifier 模型文件
- `improved_verifier_model/`: 改进版 verifier 模型
- `verifier_model/`: 基础 verifier 模型

## 🐛 故障排除

### 常见问题

**API 连接失败:**
- 检查 `config.py` 中的 DeepSeek API 密钥
- 验证网络连接
- 确保 API 密钥有足够的额度

**CUDA 内存不足:**
- 减少 `config.py` 中的 `batch_size`
- 设置 CUDA_VISIBLE_DEVICES="" 使用CPU训练
- 减少 `max_train_samples`

**依赖缺失:**
```bash
wsl bash -c "cd /c/Users/14421/nlp && source nlp_env/bin/activate && pip install torch transformers datasets accelerate"
```

**数据集未找到:**
- 确保 `AQuA-master/` 目录包含数据集文件
- 检查 `train.json`, `dev.json`, 和 `test.json` 是否存在

## 📈 性能优化

### 加快训练速度:
- 使用具有足够显存的 GPU（推荐8GB+）
- 减少 `max_train_samples` 进行快速实验
- 首次运行后使用 `--skip-data-generation`

### 提高准确率:
- 增加 `num_prompts` 和 `num_samples_per_prompt`
- 增加 verifier 训练的 `max_train_samples`
- 调整 `step_aware_alpha` 参数

## 🔍 结果理解

评估产生几个关键指标：

- **准确率**: 正确预测的百分比
- **改进幅度**: 相对基线的绝对改进
- **相对改进**: 相对基线的百分比改进
- **区分度**: Verifier 区分高低质量推理的能力
- **置信度**: 模型置信度分数

示例输出：
```
测试样本总数: 25
增强版 DIVERSE 准确率: 92.00% (23/25)
基线准确率: 92.00% (23/25)
准确率提升: +0.00%
区分度: 0.455
评估时间: 5.2 分钟
```

## 📚 实验记录

详细的实验过程和结果记录在：
- `增强版Verifier实验记录.md`: 完整的技术开发和测试过程
- `实验报告.md`: 实验总结报告

## 🤝 贡献指南

扩展本项目的方法：
1. 在 `config.py` 中添加新的提示模板
2. 在 `diverse_cot.py` 中实现额外的投票机制
3. 在比较函数中添加新的评估指标
4. 在 `verifier.py` 中尝试不同的 verifier 架构

## 📚 参考文献

- DIVERSE 原始论文："Diverse Verifier on Reasoning Step"
- AQuA 数据集："Program Induction by Rationale Generation"
- RoBERTa："A Robustly Optimized BERT Pretraining Approach"
- DeepSeek：大语言模型 API 服务

## ⚠ 重要说明

- API 调用可能产生费用 - 监控您的 DeepSeek API 使用情况
- 训练 verifier 需要大量计算资源
- 由于 API 响应变化，结果可能有所不同
- 保存训练好的模型以避免重新训练
- 在 WSL 环境中运行时，请使用 `wsl bash -c "..."` 包装命令
- 不要忘记激活虚拟环境

---

如有问题或疑难，请查看故障排除部分或参考源代码中的详细注释。

*最后更新：2025年6月23日* 