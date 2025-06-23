#!/bin/bash

echo "=========================================="
echo "🧠 启动Verifier模型训练"
echo "=========================================="

# 进入项目目录
cd /c/Users/14421/nlp

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source nlp_env/bin/activate

# 检查虚拟环境是否激活成功
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
else
    echo "❌ 虚拟环境激活失败"
    exit 1
fi

# 显示Python版本和PyTorch信息
echo ""
echo "🐍 Python环境信息:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo ""
echo "🚀 开始运行Verifier训练..."
echo "=========================================="

# 运行verifier训练脚本
python train_verifier_only.py

echo ""
echo "🎉 Verifier训练完成!"
echo "==========================================" 