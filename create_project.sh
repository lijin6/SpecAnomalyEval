#!/bin/bash
# 创建项目结构
mkdir -p detectors evaluation utils experiments results

# 创建核心文件
touch detectors/{crd,frfe,unrs}_detector.py
touch evaluation/metrics.py
touch utils/{data_loader,visualization}.py
touch experiments/{compare_detectors,config}.py
touch README.md

# 初始化README内容
echo "# Anomaly Detection Project" > README.md
echo -e "\n## Directory Structure" >> README.md
echo '```' >> README.md
tree >> README.md
echo '```' >> README.md

# 显示创建结果
echo "Project structure created:"
tree