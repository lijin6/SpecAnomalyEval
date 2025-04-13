# Hyperspectral Anomaly Detection Comparison

该项目用于对比多种高光谱异常检测算法在不同数据集上的检测表现，并自动生成图像可视化和评价指标报告。

## 📦 项目结构

```
.
|-- data/                    # 存放数据集
├── detectors/               # 各检测算法实现（如 CRD, FRFE, UNRS, LocalRX, GlobalRX）
├── experiments/
│   └── compare_detectors.py # 主运行逻辑文件（可由 run.py 调用）
├── utils/
│   ├── data_loader.py       # 加载 .mat 格式高光谱数据
│   ├── metrics.py           # 计算 AUC、Precision 等评价指标
│   └── visualization.py     # 绘图与结果可视化
├── results/
│   ├── figures/             # 保存检测结果图像
│   ├── exp.yaml             # 实验配置及摘要结果
│   └── metrics.csv          # 各检测器指标结果汇总表
├── run.py                   # 项目入口脚本
└── README.md                # 项目说明
```

## 🧪 支持的检测器

- `CRDDetector`
- `FRFEDetector`
- `UNRSDetector`
- `LocalRXDetector`
- `GlobalRXDetector`

你可以在 `detectors/` 中自定义实现自己的检测算法，并添加到 `run_comparison` 函数中统一运行。

## 📁 数据要求

项目读取 `.mat` 格式的高光谱数据，数据应包含键：

- `data`：3D 高光谱图像数据，形状为 `(H, W, C)`
- `gt`（可选）：二维 ground truth mask，形状为 `(H, W)`，值为 0/1，表示异常位置

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主程序

```bash
python run.py
```

### 3. 查看结果

- `results/figures/` 中为每个样本对应的检测结果图；
- `results/metrics.csv` 中为各个检测器在每张图上的评价指标（AUC、Precision、Recall、F1、OA）；
- `results/exp.yaml` 为 YAML 格式的运行摘要。

## 📊 输出示例（CSV）

| filename     | detector | AUC   | Precision | Recall | F1    | OA    |
|--------------|----------|-------|-----------|--------|-------|-------|
| sample1.mat  | CRD      | 0.912 | 0.83      | 0.78   | 0.80  | 0.96  |
| sample1.mat  | FRFE     | 0.875 | 0.79      | 0.74   | 0.76  | 0.95  |
| ...          | ...      | ...   | ...       | ...    | ...   | ...   |

## 📌 注意事项

- 若 `.mat` 文件中缺少 `gt`，则跳过该样本的评价指标计算；
- 所有检测器返回结果必须是二维检测图（与原图空间尺寸一致）；
- 支持对整个数据集批量运行、保存可视化与指标。

## 🧑‍💻 贡献者

- 开发者：@lijin6
- 日期：2025年

---

欢迎大家反馈建议或提 PR 优化算法框架～
```
