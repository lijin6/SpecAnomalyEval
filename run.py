import sys
import os
from experiments.compare_detectors import run_comparison

if __name__ == "__main__":
    # 配置参数
    config = {
        'win_out': 5,
        'win_in': 3,
        'lambda_': 0.1,
        'dataset_path': 'DATASET/abu'
    }
    
    # 运行实验
    results = run_comparison(config['dataset_path'], config)
    print(f"实验完成，结果已保存到 results/ 目录")