import os
import yaml
import csv
from datetime import datetime
from detectors import CRDDetector, FRFEDetector, UNRSDetector, LocalRXDetector, GlobalRXDetector
from utils.data_loader import load_mat_file
from utils.visualization import plot_comparison_metrics, plot_detection_results
from utils.metrics import compute_metrics
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_metrics_to_csv(results, save_path='results/metrics.csv'):
    headers = ['filename', 'detector', 'AUC', 'Precision', 'Recall', 'F1', 'OA']
    rows = []

    for filename, file_result in results.items():
        if 'metrics' not in file_result:
            continue
        for detector, metrics in file_result['metrics'].items():
            if metrics is None:
                continue
            row = [filename, detector] + [metrics.get(k) for k in ['AUC', 'Precision', 'Recall', 'F1', 'OA']]
            rows.append(row)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def run_comparison(dataset_path: str, config: dict):
    ensure_dir('results/figures')

    detectors = {
        'CRD': CRDDetector(),
        # 'FRFE': FRFEDetector(),
        'UNRS': UNRSDetector(),
        # 'LocalRX': LocalRXDetector(),
        'GlobalRX': GlobalRXDetector(),
    }

    results = {}
    for file in os.listdir(dataset_path):
        if not file.endswith('.mat'):
            continue

        try:
            data_dict = load_mat_file(os.path.join(dataset_path, file))
            if data_dict['data'] is None:
                print(f"跳过 {file}: 无有效数据")
                continue

            res = {}
            metrics_all = {}

            for name, det in detectors.items():
                try:
                    # Detect results
                    det_result = det.detect(data_dict['data'])
                    if isinstance(det_result, dict):
                        det_result = det_result.get('detection_map', 
                                                    det_result.get('result', 
                                                                  list(det_result.values())[0]))
                    res[name] = np.asarray(det_result, dtype=np.float32)

                    # Compute metrics if Ground Truth exists
                    if 'gt' in data_dict and data_dict['gt'].size > 0:
                        metrics_all[name] = compute_metrics(res[name], data_dict['gt'])
                    else:
                        metrics_all[name] = None

                except Exception as e:
                    print(f"{name} 处理 {file} 时出错: {str(e)}")
                    res[name] = np.zeros(data_dict['data'].shape[:2], dtype=np.float32)
                    metrics_all[name] = None

            # Save detection results and metrics
            results[file] = {
                'detections': res,
                'metrics': metrics_all
            }

            # Visualize and save results
            plot_detection_results(
                original=data_dict['data'],
                results=res,
                gt=data_dict['gt'] if 'gt' in data_dict and data_dict['gt'].size > 0 else None,
                filename=file
            )

        except Exception as e:
            print(f"处理文件 {file} 时发生错误: {str(e)}")
            continue

    save_metrics_to_csv(results, 'results/metrics.csv')

    # Call the plot_comparison_metrics function from visualization
    plot_comparison_metrics(results)

    return results
